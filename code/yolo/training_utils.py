"""Internal split, safe export and quality-sampling helpers."""
import hashlib
import heapq
import io
import json
import math
import os
import random
import shutil
import tempfile
from collections import Counter,defaultdict
from pathlib import Path

def _read_json(value):
    return json.loads(value or '[]')


def image_names(rows,source):
    """Preserve source filenames under their archive class, resolving label-stem collisions."""
    source=Path(source).resolve()
    names={}
    for row in rows:
        original=Path(row['image_path'])
        relative=original.relative_to(source)
        parts=relative.parts
        if parts[0]=='images':
            parts=parts[1:]
            if parts[0] in ('train','val','test'): parts=parts[1:]
        category=parts[0]
        names[row['uid']]=Path(category)/original.name
    counts=Counter(str(name.with_suffix('.txt')) for name in names.values())
    return {uid:Path('images')/(name.parent/uid/name.name if counts[str(name.with_suffix('.txt'))]>1 else name)
            for uid,name in names.items()}

def split_groups(rows, classes, seed=42, val_fraction=.2, quality_weights=None):
    """Seeded rare-label buckets in O(N log N); identical content stays together."""
    if not 0 < val_fraction < 1:
        raise ValueError('val_fraction 应在 0～1 之间')
    groups = defaultdict(list)
    for row in rows:
        if not row['image_sha']:
            raise ValueError('可训练样本缺少 image_sha')
        groups[row.get('split_group') or row['image_sha']].append(row)
    if len(groups) < 2:
        raise ValueError('至少需要两个可训练的图片内容分组，才能独立划分训练集和验证集')
    labels = {sha: {box['class'] for row in values for box in _read_json(
                    row['boxes_json'] if row['baseline_eligible'] else row['clean_boxes_json'])
                    if box['class'] in classes} for sha, values in groups.items()}
    frequency = Counter(label for values in labels.values() for label in values)
    needed = min(len(groups)-1, max(1, round(len(groups)*val_fraction)))
    if quality_weights is None:
        ordered = sorted(groups, key=lambda sha: hashlib.sha256(f'{seed}:{sha}'.encode()).hexdigest())
        group_weights=None
    else:
        group_weights={sha:sum(quality_weights[row['uid']] for row in values)/len(values) for sha,values in groups.items()}
        if any(not math.isfinite(w) or w<=0 for w in group_weights.values()):
            raise ValueError('质量抽样权重必须是正数')
        rng=random.Random(seed)
        # Validation/test draws use inverse quality weights; higher quality is
        # mildly favored in the complementary training pool. No repeated images.
        ranks={sha:-math.log(max(rng.random(),1e-15))*group_weights[sha] for sha in sorted(groups)}
        ordered=sorted(groups,key=lambda sha:(ranks[sha],sha))
    by_label=defaultdict(list)
    for sha in ordered:
        for label in labels[sha]:
            by_label[label].append(sha)
    # Reserve one training group per label, reusing multilabel groups when possible.
    reserved, protected_labels=set(),set()
    for label in sorted(frequency,key=lambda value:(frequency[value],value)):
        if label not in protected_labels:
            chosen=by_label[label][0] if group_weights is None else max(by_label[label],key=lambda sha:group_weights[sha])
            reserved.add(chosen)
            protected_labels.update(labels[chosen])
    if needed > len(groups)-len(reserved):
        raise ValueError('可用内容分组过少，无法同时按配置划分验证集并为各类别保留训练样本')
    # Bucket by the rarest label; this approximates multilabel stratification without
    # scanning every remaining group for each validation slot.
    buckets=defaultdict(list)
    bucket_totals=Counter()
    for sha in ordered:
        label=min(labels[sha],key=lambda value:(frequency[value],value)) if labels[sha] else ''
        bucket_totals[label]+=1
        if sha not in reserved:
            buckets[label].append(sha)
    targets={label:needed*count/len(groups) for label,count in bucket_totals.items()}
    quota={label:min(len(buckets[label]),math.floor(targets[label])) for label in bucket_totals}
    heap=[]
    for label in sorted(bucket_totals):
        if quota[label]<len(buckets[label]):
            heapq.heappush(heap,(quota[label]-targets[label],label))
    for _ in range(needed-sum(quota.values())):
        _,label=heapq.heappop(heap)
        quota[label]+=1
        if quota[label]<len(buckets[label]):
            heapq.heappush(heap,(quota[label]-targets[label],label))
    selected={sha for label,values in buckets.items() for sha in values[:quota[label]]}
    return {row['uid']: ('val' if sha in selected else 'train')
            for sha, values in groups.items() for row in values}

def quality_resample(rows, seed=42, score_field='quality_weighted_score', min_weight=.1, power=2):
    """Fixed deterministic systematic resampling with the same epoch length as cleaned."""
    if not 0 < min_weight <= 1 or not math.isfinite(power) or power <= 0:
        raise ValueError('重采样 min_weight 应在 (0,1]，power 应为正数')
    if not rows:
        raise ValueError('筛选后训练集为空')
    ordered = sorted(rows, key=lambda row: row['uid'])
    weights = []
    for row in ordered:
        value = row.get(score_field)
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 5:
            raise ValueError(f'训练样本 {row["uid"]} 的 {score_field} 缺失或超出 0～5；请先完成评分')
        weights.append(min_weight + (1-min_weight)*(float(value)/5)**power)
    count, total = len(ordered), sum(weights)
    start = random.Random(seed).random()*total/count
    selected, cursor, cumulative = [], 0, weights[0]
    for index in range(count):
        location = start + index*total/count
        while location >= cumulative and cursor < count-1:
            cursor += 1
            cumulative += weights[cursor]
        selected.append(ordered[cursor]['uid'])
    multiplicity = Counter(selected)
    initial_multiplicity=multiplicity.copy()
    lookup={row['uid']:row for row in ordered}
    labels={row['uid']:{box['class'] for box in _read_json(row.get('clean_boxes_json',row.get('boxes_json','[]')))}
            for row in ordered}
    frequency=Counter(label for values in labels.values() for label in values)
    coverage=Counter(label for uid in selected for label in labels[uid])
    # Fixed-size sampling can omit low-scoring rare classes. Repair only the missing
    # coverage, choosing its highest-quality image and a safe removable draw.
    for label in sorted(frequency,key=lambda value:(frequency[value],value)):
        if coverage[label]:
            continue
        chosen=min((row for row in ordered if label in labels[row['uid']]),
                   key=lambda row:(-row[score_field],row['uid']))['uid']
        donors=[index for index,uid in enumerate(selected) if all(
            coverage[value]-1+int(value in labels[chosen])>=1 for value in labels[uid])]
        if not donors:
            raise ValueError(f'固定重采样无法为类别 {label} 保留样本，请增加训练样本或调整权重')
        position=min(donors,key=lambda index:(lookup[selected[index]][score_field],selected[index],index))
        removed=selected[position]
        coverage.subtract(labels[removed])
        coverage.update(labels[chosen])
        selected[position]=chosen
    multiplicity=Counter(selected)
    return selected, {row['uid']: {'score': row[score_field], 'weight': weight,
                                 'multiplicity': multiplicity[row['uid']],
                                 'initial_multiplicity': initial_multiplicity[row['uid']],
                                 'class_coverage_repair_delta':multiplicity[row['uid']]-initial_multiplicity[row['uid']]}
                      for row, weight in zip(ordered, weights)}

def _valid_boxes(boxes, width, height, classes):
    if not boxes:
        raise ValueError('可训练正样本至少需要一个标签框')
    converted = []
    for box in boxes:
        coords = box.get('coords')
        if box.get('class') not in classes or not coords or len(coords) != 4 or not all(
                isinstance(value, (int, float)) and math.isfinite(value) for value in coords):
            raise ValueError('标签类别或框坐标不可转换')
        x1, y1, x2, y2 = coords
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            raise ValueError('标签框倒置、面积为零或越界')
        converted.append((classes.index(box['class']), (x1+x2)/(2*width), (y1+y2)/(2*height),
                          (x2-x1)/width, (y2-y1)/height))
    return converted

def _atomic_bytes(path, data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name+'.',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name)
            stream.write(data)
        os.replace(temporary,path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_yolo_labels(path, converted):
    path=Path(path)
    parts=list(path.parts)
    index=max(i for i,value in enumerate(parts) if value=='images')
    parts[index]='labels'
    label=Path(*parts).with_suffix('.txt')
    content=''.join(f'{cls} '+' '.join(f'{v:.8f}' for v in values)+'\n' for cls,*values in converted)
    _atomic_bytes(label,content.encode('utf-8'))


def _jpeg_without_exif(data):
    """Drop JPEG EXIF APP1 segments while preserving the compressed scan bytes."""
    if not data.startswith(b'\xff\xd8'):
        return None
    output=bytearray(data[:2]);cursor=2;removed=False
    while cursor<len(data):
        start=cursor
        if data[cursor]!=0xff:
            return None
        while cursor<len(data) and data[cursor]==0xff:
            cursor+=1
        if cursor>=len(data):return None
        marker=data[cursor];cursor+=1
        if marker in (0xda,0xd9):
            output.extend(data[start:])
            return bytes(output) if removed else None
        if marker in (0x01,0xd8,*range(0xd0,0xd8)):
            output.extend(data[start:cursor]);continue
        if cursor+2>len(data):return None
        length=int.from_bytes(data[cursor:cursor+2],'big');end=cursor+length
        if length<2 or end>len(data):return None
        if marker==0xe1 and data[cursor+2:end].startswith(b'Exif\x00\x00'):
            removed=True
        else:
            output.extend(data[start:end])
        cursor=end
    return None


def export_original_reference(row, path, boxes, classes, mode='link'):
    """Link existing source bytes and write independent YOLO labels; never edit sources."""
    if mode not in ('link','copy'): raise ValueError('storage.original_images仅支持link或copy')
    if row.get('normalize_exif'):
        data=Path(row['image_path']).read_bytes()
        if row.get('source_fingerprint'):
            from quality_pipeline import fingerprint
            if fingerprint(Path(row['image_path']),Path(row['label_path']) if row.get('label_path') else None)!=row['source_fingerprint']:
                raise ValueError('源图片/XML在扫描后变化：'+row['image_path'])
        elif hashlib.sha256(data).hexdigest()!=row['image_sha']:
            raise ValueError('源图片在扫描后变化：'+row['image_path'])
        normalized=_jpeg_without_exif(data)
        if normalized is not None:
            converted=_valid_boxes(boxes,row['width'],row['height'],classes)
            path=Path(path)
            if not path.is_file() or path.read_bytes()!=normalized:
                _atomic_bytes(path,normalized)
            _write_yolo_labels(path,converted)
            return 'jpeg_metadata_copy'
        _export_image(row,path,boxes,classes,force_reencode=True)
        return 'encoded_canvas_copy'
    source=Path(row['image_path']);path=Path(path)
    converted=_valid_boxes(boxes,row['width'],row['height'],classes)
    if row.get('source_fingerprint'):
        from quality_pipeline import fingerprint
        if fingerprint(source,Path(row['label_path']) if row.get('label_path') else None)!=row['source_fingerprint']:
            raise ValueError(f'源图片或XML在扫描后变化：{source}')
    else:
        digest=hashlib.sha256()
        with source.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
        if digest.hexdigest()!=row['image_sha']: raise ValueError(f'源图片在扫描后变化：{source}')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.is_symlink():
        # A prior export may have stopped after creating the link/image.
        same_file=path.is_file() and os.path.samefile(source,path)
        if not same_file and (not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['image_sha']):
            raise ValueError(f'已有原始对照图片与扫描缓存不同：{path}')
        kind='symlink' if path.is_symlink() else ('hardlink' if same_file else 'copy')
    elif mode=='copy':
        _atomic_bytes(path,source.read_bytes());kind='copy'
    else:
        try:
            os.link(source,path);kind='hardlink'
        except OSError:
            try:
                try: target=os.path.relpath(source,path.parent)
                except ValueError: target=str(source)
                path.symlink_to(target);kind='symlink'
            except OSError as exc:
                raise ValueError('无法创建省空间图片链接；Linux请检查目录权限，Windows需链接权限，'
                                 '或将storage.original_images改为copy（会额外占用一套原始图片空间）') from exc
    _write_yolo_labels(path,converted)
    return kind


def _export_image(row, path, boxes, classes, masks=(), force_reencode=False):
    from PIL import Image, ImageDraw
    width, height = row['width'], row['height']
    converted = _valid_boxes(boxes, width, height, classes)
    data=Path(row['image_path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != row['image_sha']:
        raise ValueError(f'图片内容在扫描后变化：{row["image_path"]}')
    with Image.open(io.BytesIO(data)) as opened:
        # VOC/manifest boxes use encoded pixels. Drop EXIF so YOLO uses this same canvas.
        image = opened.convert('RGB')
        if image.size != (width, height):
            raise ValueError(f'图片尺寸在扫描后变化：{row["image_path"]}')
        drawing = ImageDraw.Draw(image)
        for mask in masks:
            coords = mask['coords'] if isinstance(mask, dict) else mask
            if len(coords) != 4 or not all(isinstance(v, (int,float)) and math.isfinite(v) for v in coords):
                raise ValueError('遮罩坐标无效')
            x1, y1, x2, y2 = coords
            if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
                raise ValueError('遮罩坐标越界或面积为零')
            if any(min(x2, b['coords'][2]) > max(x1, b['coords'][0]) and
                   min(y2, b['coords'][3]) > max(y1, b['coords'][1]) for b in boxes):
                raise ValueError('待遮罩区域与保留目标相交，应送人工复核')
            # PIL endpoints are inclusive; ceil-1 matches half-open pixel rectangles.
            drawing.rectangle((math.floor(x1),math.floor(y1),math.ceil(x2)-1,math.ceil(y2)-1), fill=(114,114,114))
        path.parent.mkdir(parents=True, exist_ok=True)
        if not masks and not force_reencode and path.suffix.lower()==Path(row['image_path']).suffix.lower():
            exported=data
        else:
            buffer=io.BytesIO()
            image.save(buffer, format='JPEG', quality=95, subsampling=0)
            exported=buffer.getvalue()
        if not path.is_file() or path.read_bytes()!=exported:
            _atomic_bytes(path,exported)
    _write_yolo_labels(path,converted)
