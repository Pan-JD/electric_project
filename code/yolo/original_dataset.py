"""Export an unfiltered six-class YOLO baseline from cached source annotations."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
import hashlib
import json
from pathlib import Path

from simple_paths import write_json
from quality_experiment_pipeline import REASON_NAMES
from quality_experiment_rules import baseline_record
from quality_logging import log, progress_stage, update_progress
from dataset_splits import split_settings, split_directory, write_seed_split
from yolo.training_utils import export_original_reference, image_names


def _csv(path, headers, rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(headers);writer.writerows(rows)


def source_fingerprints(db):
    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='processed_files'").fetchone():
        return dict(db.execute('SELECT image_path,fingerprint FROM processed_files'))
    return {}


def reference_original_dataset(dataset, classes, split):
    """Build project split lists using the existing independent original image pool."""
    dataset=Path(dataset).resolve()
    data=json.loads((dataset/'data.yaml').read_text(encoding='utf-8-sig'))
    if data['names']!=list(classes):
        raise ValueError('原名副本的类别顺序须与筛选classes一致，以对应现有YOLO标签编号')
    index=json.loads((dataset/'图片索引.json').read_text(encoding='utf-8-sig'))
    rows=[{**row,'baseline_eligible':True} for row in index['samples']]
    images={row['uid']:dataset/row['export_image'] for row in rows}
    if any(not image.resolve().is_relative_to(dataset) for image in images.values()):
        raise ValueError('原始图片索引越过数据集目录')
    seeds,fraction=split_settings(split)
    split_output=split_directory(split,dataset)
    paths={};summaries={}
    for seed in seeds:
        result=write_seed_split(dataset,rows,images,classes,seed,fraction,original=True,
                                validation_cleanup=False,split_output=split_output)
        paths[str(seed)]=result['yaml'].as_posix()
        summaries[str(seed)]=result['summary']
        log(f"original seed={seed}：训练 {len(result['train_uids']):,}，验证 {len(result['holdout_uids']):,}（引用现有原名副本）")
    write_json(split_output/'划分参数.json',split)
    return {'output':str(dataset),'images':len(rows),'reused':True,
            'storage_counts':index.get('storage_counts',{'copy':len(rows)}),
            'splits_output':str(split_output),'splits':paths,'summaries':summaries}


def export_original_dataset(db, output, source, classes, split, mode='link', workers=8):
    """Independent of clean_status: retain duplicates, low resolution and all valid boxes."""
    output, source=Path(output).resolve(),Path(source).resolve()
    if output==source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('原始YOLO输出须与by_label图片目录独立')
    seeds,fraction=split_settings(split)
    split_output=split_directory(split,output)
    catalog=hashlib.sha256()
    for row in sorted(db.execute('SELECT uid,image_sha FROM samples NOT INDEXED'),key=lambda row:row[0]):
        catalog.update(f'{row[0]}:{row[1]}\n'.encode())
    identity={'source':str(source),'classes':list(classes),'storage':{'original_images':mode},
              'conversion_version':1,'catalog_sha256':catalog.hexdigest()}
    index=output/'图片索引.json';state=output/'_导出状态.json'
    reused=index.is_file()
    if reused:
        saved=json.loads(index.read_text(encoding='utf-8-sig'))
        if saved.get('identity')!=identity:
            raise ValueError(f'已有原始基线目录的样本或类别不同，请更换output：{output}')
        rows=saved['samples'];storage=Counter(saved['storage_counts']);conversion=saved['conversion']
        compacted=False
        for row in rows:
            boxes=json.loads(row['boxes_json'])
            if any(set(box)-{'class','coords','index'} for box in boxes):
                row['boxes_json']=json.dumps([{key:box[key] for key in ('class','coords','index') if key in box}
                                              for box in boxes],ensure_ascii=False)
                compacted=True
        if compacted:
            saved['samples']=rows
            write_json(index,saved)
        log('复用原始全量基线：'+str(output))
    else:
        if output.exists() and any(output.iterdir()):
            if not state.is_file() or json.loads(state.read_text(encoding='utf-8-sig'))!=identity:
                raise ValueError(f'原始基线目录已有其他内容：{output}')
        write_json(state,identity)
        rows=[];details=[];storage=Counter();skip=Counter();skipped_boxes=0
        fingerprints=source_fingerprints(db)
        total=db.execute('SELECT COUNT(*) FROM samples').fetchone()[0]
        with progress_stage('整理原始全量标注',total):
            for number,record in enumerate(db.execute('SELECT uid,payload_json FROM samples NOT INDEXED'),1):
                payload=json.loads(record['payload_json'])
                decision=baseline_record(payload,classes)
                selected=decision['boxes'];skipped_boxes+=decision['skipped_boxes']
                details.append([record['uid'],payload['path'],payload['label_path'],
                                '纳入' if selected else '跳过',len(selected),decision['skipped_boxes'],
                                REASON_NAMES.get(decision['reason'],decision['reason'])])
                if not selected:
                    skip[decision['reason']]+=1
                    update_progress(number)
                    continue
                row={'uid':record['uid'],'image_path':payload['path'],'label_path':payload['label_path'],
                     'image_sha':payload['sha256'],'split_group':payload['sha256'],
                     'width':payload['width'],'height':payload['height'],'baseline_eligible':True,
                     'boxes_json':json.dumps(selected,ensure_ascii=False),'normalize_exif':decision['normalize_exif'],
                     'source_fingerprint':fingerprints.get(payload['path'])}
                rows.append(row);update_progress(number)
        if not rows:
            raise ValueError('没有可转换为YOLO的选定类别图片')
        rows.sort(key=lambda row:row['uid'])
        names=image_names(rows,source)
        for row in rows:
            row['export_image']=names[row['uid']].as_posix()
        log(f'原始转换：{len(rows):,} 张；方向元数据需规范化 {sum(row["normalize_exif"] for row in rows):,} 张；其他直接建立链接。')
        with progress_stage('原始图片链接与标签导出',len(rows),note=f'workers={workers}'),ThreadPoolExecutor(max_workers=workers) as executor:
            pending=set();iterator=iter(rows);done=0
            def export(row):
                return export_original_reference(row,output/row['export_image'],json.loads(row['boxes_json']),classes,mode)
            def fill():
                while len(pending)<2*workers:
                    row=next(iterator,None)
                    if row is None:break
                    pending.add(executor.submit(export,row))
            fill()
            while pending:
                ready,pending=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                for future in ready:
                    storage[future.result()]+=1;done+=1;update_progress(done)
                fill()
        conversion={'merged_samples':total,'included_images':len(rows),'skipped_images':sum(skip.values()),
                    'skipped_boxes':skipped_boxes,'skip_reasons':dict(skip),
                    'archive_images':db.execute('SELECT COUNT(*) FROM occurrences').fetchone()[0]}
        _csv(output/'原始转换清单.csv',['样本ID','原图片路径','原标签路径','转换状态','输出框数','无法转换框数','跳过原因'],details)
        write_json(index,{'identity':identity,'samples':rows,'storage_counts':dict(storage),'conversion':conversion})
        state.unlink(missing_ok=True)
    images={row['uid']:output/row['export_image'] for row in rows}
    stats=[];paths={};summaries={};lookup={row['uid']:row for row in rows}
    for seed in seeds:
        result=write_seed_split(output,rows,images,classes,seed,fraction,original=True,
                                validation_cleanup=False,split_output=split_output)
        paths[str(seed)]=result['yaml'].as_posix();summaries[str(seed)]=result['summary']
        log(f'original seed={seed}：训练 {len(result["train_uids"]):,}，验证 {len(result["holdout_uids"]):,}（保留完整可转换样本）')
        for category in classes:
            counts=[]
            for uids in (result['train_uids'],result['holdout_uids']):
                boxes=[[b for b in json.loads(lookup[uid]['boxes_json']) if b['class']==category] for uid in uids]
                counts.extend([sum(bool(b) for b in boxes),sum(len(b) for b in boxes)])
            stats.append([seed,category,*counts])
    _csv(split_output/'数据集统计.csv',['划分种子','类别','训练图片数','训练框数','验证图片数','验证框数'],stats)
    write_json(split_output/'划分参数.json',split)
    log(f'原始基线：纳入 {len(rows):,} 张，无法转换跳过 {conversion["skipped_images"]:,} 张；来源归档 {conversion["archive_images"]:,}，合并样本 {conversion["merged_samples"]:,}')
    return {'output':str(output),'images':len(rows),'reused':reused,'storage_counts':dict(storage),
            'conversion':conversion,'splits_output':str(split_output),'splits':paths,'summaries':summaries}


