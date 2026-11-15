"""Copy one category's selected images and annotations into a portable train/val dataset."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image
from simple_paths import load_json, resolve_path, write_json
from count_labels import IMAGE_EXTENSIONS
from copy_named_dataset import independent_copy
from quality_logging import log, progress_stage, run_logging, update_progress
from yolo.train import _dataset_snapshot
from yolo.training_utils import _atomic_bytes


def image_files(inputs):
    result = []
    for value in inputs if isinstance(inputs, list) else [inputs]:
        path = Path(value)
        if path.is_dir():
            result.extend(sorted(p.resolve() for p in path.rglob('*')
                                 if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS))
        elif path.suffix.lower() == '.txt':
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                if line.strip():
                    image = Path(line.strip())
                    result.append(image.resolve() if image.is_absolute() else (path.parent / image).resolve())
        else:
            raise ValueError(f'划分输入须为图片目录或TXT清单：{path}')
    return list(dict.fromkeys(result))


def source_paths(image, data):
    if data.get('image_root'):
        image_root = Path(data['image_root'])
        relative = image.relative_to(image_root)
        label = (Path(data['label_root']) / relative).with_suffix('.txt')
    else:
        positions = [index for index, name in enumerate(image.parts) if name == 'images']
        if not positions:
            raise ValueError(f'图片路径须包含images目录，或配置image_root和label_root：{image}')
        position = positions[-1]
        image_root = Path(*image.parts[:position + 1])
        relative = image.relative_to(image_root)
        label = (image_root.parent / 'labels' / relative).with_suffix('.txt')
    original_xml = (image_root.parent / 'annotations' / relative).with_suffix('.xml')
    return image_root, relative, label, original_xml


def read_boxes(label, class_count):
    boxes = []
    for number, line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        try:
            category, xc, yc, width, height = map(float, fields)
            valid = (category.is_integer() and 0 <= category < class_count
                     and all(math.isfinite(value) for value in (xc, yc, width, height))
                     and 0 <= xc <= 1 and 0 <= yc <= 1 and 0 < width <= 1 and 0 < height <= 1)
        except (ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(f'YOLO标签异常：{label}:{number}')
        boxes.append((int(category), xc, yc, width, height))
    return boxes


def voc_bytes(image, boxes, names):
    with Image.open(image) as opened:
        width, height = opened.size
    root = ET.Element('annotation')
    ET.SubElement(root, 'filename').text = image.name
    size = ET.SubElement(root, 'size')
    for key, value in (('width', width), ('height', height), ('depth', 3)):
        ET.SubElement(size, key).text = str(value)
    for category, xc, yc, bw, bh in boxes:
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = names[category]
        box = ET.SubElement(obj, 'bndbox')
        coordinates = ((xc - bw / 2) * width, (yc - bh / 2) * height,
                       (xc + bw / 2) * width, (yc + bh / 2) * height)
        for key, value in zip(('xmin', 'ymin', 'xmax', 'ymax'), coordinates):
            ET.SubElement(box, key).text = str(round(value, 6))
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)


def export_class(data_path, output, category='jyz_pl', workers=4, dry_run=False, single_class=False):
    if workers < 1:
        raise ValueError('workers须为正整数')
    data_path, output = resolve_path(data_path), resolve_path(output)
    data = _dataset_snapshot(data_path)
    names = data['names']
    if isinstance(names, dict):
        names = [names[index] for index in range(len(names))]
    if category not in names:
        raise ValueError(f'数据配置中没有类别：{category}')
    if not category or Path(category).name != category or '/' in category or '\\' in category or category in ('.', '..'):
        raise ValueError('类别名称须能作为单个目录名')
    target_id = names.index(category)
    export_names = [category] if single_class else names
    entries, uid_maps = [], {}
    with progress_stage('选择含目标类别的评分筛选样本'):
        for split in ('train', 'val'):
            for image in image_files(data[split]):
                image_root, relative, label, original_xml = source_paths(image, data)
                for root in (image_root, label.parent):
                    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
                        raise ValueError('导出目录请使用输入图片和标签之外的新目录')
                boxes = read_boxes(label, len(names))
                if not any(box[0] == target_id for box in boxes):
                    continue
                if single_class:
                    boxes = [(0, *box[1:]) for box in boxes if box[0] == target_id]
                if image_root not in uid_maps:
                    index = image_root.parent / '数据路径.json'
                    uid_maps[image_root] = {
                        (image_root.parent / row['image_path']).resolve(): row['uid']
                        for row in load_json(index)['filtered_samples']
                    } if index.is_file() else {}
                entries.append({'split': split, 'source': image, 'label': label,
                                'original_xml': original_xml, 'relative': relative, 'boxes': boxes,
                                'sample_id': uid_maps[image_root].get(image, '')})
                update_progress(len(entries))
    counts = Counter(entry['split'] for entry in entries)
    if not counts['train'] or not counts['val']:
        raise ValueError('所选类别的train或val为空，无法保留现有两组划分')
    collisions = Counter((entry['split'], entry['source'].stem.casefold()) for entry in entries)
    targets = set()
    for entry in entries:
        relative = Path(entry['split']) / category / entry['source'].name
        if collisions[entry['split'], entry['source'].stem.casefold()] > 1:
            # Preserve original filenames by retaining source subfolders for collisions.
            original = entry['relative']
            parts = original.parts[1:] if original.parts[0] in ('train', 'val', 'test') else original.parts
            relative = Path(entry['split']) / category / Path(*parts)
        key = relative.with_suffix('.txt').as_posix().casefold()
        if key in targets:
            raise ValueError(f'原文件名及来源子目录仍有冲突：{relative}')
        targets.add(key)
        entry['image_out'] = Path('images') / relative
        entry['label_out'] = (Path('labels') / relative).with_suffix('.txt')
        entry['xml_out'] = (Path('annotations') / relative).with_suffix('.xml')
    summary = {'source_data': str(data_path), 'category': category, 'classes': export_names,
               'source_classes': names,
               'samples': len(entries), 'split_counts': dict(counts),
               'storage': 'independent_copy', 'split_policy': 'preserve_source_train_val',
               'annotation_policy': 'single_category_id_zero' if single_class else 'preserve_all_training_classes_and_ids',
               'output': str(output)}
    log(f'{category}：train {counts["train"]:,}，val {counts["val"]:,}；输出 {output}')
    if dry_run:
        return summary
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)

    def copy(entry):
        independent_copy(entry['source'], output / entry['image_out'])
        if single_class:
            text = ''.join('0 ' + ' '.join(f'{value:.12g}' for value in box[1:]) + '\n'
                           for box in entry['boxes'])
            _atomic_bytes(output / entry['label_out'], text.encode('utf-8'))
        else:
            independent_copy(entry['label'], output / entry['label_out'])
        _atomic_bytes(output / entry['xml_out'], voc_bytes(entry['source'], entry['boxes'], export_names))

    with progress_stage('复制独立图片及训练标注', len(entries)), ThreadPoolExecutor(max_workers=workers) as executor:
        for completed, _ in enumerate(executor.map(copy, entries), 1):
            update_progress(completed)
    for split in ('train', 'val'):
        lines = ['./' + entry['image_out'].as_posix() + '\n' for entry in entries if entry['split'] == split]
        _atomic_bytes(output / f'{split}.txt', ''.join(lines).encode('utf-8'))
    with (output / '样本对应表.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'split', 'category', 'image_path', 'label_path', 'yolo_label_path',
                         'source_image_path', 'source_yolo_label_path', 'source_xml_path'])
        writer.writerows([entry['sample_id'], entry['split'], category, entry['image_out'].as_posix(),
                          entry['xml_out'].as_posix(), entry['label_out'].as_posix(),
                          str(entry['source']), str(entry['label']), str(entry['original_xml'])]
                         for entry in entries)
    write_json(output / 'data.yaml', {'train': 'train.txt', 'val': 'val.txt', 'names': export_names,
                                     'preserve_manifest_order': True})
    write_json(output / '导出汇总.json', {**summary, 'status': 'complete'})
    log('导出完成：' + str(output / 'data.yaml'))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='datasets/scored_filtered_v2/seed_42/data.yaml')
    parser.add_argument('--category', default='jyz_pl')
    parser.add_argument('--output', required=True, help='新建独立数据集目录')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true', help='只核对选择数量和输出路径')
    parser.add_argument('--single-class', action='store_true', help='仅保留所选类别的框并重编号为0，用于单类训练')
    args = parser.parse_args()
    try:
        with run_logging(resolve_path('logs'), 'export_scored_class', 10):
            export_class(args.data, args.output, args.category, args.workers, args.dry_run, args.single_class)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        parser.exit(2, str(exc) + '\n')


if __name__ == '__main__':
    main()
