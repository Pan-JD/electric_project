"""Select scored train/val image paths while reusing the existing YOLO dataset."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from simple_paths import load_json, resolve_path, write_json
from score_dataset import _source_path
from quality_logging import log, progress_stage, run_logging, update_progress


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.part')
    temporary.write_text(text, encoding='utf-8')
    os.replace(temporary, path)


def write_training_labels(entries, source, splits, source_classes, classes):
    """Write small four-class labels; image files and original labels remain in place."""
    mapping = {index: classes.index(name) for index, name in enumerate(source_classes) if name in classes}
    written = set()
    for entry in entries:
        relative = entry['image'].relative_to(source / 'images').with_suffix('.txt')
        original = source / 'labels' / relative
        target = splits / 'labels' / relative
        lines = []
        for number, line in enumerate(original.read_text(encoding='utf-8-sig').splitlines(), 1):
            values = line.split()
            if not values:
                continue
            if len(values) != 5 or not values[0].isdigit() or int(values[0]) >= len(source_classes):
                raise ValueError(f'YOLO标签格式异常：{original}:{number}')
            if int(values[0]) in mapping:
                lines.append(str(mapping[int(values[0])]) + ' ' + ' '.join(values[1:]) + '\n')
        if not lines:
            raise ValueError(f'图片缺少本次实验类别的有效框：{original}')
        write_text(target, ''.join(lines))
        written.add(target)
    # Remove only this experiment's obsolete generated labels after successful export.
    label_root = splits / 'labels'
    for target in label_root.rglob('*.txt'):
        if target not in written:
            target.unlink()
    for folder in sorted((p for p in label_root.rglob('*') if p.is_dir()),
                         key=lambda p: len(p.parts), reverse=True):
        if not any(folder.iterdir()):
            folder.rmdir()


def export_dataset(entries, source, splits, output, classes, workers):
    """Copy selected images and four-class labels, retaining names and split membership."""
    output.mkdir(parents=True, exist_ok=False)

    def copy(entry):
        relative = entry['image'].relative_to(source / 'images')
        target = output / 'images' / relative
        label = relative.with_suffix('.txt')
        xml = relative.with_suffix('.xml')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry['image'], target)
        for kind, relative_label in (('labels', label), ('annotations', xml)):
            (output / kind / relative_label).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(splits / 'labels' / label, output / 'labels' / label)
        tree = ET.parse(source / 'annotations' / xml)
        root = tree.getroot()
        for obj in list(root.findall('object')):
            if obj.findtext('name') not in classes:
                root.remove(obj)
        for node in root.findall('path'):
            root.remove(node)
        if root.find('filename') is not None:
            root.find('filename').text = target.name
        tree.write(output / 'annotations' / xml, encoding='utf-8', xml_declaration=True)
        return {**entry, 'source_image': entry['image'], 'image': target}

    copied = []
    with progress_stage('复制综合评分筛选数据', len(entries)), ThreadPoolExecutor(max_workers=workers) as executor:
        for completed, entry in enumerate(executor.map(copy, entries), 1):
            copied.append(entry)
            update_progress(completed)
    for split in ('train', 'val'):
        write_text(output / (split + '.txt'), ''.join(
            './' + row['image'].relative_to(output).as_posix() + '\n'
            for row in copied if row['split'] == split))
    write_json(output / 'data.yaml', {'train': 'train.txt', 'val': 'val.txt', 'names': classes,
                                      'preserve_manifest_order': True})
    with (output / '样本对应表.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'split', 'category', 'image_path', 'label_path', 'source_image_path'])
        for row in copied:
            relative = row['image'].relative_to(output)
            xml = (Path('annotations') / relative.relative_to('images')).with_suffix('.xml')
            writer.writerow([row['uid'], row['split'], ';'.join(row['classes']), relative.as_posix(),
                             xml.as_posix(), row['source_image'].as_posix()])
    return copied


def filter_scored(config):
    scores = resolve_path(config['scores'])
    reports = resolve_path(config['reports'])
    splits = resolve_path(config['split_output'])
    dataset_output = resolve_path(config['dataset_output']) if config.get('dataset_output') else None
    export_workers = config.get('export_workers', 4)
    if not isinstance(export_workers, int) or isinstance(export_workers, bool) or export_workers < 1:
        raise ValueError('export_workers须为正整数')
    score_config = load_json(scores / '评分配置.json')
    source = resolve_path(score_config['input'])
    score_csv = scores / '逐图评分.csv'
    paths = load_json(source / '数据路径.json')
    classes = score_config.get('classes') or paths['classes']
    included_ids = [index for index, name in enumerate(paths['classes']) if name in classes]
    if len(included_ids) != len(classes):
        raise ValueError('评分类别与输入数据集的类别名称不一致')
    minimum_scores = config.get('minimum_scores', {})
    maximum_scores = config.get('maximum_scores', {})
    require_complete = config.get('require_complete_scores', False)
    if not isinstance(require_complete, bool):
        raise ValueError('require_complete_scores须为true或false')
    if not isinstance(minimum_scores, dict) or not isinstance(maximum_scores, dict):
        raise ValueError('minimum_scores和maximum_scores须为评分列名与分数的对应表')
    if not minimum_scores and not maximum_scores:
        raise ValueError('请设置minimum_scores或maximum_scores')
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) or not 0 <= value <= 5
           for value in [*minimum_scores.values(), *maximum_scores.values()]):
        raise ValueError('评分上下限须在0～5之间')
    if any(minimum > maximum_scores[column] for column, minimum in minimum_scores.items()
           if column in maximum_scores):
        raise ValueError('评分下限不能高于上限')
    score_columns = list(dict.fromkeys([*minimum_scores, *maximum_scores]))
    for folder in (reports, splits):
        if folder == source or folder.is_relative_to(source):
            raise ValueError('筛选记录和图片列表请保存在项目目录')
    if dataset_output:
        for folder in (source, reports, splits):
            if (dataset_output == folder or dataset_output.is_relative_to(folder)
                    or folder.is_relative_to(dataset_output)):
                raise ValueError('dataset_output须与输入数据和项目记录使用互不包含的新目录')
        if dataset_output.exists():
            raise ValueError(f'新数据集目录已存在，请使用新的dataset_output：{dataset_output}')

    lookup = {entry['uid']: entry for entry in paths['filtered_samples']}
    entries, decisions, seen = [], [], set()
    with score_csv.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if set(score_columns) - set(reader.fieldnames or []):
            raise ValueError('筛选配置中的评分列名不存在于逐图评分.csv')
        if require_complete and '评分覆盖率(%)' not in (reader.fieldnames or []):
            raise ValueError('完整评分筛选需要评分覆盖率(%)列')
        for row in reader:
            uid = row['样本ID']
            if uid in seen:
                raise ValueError(f'逐图评分中样本ID重复：{uid}')
            seen.add(uid)
            reasons = []
            if uid not in lookup:
                raise ValueError(f'评分样本不在输入图片索引中：{uid}')
            image = _source_path(source, lookup[uid]['image_path'])
            relative = image.relative_to(source / 'images')
            if len(relative.parts) < 3 or relative.parts[0] not in ('train', 'val'):
                raise ValueError(f'输入须为已划分train/val的原名数据集：{image}')
            if relative.parts[1] not in classes:
                reasons.append('归档类别不在本次实验范围')
            if require_complete:
                try:
                    coverage = float(row['评分覆盖率(%)'])
                except (TypeError, ValueError):
                    coverage = float('nan')
                if not math.isfinite(coverage) or not math.isclose(coverage, 100., abs_tol=1e-6):
                    reasons.append('评分未覆盖全部指标')
            for column in score_columns:
                try:
                    value = float(row[column])
                except (TypeError, ValueError):
                    value = float('nan')
                if not math.isfinite(value) or not 0 <= value <= 5:
                    reasons.append(column + '缺失或无效')
                elif column in minimum_scores and value < minimum_scores[column]:
                    reasons.append(f'{column}={value:g}<{minimum_scores[column]:g}')
                elif column in maximum_scores and value > maximum_scores[column]:
                    reasons.append(f'{column}={value:g}>{maximum_scores[column]:g}')
            decisions.append([uid, row['原文件名'], row['类别'], row['图像质量分'],
                              row['综合质量分'], '筛除' if reasons else '保留', '；'.join(reasons)])
            if reasons:
                continue
            categories = sorted(set(row['类别'].split(';')) & set(classes))
            if not categories:
                raise ValueError(f'样本没有本次实验类别：{uid}')
            entries.append({'uid': uid, 'image': image, 'split': relative.parts[0], 'classes': categories})
    counts = Counter(entry['split'] for entry in entries)
    if not counts['train'] or not counts['val']:
        raise ValueError('筛选后训练集或验证集为空，请调整评分上下限')
    identity = {'source': str(source), 'scores': str(scores),
                'reports': str(reports), 'split_output': str(splits), 'classes': classes,
                'minimum_scores': minimum_scores, 'maximum_scores': maximum_scores,
                'require_complete_scores': require_complete,
                'score_sha256': hashlib.sha256(score_csv.read_bytes()).hexdigest(),
                'dataset_output': str(dataset_output) if dataset_output else None,
                'split_policy': 'preserve_source_train_val',
                'storage': 'independent_copy' if dataset_output else 'reference'}
    marker = reports / '筛选配置.json'
    own_lists = marker.exists() and load_json(marker).get('split_output') == str(splits)
    if splits.exists() and any(splits.iterdir()) and not own_lists:
        raise ValueError(f'划分目录已有其他实验内容，请修改split_output：{splits}')
    write_json(marker, identity)
    write_json(reports / '筛选汇总.json', {'status': 'building', 'training_yaml': str(splits / 'data.yaml')})

    write_training_labels(entries, source, splits, paths['classes'], classes)
    image_root, label_root = source / 'images', splits / 'labels'
    if dataset_output:
        entries = export_dataset(entries, source, splits, dataset_output, classes, export_workers)
        image_root, label_root = dataset_output / 'images', dataset_output / 'labels'
    for split in ('train', 'val'):
        images = [entry['image'] for entry in entries if entry['split'] == split]
        write_text(splits / (split + '.txt'), ''.join(path.as_posix() + '\n' for path in images))
    write_json(splits / 'data.yaml', {'train': 'train.txt', 'val': 'val.txt', 'names': classes,
                                     'image_root': image_root.as_posix(),
                                     'label_root': label_root.as_posix(),
                                     'preserve_manifest_order': True})
    record = reports / '筛选记录.csv'
    with record.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['样本ID', '原文件名', '类别', '图像质量分', '综合质量分', '处理结果', '筛除原因'])
        writer.writerows(decisions)
    summary = {'input_images': len(decisions), 'kept_images': len(entries),
               'excluded_images': len(decisions) - len(entries), 'split_counts': dict(counts),
               'class_counts': dict(Counter(category for entry in entries for category in entry['classes'])),
               'dataset': str(dataset_output or source), 'source_dataset': str(source),
               'storage': 'independent_copy' if dataset_output else 'reference_images_generated_labels',
               'classes': classes, 'selected_class_ids': list(range(len(classes))),
               'require_complete_scores': require_complete,
               'training_yaml': str(splits / 'data.yaml'), 'status': 'complete'}
    write_json(reports / '筛选汇总.json', summary)
    if dataset_output:
        write_json(dataset_output / '导出汇总.json', summary)
    log(f'评分样本 {len(decisions):,} 张，训练引用 {counts["train"]:,} 张，验证引用 {counts["val"]:,} 张')
    log('训练配置：' + str(splits / 'data.yaml'))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/filter_scored.json')
    args = parser.parse_args()
    try:
        with run_logging(resolve_path('logs'), 'filter_scored', 10):
            filter_scored(load_json(args.config))
    except (OSError, ValueError, KeyError, IndexError) as exc:
        parser.exit(2, str(exc) + '\n')


if __name__ == '__main__':
    main()
