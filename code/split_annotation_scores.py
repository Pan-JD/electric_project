"""Match model ratings to retained images and preserve their existing train/val split."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path, PurePosixPath

from simple_paths import resolve_path, write_json
from quality_experiment_scoring import read_annotation_scores

EXPERIMENT_CLASSES = ('fhz_f', 'fhz_h', 'jyz_pl', 'jyz_zc')


def filename(value):
    return str(value or '').replace('\\', '/').rsplit('/', 1)[-1].casefold()


def retained_index(path, classes=EXPERIMENT_CLASSES):
    """Use the hard-filter scoring table; image bytes and source XML are not read."""
    candidates = defaultdict(dict)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'样本ID', '原文件名', '类别', '图片路径'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('index须为硬筛样本对应的逐图评分.csv')
        for row in reader:
            parts = PurePosixPath(row['图片路径'].replace('\\', '/')).parts
            positions = [i for i, value in enumerate(parts) if value == 'images']
            if not positions:
                raise ValueError('图片路径中缺少images目录')
            image = PurePosixPath(*parts[positions[-1]:])
            if len(image.parts) < 4 or image.parts[1] not in ('train', 'val'):
                raise ValueError('图片路径须包含images/train或images/val')
            if image.parts[2] not in classes:
                continue
            label = PurePosixPath('annotations', *image.parts[1:]).with_suffix('.xml')
            item = {'uid': row['样本ID'], 'split': image.parts[1],
                    'image': image.as_posix(), 'label': label.as_posix()}
            for category in row['类别'].split(';'):
                if category in classes:
                    candidates[category, filename(row['原文件名'])][item['uid']] = item
    return candidates


def split_scores(inputs, index, output, default_class='', classes=EXPERIMENT_CLASSES):
    index, output = Path(index).resolve(), Path(output).resolve()
    candidates = retained_index(index, classes)
    assignments = defaultdict(list)
    counts = Counter()
    categories = set()
    sources = []
    for path in inputs:
        path = Path(path).resolve()
        incoming = read_annotation_scores(path)
        sources.append(str(path))
        counts.update(incoming['stats'])
        for record in incoming['records']:
            category = record['category'] or default_class
            if not category:
                raise ValueError('CSV缺少category时，请通过--category指定类别')
            if category not in classes:
                counts['excluded_class_rows'] += 1
                continue
            categories.add(category)
            choices = candidates.get((category, filename(record['image_path'])), {})
            if len(choices) != 1:
                counts['unmatched_rows' if not choices else 'ambiguous_rows'] += 1
                continue
            item = next(iter(choices.values()))
            if record['label_path'] and filename(record['label_path']) != filename(item['label']):
                counts['label_name_mismatch_rows'] += 1
                continue
            assignments[item['uid'], category].append((item, record))

    selected = []
    for (uid, category), values in assignments.items():
        signatures = {(record['legacy_score'], record['b1'], record['b2']) for _, record in values}
        if len(signatures) != 1:
            counts['conflicting_samples'] += 1
            continue
        counts['duplicate_rows'] += len(values) - 1
        item, record = values[0]
        selected.append({**item, 'category': category, 'score': record['legacy_score'],
                         'b1': record['b1'], 'b2': record['b2']})
    output.mkdir(parents=True, exist_ok=True)
    for excluded in ('aqmzc', 'wcaqm'):
        if excluded not in classes:
            for split in ('train', 'val'):
                (output / f'{excluded}_{split}_scores.csv').unlink(missing_ok=True)
    columns = ['image_path', 'label_path', 'category', 'score', 'sample_id']
    if any(row['b1'] is not None or row['b2'] is not None for row in selected):
        columns.extend(['b1', 'b2'])
    coverage = {}
    for category in sorted(categories):
        totals = Counter()
        for split in ('train', 'val'):
            rows = sorted((row for row in selected if row['category'] == category and row['split'] == split),
                          key=lambda row: row['image'])
            target = output / f'{category}_{split}_scores.csv'
            temporary = target.with_name(target.name + '.part')
            with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for row in rows:
                    values = {'image_path': row['image'], 'label_path': row['label'], 'category': category,
                              'score': row['score'], 'sample_id': row['uid'], 'b1': row['b1'], 'b2': row['b2']}
                    writer.writerow({key: values[key] for key in columns})
            os.replace(temporary, target)
            totals[split] = len(rows)
        retained = len({uid for (label, _), matches in candidates.items() if label == category for uid in matches})
        coverage[category] = {**totals, 'matched': sum(totals.values()), 'retained_samples': retained,
                              'without_rating': retained - sum(totals.values())}
    summary = {'sources': sources, 'index': str(index), 'counts': dict(counts),
               'matched_class_samples': len(selected), 'by_class': coverage,
               'classes': list(classes),
               'policy': 'selected classes and archive folders; category + original filename; preserve train/val'}
    previous_summary = output / '匹配汇总.json'
    if previous_summary.is_file():
        for key, value in json.loads(previous_summary.read_text(encoding='utf-8-sig')).items():
            if key.endswith('_category_boxes_changed_by_hard_filter'):
                summary[key] = value
    write_json(previous_summary, summary)
    # Missing scores concern only the categories supplied in this invocation.
    assigned = {(row['uid'], row['category']) for row in selected}
    missing = []
    for (category, _), matches in candidates.items():
        if category not in categories:
            continue
        for uid, item in matches.items():
            if (uid, category) not in assigned:
                missing.append({'image_path': item['image'], 'label_path': item['label'],
                                'category': category, 'split': item['split'], 'score': ''})
    missing_path = output.parent / '待补充标注评分.csv'
    try:
        stream = missing_path.open('w', encoding='utf-8-sig', newline='')
    except PermissionError:
        missing_path = output.parent / '待补充标注评分_四类.csv'
        print('原待补充CSV正被占用，更新结果另存为四类版本。', flush=True)
        stream = missing_path.open('w', encoding='utf-8-sig', newline='')
    with stream:
        writer = csv.DictWriter(stream, fieldnames=['image_path','label_path','category','split','score'])
        writer.writeheader()
        writer.writerows(sorted(missing, key=lambda row:(row['category'],row['split'],row['image_path'])))
    print(f'待补充评分 {len(missing):,} 条：{missing_path}', flush=True)
    for category, values in coverage.items():
        print(f'{category}：train {values["train"]:,}，val {values["val"]:,}，'
              f'已匹配 {values["matched"]:,} / 硬筛保留 {values["retained_samples"]:,}', flush=True)
    print(f'未匹配 {counts["unmatched_rows"]:,}，歧义 {counts["ambiguous_rows"]:,}，'
          f'冲突 {counts["conflicting_samples"]:,}，XML名称不一致 {counts["label_name_mismatch_rows"]:,}', flush=True)
    print('输出目录：' + str(output), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', nargs='+', required=True, help='同一版本的大模型评分CSV')
    parser.add_argument('--index', default='electronic_data/quality_scores_v2/逐图评分.csv',
                        help='硬筛后全量评分表，用于匹配保留样本和train/val归属')
    parser.add_argument('--output', default='run_results/label_detection/current')
    parser.add_argument('--category', default='', help='CSV没有category时使用的类别')
    parser.add_argument('--classes', nargs='+', default=list(EXPERIMENT_CLASSES), help='本次实验类别及归档目录范围')
    args = parser.parse_args()
    try:
        split_scores([resolve_path(path) for path in args.scores], resolve_path(args.index),
                     resolve_path(args.output), args.category, args.classes)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, str(exc) + '\n')


if __name__ == '__main__':
    main()
