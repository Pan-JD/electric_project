"""Export validation images after exact and perceptual deduplication against training images."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import time

import yaml

from simple_paths import resolve_path, write_json
from copy_named_dataset import duration, independent_copy, log, sha, write_csv
from quality_metrics import load_config
from validation_similarity import find_near_train_matches
from yolo.training_utils import _atomic_bytes


def load_split(dataset, data, split, mapping):
    base = Path(data.get('path') or dataset)
    if not base.is_absolute():
        base = dataset/base
    listing = Path(data[split])
    listing = listing if listing.is_absolute() else base/listing
    if listing.suffix.lower() != '.txt' or not listing.is_file():
        raise ValueError(f'{split}须指向现有图片名单TXT：{listing}')
    rows = []
    seen = set()
    for line in listing.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        image = Path(line)
        image = Path(os.path.abspath(image if image.is_absolute() else listing.parent/image))
        if str(image) in seen:
            raise ValueError(f'{split}名单重复引用同一文件：{image}')
        seen.add(str(image))
        row = dict(mapping[str(image)])
        row['image'] = str(image)
        row['relative'] = image.relative_to(dataset/'images'/split).as_posix()
        checksum = row['image_sha256']
        if not re.fullmatch('[0-9a-f]{64}', checksum):
            raise ValueError(f'对应表中的图片SHA256无效：{image}')
        rows.append(row)
    if not rows:
        raise ValueError(f'{split}名单为空')
    return rows


def dataset_digest(rows):
    value = hashlib.sha256()
    for row in rows:
        value.update((row['image']+'\0'+row['image_sha256']+'\n').encode('utf-8'))
    return value.hexdigest()


def export_validation(dataset, name='val_no_train_duplicates', workers=4, rehash=False, ssim=0.95):
    dataset = resolve_path(dataset)
    if workers < 1 or not re.fullmatch('val_[a-zA-Z0-9_]+', name) or not 0 <= ssim <= 1:
        raise ValueError('workers须为正整数，SSIM须介于0和1之间，输出名称须为val_开头的英文、数字或下划线')
    with (dataset/'样本对应表.csv').open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    mapping = {os.path.abspath(dataset/row['image_path']):row for row in records}
    if len(mapping) != len(records):
        raise ValueError('样本对应表存在重复图片路径')
    data = yaml.safe_load((dataset/'data.yaml').read_text(encoding='utf-8-sig'))
    train = load_split(dataset, data, 'train', mapping)
    validation = load_split(dataset, data, 'val', mapping)
    log(f'训练 {len(train):,} 张，验证 {len(validation):,} 张；检查精确重复及SSIM近重复')
    if rehash:
        rows = train+validation
        last = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for count, checksum in enumerate(executor.map(sha, (Path(row['image']) for row in rows)), 1):
                rows[count-1]['image_sha256'] = checksum
                if time.monotonic()-last >= 10:
                    log(f'重新计算图片哈希 {count:,}/{len(rows):,}')
                    last = time.monotonic()
    else:
        log('复用独立副本生成时已经校验的图片哈希')
    train_by_hash = {}
    for row in train:
        train_by_hash.setdefault(row['image_sha256'], row)
    rules = load_config()
    rules['near_ssim'] = ssim
    rules = {key:rules[key] for key in ('near_phash_distance','near_dhash_distance',
             'near_ssim','near_color_mae','near_aspect_relative_difference')}
    identity = {'dataset':str(dataset), 'name':name, 'algorithm':'image_sha256_phash_dhash_ssim_v1',
                'train_digest':dataset_digest(train), 'val_digest':dataset_digest(validation), 'near_rules':rules}
    state = dataset/'_cache'/(name+'.json')
    targets = [dataset/folder/name for folder in ('images','labels','annotations')]
    for target in targets:
        if target.is_symlink():
            raise ValueError(f'生成目录是符号链接，请先核对：{target}')
        target.resolve().relative_to(dataset)
    if state.is_file():
        saved = json.loads(state.read_text(encoding='utf-8-sig'))
        if any(saved.get(key) != identity[key] for key in ('dataset','name','train_digest','val_digest')):
            raise ValueError('该输出名称已用于其他数据版本，请使用新的--name保存本次结果')
    elif any(target.exists() for target in targets):
        raise ValueError('输出目录已有其他内容，请指定新的--name')
    candidates = [row for row in validation if row['image_sha256'] not in train_by_hash]
    matches = {checksum:{'kind':'exact','training_image':row['image']}
               for checksum,row in train_by_hash.items()}
    matches.update(find_near_train_matches(
        train, candidates, dataset/'_cache/val_similarity_v1.sqlite', rules, workers))
    retained = [row for row in validation if row['image_sha256'] not in matches]
    duplicates = [row for row in validation if row['image_sha256'] in matches]
    if not retained:
        raise ValueError('验证图片全部与训练集重复，无法生成非空评估集')
    write_json(state, identity)
    write_json(dataset/(name+'_汇总.json'), {**identity, 'status':'building'})
    kinds = Counter(matches[row['image_sha256']]['kind'] for row in duplicates)
    log(f'精确重复 {kinds["exact"]:,} 张，近重复 {kinds["near"]:,} 张，待导出 {len(retained):,} 张')
    for row in retained:
        row['bytes'] = Path(row['image']).stat().st_size
    for folder in ('images','labels','annotations'):
        for parent in {Path(row['relative']).parent for row in retained}:
            (dataset/folder/name/parent).mkdir(parents=True, exist_ok=True)

    def copy(row):
        relative = Path(row['relative'])
        image = dataset/'images'/name/relative
        actual_sha = independent_copy(Path(row['image']), image)
        if actual_sha != row['image_sha256']:
            raise ValueError(f'图片与已有哈希记录不同，请使用--rehash及新的--name重新筛选：{row["image"]}')
        label = dataset/'labels'/name/relative.with_suffix('.txt')
        xml = dataset/'annotations'/name/relative.with_suffix('.xml')
        independent_copy(dataset/row['yolo_label_path'], label)
        independent_copy(dataset/row['label_path'], xml)
        return row['bytes']

    started = last = time.monotonic()
    total_bytes = sum(row['bytes'] for row in retained)
    completed = completed_bytes = last_bytes = 0
    pending = set()
    iterator = iter(retained)
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        while True:
            while len(pending) < 2*workers:
                row = next(iterator, None)
                if row is None:
                    break
                pending.add(executor.submit(copy, row))
            if not pending:
                break
            ready, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in ready:
                completed_bytes += future.result()
                completed += 1
            now = time.monotonic()
            if now-last >= 10:
                rate = (completed_bytes-last_bytes)/(now-last)
                eta = duration((total_bytes-completed_bytes)/rate) if rate > 0 else '等待磁盘响应'
                log(f'复制并校验 {completed:,}/{len(retained):,} 张 | '
                    f'速度 {rate/1024**2:.1f} MiB/秒 | 预计剩余 {eta}')
                last, last_bytes = now, completed_bytes
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
    listing = dataset/(name+'.txt')
    _atomic_bytes(listing, ''.join('./images/'+name+'/'+row['relative']+'\n' for row in retained).encode('utf-8'))
    evaluation = {**data, 'path':str(dataset), 'val':listing.name}
    # Anchor other relative split entries to their existing dataset base.
    original_base = Path(data.get('path') or dataset)
    if not original_base.is_absolute():
        original_base = dataset/original_base
    for split in ('train','test'):
        if isinstance(data.get(split), str):
            value = Path(data[split])
            evaluation[split] = str(value if value.is_absolute() else original_base/value)
    evaluation_path = dataset/('data_'+name+'.yaml')
    write_json(evaluation_path, evaluation)
    # Upgrade this generated result in place after publishing the new, complete list.
    for row in duplicates:
        relative = Path(row['relative'])
        for folder, suffix in (('images',relative.suffix),('labels','.txt'),('annotations','.xml')):
            target_root = dataset/folder/name
            target = target_root/relative.with_suffix(suffix)
            if target.exists() or target.is_symlink():
                target.resolve().relative_to(target_root.resolve())
                if target.is_symlink():
                    raise ValueError(f'生成目录中存在符号链接，请先核对：{target}')
                target.unlink()
    write_csv(dataset/(name+'_去重记录.csv'),
              ['验证图片', '验证XML', '训练集对应图片', '重复类型', 'SSIM', 'pHash距离', 'dHash距离', '颜色差', '图片SHA256'],
              [(row['image'],str(dataset/row['label_path']),
                matches[row['image_sha256']]['training_image'],
                '精确重复' if matches[row['image_sha256']]['kind']=='exact' else '近重复',
                *[matches[row['image_sha256']].get(key,'') for key in ('ssim','phash_distance','dhash_distance','color_mae')],
                row['image_sha256']) for row in duplicates])
    before, exact, near = Counter(), Counter(), Counter()
    exact_rows = [row for row in duplicates if matches[row['image_sha256']]['kind']=='exact']
    near_rows = [row for row in duplicates if matches[row['image_sha256']]['kind']=='near']
    for rows, counts in ((validation,before),(exact_rows,exact),(near_rows,near)):
        for row in rows:
            for category in row['classes'].split(';'):
                if category:
                    counts[category] += 1
    categories = sorted(before)
    write_csv(dataset/(name+'_各类统计.csv'), ['类别', '原验证图片数', '精确重复图片数', '近重复图片数', '保留图片数'],
              [['总体',len(validation),len(exact_rows),len(near_rows),len(retained)]]+
              [[category,before[category],exact[category],near[category],before[category]-exact[category]-near[category]]
               for category in categories])
    summary = {**identity, 'status':'complete', 'hash_source':'recomputed' if rehash else 'verified_copy_manifest',
               'train_images':len(train), 'original_val_images':len(validation),
               'duplicates':len(duplicates), 'retained_val_images':len(retained),
               'exact_duplicates':kinds['exact'], 'near_duplicates':kinds['near'],
               'evaluation_config':str(evaluation_path), 'seconds':round(time.monotonic()-started,3)}
    write_json(dataset/(name+'_汇总.json'), summary)
    log(f'完成，保留 {len(retained):,} 张，评估配置 {evaluation_path}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='/mnt/data/panjinduo/datasets/electronic_data/sample_original_named_seed42')
    parser.add_argument('--name', default='val_no_train_duplicates')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--rehash', action='store_true', help='原图修改后重新计算train/val全部图片SHA256')
    parser.add_argument('--ssim', type=float, default=0.95, help='近重复SSIM阈值，默认0.95')
    args = parser.parse_args()
    try:
        export_validation(**vars(args))
    except KeyboardInterrupt:
        log('已中断，重复相同命令可校验并复用已完成文件')
        raise SystemExit(130)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, str(exc)+'\n')


if __name__ == '__main__':
    main()
