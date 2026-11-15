"""Find validation images perceptually matching any training image."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
import hashlib
from pathlib import Path
import sqlite3
import time

import cv2
import numpy as np

from quality_duplicates import BKTree
from quality_metrics import image_features, near_match
from copy_named_dataset import duration, log


def extract_feature(row):
    data = Path(row['image']).read_bytes()
    checksum = hashlib.sha256(data).hexdigest()
    if checksum != row['image_sha256']:
        raise ValueError(f'图片内容已改变，请增加--rehash重新计算：{row["image"]}')
    image = cv2.imdecode(np.frombuffer(data, np.uint8),
                         cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f'图片解码失败：{row["image"]}')
    height, width = image.shape[:2]
    return checksum, {'width':width, 'height':height, **image_features(image)}


def load_features(representatives, cache, workers):
    cache.parent.mkdir(parents=True, exist_ok=True)
    keys = ('width','height','phash','dhash','gray64','color16')
    with closing(sqlite3.connect(cache, timeout=60)) as db:
        db.execute('''CREATE TABLE IF NOT EXISTS features(
            sha TEXT PRIMARY KEY,width INTEGER,height INTEGER,phash TEXT,dhash TEXT,
            gray64 BLOB,color16 BLOB)''')
        features = {row[0]:dict(zip(keys, row[1:])) for row in db.execute('SELECT * FROM features')
                    if row[0] in representatives}
        jobs = [row for checksum,row in representatives.items() if checksum not in features]
        reused = len(features)
        log(f'近重复图像特征共 {len(representatives):,} 份，缓存复用 {reused:,}，待计算 {len(jobs):,}')
        cv2.setNumThreads(1)
        started = last = time.monotonic()
        iterator = iter(jobs)
        pending = set()
        executor = ThreadPoolExecutor(max_workers=workers)
        try:
            while True:
                while len(pending) < 2*workers:
                    row = next(iterator, None)
                    if row is None:
                        break
                    pending.add(executor.submit(extract_feature, row))
                if not pending:
                    break
                ready, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in ready:
                    checksum, value = future.result()
                    features[checksum] = value
                    db.execute('INSERT OR REPLACE INTO features VALUES (?,?,?,?,?,?,?)',
                               (checksum, *[value[key] for key in keys]))
                if time.monotonic()-last >= 10:
                    db.commit()
                    computed = len(features)-reused
                    rate = computed/max(time.monotonic()-started, 0.001)
                    eta = duration((len(jobs)-computed)/rate) if rate else '等待磁盘响应'
                    log(f'近重复特征 {len(features):,}/{len(representatives):,} | 预计剩余 {eta}')
                    last = time.monotonic()
            db.commit()
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            db.commit()
    return features


def find_near_train_matches(train, validation, cache, rules, workers):
    train_representatives = {}
    validation_representatives = {}
    for row in train:
        train_representatives.setdefault(row['image_sha256'], row)
    for row in validation:
        validation_representatives.setdefault(row['image_sha256'], row)
    if not validation_representatives:
        return {}
    features = load_features({**validation_representatives, **train_representatives}, cache, workers)
    tree = BKTree()
    for checksum in sorted(train_representatives):
        tree.add(int(features[checksum]['phash'], 16), checksum)
    matches = {}
    started = last = time.monotonic()
    total = len(validation_representatives)
    log(f'开始逐组比对val与train近重复，SSIM阈值 {rules["near_ssim"]}')
    for count, (checksum, row) in enumerate(validation_representatives.items(), 1):
        current = features[checksum]
        candidates = sorted(tree.query(int(current['phash'], 16), rules['near_phash_distance']))
        for _, reference in candidates:
            evidence = near_match(current, features[reference], rules)
            if evidence:
                matches[checksum] = {'kind':'near', 'training_image':train_representatives[reference]['image'],
                                     **evidence}
                break
        if time.monotonic()-last >= 10:
            rate = count/max(time.monotonic()-started, 0.001)
            log(f'近重复检索 {count:,}/{total:,} 个图像内容组 | '
                f'已命中 {len(matches):,} 组 | 预计剩余 {duration((total-count)/rate)}')
            last = time.monotonic()
    return matches
