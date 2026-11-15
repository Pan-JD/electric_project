"""Apply the existing hard rules to classified VOC files using same-volume moves."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import xml.etree.ElementTree as ET

from simple_paths import load_json, write_json
from quality_experiment_rules import assess_record, FILTER_RULES_VERSION
from quality_metrics import read_labels, IMAGE_EXTENSIONS

import cv2
import numpy as np
from PIL import Image


REASONS = {
    'image_unavailable': '图像不可用', 'annotation_unavailable': 'XML缺失或损坏',
    'annotation_frame': '图像与XML坐标系不一致', 'low_resolution': '图像任一边小于阈值',
    'box_geometry': '标注框几何异常', 'tiny_target': '缩放后目标过小',
    'large_box': '标注框面积过大', 'duplicate_box': '重复标注框',
    'no_valid_targets': '没有保留的有效目标', 'exact_duplicate_extra': '全局精确重复副本',
}


def emit(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stat_stamp(info):
    return [info.st_size, info.st_mtime_ns]


@lru_cache(maxsize=256)
def resolved_parent(path):
    return path.resolve()


def within(path, root):
    path = Path(os.path.abspath(path))
    resolved = resolved_parent(path.parent) / path.name
    resolved.relative_to(root)
    return resolved


def read_checked(path):
    """Query the open handle, avoiding repeated exFAT filename lookups."""
    with path.open('rb') as stream:
        before = stat_stamp(os.fstat(stream.fileno()))
        data = stream.read()
        if before != stat_stamp(os.fstat(stream.fileno())):
            raise ValueError(f'读取时文件发生变化：{path}')
    return data, before


def directory_jobs(folder, category):
    """DirEntry caches Windows size/mtime and link flags from directory enumeration."""
    folders = [folder]
    while folders:
        directory = folders.pop()
        with os.scandir(directory) as stream:
            entries = list(stream)
        files = {}
        for entry in entries:
            if entry.is_symlink():
                if Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS | {'.xml'} or entry.is_dir():
                    raise ValueError(f'请先处理符号链接：{entry.path}')
                continue
            if entry.is_dir(follow_symlinks=False):
                if not entry.name.startswith('.'):
                    folders.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                files[entry.name.casefold()] = entry
        for entry in files.values():
            image = Path(entry.path)
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = files.get((image.stem + '.xml').casefold())
            yield (image, Path(label.path) if label else None, category,
                   [stat_stamp(entry.stat(follow_symlinks=False)),
                    stat_stamp(label.stat(follow_symlinks=False)) if label else None])


def atomic_bytes(path, data):
    temporary = path.with_name(path.name + '.hardfilter.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def inspect_job(job, config):
    """Use the audit's VOC parser, raw canvas/EXIF checks and hard-rule function."""
    image_path, label_path, category, expected_stamp = job
    source = Path(config['source'])
    image_path = within(image_path, source)
    label_path = within(label_path, source) if label_path else None
    data, image_stamp = read_checked(image_path)
    label_data, label_stamp = read_checked(label_path) if label_path else (None, None)
    before = [image_stamp, label_stamp]
    if expected_stamp != before:
        raise ValueError(f'枚举后文件发生变化：{image_path}')
    record = {'ok': False, 'boxes': [], 'annotation': {'status': 'missing'}}
    sha = digest(data)
    try:
        if not data:
            raise ValueError('empty image')
        try:
            with Image.open(io.BytesIO(data)) as metadata:
                orientation = int(metadata.getexif().get(274, 1))
        except (OSError, ValueError, SyntaxError):
            orientation = 0
        canvas = cv2.imdecode(np.frombuffer(data, np.uint8),
                              cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if canvas is None:
            raise ValueError('image decode failed')
        height, width = canvas.shape[:2]
        annotation = read_labels(label_path, width, height, config['classes'], data=label_data)
        reasons = []
        if orientation != 1:
            reasons.append('EXIF_orientation_requires_box_frame')
        if annotation['declared_size'] and annotation['declared_size'] != [width, height]:
            reasons.append('XML_size_differs_from_selected_canvas')
        record.update(ok=True, width=width, height=height, annotation=annotation,
                      frame_ok=not reasons, frame_reasons=reasons, boxes=annotation['objects'])
    except (OSError, ValueError, cv2.error, SyntaxError) as exc:
        record['error'] = str(exc)
    decision = assess_record(record, config)
    signature = record['annotation'].get('signature', '')
    uid = digest(f'{image_path.stem}\0{sha}\0{signature}'.encode())
    return {
        'path': str(image_path), 'label': str(label_path or ''), 'category': category,
        'sha': sha, 'uid': uid, 'stamp': before,
        'label_sha': digest(label_data) if label_path else '',
        'width': record.get('width', 0), 'height': record.get('height', 0),
        'decision': decision,
    }


def scan(db, source, config, workers):
    jobs = []
    for category in config['classes']:
        folder = (source / category).resolve()
        folder.relative_to(source)
        if not folder.is_dir():
            raise ValueError(f'类别目录不存在：{folder}')
        jobs.extend(directory_jobs(folder, category))
    jobs.sort(key=lambda item: str(item[0]))
    previous = {row['path']: json.loads(row['payload'])
                for row in db.execute('SELECT path,payload FROM images')}
    todo = []
    for job in jobs:
        cached = previous.get(str(job[0]))
        if cached is None or cached['stamp'] != job[3]:
            todo.append(job)
    current = {str(job[0]) for job in jobs}
    for path in previous.keys() - current:
        db.execute('DELETE FROM images WHERE path=?', (path,))
    db.commit()
    emit(f'六类图片 {len(jobs):,} 张，缓存复用 {len(jobs)-len(todo):,} 张，待扫描 {len(todo):,} 张')
    cv2.setNumThreads(1)
    started = last = time.monotonic()
    iterator = iter(todo)
    pending = set()
    done = 0
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        while True:
            while len(pending) < workers * 2:
                job = next(iterator, None)
                if job is None:
                    break
                pending.add(executor.submit(inspect_job, job, config))
            if not pending:
                break
            ready, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in ready:
                record = future.result()
                db.execute('INSERT OR REPLACE INTO images(path,payload) VALUES (?,?)',
                           (record['path'], json.dumps(record, ensure_ascii=False)))
                done += 1
            if time.monotonic() - last >= 10:
                db.commit()
                elapsed = time.monotonic() - started
                emit(f'硬筛扫描 {len(jobs)-len(todo)+done:,}/{len(jobs):,} 张，新增计算 {done/max(elapsed, .001):.1f} 张/秒')
                last = time.monotonic()
        db.commit()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        db.commit()


def plan(db, source, quarantine):
    records = [json.loads(row['payload']) for row in db.execute('SELECT payload FROM images ORDER BY path')]
    groups = defaultdict(list)
    for record in records:
        groups[record['sha']].append(record)
    for group in groups.values():
        def rank(record):
            decision = record['decision']
            boxes = decision['clean_boxes_json']
            return (decision['clean_status'] != 'keep', -len({b['class'] for b in boxes}),
                    -len(boxes), record['uid'], record['path'])
        ordered = sorted(group, key=rank)
        for record in ordered[1:]:
            record['decision']['clean_status'] = 'reject'
            record['decision']['reject_reasons_json'].append('exact_duplicate_extra')
            record['representative'] = ordered[0]['path']
    label_groups = defaultdict(list)
    operations = []
    for record in records:
        decision = record['decision']
        rejected = decision['clean_status'] == 'reject'
        record['action'] = '移出图片' if rejected else '保留图片'
        record['destination'] = ''
        if rejected:
            bucket = '精确重复' if 'exact_duplicate_extra' in decision['reject_reasons_json'] else '硬筛不通过'
            target = quarantine / bucket / Path(record['path']).relative_to(source)
            record['destination'] = str(target)
            operations.append((record['path'], 'move', str(target),
                               {'stamp': record['stamp'][0], 'sha': record['sha']}))
        if record['label']:
            label_groups[record['label']].append(record)
    for label, references in label_groups.items():
        retained = [r for r in references if r['decision']['clean_status'] == 'keep']
        relative = Path(label).relative_to(source)
        if not retained:
            target = Path(references[0]['destination']).with_suffix(Path(label).suffix)
            operations.append((label, 'move', str(target),
                               {'stamp': references[0]['stamp'][1], 'sha': references[0]['label_sha']}))
            continue
        removals = {tuple(d['index'] for d in r['decision']['box_decisions']
                          if d['action'].startswith('remove')) for r in retained}
        if len(removals) != 1:
            raise ValueError(f'同一XML被不同图片共用且框级处理不同，请先处理：{label}')
        indices = next(iter(removals))
        if not indices:
            continue
        tree = ET.parse(label)
        root = tree.getroot()
        objects = root.findall('object')
        for index in indices:
            root.remove(objects[index])
        data = ET.tostring(root, encoding='utf-8', xml_declaration=True)
        target = quarantine / '原始XML备份' / relative
        operations.append((label, 'rewrite', str(target),
                           {'stamp': retained[0]['stamp'][1], 'sha': retained[0]['label_sha'],
                            'replacement': data.decode('utf-8'), 'replacement_sha': digest(data)}))
        for record in retained:
            record['action'] = '保留图片并移除异常框'
    targets = set()
    for path, kind, target, payload in operations:
        within(path, source)
        within(target, quarantine)
        if target in targets or Path(target).exists():
            raise ValueError(f'隔离目标路径已存在或重复：{target}')
        targets.add(target)
    with db:
        for record in records:
            db.execute('UPDATE images SET payload=? WHERE path=?',
                       (json.dumps(record, ensure_ascii=False), record['path']))
        db.executemany('INSERT INTO operations(source,kind,target,payload,status) VALUES (?,?,?,?,0)',
                       [(p,k,t,json.dumps(v, ensure_ascii=False)) for p,k,t,v in operations])
        db.execute("INSERT OR REPLACE INTO metadata VALUES ('stage','planned')")
    emit(f'计划保留 {sum(r["decision"]["clean_status"]=="keep" for r in records):,} 张，移出 {sum(r["decision"]["clean_status"]=="reject" for r in records):,} 张')


def apply_operations(db, source, quarantine):
    total = db.execute('SELECT COUNT(*) FROM operations').fetchone()[0]
    completed = db.execute('SELECT COUNT(*) FROM operations WHERE status=1').fetchone()[0]
    last = time.monotonic()
    for row in db.execute('SELECT * FROM operations WHERE status=0 ORDER BY id').fetchall():
        path = within(row['source'], source)
        target = within(row['target'], quarantine)
        payload = json.loads(row['payload'])
        target.parent.mkdir(parents=True, exist_ok=True)
        if row['kind'] == 'move':
            try:
                current_stamp = stat_stamp(path.stat())
            except FileNotFoundError:
                current_stamp = None
            if current_stamp is not None:
                if current_stamp != payload['stamp'] or (os.name != 'nt' and target.exists()):
                    raise ValueError(f'待移文件发生变化或目标已存在：{path}')
                # os.rename is a same-volume move. It never copies image contents.
                os.rename(path, target)
            elif not target.is_file() or digest(target.read_bytes()) != payload['sha']:
                raise ValueError(f'待移文件和隔离文件均无法核对：{path}')
        else:
            current_sha = digest(path.read_bytes())
            if current_sha == payload['replacement_sha'] and target.is_file() and digest(target.read_bytes()) == payload['sha']:
                pass
            else:
                if current_sha != payload['sha']:
                    raise ValueError(f'待修改XML发生变化：{path}')
                if target.exists():
                    if digest(target.read_bytes()) != payload['sha']:
                        raise ValueError(f'XML备份内容不一致：{target}')
                else:
                    shutil.copy2(path, target)
                atomic_bytes(path, payload['replacement'].encode('utf-8'))
        db.execute('UPDATE operations SET status=1 WHERE id=?', (row['id'],))
        completed += 1
        if completed % 100 == 0:
            db.commit()
        if time.monotonic() - last >= 10:
            db.commit()
            emit(f'文件处理 {completed:,}/{total:,} 项')
            last = time.monotonic()
    with db:
        db.execute("INSERT OR REPLACE INTO metadata VALUES ('stage','complete')")


def reports(db, quarantine, config):
    totals = Counter()
    categories = defaultdict(Counter)
    reasons = Counter()
    destinations = {}
    for row in db.execute('SELECT payload FROM images'):
        record = json.loads(row['payload'])
        destinations[record['path']] = record['destination'] or record['path']
    with (quarantine / '筛选清单.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['原类别', '原图片路径', '原XML路径', '处理', '原因', '移出图片路径', '重复对照图片', '移除框数'])
        for row in db.execute('SELECT payload FROM images ORDER BY path'):
            record = json.loads(row['payload'])
            decision = record['decision']
            rejected = decision['clean_status'] == 'reject'
            key = '移出' if rejected else '保留'
            totals[key] += 1
            categories[record['category']][key] += 1
            removed = sum(d['action'].startswith('remove') for d in decision['box_decisions'])
            if not rejected and removed:
                totals['保留图片中修改XML'] += 1
                totals['保留图片中移除异常框'] += removed
            codes = list(dict.fromkeys(decision['reject_reasons_json']))
            reasons.update(REASONS.get(code, code) for code in codes)
            writer.writerow([record['category'],record['path'],record['label'],record['action'],
                             '、'.join(REASONS.get(code,code) for code in codes),
                             record['destination'],destinations.get(record.get('representative',''),''),removed])
    summary = {'源目录': str(config['source']), '隔离目录': str(quarantine),
               '硬筛阈值': config['hard_filter'], '去重范围': '六类全局，图片文件SHA256相同',
               '总计': dict(totals), '按原归档类别统计': {k:dict(v) for k,v in categories.items()},
               '触发原因次数': dict(reasons)}
    write_json(quarantine / '筛选汇总.json', summary)
    emit(json.dumps(summary['总计'], ensure_ascii=False))
    for category, values in categories.items():
        emit(f'{category}：保留 {values["保留"]:,}，移出 {values["移出"]:,}')
    emit(f'完整清单 {quarantine / "筛选清单.csv"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='G:/sample_classified/by_label')
    parser.add_argument('--quarantine', default='G:/sample_classified/hard_rejected')
    parser.add_argument('--config', default='configs/filter.json')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    config = load_json(args.config)
    source, quarantine = Path(args.source).resolve(), Path(args.quarantine).resolve()
    if not source.is_dir() or source == quarantine or source in quarantine.parents or quarantine in source.parents:
        raise ValueError('源目录必须存在，隔离目录须位于源目录之外')
    if source.anchor.casefold() != quarantine.anchor.casefold():
        raise ValueError('隔离目录必须与源目录位于同一磁盘')
    if args.workers < 1:
        raise ValueError('workers须为正整数')
    classes = config.get('classes', [])
    if not classes or len(set(classes)) != len(classes) or any(not c or c in ('.','..') or '/' in c or '\\' in c for c in classes):
        raise ValueError('配置中的classes须为不重复的类别目录名')
    # Validate the shared hard-rule configuration before creating any output.
    assess_record({'ok':False}, config)
    config = {'source': str(source), 'classes': classes, 'hard_filter':config.get('hard_filter', {}),
              'rules_version': FILTER_RULES_VERSION}
    quarantine.mkdir(parents=True, exist_ok=True)
    lock = quarantine / '.hardfilter.lock'
    with lock.open('a+b') as lock_stream:
        if os.name == 'nt':
            import msvcrt
            lock_stream.seek(0)
            if not lock_stream.read(1):
                lock_stream.write(b'0')
                lock_stream.flush()
            lock_stream.seek(0)
            msvcrt.locking(lock_stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = sqlite3.connect(quarantine / '.hardfilter.sqlite')
        db.row_factory = sqlite3.Row
        try:
            db.executescript('''CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT);
                CREATE TABLE IF NOT EXISTS images(path TEXT PRIMARY KEY,payload TEXT);
                CREATE TABLE IF NOT EXISTS operations(id INTEGER PRIMARY KEY,source TEXT UNIQUE,
                    kind TEXT,target TEXT UNIQUE,payload TEXT,status INTEGER);''')
            stored = dict(db.execute('SELECT key,value FROM metadata'))
            identity = json.dumps(config, sort_keys=True, ensure_ascii=False)
            if stored.get('config', identity) != identity:
                raise ValueError('隔离目录已有不同参数的任务，请使用另一个隔离目录')
            with db:
                db.execute("INSERT OR IGNORE INTO metadata VALUES ('config',?)", (identity,))
            emit(f'源目录 {source}；隔离目录 {quarantine}；并发 {args.workers}')
            stage = stored.get('stage', 'scan')
            if stage == 'scan':
                scan(db, source, config, args.workers)
                plan(db, source, quarantine)
            if stage != 'complete':
                apply_operations(db, source, quarantine)
            reports(db, quarantine, config)
        finally:
            db.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        emit('任务已中断，重新运行相同命令可继续')
        raise SystemExit(130)
    except (OSError, ValueError, sqlite3.Error) as exc:
        emit(f'停止处理：{exc}')
        raise SystemExit(1)
