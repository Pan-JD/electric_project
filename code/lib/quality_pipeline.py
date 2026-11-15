"""Bounded-concurrency extraction, resumable SQLite cache and screening analysis."""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
import os
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

import cv2

from quality_metrics import (IMAGE_EXTENSIONS, OCRReader, blur_flag, box_geometry, crossing_text_hint,
                             class_rules, exposure_flags, inspect_file, overlap_fraction,
                             text_key, watermark_hint)
from quality_logging import log, progress_items, update_progress

VERSION = '1.3.0'


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def open_cache(output, settings, resume=False):
    output.mkdir(parents=True, exist_ok=True)
    target = output / '_cache' / 'audit.sqlite'
    if not target.exists() and any(output.iterdir()):
        raise ValueError(f'Use an empty output directory: {output}')
    if target.exists() and not resume:
        raise ValueError(f'Output already contains audit.sqlite; use --resume or a new --output: {output}')
    target.parent.mkdir(exist_ok=True)
    db = sqlite3.connect(target)
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY, path TEXT UNIQUE, label_path TEXT, fingerprint TEXT,
            seen TEXT, uid TEXT, payload TEXT);
        CREATE TABLE IF NOT EXISTS features (
            sha TEXT PRIMARY KEY, phash TEXT, dhash TEXT, gray64 BLOB, color16 BLOB,
            width INTEGER, height INTEGER);
        CREATE TABLE IF NOT EXISTS image_cache (sha TEXT PRIMARY KEY, payload TEXT);
    ''')
    serialized = json_text(settings)
    old = db.execute("SELECT value FROM meta WHERE key='settings'").fetchone()
    if old and old[0] != serialized:
        db.close()
        raise ValueError('Resume settings differ from the cached run. Use a new output directory.')
    db.execute("INSERT OR REPLACE INTO meta VALUES ('settings',?)", (serialized,))
    db.execute("INSERT OR REPLACE INTO meta VALUES ('status','extracting')")
    db.commit()
    return db


def walk_error(error):
    raise error


def count_images(source, limit=0):
    """Count filenames only; no image decoding, label reads or in-memory path list."""
    count = 0
    for directory, directories, filenames in os.walk(source, followlinks=False, onerror=walk_error):
        directories[:] = sorted(d for d in directories if not d.startswith('.'))
        for name in filenames:
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                count += 1
                if limit and count >= limit:
                    update_progress(count, note=f'已达到 --limit={limit}')
                    return count
        update_progress(count, note=f'当前目录：{directory}')
    return count


def sample_classified_images(source, ratio, seed=42):
    """Select ceil(N * ratio) paths per first-level class, without opening images/XML.

    Hash ranking of POSIX relative paths is stable across mounts/platforms and
    produces nested samples for different ratios with the same seed.
    """
    fraction = Decimal(str(ratio))
    if not fraction.is_finite() or not 0 < fraction <= 1:
        raise ValueError('--sample-ratio must be in (0, 1]')
    groups = defaultdict(list)
    discovered = 0
    for directory, directories, filenames in os.walk(source, followlinks=False, onerror=walk_error):
        directories[:] = sorted(d for d in directories if not d.startswith('.'))
        for name in filenames:
            if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = (Path(directory) / name).relative_to(source)
            if len(relative.parts) < 2:
                raise ValueError('Per-class sampling requires source/<class>/.../image; '
                                 f'image found directly under source: {name}')
            groups[relative.parts[0]].append(relative.as_posix())
            discovered += 1
        update_progress(discovered, note=f'抽样枚举：{directory}')
    selected, counts = set(), {}
    def rank(relative):
        digest = hashlib.sha256(f'{seed}\0{relative}'.encode('utf-8')).digest()
        return digest, relative
    for category, paths in sorted(groups.items()):
        number = int((len(paths) * fraction).to_integral_value(rounding=ROUND_CEILING))
        chosen = heapq.nsmallest(number, paths, key=rank)
        selected.update(chosen)
        counts[category] = {'available': len(paths), 'selected': number}
        log(f'类别抽样 {category}：{number:,}/{len(paths):,} 张 ({number/len(paths):.2%})')
    return selected, counts


def iter_images(source, labels_root, annotation, selected_paths=None):
    """Only enumerates the selected source. Ignores path/filename fields inside XML."""
    for directory, directories, filenames in os.walk(source, followlinks=False, onerror=walk_error):
        directories[:] = sorted(d for d in directories if not d.startswith('.'))
        file_index = {n.casefold(): n for n in filenames}
        for name in sorted(filenames):
            path = Path(directory) / name
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if selected_paths is not None and path.relative_to(source).as_posix() not in selected_paths:
                continue
            if annotation == 'none':
                yield path, None
                continue
            base = (labels_root / path.relative_to(source)).with_suffix('') if labels_root else path.with_suffix('')
            candidates = []
            suffixes = ['.xml', '.txt'] if annotation == 'auto' else ['.xml' if annotation == 'voc' else '.txt']
            for suffix in suffixes:
                # Append to the stem: names such as image.version.jpg retain '.version'.
                candidate = base.parent / (base.name + suffix)
                if not labels_root:
                    matched = file_index.get(candidate.name.casefold())
                    if matched:
                        candidates.append(path.parent / matched)
                elif candidate.is_file():
                    candidates.append(candidate)
            if not candidates and not labels_root and annotation in ('auto', 'yolo'):
                parts = list(path.parts)
                indices = [i for i, part in enumerate(parts) if part.casefold() == 'images']
                if indices:
                    parts[indices[-1]] = 'labels'
                    candidate = Path(*parts).with_suffix('.txt')
                    if candidate.is_file():
                        candidates.append(candidate)
            yield path, candidates[0] if candidates else None


def fingerprint(path, label):
    def stat(p):
        if p is None:
            return None
        try:
            info = p.stat()
            return [str(p), info.st_size, info.st_mtime_ns]
        except OSError as exc:
            return [str(p), type(exc).__name__]
    return json_text([stat(path), stat(label)])


def store_result(db, path, label_path, stamp, token, record):
    features = record.get('features')
    if features:
        db.execute('INSERT OR REPLACE INTO features VALUES (?,?,?,?,?,?,?)',
                   (record['sha256'], features['phash'], features['dhash'], features['gray64'],
                    features['color16'], record['width'], record['height']))
    # Keep image-only computations separate from the current file's XML/classes.
    shared = None
    if record['ok'] and '_regions' in record:
        keys = ('sha256', 'width', 'height', 'raw_width', 'raw_height', 'orientation',
                'metadata_error', 'metrics', 'ocr_status', 'ocr', 'ocr_error', 'ocr_device', 'ocr_providers')
        shared = {key: record[key] for key in keys if key in record}
        shared['regions'] = record['_regions']
        db.execute('INSERT OR REPLACE INTO image_cache VALUES (?,?)',
                   (record['sha256'], json_text(shared)))
    payload = {key: value for key, value in record.items() if key not in ('features', '_regions')}
    db.execute('''INSERT INTO files(path,label_path,fingerprint,seen,payload) VALUES (?,?,?,?,?)
                  ON CONFLICT(path) DO UPDATE SET label_path=excluded.label_path,
                  fingerprint=excluded.fingerprint,seen=excluded.seen,payload=excluded.payload''',
               (str(path), str(label_path or ''), stamp, token, json_text(payload)))
    return shared


def extract(db, source, options, config, workers, limit, selected_paths=None):
    cv2.setNumThreads(1)
    token = uuid.uuid4().hex
    local = threading.local()
    gpu_ocr = None
    if options['ocr'] and options.get('ocr_device', 'cpu') == 'cuda':
        log('正在初始化 GPU OCR（文字检测、方向分类、文字识别）…')
        gpu_ocr = OCRReader(config, device='cuda')
        log('GPU OCR 实际后端：'+json_text(gpu_ocr.providers)+'；所有 worker 共用一套模型')
        db.execute("INSERT OR REPLACE INTO meta VALUES ('ocr_runtime',?)",
                   (json_text({'device':'cuda','providers':gpu_ocr.providers}),))
        db.commit()
    def prepare(path):
        try:
            data = path.read_bytes()
            return data, hashlib.sha256(data).hexdigest()
        except OSError:
            # inspect_file produces the standard unreadable-image record.
            return None, None

    def process(path, label, data, sha, shared):
        needs_ocr = options['ocr'] and not (shared and shared.get('ocr_status') == 'ok')
        if needs_ocr and gpu_ocr is None and not hasattr(local, 'ocr'):
            local.ocr = OCRReader(config)
        return inspect_file(path, label, options, config,
                            (gpu_ocr or getattr(local, 'ocr', None)) if needs_ocr else None,
                            data=data, sha=sha, shared=shared)

    count, reused, completed, content_reused, last = 0, 0, 0, 0, time.monotonic()
    pending = {}
    active = set()
    waiting = defaultdict(deque)
    memory_cache = OrderedDict()

    def remember(sha, shared):
        memory_cache[sha] = shared
        memory_cache.move_to_end(sha)
        if len(memory_cache) > 32:
            memory_cache.popitem(last=False)

    def cached_image(sha):
        if not sha:
            return None
        if sha in memory_cache:
            memory_cache.move_to_end(sha)
            return memory_cache[sha]
        old = db.execute('SELECT payload FROM image_cache WHERE sha=?', (sha,)).fetchone()
        if old:
            shared = json.loads(old[0])
            remember(sha, shared)
            return shared

    def refresh():
        update_progress(completed+reused,
                        rate_done=completed,
                        note=f'本轮处理 {completed:,}，续跑缓存复用 {reused:,}，'
                             f'相同图片指标复用 {content_reused:,}，排队/处理中 {count-reused-completed}')

    with ThreadPoolExecutor(max_workers=workers) as pool:
        def submit_inspection(job, data, sha):
            nonlocal content_reused
            shared = cached_image(sha)
            if shared:
                content_reused += 1
            if sha:
                active.add(sha)
            future = pool.submit(process, job[0], job[1], data, sha, shared)
            pending[future] = ('inspect', job, sha)

        def drain(block=False):
            nonlocal completed, last
            if not pending:
                return
            done, _ = wait(pending, return_when=FIRST_COMPLETED, timeout=None if block else 0)
            for future in done:
                kind, job, sha = pending.pop(future)
                if kind == 'read':
                    data, sha = future.result()
                    if sha in active:
                        waiting[sha].append((job, data))
                    else:
                        submit_inspection(job, data, sha)
                else:
                    record = future.result()
                    shared = store_result(db, *job, token, record)
                    if shared:
                        remember(sha, shared)
                    completed += 1
                    if sha:
                        active.remove(sha)
                        queue = waiting.get(sha)
                        if queue:
                            next_job, data = queue.popleft()
                            if not queue:
                                del waiting[sha]
                            submit_inspection(next_job, data, sha)
            refresh()
            if time.monotonic() - last > 3:
                db.commit()
                last = time.monotonic()

        for path, label in iter_images(source, options['labels_root'], options['annotation'], selected_paths):
            if limit and count >= limit:
                break
            # Includes read buffers and copies waiting for the same image hash.
            while count-reused-completed >= 2 * workers:
                drain(True)
            count += 1
            stamp = fingerprint(path, label)
            old = db.execute('SELECT fingerprint FROM files WHERE path=?', (str(path),)).fetchone()
            if old and old[0] == stamp:
                db.execute('UPDATE files SET seen=? WHERE path=?', (token, str(path)))
                reused += 1
                refresh()
                if time.monotonic() - last > 3:
                    db.commit()
                    last = time.monotonic()
                drain()
                continue
            future = pool.submit(prepare, path)
            pending[future] = ('read', (path, label, stamp), None)
            refresh()
            drain()
        while pending:
            drain(True)
    if count == 0:
        raise ValueError(f'No supported images under {source}')
    db.execute('DELETE FROM files WHERE seen<>? OR seen IS NULL', (token,))
    db.commit()
    update_progress(count, total=count, rate_done=completed)
    log(f'图像提取完成：文件 {count:,}，本轮处理 {completed:,}，续跑缓存复用 {reused:,}，'
        f'相同图片指标复用 {content_reused:,}')


def reset_analysis(db):
    for name in ('units', 'unit_classes', 'boxes', 'findings', 'evaluated', 'ocr_lines',
                 'duplicate_members', 'near_search'):
        db.execute(f'DROP TABLE IF EXISTS {name}')
    db.executescript('''
        CREATE TABLE units (uid TEXT PRIMARY KEY, file_id INTEGER, sha TEXT, ann_sig TEXT,
                            width INTEGER, height INTEGER, ok INTEGER);
        CREATE TABLE unit_classes (uid TEXT, class TEXT, PRIMARY KEY(uid,class));
        CREATE TABLE boxes (uid TEXT, object_index INTEGER, class TEXT, eligible INTEGER,
                            area REAL, short_at_model REAL, payload TEXT);
        CREATE TABLE findings (uid TEXT, issue TEXT, status TEXT, severity TEXT,
                               object_index INTEGER, evidence TEXT);
        CREATE TABLE evaluated (uid TEXT, issue TEXT, PRIMARY KEY(uid,issue));
        CREATE TABLE ocr_lines (uid TEXT, line_index INTEGER, sha TEXT, repeat_key TEXT,
                               hint TEXT, payload TEXT);
    ''')


def consolidate(db, layout):
    """Preserve source occurrences; collapse classification copies of the same named pair."""
    reset_analysis(db)
    for file_id, path, payload in progress_items(db.execute('SELECT id,path,payload FROM files ORDER BY path')):
        row = json.loads(payload)
        sha = row.get('sha256', '')
        annotation = row.get('annotation', {})
        signature = annotation.get('signature', '')
        if layout in ('classified', 'classified_replay') and row['ok'] and signature:
            key = [Path(row['path']).stem, sha, signature]
        else:
            key = [path]
        uid = hashlib.sha256(json_text(key).encode()).hexdigest()[:24]
        db.execute('UPDATE files SET uid=? WHERE id=?', (uid, file_id))
        inserted = db.execute('INSERT OR IGNORE INTO units VALUES (?,?,?,?,?,?,?)',
                              (uid, file_id, sha, signature, row.get('width'), row.get('height'), row['ok'])).rowcount
        for label in row['classes'] or ['__unlabeled__']:
            db.execute('INSERT OR IGNORE INTO unit_classes VALUES (?,?)', (uid, label))
        if inserted:
            for box in row['boxes']:
                db.execute('INSERT INTO boxes VALUES (?,?,?,?,?,?,?)',
                           (uid, box['index'], box['class'], box.get('eligible', False),
                            box.get('area_fraction'), box.get('short_at_model'), json_text(box)))
    if layout == 'classified_replay':
        # Historical CSVs contain only the sampled class's boxes. Combine those
        # target subsets before screening a shared image; indices refer to the XML.
        shared = db.execute('SELECT uid FROM files GROUP BY uid HAVING COUNT(*)>1').fetchall()
        for (uid,) in shared:
            records = [json.loads(p) for (p,) in db.execute('SELECT payload FROM files WHERE uid=? ORDER BY path', (uid,))]
            merged = records[0]
            boxes = {box['index']: box for r in records for box in r['boxes']}
            merged['boxes'] = [boxes[k] for k in sorted(boxes)]
            merged['classes'] = sorted({c for r in records for c in r['classes']})
            db.execute('UPDATE files SET payload=? WHERE id=(SELECT file_id FROM units WHERE uid=?)',
                       (json_text(merged), uid))
            db.execute('DELETE FROM boxes WHERE uid=?', (uid,))
            for box in merged['boxes']:
                db.execute('INSERT INTO boxes VALUES (?,?,?,?,?,?,?)',
                           (uid, box['index'], box['class'], box.get('eligible', False),
                            box.get('area_fraction'), box.get('short_at_model'), json_text(box)))
    db.executescript('''CREATE INDEX unit_sha_idx ON units(sha);
                       CREATE INDEX class_idx ON unit_classes(class,uid);
                       CREATE INDEX box_uid_idx ON boxes(uid);
                       CREATE INDEX IF NOT EXISTS file_uid_idx ON files(uid);''')
    db.commit()
    log(f"consolidated: {db.execute('SELECT COUNT(*) FROM files').fetchone()[0]} paths -> "
        f"{db.execute('SELECT COUNT(*) FROM units').fetchone()[0]} sample units")


def add_finding(db, uid, issue, evidence, index=-1, status='candidate', severity='review'):
    db.execute('INSERT INTO findings VALUES (?,?,?,?,?,?)',
               (uid, issue, status, severity, index, json_text(evidence)))


def evaluated(db, uid, *issues):
    db.executemany('INSERT OR IGNORE INTO evaluated VALUES (?,?)', ((uid, issue) for issue in issues))


def analyze_units(db, config):
    last = time.monotonic()
    for uid, payload in progress_items(db.execute('''SELECT u.uid,f.payload FROM units u
                                                        JOIN files f ON f.id=u.file_id ORDER BY u.uid''')):
        row = json.loads(payload)
        evaluated(db, uid, 'unavailable')
        if not row['ok']:
            add_finding(db, uid, 'unavailable', {'error': row.get('error', '')}, status='measured')
            continue
        evaluated(db, uid, 'low_resolution', 'coordinate_frame', 'annotation_unavailable')
        if min(row['width'], row['height']) < config['low_resolution_short_side']:
            add_finding(db, uid, 'low_resolution', {'width': row['width'], 'height': row['height'],
                        'threshold': config['low_resolution_short_side']}, status='measured')
        if not row['frame_ok']:
            add_finding(db, uid, 'coordinate_frame', {'reasons': row['frame_reasons'],
                        'orientation': row.get('orientation')})
        annotation = row.get('annotation', {})
        if annotation.get('status') != 'ok':
            add_finding(db, uid, 'annotation_unavailable', {'status': annotation.get('status'),
                        'error': annotation.get('error', '')}, status='measured')
        else:
            evaluated(db, uid, 'annotation_unavailable', 'box_geometry', 'annotation_conflict')
        eligible = [b for b in row['boxes'] if b.get('eligible')]
        if eligible:
            evaluated(db, uid, 'small_target', 'tiny_target', 'large_box')
        for box in row['boxes']:
            if not box.get('valid') or (row['frame_ok'] and box.get('outside')):
                add_finding(db, uid, 'box_geometry', {'class': box['class'], 'coords': box['coords'],
                            'reason': box.get('reason', 'outside_canvas')}, box['index'], status='measured')
            if not box.get('eligible'):
                continue
            rules = class_rules(config, box['class'])
            evidence = {'class': box['class'], 'short_at_model': box['short_at_model'],
                        'area_fraction': box['area_fraction'], 'coords': box['coords']}
            if box['short_at_model'] < rules['small_target_pixels'] or box['area_fraction'] < rules['small_target_area_fraction']:
                add_finding(db, uid, 'small_target', evidence, box['index'], status='measured')
            if box['short_at_model'] < rules['tiny_target_pixels']:
                add_finding(db, uid, 'tiny_target', evidence, box['index'], status='measured', severity='high')
            if box['area_fraction'] >= rules['large_box_area_fraction']:
                add_finding(db, uid, 'large_box', {**evidence, 'rule': 'large_canvas_fraction'}, box['index'])

        exposure_boxes = [b for b in eligible if b['metrics'] and
                          not class_rules(config, b['class']).get('skip_roi_exposure')]
        exposure_regions = [(b['index'], b['class'], b['metrics']) for b in exposure_boxes]
        if not exposure_regions:
            exposure_regions = [(-1, '', row['metrics'])]
        evaluated(db, uid, 'exposure')
        for index, label, metrics in exposure_regions:
            flags = exposure_flags(metrics, class_rules(config, label))
            if flags:
                add_finding(db, uid, 'exposure', {'scope': 'image' if index < 0 else 'target',
                            'class': label, 'flags': flags, 'metrics': metrics}, index,
                            severity='high' if metrics['mean_gray'] < 30 else 'review')

        # Every labeled target is assessed only if it has enough actual pixels.
        blur_regions = []
        for box in eligible:
            rules = class_rules(config, box['class'])
            if (not rules.get('skip_blur') and box['metrics'] and
                    box['short_raw'] >= rules['roi_min_raw_side'] and
                    box['short_at_model'] >= rules['tiny_target_pixels']):
                blur_regions.append((box['index'], box['class'], box['metrics']))
        if not row['boxes'] or not row['frame_ok']:
            if not row['classes'] or any(not class_rules(config, c).get('skip_blur') for c in row['classes']):
                blur_regions = [(-1, '', row['metrics'])]
        if blur_regions:
            evaluated(db, uid, 'blur')
        for index, label, metrics in blur_regions:
            if blur_flag(metrics, class_rules(config, label)):
                add_finding(db, uid, 'blur', {'scope': 'image' if index < 0 else 'target',
                            'class': label, 'metrics': metrics}, index,
                            severity='high' if metrics['laplacian_variance'] < 20 and metrics['tenengrad'] < 300 else 'review')
        if row.get('ocr_status') == 'ok':
            evaluated(db, uid, 'watermark_present')
            if eligible:
                evaluated(db, uid, 'text_overlap', 'watermark_occlusion')
            for index, line in enumerate(row['ocr']):
                db.execute('INSERT INTO ocr_lines VALUES (?,?,?,?,?,?)',
                           (uid, index, row['sha256'],
                            text_key(line['text'], line['polygon'], row['width'], row['height']),
                            watermark_hint(line['text'], config), json_text(line)))
        if time.monotonic() - last > 3:
            db.commit()
            last = time.monotonic()
    db.executescript('''CREATE INDEX finding_idx ON findings(issue,uid);
                       CREATE INDEX finding_unit_idx ON findings(uid);
                       CREATE INDEX evaluated_issue_idx ON evaluated(issue,uid);
                       CREATE INDEX box_class_idx ON boxes(class,eligible);
                       CREATE INDEX ocr_uid_idx ON ocr_lines(uid);
                       CREATE INDEX ocr_key_idx ON ocr_lines(repeat_key,sha);''')
    db.commit()


def analyze_annotations(db):
    groups = db.execute('''SELECT sha,COUNT(DISTINCT ann_sig) FROM units
        WHERE ok=1 AND ann_sig<>'' GROUP BY sha HAVING COUNT(DISTINCT ann_sig)>1''').fetchall()
    update_progress(0, total=len(groups), note='检查同图标注版本差异')
    for sha, versions in progress_items(groups):
        for (uid,) in db.execute('SELECT uid FROM units WHERE sha=? AND ann_sig<>\'\'', (sha,)):
            add_finding(db, uid, 'annotation_conflict', {'image_sha256': sha, 'annotation_versions': versions})
    db.commit()


def analyze_ocr(db, config):
    repeated = {r[0] for r in db.execute('''SELECT repeat_key FROM ocr_lines GROUP BY repeat_key
                                          HAVING COUNT(DISTINCT sha)>=?''', (config['ocr_repeat_min_images'],))}
    for (uid,) in progress_items(db.execute("SELECT uid FROM evaluated WHERE issue='watermark_present'")):
        all_lines, overlay_lines = [], []
        for key, hint, payload in db.execute('SELECT repeat_key,hint,payload FROM ocr_lines WHERE uid=?', (uid,)):
            line = json.loads(payload)
            all_lines.append(line)
            if hint or key in repeated:
                overlay_lines.append({**line, 'reason': hint or 'repeated_text_at_same_position'})
        if overlay_lines:
            add_finding(db, uid, 'watermark_present', {'lines': overlay_lines})
        for index, label, payload in db.execute('SELECT object_index,class,payload FROM boxes WHERE uid=? AND eligible=1', (uid,)):
            box = json.loads(payload)
            crossing = [line for line in all_lines if crossing_text_hint(box['coords'], line['polygon'])]
            for issue, lines in [('text_overlap', all_lines), ('watermark_occlusion', overlay_lines+crossing)]:
                fraction = overlap_fraction(box['coords'], [line['polygon'] for line in lines])
                if fraction >= config['ocr_overlap_fraction']:
                    touching = [line for line in lines if overlap_fraction(box['coords'], [line['polygon']]) > 0]
                    add_finding(db, uid, issue, {'class': label, 'overlap_fraction': fraction,
                                'coords': box['coords'], 'text': list(dict.fromkeys(line['text'] for line in touching)),
                                'crossing_text_cue': bool(crossing)}, index,
                                severity='high' if fraction >= .20 else 'review')
    db.commit()


def import_existing(db, audit_dir, options):
    """Read only the two audit CSV files. Referenced source paths remain plain strings."""
    with (audit_dir / 'image_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        images = list(csv.DictReader(stream))
    by_id = {}
    with (audit_dir / 'target_box_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        for box in csv.DictReader(stream):
            by_id.setdefault(box['id'], []).append(box)
    metric_names = ('mean_gray', 'dark_fraction_le10', 'bright_fraction_ge245',
                    'laplacian_variance', 'tenengrad', 'gray_entropy')
    locations = {}
    location_file = audit_dir / 'sample_file_locations.csv'
    if location_file.is_file():
        with location_file.open(encoding='utf-8-sig', newline='') as stream:
            locations = {r['id']:r for r in csv.DictReader(stream)}
    token = uuid.uuid4().hex
    update_progress(0, total=len(images))
    for image in progress_items(images):
        ok = image['decode_ok'].lower() == 'true'
        width, height = int(image['width'] or 0), int(image['height'] or 0)
        orientation = int(image.get('exif_orientation') or 1)
        frame_reasons = []
        if orientation != 1:
            frame_reasons.append('EXIF_orientation_requires_box_frame')
        if image.get('xml_size_mismatch_raw', '').lower() == 'true':
            frame_reasons.append('XML_size_differs_from_selected_canvas')
        location = locations.get(image['id'], {})
        row = {'path': location.get('classified_image') or image['relative_path'],
               'label_path': location.get('classified_xml') or str(Path(image['relative_path']).with_suffix('.xml')),
               'sample_id': image['id'], 'ok': ok, 'width': width, 'height': height,
               'raw_width': width, 'raw_height': height, 'orientation': orientation,
               'frame_ok': not frame_reasons, 'frame_reasons': frame_reasons,
               'sha256': image['sha256'], 'classes': [image['class_code']],
               'annotation': {'status': 'error' if image['xml_error'] else 'ok',
                              'signature': image.get('xml_sha256', ''),
                              'signature_kind': 'xml_file_sha256', 'error': image['xml_error']},
               'metrics': {key: float(image[key]) for key in metric_names if image.get(key)},
               'ocr_status': 'not_available_in_existing_metrics', 'ocr': [], 'boxes': []}
        for b in by_id.get(image['id'], []):
            coords = [float(b[k]) for k in ('xmin','ymin','xmax','ymax')] if all(b.get(k) for k in ('xmin','ymin','xmax','ymax')) else None
            geom = box_geometry(coords, width, height, options['imgsz'], options['scaleup']) if ok else {'valid': False}
            metrics = {key: float(b['roi_'+key]) for key in metric_names if b.get('roi_'+key)}
            row['boxes'].append({'index': int(b['object_index_all_xml']), 'class': b['class_code'],
                                 'coords': coords, **geom, 'metrics': metrics,
                                 'eligible': geom['valid'] and not geom.get('outside') and not frame_reasons})
        # Use the audit ID as a stable sample-copy key; do not open relative_path.
        store_result(db, image['id'], row['label_path'], 'existing_csv', token, row)
    db.execute('DELETE FROM files WHERE seen<>?', (token,))
    db.commit()
    log(f'imported {len(images)} existing sample records; no source image paths opened')
