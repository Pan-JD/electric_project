"""Audit selected class folders into a new experiment manifest; never alters sources."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import signal
import sqlite3
import tempfile
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

from quality_logging import log, progress_stage, update_progress
from quality_metrics import inspect_file, load_config, OCRReader, annotation_signature
from quality_pipeline import iter_images, fingerprint
from quality_experiment_rules import assess_record, FILTER_RULES_VERSION
from quality_experiment_scoring import score_record, read_annotation_scores, combine_dimensions, validate_weights


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


class ManifestConnection(sqlite3.Connection):
    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()


def open_manifest(path, readonly=False):
    path = Path(path).resolve()
    db = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True, factory=ManifestConnection) if readonly else sqlite3.connect(path,factory=ManifestConnection)
    db.row_factory = sqlite3.Row
    return db


def initialize(db, config):
    db.executescript('''
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE samples(uid TEXT PRIMARY KEY,image_path TEXT,label_path TEXT,image_sha TEXT,
          annotation_sig TEXT,split_group TEXT,width INTEGER,height INTEGER,classes_json TEXT,
          baseline_eligible INTEGER,clean_status TEXT,boxes_json TEXT,clean_boxes_json TEXT,
          mask_regions_json TEXT,reject_reasons_json TEXT,box_decisions_json TEXT,kept_indices_json TEXT,
          image_quality_score REAL,total_quality_score REAL,quality_weighted_score REAL,
          scores_json TEXT,payload_json TEXT);
        CREATE TABLE occurrences(uid TEXT,image_path TEXT,label_path TEXT,archive_class TEXT);
        CREATE TABLE features(uid TEXT PRIMARY KEY,phash TEXT,dhash TEXT,gray64 BLOB,color16 BLOB);
        CREATE TABLE ratings(uid TEXT,category TEXT,b1 INTEGER,b2 INTEGER,source_json TEXT,PRIMARY KEY(uid,category));
        CREATE INDEX sample_sha ON samples(image_sha);
        CREATE INDEX occurrence_uid ON occurrences(uid);
    ''')
    db.execute('INSERT INTO metadata VALUES (?,?)', ('config', dumps(config)))
    db.execute('INSERT INTO metadata VALUES (?,?)', ('status', 'scanning'))
    db.execute('INSERT INTO metadata VALUES (?,?)', ('filter_rules_version', str(FILTER_RULES_VERSION)))
    db.commit()


def clone_for_rescore(previous, output):
    """Copy a completed metrics manifest; source image/XML files are never opened."""
    previous,output=Path(previous).resolve(),Path(output).resolve()
    if previous==output: raise ValueError('--from-run 必须与当前 run_name 不同')
    if not (previous/'experiment.sqlite').is_file(): raise ValueError(f'旧实验缓存不存在：{previous}')
    if output.exists() and any(output.iterdir()): raise ValueError(f'新实验目录须为空：{output}')
    with open_manifest(previous/'experiment.sqlite',readonly=True) as source:
        if dict(source.execute('SELECT key,value FROM metadata')).get('status')!='complete':
            raise ValueError('旧实验扫描尚未完成')
        output.mkdir(parents=True,exist_ok=True)
        with open_manifest(output/'experiment.sqlite') as destination:
            source.backup(destination)
            destination.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('parent_run',str(previous)))


def score_values(payload, decision, config, annotation=None):
    value = score_record(payload, decision['kept_indices'], config, annotation)
    new_weights = config.get('reweight', {}).get('weights')
    weighted = combine_dimensions(value['dimensions'], validate_weights(new_weights),config.get('scoring')) if new_weights else value
    return value['image_score'], value['total_score'], weighted['total_score'], dumps(value)


def experiment_annotation_signature(record, classes):
    """Compare annotations for the selected task, ignoring unrelated classes."""
    if record.get('annotation', {}).get('status') != 'ok':
        return ''
    selected = set(classes)
    objects = {}
    for box in record.get('boxes', []):
        if selected and box.get('class') not in selected:
            continue
        obj = {'class': box.get('class'), 'coords': box.get('coords')}
        objects[dumps(obj)] = obj
    return annotation_signature(list(objects.values())) if objects else ''


def store_record(db, record, archive_class, config):
    record.pop('_regions', None)
    features = record.pop('features', None)
    sha = record.get('sha256', '')
    signature = record.get('annotation', {}).get('signature', '')
    stem = Path(record['path']).stem
    identity = f'{stem}\0{sha}\0{signature}' if sha else record['path']
    uid = record.pop('_sample_uid',None) or hashlib.sha256(identity.encode()).hexdigest()
    db.execute('INSERT INTO occurrences VALUES (?,?,?,?)', (uid, record['path'], record['label_path'], archive_class))
    if db.execute('SELECT 1 FROM samples WHERE uid=?', (uid,)).fetchone():
        return
    decision = assess_record(record, config)
    categories = sorted(set(record.get('classes', [])) & set(config['classes'])) or [archive_class]
    values = [uid, record['path'], record['label_path'], sha,
              experiment_annotation_signature(record, config['classes']), sha or uid,
              record.get('width',0), record.get('height',0), dumps(categories),
              int(decision['baseline_eligible']), decision['clean_status'],
              dumps(decision['boxes_json']), dumps(decision['clean_boxes_json']), dumps(decision['mask_regions_json']),
              dumps(decision['reject_reasons_json']), dumps(decision['box_decisions']), dumps(decision['kept_indices']),
              *(score_values(record, decision, config) if config.get('compute_scores',True) else
                (None,None,None,dumps(combine_dimensions({k:{'score':None} for k in validate_weights({})},validate_weights({}))))), dumps(record)]
    db.execute('INSERT INTO samples VALUES ('+','.join('?' for _ in values)+')', values)
    if features:
        db.execute('INSERT INTO features VALUES (?,?,?,?,?)',
                   (uid, features['phash'], features['dhash'], features['gray64'], features['color16']))


def _scan_identity(config):
    """Worker count and split lists may change while resuming the same scan."""
    value = json.loads(dumps(config))
    for key in ('workers', 'split'):
        value.pop(key, None)
    value.get('scan', {}).pop('workers', None)
    return value


def rebuild_filter_from_cache(config, previous, output):
    """Reassess a completed metrics cache into a separate output; no source-file reads."""
    previous, output = Path(previous).resolve(), Path(output).resolve()
    source = previous if previous.is_file() else previous/'_cache/experiment.sqlite'
    if not source.is_file():
        raise ValueError(f'可复用的筛选指标缓存不存在：{source}')
    destination = output/'experiment.sqlite'
    if source == destination:
        raise ValueError('复用指标缓存时须使用新的output，保留旧实验结果')
    if destination.is_file():
        with open_manifest(destination, readonly=True) as db:
            meta = dict(db.execute('SELECT key,value FROM metadata'))
            if (meta.get('metrics_parent') != str(source)
                    or _scan_identity(json.loads(meta['config'])) != _scan_identity(config)
                    or meta.get('filter_rules_version') != str(FILTER_RULES_VERSION)):
                raise ValueError('已有重筛缓存与当前配置不同，请使用新的output')
            if meta.get('status') == 'complete':
                log('复用已完成的重筛缓存，继续报告和图片导出。')
                return
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError(f'新的缓存目录已有其他内容：{output}')
        with open_manifest(source, readonly=True) as db:
            meta = dict(db.execute('SELECT key,value FROM metadata'))
            old = json.loads(meta['config'])
            if meta.get('status') != 'complete':
                raise ValueError('只能复用已完成的指标缓存')
            if Path(old['source']).resolve() != Path(config['source']).resolve() or old['classes'] != config['classes']:
                raise ValueError('复用缓存须保持相同的数据源和选定类别')
            output.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='._clone_',dir=output) as temporary:
                staged = Path(temporary)/'experiment.sqlite'
                with open_manifest(staged) as target:
                    db.backup(target)
                    for key, value in [('metrics_parent', str(source)), ('config', dumps(config)),
                                       ('status', 'rebuilding'), ('filter_rules_version', str(FILTER_RULES_VERSION))]:
                        target.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', (key, value))
                staged.replace(destination)
    log('复用图片指标：'+str(source)+'；重新计算框级筛选和本次类别的标注分组。')
    with open_manifest(destination) as db:
        total = db.execute('SELECT COUNT(*) FROM samples').fetchone()[0]
        with progress_stage('复用指标缓存重新筛选',total):
            last, done = '', 0
            while True:
                rows = db.execute('SELECT uid,payload_json FROM samples WHERE uid>? ORDER BY uid LIMIT 100', (last,)).fetchall()
                if not rows:
                    break
                for row in rows:
                    payload = json.loads(row['payload_json'])
                    decision = assess_record(payload,config)
                    db.execute('''UPDATE samples SET annotation_sig=?,split_group=CASE WHEN image_sha!='' THEN image_sha ELSE uid END,
                        baseline_eligible=?,clean_status=?,boxes_json=?,clean_boxes_json=?,mask_regions_json=?,
                        reject_reasons_json=?,box_decisions_json=?,kept_indices_json=?,
                        image_quality_score=NULL,total_quality_score=NULL,quality_weighted_score=NULL WHERE uid=?''',
                        (experiment_annotation_signature(payload,config['classes']),int(decision['baseline_eligible']),
                         decision['clean_status'],dumps(decision['boxes_json']),dumps(decision['clean_boxes_json']),
                         dumps(decision['mask_regions_json']),dumps(decision['reject_reasons_json']),
                         dumps(decision['box_decisions']),dumps(decision['kept_indices']),row['uid']))
                    done += 1
                    update_progress(done)
                last = rows[-1]['uid']
            deduplicate(db,config)
            db.execute("UPDATE metadata SET value='complete' WHERE key='status'")
        # One transaction: a retry reapplies decisions from the unchanged metrics.
        db.commit()


def _legacy_file_matches(row):
    """Validate pre-checkpoint caches using stored image/XML hashes, without metrics."""
    payload = json.loads(row['payload_json'])
    expected = payload.get('annotation', {}).get('sha256')
    if not row['image_sha'] or not expected or not row['label_path']:
        return None
    try:
        return (hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest() == row['image_sha']
                and hashlib.sha256(Path(row['label_path']).read_bytes()).hexdigest() == expected)
    except OSError:
        return False


def _resume_jobs(db, jobs):
    """Reuse committed paths; size/mtime changes require a separate experiment."""
    db.execute('''CREATE TABLE IF NOT EXISTS processed_files(
        image_path TEXT PRIMARY KEY, fingerprint TEXT NOT NULL)''')
    db.execute('CREATE INDEX IF NOT EXISTS occurrence_path ON occurrences(image_path)')
    job_map = {str(image): (image, label, category, fingerprint(image, label))
               for image, label, category in jobs}
    previous_paths = {r[0] for r in db.execute('SELECT image_path FROM occurrences')}
    removed = previous_paths - job_map.keys()
    if removed:
        raise ValueError(f'续跑时发现已处理图片移除或更名，请使用新的output：{min(removed)}')
    checkpoints = dict(db.execute('SELECT image_path,fingerprint FROM processed_files'))
    for path, stamp in checkpoints.items():
        if path not in job_map or job_map[path][3] != stamp:
            raise ValueError(f'续跑时发现图片或XML已变化，请使用新的output：{path}')
    # The old scanner already committed samples/occurrences every 100 files.
    # Adopt these records only after checking both content hashes.
    legacy = previous_paths - checkpoints.keys()
    if legacy:
        log(f'校验旧版已提交缓存：{len(legacy):,} 张（核对图片/XML哈希，不重复计算指标）')
        for path in sorted(legacy):
            row = db.execute('''SELECT o.image_path,o.label_path,s.image_sha,s.payload_json
                FROM occurrences o JOIN samples s USING(uid) WHERE o.image_path=? LIMIT 1''', (path,)).fetchone()
            job = job_map[path]
            matched = _legacy_file_matches(row)
            if str(job[1] or '') != row['label_path'] or matched is False:
                raise ValueError(f'旧版缓存核对时发现图片或XML变化，请使用新的output：{path}')
            if matched:
                db.execute('INSERT OR REPLACE INTO processed_files VALUES (?,?)', (path, job[3]))
            else:
                db.execute('DELETE FROM occurrences WHERE image_path=?', (path,))
        db.execute('DELETE FROM features WHERE uid NOT IN (SELECT uid FROM occurrences)')
        db.execute('DELETE FROM samples WHERE uid NOT IN (SELECT uid FROM occurrences)')
    db.commit()
    processed = {r[0] for r in db.execute('SELECT image_path FROM processed_files')}
    return [job for path, job in job_map.items() if path not in processed], len(processed)


def _shutdown_workers(executor, db=None, total=None):
    # Repeated Ctrl+C must not interrupt the cache commit/worker cleanup.
    main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN) if main_thread else None
    try:
        try:
            if db is not None:
                db.commit()
                saved = db.execute('SELECT COUNT(*) FROM processed_files').fetchone()[0]
                log(f'已保存 {saved:,}/{total:,} 张缓存；停止提交新任务，等待当前worker退出。')
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    finally:
        if main_thread:
            signal.signal(signal.SIGINT, previous)


def _extract_jobs(db, jobs, reused, total, workers, process, config):
    executor = ThreadPoolExecutor(max_workers=workers)
    pending, iterator, completed = set(), iter(jobs), 0
    stopping = False
    try:
        with progress_stage('2/3 指标计算与框级筛选', total):
            def progress():
                update_progress(reused + completed, rate_done=completed,
                                note=f'本轮计算 {completed:,}，缓存复用 {reused:,}')
            def fill():
                while len(pending) < 2 * workers:
                    job = next(iterator, None)
                    if job is None:
                        break
                    pending.add(executor.submit(process, job))
            progress()
            fill()
            while pending:
                ready, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in ready:
                    record, category, stamp = future.result()
                    if not db.in_transaction:
                        db.execute('BEGIN')
                    db.execute('SAVEPOINT image_record')
                    try:
                        store_record(db, record, category, config)
                        db.execute('INSERT INTO processed_files VALUES (?,?)', (record['path'], stamp))
                        db.execute('RELEASE image_record')
                    except BaseException:
                        try:
                            db.execute('ROLLBACK TO image_record')
                            db.execute('RELEASE image_record')
                        except sqlite3.OperationalError as exc:
                            # SIGINT may arrive immediately after RELEASE succeeded.
                            if 'no such savepoint' not in str(exc):
                                raise
                        raise
                    completed += 1
                    progress()
                    if completed % 100 == 0:
                        db.commit()
                fill()
            db.commit()
    except BaseException:
        stopping = True
        _shutdown_workers(executor, db, total)
        raise
    finally:
        if not stopping:
            _shutdown_workers(executor)


def add_reason(db, uid, reason, status=None):
    row = db.execute('SELECT reject_reasons_json,clean_status FROM samples WHERE uid=?', (uid,)).fetchone()
    reasons = json.loads(row[0])
    if reason not in reasons:
        reasons.append(reason)
    db.execute('UPDATE samples SET reject_reasons_json=?,clean_status=? WHERE uid=?',
               (dumps(reasons), status or row[1], uid))


def deduplicate(db, config):
    groups = db.execute("SELECT image_sha FROM samples WHERE image_sha!='' GROUP BY image_sha HAVING COUNT(*)>1").fetchall()
    for (sha,) in groups:
        rows = db.execute('SELECT uid,annotation_sig,clean_status,image_quality_score,clean_boxes_json FROM samples WHERE image_sha=? ORDER BY uid', (sha,)).fetchall()
        signatures = {r['annotation_sig'] for r in rows if r['annotation_sig']}
        if len(signatures) > 1:
            for r in rows:
                add_reason(db, r['uid'], 'same_image_annotation_versions')
        # Select one structurally usable version; semantic differences are recorded.
        def rank(row):
            boxes = json.loads(row['clean_boxes_json'])
            return (row['clean_status'] != 'keep', -len({b['class'] for b in boxes}), -len(boxes), row['uid'])
        ordered = sorted(rows, key=rank)
        for row in ordered[1:]:
            add_reason(db, row['uid'], 'exact_duplicate_extra', 'reject')
    near = config.get('near_duplicates', {})
    if not near.get('enabled'):
        return
    from quality_duplicates import BKTree
    from quality_metrics import near_match, match_target_appearance
    if near.get('action', 'record') not in ('record', 'reject'):
        raise ValueError('near_duplicates.action 仅支持 record 或 reject')
    metric_config = load_config()
    tree, representatives = BKTree(), {}
    for row in db.execute("SELECT s.*,f.phash,f.dhash,f.gray64,f.color16 FROM samples s JOIN features f USING(uid) WHERE clean_status='keep' ORDER BY image_quality_score DESC,uid").fetchall():
        current = dict(row)
        boxes = json.loads(row['clean_boxes_json'])
        found = None
        examined = 0
        for _, key in tree.query(int(row['phash'],16), metric_config['near_phash_distance']):
            if examined >= near.get('max_candidates',200):
                add_reason(db,row['uid'],'near_search_truncated')
                break
            examined += 1
            other, other_boxes = representatives[key]
            if json.loads(row['classes_json']) != json.loads(other['classes_json']):
                continue
            match = near_match(current,other,metric_config)
            labels=json.loads(row['classes_json'])
            targets_match = match and all(match_target_appearance(
                [b for b in boxes if b['class']==label], [b for b in other_boxes if b['class']==label],
                (row['width'],row['height']),(other['width'],other['height']),metric_config) for label in labels)
            if targets_match:
                found = other
                break
        if found:
            add_reason(db,row['uid'],'near_duplicate_candidate','reject' if near.get('action','record')=='reject' else None)
            db.execute('UPDATE samples SET split_group=? WHERE uid=?', (found['split_group'],row['uid']))
        else:
            tree.add(int(row['phash'],16),row['uid'])
            representatives[row['uid']] = (current, boxes)
    # Propagate optional near groups to exact copies and annotation variants too.
    for sha, group in db.execute('SELECT image_sha,split_group FROM samples WHERE image_sha!=split_group').fetchall():
        db.execute('UPDATE samples SET split_group=? WHERE image_sha=?', (group,sha))


def scan_dataset(config, output, reports=True, resume=False):
    output = Path(output)
    manifest = output/'experiment.sqlite'
    if output.exists() and any(output.iterdir()) and not (resume and manifest.is_file()):
        raise ValueError(f'扫描缓存目录已有其他内容：{output}')
    if manifest.is_file():
        with open_manifest(manifest, readonly=True) as previous:
            meta = dict(previous.execute('SELECT key,value FROM metadata'))
            if meta.get('filter_rules_version') != str(FILTER_RULES_VERSION):
                raise ValueError('已有缓存采用旧版类别筛选规则，请使用新的output并通过reuse_metrics_from复用已完成缓存')
            if _scan_identity(json.loads(meta['config'])) != _scan_identity(config):
                raise ValueError('续跑的数据源/类别/筛选规则已改变，请使用新的output；workers和划分种子可直接调整')
            if meta.get('status') == 'complete':
                log('复用已完成扫描缓存，继续生成筛选报告和数据集。')
                return
    source = Path(config['source'])
    jobs = []
    sample_ids={}
    with progress_stage('1/3 枚举选定类别目录'):
        if config.get('input_layout')=='named':
            index=json.loads((source/'图片索引.json').read_text(encoding='utf-8-sig'))
            for row in index['samples']:
                categories={box['class'] for box in json.loads(row['boxes_json'])}&set(config['classes'])
                if not categories: continue
                image=source/row['export_image']
                relative=image.relative_to(source/'images')
                label=source/'annotations'/relative.with_suffix('.xml')
                if not image.resolve().is_relative_to(source.resolve()) or not label.resolve().is_relative_to(source.resolve()):
                    raise ValueError(f'图片索引路径越过输入目录：{image}')
                if not image.is_file() or not label.is_file():
                    raise ValueError(f'原名副本图片或XML缺失：{image}；{label}')
                category=relative.parts[1] if relative.parts[0] in ('train','val','test') else relative.parts[0]
                if category not in categories: category=sorted(categories)[0]
                sample_ids[str(image)]=row['uid']
                jobs.append((image,label,category))
                update_progress(len(jobs))
        else:
            for category in config['classes']:
                folder = source/category
                if not folder.is_dir():
                    raise ValueError(f'类别目录不存在：{folder}')
                for image, label in iter_images(folder, None, 'voc'):
                    jobs.append((image,label,category))
                    update_progress(len(jobs))
        jobs.sort(key=lambda r: str(r[0]))
        log(f'选定 {len(config["classes"])} 类，共 {len(jobs):,} 个归档图片路径')
    if not jobs:
        raise ValueError('选定目录中没有图片')
    workers = config['scan'].get('workers', 4)
    # Outer workers control parallelism; prevent one OpenCV pool per worker.
    import cv2
    cv2.setNumThreads(1)
    log(f'指标提取并发：{workers} workers；OpenCV内部线程：1；逻辑CPU：{os.cpu_count() or "未知"}')
    output.mkdir(parents=True,exist_ok=True)
    existed = manifest.is_file()
    db = open_manifest(manifest)
    try:
        if not existed:
            initialize(db,config)
        jobs_to_process, reused = _resume_jobs(db, jobs)
        if reused:
            log(f'断点续跑：缓存复用 {reused:,} 张，待计算 {len(jobs_to_process):,} 张')
        options = {'annotation':'voc','class_names':config['classes'],'box_frame':'auto',
                   'imgsz':[config['hard_filter']['imgsz']]*2,'scaleup':True,'layout':'classified',
                   'ocr':config['scan'].get('ocr',False)}
        metrics_config = load_config()
        ocr = OCRReader(metrics_config, config.get('ocr_device','cpu')) if options['ocr'] and jobs_to_process else None
        lock = threading.Lock()
        def read_ocr(image):
            with lock:
                return ocr(image)
        if ocr:
            read_ocr.device, read_ocr.providers = ocr.device, ocr.providers
        def process(job):
            image, label, category, stamp = job
            record = inspect_file(image,label,options,metrics_config,read_ocr if ocr else None)
            if str(image) in sample_ids: record['_sample_uid']=sample_ids[str(image)]
            if fingerprint(image,label) != stamp:
                raise ValueError(f'处理时图片或XML发生变化：{image}；请使用新的output重新筛选')
            return record, category, stamp
        _extract_jobs(db, jobs_to_process, reused, len(jobs), workers, process, config)
        db.execute("UPDATE metadata SET value='extracted' WHERE key='status'")
        db.commit()
        with progress_stage('3/3 重复处理与统计'):
            db.execute('BEGIN')
            deduplicate(db,config)
            if config.get('annotation_scores_csv'):
                import_ratings(db,config['annotation_scores_csv'],config)
            db.execute("UPDATE metadata SET value='complete' WHERE key='status'")
            db.commit()
            if reports: write_reports(db,output,config)
    except BaseException:
        # Deduplication is one transaction. Retry starts from the extracted records.
        db.rollback()
        raise
    finally:
        db.close()
    log(f'扫描结果：{output}')


def basename(value):
    return str(value or '').replace('\\','/').rsplit('/',1)[-1].casefold()


def import_ratings(db, path, config, images=None):
    incoming = read_annotation_scores(path)
    candidates = defaultdict(list)
    rows=list(db.execute('SELECT o.uid,o.image_path,o.label_path,s.classes_json FROM occurrences o JOIN samples s USING(uid)'))
    if images:
        lookup={row['uid']:dict(row) for row in rows}
        for uid,image in images.items():
            if uid not in lookup:
                continue
            parts=list(Path(image).parts)
            parts[max(i for i,value in enumerate(parts) if value=='images')]='labels'
            rows.append({**lookup[uid],'image_path':str(image),'label_path':str(Path(*parts).with_suffix('.txt'))})
            parts[max(i for i,value in enumerate(parts) if value=='labels')]='annotations'
            rows.append({**lookup[uid],'image_path':str(image),'label_path':str(Path(*parts).with_suffix('.xml'))})
            rows.append({**lookup[uid],'image_path':uid+Path(image).suffix,'label_path':uid+'.txt'})
    for row in rows:
        for category in json.loads(row['classes_json']):
            candidates[category,basename(row['image_path'])].append(dict(row))
    by_uid = defaultdict(list)
    for row in rows:
        by_uid[row['uid']].append(dict(row))
    assigned, counts = defaultdict(list), Counter(incoming['stats'])
    for record in incoming['records']:
        category=record['category']
        categories=[category] if category else config['classes']
        choices=[row for label in categories for row in candidates[label,basename(record['image_path'])]]
        if record['label_path']:
            choices=[r for r in choices if basename(r['label_path'])==basename(record['label_path'])]
        normalize=lambda value: value.replace('\\','/').casefold()
        relative=Path(record['image_path'])
        references={normalize(record['image_path']),normalize(str((Path(record['source_file']).parent/relative).resolve()))}
        exact=[r for r in choices if normalize(r['image_path']) in references]
        if not exact and not relative.is_absolute() and '..' not in relative.parts:
            suffix='/'+normalize(record['image_path'])
            exact=[r for r in choices if normalize(r['image_path']).endswith(suffix)]
        in_class=[r for r in choices if category and category in r['image_path'].replace('\\','/').split('/')[:-1]]
        choices=exact or in_class or choices
        if record.get('sample_id'):
            choices=[row for row in by_uid.get(record['sample_id'], [])
                     if not category or category in json.loads(row['classes_json'])]
        matches={r['uid'] for r in choices}
        if len(matches)!=1:
            counts['unmatched_rows' if not matches else 'ambiguous_rows']+=1
            continue
        if record['b1'] is None and record['b2'] is None:
            counts['overall_score_rows']+=1
        uid=next(iter(matches))
        retained=json.loads(choices[0]['classes_json'])
        for label in [category] if category else retained:
            assigned[uid,label].append(record)
    db.execute('DELETE FROM ratings')
    for (uid,category),records in assigned.items():
        values={key:{row[key] for row in records if row[key] is not None} for key in ('b1','b2','legacy_score')}
        if values['b1'] or values['b2']:
            values['legacy_score']=set()
        if any(len(numbers)>1 for numbers in values.values()):
            counts['conflicting_samples']+=1
            continue
        r={**records[0],**{key:next(iter(numbers),None) for key,numbers in values.items()}}
        db.execute('INSERT INTO ratings VALUES (?,?,?,?,?)',(uid,category,r['b1'],r['b2'],dumps(r)))
        counts['matched_class_samples']+=1
    db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('rating_import',dumps(dict(counts))))
    log('大模型评分：'+str(path))
    update_scores(db,config)


def update_scores(db,config):
    for row in db.execute('SELECT uid,payload_json,kept_indices_json FROM samples').fetchall():
        values = list(db.execute('SELECT b1,b2,source_json FROM ratings WHERE uid=?',(row['uid'],)))
        categories=json.loads(db.execute('SELECT classes_json FROM samples WHERE uid=?',(row['uid'],)).fetchone()[0])
        annotation={}
        for i,key in enumerate(('b1','b2')):
            numbers=[r[i] for r in values if r[i] is not None]
            # category ratings remain separate; combine category dimension scores after scoring.
            if numbers: annotation[key]=numbers[0]
        overall=[json.loads(r[2]).get('legacy_score') for r in values]
        overall=[number for number in overall if number is not None]
        if overall and not annotation:
            annotation['score']=overall[0]
        payload=json.loads(row['payload_json'])
        score=score_record(payload,json.loads(row['kept_indices_json']),config,annotation)
        for i,key in enumerate(('b1','b2')):
            numbers=[r[i] for r in values if r[i] is not None]
            if numbers:
                score['dimensions'][key]['score']=sum(numbers)/len(numbers)
                score['dimensions'][key]['evidence']={'ratings_by_category':numbers,'aggregation':'equal_category_mean'}
                score['dimensions'][key]['coverage']=len(numbers)/len(categories) if categories else 0
        if overall and all(annotation.get(key) is None for key in ('b1','b2')):
            score['dimensions']['b']['score']=sum(overall)/len(overall)
            score['dimensions']['b']['coverage']=len(overall)/len(categories) if categories else 0
            score['dimensions']['b']['evidence']={'ratings_by_category':overall,'aggregation':'equal_category_mean'}
        score=combine_dimensions(score['dimensions'],validate_weights(config.get('weights',{})),config.get('scoring'))
        new=config.get('reweight',{}).get('weights')
        weighted=combine_dimensions(score['dimensions'],validate_weights(new),config.get('scoring')) if new else score
        basis=config.get('reweight',{}).get('score_source','image')
        if basis=='image':
            sampling_score=weighted['image_score']
        elif basis=='complete':
            sampling_score=weighted['total_score'] if weighted['complete'] else None
        elif basis=='available':
            sampling_score=weighted['total_score']
        else:
            raise ValueError('reweight.score_source须为image、complete或available')
        db.execute('UPDATE samples SET image_quality_score=?,total_quality_score=?,quality_weighted_score=?,scores_json=? WHERE uid=?',
                   (score['image_score'],score['total_score'],sampling_score,dumps(score),row['uid']))


def rescore(config,output):
    with open_manifest(Path(output)/'experiment.sqlite') as db:
        meta=dict(db.execute('SELECT key,value FROM metadata'))
        if meta.get('status')!='complete': raise ValueError('扫描尚未完成')
        previous=json.loads(meta['config'])
        # Rescore changes weights/anchors/CSV only; hard decisions require a new scan.
        for key in ('classes','hard_filter','source','near_duplicates'):
            if previous.get(key)!=config.get(key): raise ValueError(f'{key} 已变化，请使用新 run_name 重新 scan')
        if previous.get('scan',{}).get('ocr',False)!=config.get('scan',{}).get('ocr',False):
            raise ValueError('OCR启用状态已变化，请使用新 run_name 重新 scan')
        if config.get('annotation_scores_csv'):
            import_ratings(db,config['annotation_scores_csv'],config)
        else:
            db.execute('DELETE FROM ratings')
            db.execute("DELETE FROM metadata WHERE key='rating_import'")
            update_scores(db,config)
        db.execute('UPDATE metadata SET value=? WHERE key=?',(dumps(config),'config'))
        write_reports(db,output,config)


REASON_NAMES = {
    'exact_duplicate_extra':'精确重复多余样本','same_image_annotation_versions':'同图标注版本差异（记录）',
    'no_convertible_selected_targets':'没有可转换的选定类别框',
    'near_duplicate_candidate':'近重复候选','near_search_truncated':'近重复搜索达上限',
    'image_unavailable':'图像不可用','annotation_unavailable':'标签缺失或损坏',
    'annotation_frame':'图像与标签坐标系待复核','low_resolution':'图像短边低于阈值',
    'box_geometry':'标注框几何异常','unknown_class':'标签类别不在实验列表',
    'tiny_target':'缩放后目标短边不超过阈值','large_box':'框面积占比超过阈值',
    'duplicate_box':'同类别同坐标重复框','no_valid_targets':'没有保留目标',
    'unlocatable_target':'问题目标无法定位','mask_overlap':'问题区域与保留目标相交',
}


def write_reports(db,output,config):
    output=Path(output)
    (output/'run_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    counts=Counter(); sums=defaultdict(list); reason_counts=Counter()
    with (output/'逐图筛选评分.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['样本ID','图片路径','标签路径','类别','筛选状态','筛选原因','保留框数','问题框数',
                         '图像质量分','当前综合分','评分覆盖率(%)','重采样依据分','a1亮度曝光','a2清晰度','a3文字水印遮挡','a4目标占比','b1类别准确性','b2区域一致性'])
        for row in db.execute('SELECT * FROM samples ORDER BY uid'):
            categories=json.loads(row['classes_json']); score=json.loads(row['scores_json']); dimensions=score['dimensions']
            reasons=json.loads(row['reject_reasons_json']); decisions=json.loads(row['box_decisions_json'])
            status={'keep':'保留','review':'待复核','reject':'筛除'}[row['clean_status']]
            writer.writerow([row['uid'],row['image_path'],row['label_path'],';'.join(categories),status,
                             ';'.join(REASON_NAMES.get(r,r) for r in reasons),len(json.loads(row['clean_boxes_json'])),
                             sum(bool(d.get('reasons')) for d in decisions),row['image_quality_score'],row['total_quality_score'],
                             round(score['coverage']*100,2),row['quality_weighted_score'],
                             *[dimensions[k]['score'] for k in ('a1','a2','a3','a4','b1','b2')]])
            for category in ['总体',*categories]:
                counts[category,'总数']+=1; counts[category,status]+=1
                if row['baseline_eligible']: counts[category,'原始可训练']+=1
                if row['clean_status']=='keep' and not row['baseline_eligible']: counts[category,'局部修复']+=1
                if row['clean_status']=='keep':
                    for key in ('image_quality_score','total_quality_score'):
                        if row[key] is not None: sums[category,key].append(row[key])
                    sums[category,'coverage'].append(score['coverage']*100)
                    for key in ('a1','a2','a3','a4','b1','b2'):
                        if dimensions[key]['score'] is not None: sums[category,key].append(dimensions[key]['score'])
                for reason in set(reasons): reason_counts[category,reason]+=1
    with (output/'类别筛选统计.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['类别','统计样本数','原始可训练数','保留数','待复核数','筛除数','筛除比例(%)','局部修复保留数','保留图像质量均分','保留当前综合均分','保留评分覆盖率均值(%)',
                         'a1亮度曝光均分','a2清晰度均分','a3文字水印遮挡均分','a4目标占比均分','b1类别准确性均分','b2区域一致性均分'])
        for category in ['总体',*config['classes']]:
            mean=lambda key: round(sum(sums[category,key])/len(sums[category,key]),2) if sums[category,key] else ''
            total=counts[category,'总数']
            writer.writerow([category,total,counts[category,'原始可训练'],counts[category,'保留'],counts[category,'待复核'],counts[category,'筛除'],
                             round(100*counts[category,'筛除']/total,2) if total else 0,counts[category,'局部修复'],mean('image_quality_score'),mean('total_quality_score'),mean('coverage'),
                             *[mean(k) for k in ('a1','a2','a3','a4','b1','b2')]])
    with (output/'筛选原因统计.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['类别','原因','涉及样本数','占该类别比例(%)'])
        for (category,reason),number in sorted(reason_counts.items()):
            writer.writerow([category,REASON_NAMES.get(reason,reason),number,round(100*number/counts[category,'总数'],2)])
    db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('summary',dumps({'sample_units':counts['总体','总数'],
               'archive_paths':db.execute('SELECT COUNT(*) FROM occurrences').fetchone()[0],
               'keep':counts['总体','保留'],'review':counts['总体','待复核'],'reject':counts['总体','筛除']})))
    db.commit()
