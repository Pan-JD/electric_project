"""Index saved scalar measurements for interactive threshold queries."""
from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
import threading
import time
from contextlib import closing
from pathlib import Path


METRICS = ('mean_gray', 'laplacian_variance', 'tenengrad')
INDEX_VERSION = 1


def finite_number(value):
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def parse_thresholds(values=None):
    """Normalize public query parameters; blank values impose no condition."""
    values = values or {}
    result = {'metric_scope': values.get('metric_scope', 'target'),
              'gray_mode': values.get('gray_mode', 'below')}
    if result['metric_scope'] not in ('image', 'target'):
        raise ValueError('指标区域需选择整图或目标区域')
    if result['gray_mode'] not in ('below', 'above', 'inside', 'outside'):
        raise ValueError('未知灰度筛选方式')
    for key in ('gray_low', 'gray_high', 'laplacian_max', 'tenengrad_max'):
        value = values.get(key)
        if value in (None, ''):
            continue
        number = finite_number(value)
        if number is None or number < 0 or (key.startswith('gray_') and number > 255):
            raise ValueError('灰度须为 0–255 的有限数值；方差和梯度能量须为非负有限数值')
        result[key] = number
    gray = any(key in result for key in ('gray_low', 'gray_high'))
    if gray:
        required = {'below': ('gray_high',), 'above': ('gray_low',),
                    'inside': ('gray_low', 'gray_high'), 'outside': ('gray_low', 'gray_high')}
        if any(key not in result for key in required[result['gray_mode']]):
            raise ValueError('请完整填写当前灰度条件的阈值')
        if result['gray_mode'] in ('inside', 'outside') and result['gray_low'] > result['gray_high']:
            raise ValueError('灰度区间下限须小于或等于上限')
    result['active'] = gray or any(key in result for key in ('laplacian_max', 'tenengrad_max'))
    return result


def threshold_sql(filters, label='', alias='r'):
    """All conditions apply to the same region, rather than different boxes."""
    conditions = [f'{alias}.scope=?']
    parameters = [filters['metric_scope']]
    if label and filters['metric_scope'] == 'target':
        conditions.append(f'{alias}.class=?')
        parameters.append(label)
    if 'gray_low' in filters or 'gray_high' in filters:
        gray = f'{alias}.mean_gray'
        mode = filters['gray_mode']
        if mode == 'below':
            conditions.append(f'{gray}<?')
            parameters.append(filters['gray_high'])
        elif mode == 'above':
            conditions.append(f'{gray}>?')
            parameters.append(filters['gray_low'])
        elif mode == 'inside':
            conditions.append(f'{gray} BETWEEN ? AND ?')
            parameters.extend((filters['gray_low'], filters['gray_high']))
        else:
            conditions.append(f'({gray}<? OR {gray}>?)')
            parameters.extend((filters['gray_low'], filters['gray_high']))
    for key, column in (('laplacian_max', 'laplacian_variance'), ('tenengrad_max', 'tenengrad')):
        if key in filters:
            conditions.append(f'{alias}.{column}<?')
            parameters.append(filters[key])
    return ' AND '.join(conditions), parameters


class MetricIndex:
    """Build lazily from the audit cache; never read images or original labels."""
    def __init__(self, database, total):
        self.source = database
        self.path = database.with_name('review_metrics.sqlite')
        stat = database.stat()
        self.identity = json.dumps([INDEX_VERSION, stat.st_size, stat.st_mtime_ns])
        self.total = total
        self.lock = threading.Lock()
        self.finished = threading.Event()
        self.state = 'ready' if self.is_current() else 'idle'
        self.processed = total if self.state == 'ready' else 0
        self.error = ''
        if self.state == 'ready':
            self.finished.set()

    def is_current(self):
        if not self.path.is_file():
            return False
        try:
            with closing(sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)) as db:
                row = db.execute("SELECT value FROM meta WHERE key='source'").fetchone()
                return row is not None and row[0] == self.identity
        except sqlite3.Error:
            return False

    def status(self):
        with self.lock:
            return {'state': self.state, 'processed': self.processed, 'total': self.total,
                    'error': self.error}

    def ensure_ready(self):
        with self.lock:
            if self.state == 'idle':
                self.state = 'building'
                threading.Thread(target=self.build, name='review-metrics', daemon=True).start()
            return self.state == 'ready'

    def build(self):
        temporary = None
        try:
            fd, filename = tempfile.mkstemp(prefix='review_metrics_', suffix='.building', dir=self.path.parent)
            os.close(fd)
            temporary = Path(filename)
            print('正在从已有扫描缓存建立阈值索引，无需读取原图…', flush=True)
            with closing(sqlite3.connect(temporary)) as target:
                target.executescript('''PRAGMA journal_mode=OFF;
                    CREATE TABLE meta (key TEXT PRIMARY KEY,value TEXT);
                    CREATE TABLE regions (
                        uid TEXT,object_index INTEGER,scope TEXT,class TEXT,
                        mean_gray REAL,laplacian_variance REAL,tenengrad REAL,
                        measurement_width INTEGER,measurement_height INTEGER,
                        PRIMARY KEY(uid,object_index)) WITHOUT ROWID;''')
                source = sqlite3.connect(self.source.as_uri() + '?mode=ro', uri=True)
                pending = []
                last = time.monotonic()
                try:
                    records = source.execute('''SELECT u.uid,f.payload FROM units u
                        JOIN files f ON f.id=u.file_id''')
                    for processed, (uid, payload) in enumerate(records, 1):
                        record = json.loads(payload)
                        regions = [(-1, 'image', '', record.get('metrics', {}))] if record.get('ok') else []
                        regions.extend((box['index'], 'target', box['class'], box.get('metrics', {}))
                                       for box in record.get('boxes', []) if box.get('eligible'))
                        for index, scope, label, metrics in regions:
                            numbers = [finite_number(metrics.get(key)) for key in METRICS]
                            if any(value is not None for value in numbers):
                                pending.append((uid, index, scope, label, *numbers,
                                                metrics.get('measurement_width'), metrics.get('measurement_height')))
                        if len(pending) >= 2000:
                            target.executemany('INSERT INTO regions VALUES (?,?,?,?,?,?,?,?,?)', pending)
                            pending.clear()
                        if processed % 1000 == 0:
                            with self.lock:
                                self.processed = processed
                        if time.monotonic() - last >= 3:
                            print(f'阈值索引 {processed:,}/{self.total:,} 样本 ({100*processed/max(1,self.total):.1f}%)', flush=True)
                            last = time.monotonic()
                    target.executemany('INSERT INTO regions VALUES (?,?,?,?,?,?,?,?,?)', pending)
                finally:
                    source.close()
                with self.lock:
                    self.processed = self.total
                print('阈值指标提取完成，正在建立查询索引…', flush=True)
                for metric in METRICS:
                    target.execute(f'CREATE INDEX region_{metric}_idx ON regions(scope,{metric})')
                target.execute('INSERT INTO meta VALUES (?,?)', ('source', self.identity))
                target.commit()
            temporary.replace(self.path)
            with self.lock:
                self.state, self.processed = 'ready', self.total
            print('阈值索引已就绪；后续启动直接复用。', flush=True)
        except (OSError, ValueError, sqlite3.Error) as exc:
            with self.lock:
                self.state, self.error = 'error', str(exc)
            print(f'阈值索引建立失败：{exc}', flush=True)
        finally:
            try:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
            finally:
                self.finished.set()
