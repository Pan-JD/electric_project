"""Attach partial model scores to cached image/annotation identities; never opens originals."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path


def filename(value):
    return str(value or '').replace('\\', '/').rsplit('/', 1)[-1].casefold()


def normalized_path(value):
    return str(value or '').replace('\\', '/').rstrip('/').casefold()


class ScoreIndex:
    def __init__(self, database, directory=None, default_class='jyz_pl', classes=(), alias='scoredb'):
        if alias not in ('scoredb', 'afterscoredb'):
            raise ValueError('Invalid score database alias')
        self.alias = alias
        self.path = None
        self.enabled = directory is not None
        self.info = {'enabled': self.enabled}
        if not self.enabled:
            return
        directory = Path(directory).resolve()
        if not directory.is_dir() and not (directory.is_file() and directory.suffix.lower() == '.csv'):
            raise ValueError(f'评分目录不存在：{directory}')
        if default_class and default_class not in classes:
            raise ValueError(f'评分默认类别不存在于本轮扫描：{default_class}，请通过 --score-class 指定')
        files = [directory] if directory.is_file() else sorted(p for p in directory.rglob('*') if p.is_file() and p.suffix.lower()=='.csv')
        if not files:
            raise ValueError(f'评分目录中没有 CSV：{directory}')
        stamp = hashlib.sha256()
        stamp.update(f'scores-v4\0{database.resolve()}\0{database.stat().st_size}\0{database.stat().st_mtime_ns}\0{default_class}'.encode())
        contents = []
        for path in files:
            data = path.read_bytes()
            stamp.update(str(path).encode())
            stamp.update(data)
            contents.append((path, data))
        self.path = database.parent/f'review_scores_{stamp.hexdigest()[:20]}.sqlite'
        if self.path.is_file():
            try:
                with sqlite3.connect(self.path.as_uri()+'?mode=ro', uri=True) as db:
                    self.info = json.loads(db.execute('SELECT payload FROM metadata').fetchone()[0])
                return
            except (sqlite3.Error, TypeError, ValueError):
                # A failed prior import can be rebuilt without changing the audit database.
                pass
        self.build(database, directory, default_class, set(classes), contents)

    def build(self, database, directory, default_class, classes, contents):
        counters = Counter()
        records = defaultdict(list)
        accepted_files = []
        for path, data in contents:
            reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
            if not {'image_path','score'} <= set(reader.fieldnames or []):
                counters['ignored_csv_files'] += 1
                continue
            accepted_files.append(str(path))
            for number, row in enumerate(reader, 2):
                counters['input_rows'] += 1
                image, label = (row.get('image_path') or '').strip(), (row.get('label_path') or '').strip()
                category = (row.get('class') or row.get('class_code') or row.get('category') or row.get('类别') or '').strip()
                if not category:
                    parts = image.replace('\\','/').split('/')[:-1]
                    category = next((part for part in reversed(parts) if part in classes), '')
                if not category:
                    category = path.parent.name if path.parent.name in classes else default_class
                try:
                    value = Decimal(str(row.get('score','')).strip())
                    if not value.is_finite() or value != int(value) or not 0 <= value <= 5:
                        raise ValueError()
                    score = int(value)
                except (InvalidOperation, ValueError, OverflowError):
                    counters['invalid_rows'] += 1
                    continue
                if not image or category not in classes:
                    counters['invalid_rows'] += 1
                    continue
                records[category].append({'image':image, 'label':label, 'score':score,
                                          'source_file':str(path), 'source_row':number})
        if not accepted_files:
            raise ValueError('评分目录中没有包含 image_path、score 字段的 CSV')

        # Restrict filename enumeration to scored classes; only reads cached path columns.
        assignments = defaultdict(list)
        with sqlite3.connect(database.as_uri()+'?mode=ro', uri=True) as audit:
            for category, incoming in records.items():
                candidates = defaultdict(list)
                wanted = {filename(r['image']) for r in incoming}
                for uid, image, label in audit.execute('''SELECT f.uid,f.path,f.label_path FROM files f
                        JOIN unit_classes c ON c.uid=f.uid WHERE c.class=?''', (category,)):
                    if filename(image) in wanted:
                        candidates[filename(image)].append((uid,image,label))
                for record in incoming:
                    choices = candidates[filename(record['image'])]
                    if record['label']:
                        choices = [r for r in choices if filename(r[2]) == filename(record['label'])]
                    exact = [r for r in choices if normalized_path(r[1]) == normalized_path(record['image'])]
                    if exact:
                        choices = exact
                    else:
                        # Prefer the occurrence stored in the requested class directory.
                        in_class = [r for r in choices if category in r[1].replace('\\','/').split('/')[:-1]]
                        if in_class:
                            choices = in_class
                    identities = {r[0] for r in choices}
                    if not identities:
                        counters['unmatched_rows'] += 1
                    elif len(identities) > 1:
                        counters['ambiguous_rows'] += 1
                    else:
                        assignments[next(iter(identities)),category].append(record)

        rows = []
        for (uid, category), values in assignments.items():
            if len({v['score'] for v in values}) != 1:
                counters['conflicting_samples'] += 1
                continue
            counters['duplicate_rows'] += len(values)-1
            record = values[0]
            rows.append((uid,category,record['score'],record['source_file'],record['source_row'],
                         record['image'],record['label']))
        by_class = {}
        for _, category, score, *_ in rows:
            counts = by_class.setdefault(category, {str(i):0 for i in range(0,6)})
            counts[str(score)] += 1
        self.info = {'enabled':True,'directory':str(directory),'default_class':default_class,
                     'csv_files':accepted_files,'matched_samples':len({r[0] for r in rows}),
                     'matched_class_samples':len(rows),'by_class':by_class,
                     'score_counts':{str(score):len({r[0] for r in rows if r[2]==score}) for score in range(0,6)},
                     **{key:counters[key] for key in ('input_rows','invalid_rows','unmatched_rows',
                        'ambiguous_rows','conflicting_samples','duplicate_rows','ignored_csv_files')}}
        with sqlite3.connect(self.path) as db:
            db.executescript('''DROP TABLE IF EXISTS scores; DROP TABLE IF EXISTS metadata;
                CREATE TABLE scores(uid TEXT, class TEXT, score INTEGER, source_file TEXT,
                    source_row INTEGER, image_name TEXT, label_name TEXT, PRIMARY KEY(uid,class));
                CREATE INDEX score_filter ON scores(score,class,uid);
                CREATE TABLE metadata(payload TEXT);''')
            db.executemany('INSERT INTO scores VALUES (?,?,?,?,?,?,?)', rows)
            db.execute('INSERT INTO metadata VALUES (?)', (json.dumps(self.info,ensure_ascii=False),))

    def attach(self, db):
        if self.enabled:
            db.execute(f'ATTACH DATABASE ? AS {self.alias}', (self.path.as_uri()+'?mode=ro',))

    def values(self, db, uid, category=''):
        if not self.enabled:
            return []
        sql = f'SELECT class,score,source_file,source_row,image_name,label_name FROM {self.alias}.scores WHERE uid=?'
        parameters = [uid]
        if category:
            sql += ' AND class=?'
            parameters.append(category)
        return [dict(row) for row in db.execute(sql+' ORDER BY class', parameters)]

    def condition(self, selected, category=''):
        selected = str(selected or '')
        if selected not in ('','scored','unscored','0','1','2','3','4','5'):
            raise ValueError('评分筛选应为 0～5、scored 或 unscored')
        if not selected:
            return '', []
        if not self.enabled:
            raise ValueError('请通过 --scores-dir 指定大模型评分 CSV 目录')
        clause, parameters = 's.uid=u.uid', []
        if category:
            clause += ' AND s.class=?'
            parameters.append(category)
        if selected.isdigit():
            clause += ' AND s.score=?'
            parameters.append(int(selected))
        exists = f'EXISTS (SELECT 1 FROM {self.alias}.scores s WHERE {clause})'
        return ('NOT '+exists if selected == 'unscored' else exists), parameters
