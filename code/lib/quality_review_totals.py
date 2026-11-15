"""Join exported quality scores to the existing audit by category + original filename."""
import csv
import hashlib
import io
import json
import math
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path


FIELDS = {'total':'综合质量分','image':'图像质量分','a1':'a1曝光亮度','a2':'a2整体清晰度',
          'a3':'a3文字水印遮挡','a4':'a4目标区域占比','b':'b整体标注质量',
          'b1':'b1类别准确性','b2':'b2区域一致性'}


def filename(value):
    return str(value or '').replace('\\','/').rsplit('/',1)[-1].casefold()


class QualityIndex:
    def __init__(self,database,source=None):
        self.info={'enabled':source is not None,'fields':FIELDS}
        self.path=None
        if source is None: return
        source=Path(source).resolve()
        if source.is_dir(): source=source/'逐图评分.csv'
        data=source.read_bytes()
        stamp=f'quality-v2:{database.resolve()}:{database.stat().st_size}:{database.stat().st_mtime_ns}:{source}'.encode()
        self.path=database.parent/('review_quality_'+hashlib.sha256(stamp+data).hexdigest()[:20]+'.sqlite')
        if self.path.exists():
            try:
                with closing(sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True)) as db:
                    self.info=json.loads(db.execute('SELECT value FROM metadata').fetchone()[0])
                return
            except (sqlite3.Error,TypeError,ValueError): pass
        reader=csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
        if not {'原文件名','类别','综合质量分','图像质量分','评分覆盖率(%)'}<=set(reader.fieldnames or []):
            raise ValueError('请提供评分程序输出的逐图评分.csv')
        incoming=defaultdict(list);counts=Counter()
        for number,row in enumerate(reader,2):
            counts['input_rows']+=1
            name=filename(row['原文件名']);categories=set(row['类别'].split(';'))-{''}
            try:
                if not name or not categories: raise ValueError()
                values={}
                for key,column in FIELDS.items():
                    raw=(row.get(column) or '').strip()
                    value=float(raw) if raw else None
                    if value is not None and (not math.isfinite(value) or not 0<=value<=5): raise ValueError()
                    values[key]=value
                coverage=float(row['评分覆盖率(%)'])
                if not math.isfinite(coverage) or not 0<=coverage<=100: raise ValueError()
            except (ValueError,TypeError):
                counts['invalid_rows']+=1;continue
            entry={'name':name,'values':values,'coverage':coverage,'sample_id':row.get('样本ID',''),'source_row':number}
            for category in categories: incoming[category].append(entry)
        assignments=defaultdict(list)
        with closing(sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)) as audit:
            for category,entries in incoming.items():
                wanted={e['name'] for e in entries};matches=defaultdict(set)
                for uid,path in audit.execute('''SELECT f.uid,f.path FROM unit_classes c
                         JOIN files f ON f.uid=c.uid WHERE c.class=?''',(category,)):
                    name=filename(path)
                    if name in wanted: matches[name].add(uid)
                for entry in entries:
                    candidates=matches[entry['name']]
                    if len(candidates)!=1:
                        counts['unmatched_class_rows' if not candidates else 'ambiguous_class_rows']+=1
                        continue
                    assignments[next(iter(candidates)),category].append(entry)
        rows=[]
        for (uid,category),entries in assignments.items():
            signatures={json.dumps([e['values'],e['coverage'],e['sample_id']],sort_keys=True) for e in entries}
            if len(signatures)>1:
                counts['conflicting_class_samples']+=1;continue
            entry=entries[0]
            rows.append((uid,category,*entry['values'].values(),entry['coverage'],entry['sample_id'],entry['source_row']))
        self.info={**self.info,**{key:counts[key] for key in ('input_rows','invalid_rows','unmatched_class_rows',
                    'ambiguous_class_rows','conflicting_class_samples')},'source':str(source),
                   'matched_samples':len({r[0] for r in rows}),'matched_class_samples':len(rows),
                   'by_class':dict(Counter(r[1] for r in rows))}
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executescript('DROP TABLE IF EXISTS scores; DROP TABLE IF EXISTS metadata;')
            db.execute('CREATE TABLE scores(uid TEXT,class TEXT,'+','.join(k+' REAL' for k in FIELDS)+
                       ',coverage REAL,sample_id TEXT,source_row INTEGER,PRIMARY KEY(uid,class))')
            db.executemany('INSERT INTO scores VALUES ('+','.join('?' for _ in range(len(FIELDS)+5))+')',rows)
            db.execute('CREATE TABLE metadata(value TEXT)')
            db.execute('INSERT INTO metadata VALUES (?)',(json.dumps(self.info,ensure_ascii=False),))

    def attach(self,db):
        if self.path: db.execute('ATTACH DATABASE ? AS qualitydb',(self.path.as_uri()+'?mode=ro',))

    def values(self,db,uid,label=''):
        if not self.path: return []
        return [dict(row) for row in db.execute('SELECT * FROM qualitydb.scores WHERE uid=?'+
                  (' AND class=?' if label else '')+' ORDER BY class',[uid,label] if label else [uid])]

    def condition(self,args,label=''):
        field=args.get('quality_field','')
        if not field: return '',[]
        if not self.path: raise ValueError('未加载综合评分文件')
        if field not in FIELDS: raise ValueError('未知综合评分指标')
        low=float(args.get('quality_min') or 0);high=float(args.get('quality_max') or 5)
        coverage=float(args.get('quality_coverage') or 0)
        if not 0<=low<=high<=5 or not 0<=coverage<=100: raise ValueError('分数范围为0～5，覆盖率为0～100')
        sql=f'EXISTS (SELECT 1 FROM qualitydb.scores qs WHERE qs.uid=u.uid AND qs.{field} BETWEEN ? AND ? AND qs.coverage>=?'
        parameters=[low,high,coverage]
        if label: sql+=' AND qs.class=?';parameters.append(label)
        return sql+')',parameters
