"""Serve a paginated, local review UI for an existing quality audit cache."""
from __future__ import annotations

import simple_paths  # Project-internal support modules.

import argparse
import csv
import io
import json
import sqlite3
import sys
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from PIL import Image, ImageOps, UnidentifiedImageError

from quality_reports import DUPLICATES, ISSUES, EXTRA_ISSUES, evidence_text
from quality_review_metrics import MetricIndex, parse_thresholds, threshold_sql
from quality_paths import environment_path_maps, map_stored_path, run_directory, resolve_project_input
from quality_review_scores import ScoreIndex
from quality_review_comparison import ScoreComparison
from quality_review_totals import QualityIndex

# Reuse the standard-library XLSX reader from the classification tools.
from count_labels import read_label_table


ASSETS = Path(__file__).with_name('quality_review')
DEFAULT_LABEL_TABLE = Path(__file__).resolve().parents[1] / 'Docs' / '变电领域标注标签.xlsx'
REASONS = {
    'EXIF_orientation_requires_box_frame': 'EXIF 带有方向信息，需要确认框基于编码像素还是显示方向',
    'XML_size_differs_from_selected_canvas': 'XML 声明尺寸与扫描使用的图片尺寸不同',
    'invalid_EXIF_orientation': 'EXIF 方向值无效',
    'reversed_or_zero_area': '框坐标倒置或框面积为零',
    'missing_or_nonfinite_coordinates': '框坐标缺失或含无效数值',
    'outside_canvas': '标注框超出图片边界',
}


def box_key(box):
    return box['class'], tuple(box['coords']) if box.get('coords') is not None else None


def annotation_difference(left, right):
    """Compare class/coordinate multisets, preserving repeated identical boxes."""
    def compare(boxes, other):
        remaining = Counter(box_key(box) for box in other)
        result = []
        for box in boxes:
            key = box_key(box)
            common = remaining[key] > 0
            if common:
                remaining[key] -= 1
            result.append({**box, 'difference': not common})
        return result

    return {'left': compare(left, right), 'right': compare(right, left)}


class ReviewStore:
    def __init__(self, results, path_maps=(), label_table=DEFAULT_LABEL_TABLE, scores_dir=None, score_class='jyz_pl', scores_before=None, scores_after=None, quality_scores=None):
        self.results = resolve_project_input(results)
        self.database = self.results / '_cache' / 'audit.sqlite'
        if not self.database.is_file():
            raise ValueError(f'找不到扫描缓存 {self.database}')
        self.path_maps = list(path_maps)
        with self.connect() as db:
            meta = dict(db.execute('SELECT key,value FROM meta'))
        if meta.get('status') != 'complete':
            raise ValueError('请等待扫描完成后启动复核网页')
        self.summary = json.loads(meta['summary'])
        self.settings = json.loads(meta['settings'])
        self.stats = self.read_stats('总体问题统计.csv')
        self.class_stats = self.read_stats('各类问题统计.csv')
        self.classes = sorted({r['类别'] for r in self.class_stats})
        self.label_table = resolve_project_input(label_table)
        labels = read_label_table(self.label_table) if self.label_table.is_file() else {}
        self.class_names = {code: name or business for code, (name, business) in labels.items()
                            if code in self.classes and (name or business)}
        self.titles = dict(ISSUES + EXTRA_ISSUES)
        self.metric_index = MetricIndex(self.database, self.summary['sample_units'])
        if (scores_before is None) != (scores_after is None):
            raise ValueError('--scores-before 和 --scores-after 需要同时指定')
        directory = resolve_project_input(scores_dir) if scores_dir is not None else None
        if scores_before is None and directory is not None:
            baseline, revised = directory/'scores.csv', directory/'jyz_pl_v3_scores.csv'
            if baseline.is_file() and revised.is_file():
                scores_before, scores_after = baseline, revised
        before = resolve_project_input(scores_before) if scores_before is not None else directory
        after = resolve_project_input(scores_after) if scores_after is not None else None
        if after is not None and (not before.is_file() or not after.is_file() or before == after):
            raise ValueError('前后版本请指定两个不同的 CSV 文件')
        self.score_index = ScoreIndex(self.database, before, score_class, self.classes)
        self.after_score_index = ScoreIndex(self.database, after, score_class, self.classes, alias='afterscoredb')
        self.score_comparison = ScoreComparison(self.score_index, self.after_score_index)
        self.quality_index = QualityIndex(self.database, resolve_project_input(quality_scores) if quality_scores else None)

    @contextmanager
    def connect(self):
        # Audit results are read-only. A request owns its connection and cursor.
        db = sqlite3.connect(self.database.as_uri() + '?mode=ro', uri=True, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        if getattr(self, 'score_index', None):
            self.score_index.attach(db)
        if getattr(self, 'after_score_index', None):
            self.after_score_index.attach(db)
        if getattr(self, 'quality_index', None):
            self.quality_index.attach(db)
        try:
            yield db
        finally:
            db.close()

    def read_stats(self, name):
        with (self.results / name).open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))

    def metadata(self):
        return {'summary': {key: self.summary[key] for key in
                            ('file_paths', 'sample_units', 'collapsed_classification_copies', 'issue_union_count')},
                'stats': self.stats, 'class_stats': self.class_stats, 'classes': self.classes,
                'class_names': self.class_names,
                'label_scores': self.score_index.info,
                'score_comparison': self.score_comparison.info,
                'quality_scores': self.quality_index.info,
                'issues': [{'key': key, 'title': title} for key, title in self.titles.items()],
                'results': str(self.results), 'imgsz': self.settings['options']['imgsz'],
                'metric_scales': {key: self.settings.get('config', {}).get(key) for key in
                                  ('image_metric_side', 'roi_metric_side')},
                'metric_index': self.metric_index.status()}

    @lru_cache(maxsize=64)
    def filtered_count(self, sql, parameters):
        with self.connect() as db:
            db.execute('ATTACH DATABASE ? AS metric', (self.metric_index.path.as_uri() + '?mode=ro',))
            return db.execute('SELECT COUNT(*) ' + sql, parameters).fetchone()[0]

    def samples(self, label='', issue='', after='', query='', limit=24, thresholds=None, population='reported', score='', score_version='before', score_change='', before_score='', after_score=''):
        if issue and issue not in self.titles:
            raise ValueError('未知问题类型')
        if label and label not in self.classes:
            raise ValueError('未知类别')
        if population not in ('reported', 'all'):
            raise ValueError('未知样本范围')
        filters = parse_thresholds(thresholds)
        active = filters['active']
        if score_version not in ('before', 'after'):
            raise ValueError('未知评分版本')
        index = self.after_score_index if score_version == 'after' else self.score_index
        score_condition, score_parameters = index.condition(score, label)
        comparison_condition, comparison_parameters = self.score_comparison.condition(score_change, before_score, after_score, label)
        if comparison_condition:
            score_condition = f'({score_condition}) AND ({comparison_condition})' if score_condition else comparison_condition
            score_parameters.extend(comparison_parameters)
        quality_condition, quality_parameters = self.quality_index.condition(thresholds or {},label)
        if quality_condition:
            score_condition = f'({score_condition}) AND ({quality_condition})' if score_condition else quality_condition
            score_parameters.extend(quality_parameters)
        stats = [r for r in self.class_stats if r['类别'] == label] if label else self.stats
        union = next(r for r in stats if r['问题'] == '任一已检出问题（去重）')
        scope_total = int(union['图像总数'])
        issue_count = int(next((r['问题图像数'] for r in stats if issue and r['问题'] == self.titles[issue]), 0) or 0)
        if issue and not issue_count:
            return {'items': [], 'has_more': False, 'next': '', 'total': 0,
                    'scope_total': scope_total, 'thresholds_active': active}
        if active and not self.metric_index.ensure_ready():
            status = self.metric_index.status()
            if status['state'] == 'error':
                raise ValueError('阈值索引建立失败：' + status['error'])
            return {'items': [], 'has_more': False, 'next': '', 'pending': True, 'metric_index': status}
        limit = min(60, max(1, limit))
        parameters = []
        join = ''
        if label:
            join = 'JOIN unit_classes c ON c.uid=u.uid AND c.class=?'
            parameters.append(label)
        metric_condition, metric_parameters = threshold_sql(filters, label)
        if active:
            join += f''' JOIN (SELECT DISTINCT r.uid FROM metric.regions r
                        WHERE {metric_condition}) matched ON matched.uid=u.uid'''
            parameters.extend(metric_parameters)
        conditions = ['1=1']
        if score_condition:
            conditions.append(score_condition)
            parameters.extend(score_parameters)
        if query:
            conditions.append("f.path LIKE ? ESCAPE '\\'")
            escaped = query.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
            parameters.append('%' + escaped + '%')
        if issue or population == 'reported':
            finding = 'q.uid=u.uid'
            if issue:
                finding += ' AND q.issue=?'
                parameters.append(issue)
            else:
                finding += ' AND q.issue IN (' + ','.join('?' for _ in self.titles) + ')'
                parameters.extend(self.titles)
            if label:
                finding += " AND (q.issue NOT IN ('exact_duplicate','near_duplicate') OR json_extract(q.evidence,'$.class')=?)"
                parameters.append(label)
            conditions.append(f'EXISTS (SELECT 1 FROM findings q WHERE {finding})')
        base = f'''FROM units u {join} JOIN files f ON f.id=u.file_id
                   WHERE {' AND '.join(conditions)}'''
        sql = f'''SELECT u.uid,u.file_id,u.width,u.height,f.path {base}
                  AND u.uid>? ORDER BY u.uid LIMIT ?'''
        if active:
            total = self.filtered_count(base, tuple(parameters))
        elif not query and not score_condition:
            total = issue_count if issue else (
                int(union['问题图像数']) if population == 'reported' else scope_total)
        else:
            with self.connect() as db:
                total = db.execute('SELECT COUNT(*) ' + base, parameters).fetchone()[0]
        with self.connect() as db:
            if active:
                db.execute('ATTACH DATABASE ? AS metric', (self.metric_index.path.as_uri() + '?mode=ro',))
            rows = [dict(row) for row in db.execute(sql, [*parameters, after, limit + 1])]
            more = len(rows) > limit
            rows = rows[:limit]
            for row in rows:
                row['issues'] = sorted({r['issue'] for r in self.findings(db, row['uid'], label)})
                row['label_scores'] = self.score_index.values(db, row['uid'], label)
                row['score_comparison'] = self.score_comparison.values(db, row['uid'], label)
                row['quality_scores'] = self.quality_index.values(db,row['uid'],label)
                if active:
                    row['metric_matches'] = [dict(r) for r in db.execute(
                        f'''SELECT r.* FROM metric.regions r WHERE r.uid=? AND {metric_condition}
                            ORDER BY r.object_index LIMIT 3''', [row['uid'], *metric_parameters])]
        return {'items': rows, 'has_more': more, 'next': rows[-1]['uid'] if more else '',
                'total': total, 'scope_total': scope_total, 'thresholds_active': active}

    def findings(self, db, uid, label=''):
        items = []
        for row in db.execute('SELECT issue,object_index,evidence FROM findings WHERE uid=?', (uid,)):
            if row['issue'] not in self.titles:
                continue
            detail = json.loads(row['evidence'])
            if label and row['issue'] in DUPLICATES and detail.get('class') != label:
                continue
            items.append({'issue': row['issue'], 'index': row['object_index'], 'evidence': detail})
        return items

    def record(self, db, uid):
        row = db.execute('''SELECT u.uid,u.file_id,u.sha,u.ann_sig,f.payload FROM units u
                            JOIN files f ON f.id=u.file_id WHERE u.uid=?''', (uid,)).fetchone()
        if row is None:
            raise ValueError('找不到该样本')
        payload = json.loads(row['payload'])
        return row, payload

    def detail(self, uid, label=''):
        with self.connect() as db:
            row, payload = self.record(db, uid)
            items = self.findings(db, uid, label)
            for item in items:
                issue, index, evidence = item['issue'], item['index'], item['evidence']
                if issue == 'low_resolution':
                    text = f"图片 {payload['width']} × {payload['height']} 像素，短边低于 {evidence['threshold']} 像素"
                elif issue == 'coordinate_frame':
                    text = '；'.join(REASONS.get(r, r) for r in evidence['reasons'])
                    text += f"，EXIF 方向值 {evidence.get('orientation')}"
                elif issue == 'box_geometry':
                    text = f"目标 {index+1} {evidence.get('class', '')}，{REASONS.get(evidence.get('reason'), evidence.get('reason', ''))}，坐标 {evidence.get('coords')}"
                elif issue == 'exact_duplicate':
                    text = f"同类别 {evidence['class']} 中图片文件内容相同，组内 {evidence['members']} 个统计样本"
                elif issue == 'near_duplicate':
                    text = f"同类别 {evidence['class']} 中视觉相似，SSIM {evidence.get('ssim', 0):.4f}，pHash 距离 {evidence.get('phash_distance')}，dHash 距离 {evidence.get('dhash_distance')}"
                elif issue == 'annotation_conflict':
                    text = f"相同图片文件内容存在 {evidence['annotation_versions']} 种类别/框坐标组合，可展开标注版本对照"
                elif issue in ('annotation_unavailable', 'unavailable'):
                    text = evidence.get('error') or evidence.get('status') or payload.get('error', '')
                else:
                    text = evidence_text(issue, index, evidence)
                item['text'], item['title'] = text, self.titles[issue]
            occurrences = db.execute('SELECT COUNT(*) FROM files WHERE uid=?', (uid,)).fetchone()[0]
            paths = [dict(r) for r in db.execute('SELECT path,label_path FROM files WHERE uid=? LIMIT 5', (uid,))]
            label_scores = self.score_index.values(db, uid, label)
            score_comparison = self.score_comparison.values(db, uid, label)
            quality_scores = self.quality_index.values(db,uid,label)
        boxes = [{k: b.get(k) for k in ('index', 'class', 'coords', 'valid', 'eligible', 'outside',
                                       'short_at_model', 'area_fraction', 'metrics')} for b in payload['boxes']]
        return {'uid': uid, 'file_id': row['file_id'], 'path': payload['path'],
                'label_path': payload['label_path'], 'width': payload.get('width'), 'height': payload.get('height'),
                'raw_width': payload.get('raw_width'), 'raw_height': payload.get('raw_height'),
                'orientation': payload.get('orientation'), 'frame_ok': payload.get('frame_ok'),
                'frame_reasons': payload.get('frame_reasons', []),
                'declared_size': payload.get('annotation', {}).get('declared_size'),
                'annotation_status': payload.get('annotation', {}).get('status'),
                'signature': row['ann_sig'], 'classes': payload['classes'], 'boxes': boxes,
                'findings': items, 'occurrences': occurrences, 'paths': paths,
                'label_scores': label_scores,
                'score_comparison': score_comparison,
                'quality_scores': quality_scores,
                'metrics': payload.get('metrics', {}),
                'scan_frame': self.settings['options'].get('box_frame', 'auto')}

    def related(self, uid, kind, label='', offset=0, limit=12):
        if kind not in ('exact', 'near', 'annotation'):
            raise ValueError('未知对照类型')
        limit, offset = min(24, max(1, limit)), max(0, offset)
        with self.connect() as db:
            row, _ = self.record(db, uid)
            if kind == 'annotation':
                # One example for each different signature, independent of XML object order.
                sql = '''SELECT u.uid,u.file_id,u.width,u.height,f.path FROM units u
                         JOIN files f ON f.id=u.file_id WHERE u.sha=? AND u.ann_sig<>?
                         AND u.ann_sig<>'' GROUP BY u.ann_sig ORDER BY u.ann_sig LIMIT ? OFFSET ?'''
                parameters = (row['sha'], row['ann_sig'], limit + 1, offset)
            else:
                groups = db.execute('''SELECT class,group_id FROM duplicate_members
                                       WHERE uid=? AND kind=?''', (uid, kind)).fetchall()
                groups = [g for g in groups if not label or g['class'] == label]
                if not groups:
                    return {'items': [], 'has_more': False}
                condition = ' OR '.join('(d.class=? AND d.group_id=?)' for _ in groups)
                parameters = [kind, uid]
                parameters.extend(value for g in groups for value in (g['class'], g['group_id']))
                parameters.extend((limit + 1, offset))
                sql = f'''SELECT u.uid,u.file_id,u.width,u.height,f.path,MIN(d.is_extra) extra,
                          MAX(u.uid=d.representative_uid) representative
                          FROM duplicate_members d JOIN units u USING(uid) JOIN files f ON f.id=u.file_id
                          WHERE d.kind=? AND d.uid<>? AND ({condition})
                          GROUP BY u.uid ORDER BY representative DESC,extra,u.uid LIMIT ? OFFSET ?'''
            rows = [dict(r) for r in db.execute(sql, parameters)]
        return {'items': rows[:limit], 'has_more': len(rows) > limit}

    def difference(self, left, right):
        with self.connect() as db:
            a, left_payload = self.record(db, left)
            b, right_payload = self.record(db, right)
        differences = annotation_difference(left_payload['boxes'], right_payload['boxes'])
        return {**differences, 'same_image': a['sha'] == b['sha'],
                'same_annotation': a['ann_sig'] == b['ann_sig']}

    def resolve_file(self, file_id, label=False):
        with self.connect() as db:
            row = db.execute('SELECT path,label_path FROM files WHERE id=?', (file_id,)).fetchone()
        if row is None:
            raise ValueError('找不到文件记录')
        value = row['label_path'] if label else row['path']
        if not value:
            raise FileNotFoundError('此样本没有标注文件')
        return map_stored_path(value, self.path_maps)

    def image(self, file_id, frame='encoded', size=1200):
        path = self.resolve_file(file_id)
        size = min(2400, max(160, size))
        with Image.open(path) as image:
            if frame == 'display':
                image = ImageOps.exif_transpose(image)
            image.thumbnail((size, size))
            image = image.convert('RGB')
            stream = io.BytesIO()
            image.save(stream, format='JPEG', quality=88)
        return stream.getvalue()


class ReviewHandler(BaseHTTPRequestHandler):
    store: ReviewStore

    def log_message(self, format, *args):
        if args and str(args[1]) not in ('200', '304'):
            super().log_message(format, *args)

    def send_data(self, data, content_type, status=200):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, value, status=200):
        self.send_data(json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
                       'application/json; charset=utf-8', status)

    def do_GET(self):
        request = urlsplit(self.path)
        args = {k: v[0] for k, v in parse_qs(request.query).items()}
        try:
            if request.path in ('/', '/review.js', '/review.css'):
                name = {'/': 'index.html', '/review.js': 'review.js', '/review.css': 'review.css'}[request.path]
                mime = {'index.html': 'text/html', 'review.js': 'text/javascript', 'review.css': 'text/css'}[name]
                self.send_data((ASSETS / name).read_bytes(), mime + '; charset=utf-8')
            elif request.path == '/api/meta':
                self.send_json(self.store.metadata())
            elif request.path == '/api/samples':
                self.send_json(self.store.samples(args.get('class', ''), args.get('issue', ''),
                               args.get('after', ''), args.get('query', ''), int(args.get('limit', 24)),
                               thresholds=args, population=args.get('population', 'reported'),
                               score=args.get('score',''), score_version=args.get('score_version','before'),
                               score_change=args.get('score_change',''), before_score=args.get('before_score',''),
                               after_score=args.get('after_score','')))
            elif request.path == '/api/detail':
                self.send_json(self.store.detail(args['uid'], args.get('class', '')))
            elif request.path == '/api/related':
                self.send_json(self.store.related(args['uid'], args['kind'], args.get('class', ''),
                               int(args.get('offset', 0))))
            elif request.path == '/api/difference':
                self.send_json(self.store.difference(args['left'], args['right']))
            elif request.path == '/api/image':
                self.send_data(self.store.image(int(args['id']), args.get('frame', 'encoded'),
                               int(args.get('size', 1200))), 'image/jpeg')
            elif request.path == '/api/label':
                path = self.store.resolve_file(int(args['id']), label=True)
                self.send_data(path.read_bytes(), 'text/plain; charset=utf-8')
            elif request.path == '/api/original':
                path = self.store.resolve_file(int(args['id']))
                with Image.open(path) as image:
                    mime = Image.MIME.get(image.format, 'application/octet-stream')
                self.send_data(path.read_bytes(), mime)
            elif request.path == '/favicon.ico':
                self.send_data(b'', 'image/x-icon', 204)
            else:
                self.send_json({'error': '地址不存在'}, HTTPStatus.NOT_FOUND)
        except (FileNotFoundError, UnidentifiedImageError):
            self.send_json({'error': '图片或标注文件当前不可读取，请连接对应磁盘，或使用 --path-map 指定新的根目录'}, 404)
        except (ValueError, KeyError) as exc:
            self.send_json({'error': str(exc)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (sqlite3.Error, OSError) as exc:
            self.send_json({'error': str(exc)}, 500)


def make_server(store, port=8765):
    handler = type('BoundReviewHandler', (ReviewHandler,), {'store': store})
    return ThreadingHTTPServer(('127.0.0.1', port), handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument('--results', type=Path, help='已完成的质量扫描输出目录')
    destination.add_argument('--run-name', help='读取代码根目录/run_results/名称')
    parser.add_argument('--quality-scores', type=Path, help='综合评分目录或逐图评分CSV；与--results一起使用可复用历史图片索引')
    parser.add_argument('--env', choices=['auto','local','s5'], default='auto', help='选择当前数据位置，自动映射两端的 10%% 抽样数据路径')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--label-table', type=Path, default=DEFAULT_LABEL_TABLE,
                        help='中文类别名称参考表，默认读取项目根目录 Docs/变电领域标注标签.xlsx')
    parser.add_argument('--scores-dir', type=Path, help='大模型标注评分目录，递归读取 CSV（忽略重复的 JSONL）')
    parser.add_argument('--scores-before', type=Path, help='改进前评分 CSV，与 --scores-after 配对使用；优先于目录自动选择')
    parser.add_argument('--scores-after', type=Path, help='改进后评分 CSV；目录中 scores.csv 与 jyz_pl_v3_scores.csv 自动配对')
    parser.add_argument('--score-class', default='jyz_pl', help='评分文件仅含文件名时的类别，默认 jyz_pl；CSV 内类别或路径优先')
    parser.add_argument('--path-map', nargs=2, action='append', default=[], metavar=('OLD_ROOT', 'NEW_ROOT'),
                        help='原数据根目录改变后映射图片/XML路径，可重复指定')
    args = parser.parse_args(argv)
    if args.quality_scores is not None and args.results is None and args.run_name is None:
        from quality_score_viewer import serve
        return serve(resolve_project_input(args.quality_scores), args.path_map, args.port)
    if args.results is None and args.run_name is None:
        parser.error('请指定 --results、--run-name 或 --quality-scores')
    try:
        if args.run_name is not None:
            args.results = run_directory(args.run_name)
        args.label_table = resolve_project_input(args.label_table)
        if not args.label_table.is_file():
            raise ValueError(f'找不到类别名称参考表 {args.label_table}')
        store = ReviewStore(args.results, args.path_map + environment_path_maps(args.env), args.label_table,
                            args.scores_dir, args.score_class, args.scores_before, args.scores_after,args.quality_scores)
        server = make_server(store, args.port)
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(2, str(exc) + '\n')
    print(f'复核网页 http://127.0.0.1:{server.server_port}  在浏览器打开，Ctrl+C 退出', flush=True)
    print(f'结果目录 {store.results}', flush=True)
    if store.quality_index.path:
        print(f'综合评分：{store.quality_index.info["source"]}',flush=True)
    print(f'中文类别名称 {len(store.class_names)}/{len(store.classes)} 类，来源 {store.label_table}', flush=True)
    if store.score_comparison.enabled:
        print(f'改进前评分：{store.score_index.info["directory"]}', flush=True)
        print(f'改进后评分：{store.after_score_index.info["directory"]}', flush=True)
    elif store.score_index.enabled:
        print(f'评分来源：{store.score_index.info["directory"]}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n复核网页已关闭')
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
