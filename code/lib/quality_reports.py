"""Three compact Chinese business tables and optional visual/diagnostic review."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from quality_logging import update_progress


ISSUES = [
    ('exposure', '曝光异常（候选）'),
    ('watermark_occlusion', '水印遮挡（候选）'),
    ('blur', '图像模糊（候选）'),
    ('low_resolution', '低分辨率'),
    ('small_target', '目标过小'),
    ('exact_duplicate', '类内精确重复（多余）'),
    ('near_duplicate', '类内近重复（候选多余）'),
    ('large_box', '标注范围过大（候选）'),
    ('semantic_label_error', '语义错标/漏标'),
]
EXTRA_ISSUES = [
    ('annotation_conflict', '同图标注版本差异'),
    ('box_geometry', '标注框几何异常'),
    ('coordinate_frame', '图像/标注坐标系待核对'),
    ('unavailable', '图像不可用'),
    ('annotation_unavailable', '标注缺失或解析失败'),
]
TITLES = dict(ISSUES + EXTRA_ISSUES)
DUPLICATES = ('exact_duplicate', 'near_duplicate')
TOTAL_TITLE = '任一已检出问题（去重）'


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def count_union(db, table, issues, class_code=None):
    if not issues:
        return 0
    if table == 'evaluated' and issues == ['near_duplicate']:
        if class_code is not None:
            return db.execute("SELECT COUNT(DISTINCT uid) FROM near_search WHERE class=? AND status='complete'",
                              (class_code,)).fetchone()[0]
        return db.execute("SELECT COUNT(*) FROM (SELECT uid FROM near_search GROUP BY uid HAVING MIN(status='complete')=1)").fetchone()[0]
    placeholders = ','.join('?' for _ in issues)
    sql = f'SELECT COUNT(DISTINCT t.uid) FROM {table} t'
    args = list(issues)
    if class_code is not None:
        sql += ' JOIN unit_classes c ON c.uid=t.uid'
    sql += f' WHERE t.issue IN ({placeholders})'
    if class_code is not None:
        sql += ' AND c.class=?'
        args.append(class_code)
        if table == 'findings':
            sql += " AND (t.issue NOT IN ('exact_duplicate','near_duplicate') OR json_extract(t.evidence,'$.class')=?)"
            args.append(class_code)
    return db.execute(sql, args).fetchone()[0]


def pct(numerator, denominator):
    return round(100 * numerator / denominator, 2) if denominator else None


def findings_for(db, uid, class_code=None):
    items = []
    for issue, index, evidence in db.execute(
            'SELECT issue,object_index,evidence FROM findings WHERE uid=? ORDER BY issue,object_index', (uid,)):
        if issue not in TITLES:
            continue
        detail = json.loads(evidence)
        if class_code is not None and issue in DUPLICATES and detail.get('class') != class_code:
            continue
        items.append((issue, index, detail))
    return items


def problem_text(items):
    present = {issue for issue, _, _ in items}
    return '；'.join(title for issue, title in ISSUES + EXTRA_ISSUES if issue in present)


def duplicate_references(db, items):
    paths = set()
    for issue, _, detail in items:
        if issue in DUPLICATES:
            row = db.execute('SELECT f.payload FROM units u JOIN files f ON f.id=u.file_id WHERE u.uid=?',
                             (detail['representative_uid'],)).fetchone()
            if row:
                paths.add(json.loads(row[0])['path'])
    return sorted(paths)


def summary_row(db, key, title, total, label=None):
    n = count_union(db, 'findings', [key], label)
    assessed = count_union(db, 'evaluated', [key], label)
    status = '已检查' if assessed == total and total else '部分检查' if assessed or n else '未评估'
    return {'问题': title, '图像总数': total, '问题图像数': n if assessed or n else None,
            '比例(%)': pct(n, total) if assessed or n else None,
            '已检查图像数': assessed, '状态': status}


def export_reports(db, output, metadata, config, preview_limit=0, detailed=False):
    update_progress(note='正在汇总总体问题统计')
    total = db.execute('SELECT COUNT(*) FROM units').fetchone()[0]
    file_count = db.execute('SELECT COUNT(*) FROM files').fetchone()[0]
    active = ISSUES + [(k, t) for k, t in EXTRA_ISSUES if count_union(db, 'findings', [k])]
    keys = [key for key, _ in active]

    def total_row(n, label=None):
        affected = count_union(db, 'findings', keys, label)
        return {'问题': TOTAL_TITLE, '图像总数': n, '问题图像数': affected,
                '比例(%)': pct(affected, n), '已检查图像数': None, '状态': '已检出问题的并集'}

    overall = [summary_row(db, key, title, total) for key, title in active] + [total_row(total)]
    stats_fields = ['问题', '图像总数', '问题图像数', '比例(%)', '已检查图像数', '状态']
    write_csv(output/'总体问题统计.csv', stats_fields, overall)
    update_progress(1, note='总体问题统计.csv 已写入')

    def class_rows():
        for label, n in db.execute('SELECT class,COUNT(*) FROM unit_classes GROUP BY class ORDER BY class'):
            update_progress(note=f'各类问题统计：{label}')
            for key, title in active:
                yield {'类别': label, **summary_row(db, key, title, n, label)}
            yield {'类别': label, **total_row(n, label)}
    write_csv(output/'各类问题统计.csv', ['类别'] + stats_fields, class_rows())
    update_progress(2, note='各类问题统计.csv 已写入')

    def image_rows():
        for uid, label, payload in db.execute('''SELECT u.uid,c.class,f.payload FROM units u
                JOIN unit_classes c USING(uid) JOIN files f ON f.id=u.file_id ORDER BY c.class,f.path'''):
            update_progress(note=f'逐图问题清单：{label}')
            items = findings_for(db, uid, label)
            if not items:
                continue
            row = json.loads(payload)
            yield {'类别': label, '图片路径': row['path'], '标注路径': row['label_path'],
                   '问题': problem_text(items), '重复对照图片': '；'.join(duplicate_references(db, items))}
    write_csv(output/'逐图问题清单.csv', ['类别','图片路径','标注路径','问题','重复对照图片'], image_rows())
    written = 3
    update_progress(written, note='逐图问题清单.csv 已写入')

    if detailed:
        def diagnostic_rows():
            for uid, payload in db.execute('SELECT u.uid,f.payload FROM units u JOIN files f ON f.id=u.file_id ORDER BY f.path'):
                row = json.loads(payload)
                for issue, index, evidence in findings_for(db, uid):
                    yield {'图片路径': row['path'], '标注路径': row['label_path'], '问题': TITLES[issue],
                           '目标序号': index + 1 if index >= 0 else '', '目标类别': evidence.get('class',''),
                           '诊断依据': json.dumps(evidence, ensure_ascii=False)}
        write_csv(output/'诊断明细.csv', ['图片路径','标注路径','问题','目标序号','目标类别','诊断依据'], diagnostic_rows())
        written += 1
        update_progress(written, note='诊断明细.csv 已写入')
    elif (output/'诊断明细.csv').exists():
        (output/'诊断明细.csv').unlink()

    summary = {'metadata': metadata, 'file_paths': file_count, 'sample_units': total,
               'collapsed_classification_copies': file_count-total,
               'target_boxes': db.execute('SELECT COUNT(*) FROM boxes').fetchone()[0],
               'eligible_target_boxes': db.execute('SELECT COUNT(*) FROM boxes WHERE eligible=1').fetchone()[0],
               'issue_union_count': overall[-1]['问题图像数'], 'issue_summary': overall, 'config': config}
    db.execute("INSERT OR REPLACE INTO meta VALUES ('summary',?)", (json.dumps(summary, ensure_ascii=False),))
    if preview_limit:
        update_progress(note=f'正在生成预览，最多 {preview_limit} 张')
        build_review_page(db, output, metadata, preview_limit)
        written += 1
        update_progress(written, note='问题预览.html 已写入')
    elif (output/'问题预览.html').exists():
        (output/'问题预览.html').unlink()
    return summary


def evidence_text(issue, index, detail):
    prefix = f"目标 {index+1}（{detail.get('class','')}）：" if index >= 0 else '整图：'
    metrics = detail.get('metrics', {})
    if issue == 'exposure':
        value = '、'.join({'dark':'过暗','overexposed':'过曝'}.get(x,x) for x in detail['flags'])
        return prefix + f"{value}，平均灰度 {metrics.get('mean_gray',0):.1f}"
    if issue == 'blur':
        return prefix + f"清晰度偏低，拉普拉斯方差 {metrics.get('laplacian_variance',0):.1f}，梯度能量 {metrics.get('tenengrad',0):.1f}"
    if issue == 'small_target':
        return prefix + f"训练缩放后短边 {detail['short_at_model']:.1f} 像素，框面积占比 {100*detail['area_fraction']:.3f}%"
    if issue == 'watermark_occlusion':
        return prefix + f"文字覆盖 {100*detail['overlap_fraction']:.1f}%，文字：{' / '.join(detail.get('text', []))}"
    if issue == 'large_box':
        return prefix + f"框面积占比 {100*detail['area_fraction']:.1f}%"
    return prefix + TITLES[issue]


def build_review_page(db, output, metadata, limit):
    # Bound preview memory even on a full dataset: at most limit records per issue.
    buckets = {key: [] for key in [*TITLES, 'unflagged']}
    for uid, payload in db.execute('SELECT u.uid,f.payload FROM units u JOIN files f ON f.id=u.file_id ORDER BY f.path'):
        row = json.loads(payload)
        items = findings_for(db, uid)
        keys = {issue for issue, _, _ in items} or {'unflagged'}
        keys = [key for key in keys if len(buckets[key]) < limit]
        if not keys:
            continue
        path = Path(row['path'])
        if metadata['mode'] == 'replay':
            path = Path(metadata['audit_dir'])/'examples'/(row.get('sample_id','')+'.jpg')
        if path.is_file():
            for key in keys:
                buckets[key].append((uid, row, path, items))
    selected, seen = [], set()
    groups = list(buckets.values())
    position = 0
    while len(selected) < limit:
        added = False
        for group in groups:
            if position < len(group):
                record = group[position]
                if record[0] not in seen:
                    selected.append(record)
                    seen.add(record[0])
                    added = True
                    if len(selected) == limit:
                        break
        position += 1
        if not added and all(position >= len(group) for group in groups):
            break
    esc = lambda x: html.escape(str(x), quote=True)
    blocks = []
    for uid, row, path, items in selected:
        uri = path.resolve().as_uri()
        overlay = ''
        if metadata['mode'] == 'scan' and row.get('frame_ok'):
            marked = {index for _, index, _ in items if index >= 0}
            rects = []
            for box in row['boxes']:
                if not box.get('eligible'):
                    continue
                x1,y1,x2,y2 = box['coords']
                color = '#ff4040' if box['index'] in marked else '#00ceba'
                rects.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" stroke="{color}"/>'
                             f'<text x="{x1}" y="{max(12,y1-3)}" fill="{color}">{box["index"]+1} {esc(box["class"])}</text>')
            for line in row.get('ocr', []):
                points = ' '.join(f'{x},{y}' for x, y in line['polygon'])
                rects.append(f'<polygon points="{points}" fill="none" stroke="#ff9c00" '
                             f'stroke-width="2" stroke-dasharray="5 3" vector-effect="non-scaling-stroke"/>')
            overlay = f'<svg viewBox="0 0 {row["width"]} {row["height"]}">' + ''.join(rects) + '</svg>'
        notes = ''.join(f'<li>{esc(evidence_text(*item))}</li>' for item in items)
        comparisons = ''.join(f'<p>重复对照：<a href="{esc(Path(p).absolute().as_uri())}">{esc(p)}</a></p>'
                              for p in duplicate_references(db, items))
        label_uri = Path(row['label_path']).absolute().as_uri() if row['label_path'] else ''
        blocks.append(f'<article><h2>{esc(row.get("sample_id") or path.name)} · '
                      f'{esc(problem_text(items) or "未触发已执行规则")}</h2>'
                      f'<div class="picture"><img loading="lazy" src="{esc(uri)}">{overlay}</div>'
                      f'<ul>{notes}</ul><p>图片：{esc(row["path"])}</p>'
                      f'<p>标注：<a href="{esc(label_uri)}">{esc(row["label_path"])}</a></p>{comparisons}</article>')
    text = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>问题预览</title>
<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;background:#f4f5f7;color:#223}
article{padding:18px;background:white;margin:18px 0;border-radius:10px}p{overflow-wrap:anywhere}
.picture{position:relative;max-width:100%;width:fit-content}.picture img{display:block;max-width:100%;max-height:620px}
svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}rect{fill:none;stroke-width:2;vector-effect:non-scaling-stroke}
text{font:12px sans-serif;stroke:none}h2{font-size:19px}</style>
<h1>问题预览</h1><p>包含触发规则的图片和少量未触发图片，供目视复核。扫描预览中红框为触发目标，青框为其余目标，橙色虚线为 OCR 检出的文字区域（含设备自身文字）；历史回放使用已有拼图。</p>
'''
    if metadata.get('sample_ratio', 1) < 1:
        text += (f'<p>本轮按一级类别目录抽样 {metadata["sample_ratio"]:.2%}，'
                 f'种子 {metadata["sample_seed"]}。统计表的分母为抽中并合并后的样本数；'
                 '重复检查只比较本轮抽中的图片。</p>')
    text += f'<p>本页 {len(selected)} 张；完整清单见“逐图问题清单.csv”，数量与比例见两份统计表。</p>'
    (output/'问题预览.html').write_text(text + ''.join(blocks) + '</html>', encoding='utf-8')
