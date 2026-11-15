"""Score filtered data and write an optional quality-weighted YOLO training list."""
from __future__ import annotations

import argparse
from bisect import bisect_right
import csv
import json
import math
import os
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from itertools import chain
from pathlib import Path

from simple_paths import resolve_path
from quality_experiment_pipeline import open_manifest, initialize, update_scores, import_ratings
from quality_experiment_scoring import validate_weights, DEFAULT_SCORING
from quality_metrics import OCRReader, load_config
from quality_logging import progress_stage, update_progress
from dataset_splits import split_settings, split_directory, write_seed_split


DIMENSIONS = (('a1','曝光亮度'),('a2','整体清晰度'),('a3','文字水印遮挡'),
              ('a4','目标区域占比'),('b1','类别准确性'),('b2','区域一致性'),('b','整体标注质量'))


def _dump(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def _relative(path, directory):
    """Persist paths relative to the score output so a whole project can move."""
    return Path(os.path.relpath(Path(path).resolve(),Path(directory).resolve())).as_posix()


def _csv(path, headers, rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def load_scoring_config(path='configs/score.json', scores_only=False):
    path=resolve_path(path)
    config=json.loads(path.read_text(encoding='utf-8-sig'))
    for key in ('input','output'):
        config[key]=str(resolve_path(config[key]))
    config.setdefault('ocr',True)
    config.setdefault('ocr_device','cpu')
    config.setdefault('workers',2)
    config.setdefault('scores_only',False)
    classes=config.get('classes')
    if classes is not None and (not isinstance(classes,list) or not classes or
                               any(not isinstance(c,str) or not c.strip() for c in classes)):
        raise ValueError('classes 必须是非空类别名称列表')
    if classes is not None: config['classes']=sorted(set(classes))
    config.setdefault('score_bins',[0,1,2,3,4,5])
    bins=config['score_bins']
    if not isinstance(bins,list) or len(bins)<2 or any(
            not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value)
            for value in bins) or bins[0]!=0 or bins[-1]!=5 or any(a>=b for a,b in zip(bins,bins[1:])):
        raise ValueError('score_bins须从0递增到5，例如[0,1,2,3,4,4.5,5]')
    if not isinstance(config['scores_only'],bool): raise ValueError('scores_only 必须为 true 或 false')
    config['scores_only']=config['scores_only'] or scores_only
    if not isinstance(config['ocr'],bool): raise ValueError('ocr 必须为 true 或 false')
    if config['ocr_device'] not in ('cpu','cuda'): raise ValueError('ocr_device 仅支持 cpu 或 cuda')
    if not isinstance(config['workers'],int) or isinstance(config['workers'],bool) or config['workers']<1:
        raise ValueError('workers 必须为正整数')
    config['weights']=validate_weights(config.get('weights',{}))
    config.setdefault('scoring',{})
    config.setdefault('reweight',{})
    config['reweight'].setdefault('score_source','image')
    if config['reweight']['score_source'] not in ('image','complete','available'):
        raise ValueError('reweight.score_source须为image、complete或available')
    config.setdefault('split',{'seeds':[42,43,44],'test_fraction':.2})
    if not config['scores_only']: split_settings(config['split'],'test')
    if config['reweight'].get('weights') is not None:
        config['reweight']['weights']=validate_weights(config['reweight']['weights'])
    if config.get('annotation_scores_csv'):
        config['annotation_scores_csv']=str(resolve_path(config['annotation_scores_csv']))
    if config.get('reuse_scores_from'):
        config['reuse_scores_from']=str(resolve_path(config['reuse_scores_from']))
    return config


def _source_path(root, value):
    """Filter exports must stay under the input dataset folder."""
    value=Path(value)
    if value.is_absolute(): raise ValueError('数据路径.json 中的路径须相对于筛选输出目录')
    result=Path(os.path.abspath(root/value))
    if not result.is_relative_to(root.resolve()): raise ValueError('数据路径.json 中的路径越过筛选输出目录')
    return result


def _decode_image(path, expected_size):
    import cv2
    import numpy as np
    image=cv2.imdecode(np.frombuffer(Path(path).read_bytes(),dtype=np.uint8),
                       cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None: raise ValueError(f'导出图片解码失败：{path}')
    if (image.shape[1],image.shape[0]) != tuple(expected_size):
        raise ValueError(f'导出图片尺寸与筛选缓存不一致：{path}')
    return image


def _complete_ocr(db, config, images):
    """Only exported copies are opened; original image/XML paths are never used."""
    cursor=db.execute("SELECT uid,width,height,payload_json FROM samples WHERE clean_status='keep' ORDER BY uid")
    rows=(dict(row) for row in cursor)
    jobs=[{k:row[k] for k in ('uid','width','height')} for row in rows
          if json.loads(row['payload_json']).get('ocr_status')!='ok']
    total=len(jobs)
    jobs=iter(jobs)
    first=next(jobs,None)
    if first is None:
        print('复用已有OCR结果。',flush=True)
        return
    metric_config=load_config()
    metric_config['ocr_model_dir']=str(resolve_path('models/ocr'))
    reader=OCRReader(metric_config,config['ocr_device'])
    jobs=chain([first],jobs)
    workers=config['workers']
    done=0
    def process(row):
        payload=json.loads(row['payload_json'])
        if payload.get('ocr_status')=='ok': return row['uid'],payload
        image=images.get(row['uid'])
        if image is None:
            payload['ocr_status']='export_not_available'
            payload['ocr']=[]
        else:
            payload['ocr']=reader(_decode_image(image,(row['width'],row['height'])))
            payload['ocr_status']='ok'
            payload['ocr_device']=reader.device
            payload['ocr_providers']=reader.providers
        return row['uid'],payload
    with progress_stage('OCR文字覆盖评分',total), ThreadPoolExecutor(max_workers=workers) as executor:
        pending=set()
        def fill():
            while len(pending)<2*workers:
                row=next(jobs,None)
                if row is None: break
                row['payload_json']=db.execute('SELECT payload_json FROM samples WHERE uid=?',(row['uid'],)).fetchone()[0]
                pending.add(executor.submit(process,row))
        fill()
        while pending:
            ready,pending=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
            for future in ready:
                uid,payload=future.result()
                db.execute('UPDATE samples SET payload_json=? WHERE uid=?',
                           (json.dumps(payload,ensure_ascii=False,allow_nan=False),uid))
                done+=1
                update_progress(done)
                if done%100==0:
                    db.commit()
            fill()
    db.commit()


def _label_path(image):
    parts=list(Path(image).parts)
    positions=[i for i,value in enumerate(parts) if value=='images']
    if not positions: return None
    parts[positions[-1]]='labels'
    return Path(*parts).with_suffix('.txt')


def _reports(db, output, images, bins=(0,1,2,3,4,5)):
    count=Counter()
    totals=defaultdict(list)
    rows=[]
    assessed=Counter()
    grade_counts=defaultdict(Counter)
    intervals=[f'[{left:g},{right:g})' for left,right in zip(bins,bins[1:])]+['5（满分）']
    for record in db.execute("SELECT * FROM samples WHERE clean_status='keep' ORDER BY uid"):
        if record['uid'] not in images: continue
        score=json.loads(record['scores_json'])
        for key,_ in DIMENSIONS:
            assessed[key]+=score['dimensions'][key]['score'] is not None
        payload=json.loads(record['payload_json'])
        categories=json.loads(record['classes_json'])
        image=images.get(record['uid'])
        label=_label_path(image) if image else None
        rows.append([record['uid'],_relative(image,output) if image else '',
                     _relative(label,output) if label else '',
                     str(record['image_path']).replace('\\','/').rsplit('/',1)[-1],record['image_path'],record['label_path'],';'.join(categories),
                     record['image_quality_score'],record['total_quality_score'],round(score['coverage']*100,2),
                     record['quality_weighted_score'],
                     *[score['dimensions'][key]['score'] for key,_ in DIMENSIONS],payload.get('ocr_status','missing')])
        for category in ['总体',*categories]:
            count[category]+=1
            values={'图像质量分':record['image_quality_score'],'综合质量分':record['total_quality_score'],
                    '评分覆盖率(%)':score['coverage']*100,
                    **{key:score['dimensions'][key]['score'] for key,_ in DIMENSIONS}}
            for key,value in values.items():
                if value is not None: totals[category,key].append(value)
                if key!='评分覆盖率(%)' and value is not None:
                    grade_counts[category,key][bisect_right(bins,value)-1]+=1
    _csv(output/'逐图评分.csv', ['样本ID','图片路径','标注路径','原文件名','原图片路径','原XML路径','类别','图像质量分','综合质量分',
         '评分覆盖率(%)','采样依据分',*[key+name for key,name in DIMENSIONS],'OCR状态'],rows)
    classes=sorted(category for category in count if category!='总体')
    def mean(category,key):
        values=totals[category,key]
        return round(sum(values)/len(values),2) if values else ''
    keys=['图像质量分','综合质量分','评分覆盖率(%)',*[key for key,_ in DIMENSIONS]]
    _csv(output/'类别评分统计.csv',['类别','评分样本数','图像质量均分','综合质量均分','评分覆盖率均值(%)',
         *[key+name+'均分' for key,name in DIMENSIONS]],
         [[category,count[category],*[mean(category,key) for key in keys]] for category in ['总体',*classes]])
    distribution=[]
    score_names=[('图像质量分','图像质量分'),('综合质量分','综合质量分'),
                 *[(key,key+name) for key,name in DIMENSIONS]]
    for category in ['总体',*classes]:
        for key,name in score_names:
            assessed_count=len(totals[category,key])
            if not assessed_count: continue
            for index,interval in enumerate(intervals):
                number=grade_counts[category,key][index]
                distribution.append([category,name,interval,number,round(number/assessed_count*100,2),assessed_count])
    _csv(output/'分数区间统计.csv',['类别','评分项','分数区间','样本数','占已评分样本比例(%)','已评分样本数'],distribution)
    return {'scored_samples':count['总体'],
            'grade_intervals':{str(i):name for i,name in enumerate(intervals)},
            'grade_counts':{category:{key:{str(i):grade_counts[category,key][i] for i in range(len(intervals))}
                                     for key in ('图像质量分','综合质量分')}
                            for category in ['总体',*classes]},
            'dimension_coverage':{key:{'assessed_samples':assessed[key],'missing_samples':count['总体']-assessed[key],
                                       'coverage_percent':round(100*assessed[key]/count['总体'],2) if count['总体'] else 0}
                                  for key,_ in DIMENSIONS}}


def _optimized_dataset(db, output, source, paths, config):
    entries=paths['filtered_samples']
    if not entries: raise ValueError('筛选后样本为空，无法划分优化训练/测试集')
    samples={row['uid']:dict(row) for row in db.execute("SELECT uid,image_sha,split_group,clean_boxes_json,quality_weighted_score FROM samples WHERE clean_status='keep'")}
    identities=[entry['uid'] for entry in entries]
    if len(identities)!=len(set(identities)): raise ValueError('filtered_samples中样本ID重复')
    if any(uid not in samples for uid in identities): raise ValueError('图片索引与筛选缓存不一致')
    selection=config.get('reweight',{})
    import math
    minimum=float(selection.get('min_weight',.8));power=float(selection.get('power',1))
    if not .5<=minimum<=1 or not math.isfinite(power) or power<=0:
        raise ValueError('温和采样要求min_weight在0.5～1之间，power为正数')
    weights={}
    for uid in identities:
        value=samples[uid]['quality_weighted_score']
        if value is None or not math.isfinite(value) or not 0<=value<=5: raise ValueError('优化划分需要有效的0～5质量分')
        weights[uid]=minimum+(1-minimum)*(value/5)**power
    image_paths={entry['uid']:_source_path(source,entry['image_path']) for entry in entries}
    seeds,fraction=split_settings(config.get('split',{}),'test')
    split_output=split_directory(config.get('split',{}),output)
    summaries={};data_configs={}
    rows=[{**samples[uid],'baseline_eligible':False} for uid in identities]
    for seed in seeds:
        result=write_seed_split(output,rows,image_paths,paths['classes'],seed,fraction,'test',
                                split_output=split_output,sampling={'min_weight':minimum,'power':power})
        training=set(result['train_uids']);testing=set(result['holdout_uids'])
        frequency=Counter(result['train_draws'])
        if not result['reused']:
            _csv(result['folder']/'划分清单.csv',['样本ID','图片路径','质量分','抽样权重','划分','每轮训练次数'],
                 [[uid,_relative(image_paths[uid],result['folder']),samples[uid]['quality_weighted_score'],weights[uid],
                   '训练' if uid in training else '测试',frequency[uid]] for uid in identities if uid in training or uid in testing])
        summaries[str(seed)]={**result['summary'],'sampling_weight_min':minimum,'sampling_weight_max':1}
        data_configs[str(seed)]=result['yaml'].as_posix()
        print(f"优化数据 seed={seed}：训练 {len(training):,}，测试 {len(testing):,}"+
              ('（复用已有列表）' if result['reused'] else '（已生成列表）'),flush=True)
    return {'splits':summaries,'splits_output':str(split_output),'data_configs':data_configs,
            'evaluation_policy':'held-out test; final epoch weights; test not used for model selection'}


def score_identity(config):
    import hashlib
    path=config.get('annotation_scores_csv')
    digest=None
    if path:
        path=resolve_path(path)
        files=[path] if path.is_file() else sorted(file for file in path.rglob('*') if file.is_file() and file.suffix.lower()=='.csv')
        stamp=hashlib.sha256()
        for file in files: stamp.update(str(file).encode()+b'\0'+file.read_bytes())
        digest=stamp.hexdigest()
    override=config.get('reweight',{}).get('weights')
    return {'version':6,'score_scale':'0..5','input':str(resolve_path(config['input'])),'weights':validate_weights(config.get('weights',{})),
            'classes':sorted(config['classes']) if config.get('classes') else None,
            'scoring':{**DEFAULT_SCORING,**config.get('scoring',{})},'ocr':config.get('ocr',True),
            'ocr_device':config.get('ocr_device','cpu'),
            'annotation_scores_csv':str(resolve_path(path)) if path else None,'annotation_sha256':digest,
            'reweight_weights':validate_weights(override) if override is not None else None,
            'score_source':config.get('reweight',{}).get('score_source','image')}


def _copy_scoring_cache(before,after,classes=None):
    """Copy retained metrics and identity aliases, omitting rejected data and image features."""
    metadata=dict(before.execute('SELECT key,value FROM metadata'))
    if metadata.get('status')!='complete':
        raise ValueError('筛选尚未完成')
    initialize(after,json.loads(metadata['config']))
    after.executemany('INSERT OR REPLACE INTO metadata VALUES (?,?)',metadata.items())
    cursor=before.execute("SELECT * FROM samples NOT INDEXED WHERE clean_status='keep'")
    placeholders=','.join('?' for _ in cursor.description)
    while rows:=cursor.fetchmany(128):
        if classes:
            rows=[row for row in rows if any(box['class'] in classes for box in json.loads(row['clean_boxes_json']))]
        after.executemany(f'INSERT INTO samples VALUES ({placeholders})',rows)
    retained={row[0] for row in after.execute('SELECT uid FROM samples')}
    cursor=before.execute('SELECT uid,image_path,label_path,archive_class FROM occurrences')
    while rows:=cursor.fetchmany(256):
        after.executemany('INSERT INTO occurrences VALUES (?,?,?,?)',[row for row in rows if row['uid'] in retained])
    after.execute("DELETE FROM metadata WHERE key IN ('score_identity','score_status','score_config','rating_import')")
    after.commit()


def _scope_scores(db, classes, source_uids=None):
    """Scope scoring targets only; preserve complete hard-filter labels for training."""
    kept=set()
    for row in db.execute("SELECT uid,clean_boxes_json FROM samples WHERE clean_status='keep'").fetchall():
        boxes=[b for b in json.loads(row['clean_boxes_json']) if b['class'] in classes]
        if not boxes or (source_uids is not None and row['uid'] not in source_uids):
            continue
        kept.add(row['uid'])
        db.execute('UPDATE samples SET classes_json=?,kept_indices_json=? WHERE uid=?',
                   (json.dumps(sorted({b['class'] for b in boxes})),
                    json.dumps([b['index'] for b in boxes]),row['uid']))
    removed=[(row[0],) for row in db.execute('SELECT uid FROM samples') if row[0] not in kept]
    for table in ('ratings','features','occurrences','samples'):
        db.executemany(f'DELETE FROM {table} WHERE uid=?',removed)
    return kept


def score_dataset(config):
    source=resolve_path(config['input']);output=resolve_path(config['output'])
    manifest=source/'_cache/experiment.sqlite'
    if not manifest.is_file(): raise ValueError(f'筛选结果不存在：{manifest}')
    paths=json.loads((source/'数据路径.json').read_text(encoding='utf-8-sig'))
    if paths.get('layout_version')!=3: raise ValueError('请使用新版统一images/labels布局的筛选结果')
    config['classes']=sorted(config.get('classes') or paths['classes'])
    if set(config['classes'])-set(paths['classes']): raise ValueError('classes 包含数据集没有的类别')
    # Only the selected archive folders participate in this experiment.
    images={entry['uid']:_source_path(source,entry['image_path']) for entry in paths['filtered_samples']
            if Path(entry['image_path']).parts[:1]==('images',)
            and len(Path(entry['image_path']).parts)>=4
            and Path(entry['image_path']).parts[2] in config['classes']}
    if output==source or output.is_relative_to(source): raise ValueError('评分output应与筛选input分开保存')
    if not config.get('scores_only',False): split_settings(config.get('split',{}),'test')
    identity=score_identity(config)
    existing=output.exists()
    complete=False
    if existing:
        cache=output/'_cache/experiment.sqlite'
        if not cache.is_file(): raise ValueError('已有评分输出不完整，请使用新的output')
        with open_manifest(cache,readonly=True) as db:
            saved=dict(db.execute('SELECT key,value FROM metadata'))
            stored=saved.get('score_identity')
        previous_identity=json.loads(stored) if stored else None
        if previous_identity!=identity:
            raise ValueError('评分规则、OCR或大模型CSV已变化，请用新output保留实验；只增加split.seeds可复用当前output')
        complete=saved.get('score_status')=='complete'
        print('复用已有评分缓存。',flush=True)
    else:
        output.mkdir(parents=True,exist_ok=False)
        (output/'_cache').mkdir()
        previous=resolve_path(config['reuse_scores_from'])/'_cache/experiment.sqlite' if config.get('reuse_scores_from') else manifest
        with open_manifest(previous,readonly=True) as before, open_manifest(output/'_cache/experiment.sqlite') as after:
            if config.get('reuse_scores_from'):
                previous_config=json.loads(dict(before.execute('SELECT key,value FROM metadata'))['score_config'])
                if resolve_path(previous_config['input'])!=source:
                    raise ValueError('复用评分缓存与当前硬筛输入不一致')
                if set(config['classes'])-set(previous_config.get('classes') or paths['classes']):
                    raise ValueError('复用缓存未覆盖新增类别，请将reuse_scores_from设为null')
            _copy_scoring_cache(before,after,set(config['classes']))
            after.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('score_identity',json.dumps(identity,ensure_ascii=False)))
            after.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('score_status','building'))
    with open_manifest(output/'_cache/experiment.sqlite') as db:
        identities=_scope_scores(db,set(config['classes']),set(images))
        images={uid:image for uid,image in images.items() if uid in identities}
        paths={**paths,'filtered_samples':[e for e in paths['filtered_samples'] if e['uid'] in identities]}
        print(f"评分范围：{', '.join(config['classes'])}；共 {len(images):,} 张",flush=True)
        if not complete:
            filter_config=json.loads(dict(db.execute('SELECT key,value FROM metadata'))['config'])
            effective={**filter_config,**config}
            if config.get('ocr',True):
                _complete_ocr(db,config,images)
            else:
                for row in db.execute("SELECT uid,payload_json FROM samples WHERE clean_status='keep'").fetchall():
                    payload=json.loads(row['payload_json']);payload['ocr_status']='disabled';payload['ocr']=[]
                    db.execute('UPDATE samples SET payload_json=? WHERE uid=?',(json.dumps(payload,ensure_ascii=False),row['uid']))
            if config.get('annotation_scores_csv'):
                import_ratings(db,resolve_path(config['annotation_scores_csv']),effective,images)
            else:
                db.execute('DELETE FROM ratings');db.execute("DELETE FROM metadata WHERE key='rating_import'")
                update_scores(db,effective)
            db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('score_config',json.dumps(config,ensure_ascii=False)))
        summary={**_reports(db,output,images,config.get('score_bins',[0,1,2,3,4,5])),
                 'classes':config['classes'],
                 'scope_policy':'selected annotation classes and selected archive folders',
                 'score_source':config.get('reweight',{}).get('score_source','image'),
                 'scores_only':config.get('scores_only',False),'score_scale':'0..5'}
        if not config.get('scores_only',False):
            summary.update(_optimized_dataset(db,output,source,paths,config))
        rating_import=db.execute("SELECT value FROM metadata WHERE key='rating_import'").fetchone()
        summary['rating_import']=json.loads(rating_import[0]) if rating_import else {}
        db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('score_identity',json.dumps(identity,ensure_ascii=False)))
        db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('score_status','complete'))
    _dump(output/'评分配置.json',config)
    if not config.get('scores_only',False):
        _dump(split_directory(config.get('split',{}),output)/'划分参数.json',
              {'split':config.get('split',{}),'reweight':config.get('reweight',{})})
    _dump(output/'评分汇总.json',summary)
    print('评分结果：'+str(output),flush=True)
    if summary.get('data_configs'):
        print('训练输入示例：'+next(iter(summary['data_configs'].values())),flush=True)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/score.json')
    parser.add_argument('--scores-only',action='store_true',help='只输出评分结果')
    args=parser.parse_args(argv)
    try:
        config=load_scoring_config(args.config,scores_only=args.scores_only)
        from quality_logging import run_logging
        with run_logging(resolve_path('logs'),'score',10):
            score_dataset(config)
        return 0
    except (ValueError,OSError,KeyError,sqlite3.Error) as exc:
        parser.exit(2,str(exc)+'\n')


if __name__=='__main__':
    raise SystemExit(main())
