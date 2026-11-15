"""Filter selected VOC classes and directly export original/filtered YOLO datasets."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
import json
from pathlib import Path
import sqlite3

from simple_paths import load_json, resolve_path, write_json, relative_list_path
from quality_experiment_pipeline import scan_dataset, rebuild_filter_from_cache, open_manifest, REASON_NAMES
from quality_experiment_rules import FILTER_RULES_VERSION
from quality_logging import run_logging, log, progress_stage, update_progress
from yolo.training_utils import _export_image, export_original_reference, image_names
from yolo.original_dataset import export_original_dataset, reference_original_dataset, source_fingerprints
from dataset_splits import split_settings, split_directory, write_seed_split


def load_filter_config(path='configs/filter.json'):
    config=load_json(path)
    for key in ('input','output'):
        if not isinstance(config.get(key),str) or not config[key].strip():
            raise ValueError(f'请在筛选配置中填写{key}')
        config[key]=str(resolve_path(config[key]))
    classes=config.get('classes',[])
    if not classes or len(classes)!=len(set(classes)) or any(not isinstance(c,str) or not c or c in ('.','..') or '/' in c or '\\' in c for c in classes):
        raise ValueError('classes须为不重复的类别目录名')
    workers=config.setdefault('workers',4)
    if not isinstance(workers,int) or isinstance(workers,bool) or workers<1: raise ValueError('workers须为正整数')
    config.setdefault('split',{'seeds':[42,43,44],'val_fraction':.2})
    split_settings(config['split'])
    config.setdefault('hard_filter',{})
    config.setdefault('original_output',str(original_directory(config)))
    config['original_output']=str(resolve_path(config['original_output']))
    config.setdefault('storage',{'original_images':'link'})
    if config['storage'].get('original_images','link') not in ('link','copy'):
        raise ValueError('storage.original_images仅支持link或copy')
    if config['storage'].get('filtered_images','link') not in ('link','copy'):
        raise ValueError('storage.filtered_images仅支持link或copy')
    config.setdefault('near_duplicates',{'enabled':False,'action':'record','max_candidates':200})
    config.setdefault('input_layout','classified')
    if config['input_layout'] not in ('classified','named'):
        raise ValueError('input_layout须为classified或named')
    return config


def _csv(path,headers,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(headers);writer.writerows(rows)


def source_directory(config):
    root=resolve_path(config['input'])
    if config.get('input_layout')=='named': return root
    return root/'by_label' if (root/'by_label').is_dir() else root


def original_directory(config):
    if config.get('original_output'): return resolve_path(config['original_output'])
    root=resolve_path(config['input'])
    return (root.parent if root.name=='by_label' else root)/'yolo'


def _export_datasets(db,config,output,existing=None):
    original_split={**config.get('split',{}),
                    'output':config.get('original_split_output',f'datasets/{original_directory(config).name}')}
    if config.get('input_layout')=='named':
        original=reference_original_dataset(original_directory(config),config['classes'],original_split)
    else:
        original=export_original_dataset(db,original_directory(config),source_directory(config),config['classes'],
                                         original_split,config.get('storage',{}).get('original_images','link'),config.get('workers',4))
    states=dict(db.execute('SELECT clean_status,COUNT(*) FROM samples GROUP BY clean_status'))
    # Sequentially read the retained rows only. The hashed UID index otherwise
    # creates random reads of large SQLite overflow records on the data disk.
    filtered=[dict(row) for row in db.execute("SELECT uid,image_path,label_path,image_sha,split_group,width,height,baseline_eligible,clean_status,clean_boxes_json FROM samples NOT INDEXED WHERE clean_status='keep'")]
    filtered.sort(key=lambda row:row['uid'])
    if not filtered: raise ValueError('筛选后保留数据为空')
    seeds,fraction=split_settings(config.get('split',{}))
    split_output=split_directory(config.get('split',{}),output)
    if existing is not None:
        images={entry['uid']:output/entry['image_path'] for entry in existing['filtered_samples']}
        if set(images)!={row['uid'] for row in filtered}:
            raise ValueError('已有图片索引与筛选缓存不一致，请使用新输出目录')
        filtered_storage=Counter(existing['export_summary'].get('filtered_image_storage',{}))
    else:
        images={};filtered_storage=Counter()
        fingerprints=source_fingerprints(db)
        for row in filtered:
            row['source_fingerprint']=fingerprints.get(row['image_path'])
        workers=config.get('workers',4)
        names=image_names(filtered,source_directory(config))
        with progress_stage('导出硬筛保留图片和标签',len(filtered),note=f'workers={workers}'),ThreadPoolExecutor(max_workers=workers) as executor:
            def export(row):
                image=output/names[row['uid']]
                if config.get('storage',{}).get('filtered_images','link')=='link':
                    kind=export_original_reference(row,image,json.loads(row['clean_boxes_json']),config['classes'],'link')
                else:
                    _export_image(row,image,json.loads(row['clean_boxes_json']),config['classes']);kind='copy'
                return row['uid'],image,kind
            pending=set();iterator=iter(filtered);done=0
            def fill():
                while len(pending)<2*workers:
                    row=next(iterator,None)
                    if row is None:break
                    pending.add(executor.submit(export,row))
            fill()
            while pending:
                ready,pending=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                for future in ready:
                    uid,image,kind=future.result();filtered_storage[kind]+=1
                    images[uid]=image;done+=1;update_progress(done)
                fill()
    paths={};summaries={};stats=[];lookup={row['uid']:row for row in filtered}
    for seed in seeds:
        result=write_seed_split(output,filtered,images,config['classes'],seed,fraction,split_output=split_output)
        paths[str(seed)]=result['yaml'].as_posix();summaries[str(seed)]=result['summary']
        log(f"filtered seed={seed}：训练 {len(result['train_uids']):,}，验证 {len(result['holdout_uids']):,}"+
            ('（复用已有列表）' if result['reused'] else '（已生成列表）'))
        for category in config['classes']:
            counts=[]
            for uids in (result['train_uids'],result['holdout_uids']):
                boxes=[[b for b in json.loads(lookup[uid]['clean_boxes_json']) if b['class']==category] for uid in uids]
                counts.extend([sum(bool(box) for box in boxes),sum(len(box) for box in boxes)])
            stats.append([seed,category,*counts])
    _csv(split_output/'数据集统计.csv',['划分种子','类别','训练图片数','训练框数','验证图片数','验证框数'],stats)
    summary={'original_output':original['output'],'original_images':original['images'],
             'filtered_images':len(filtered),'archive_images':db.execute('SELECT COUNT(*) FROM occurrences').fetchone()[0],
             'merged_samples':sum(states.values()),'rejected_images':states.get('reject',0),
             'original_reused':original['reused'],'original_image_storage':original['storage_counts'],
             'filtered_image_copies':filtered_storage['copy'],'filtered_image_storage':dict(filtered_storage),
             'splits':{'original':original['summaries'],'filtered':summaries}}
    write_json(output/'数据路径.json',{'layout_version':3,'classes':config['classes'],
               'filter_config':filter_identity(config),'splits_output':str(split_output),'splits':paths,'export_summary':summary,
               'filtered_samples':[{'uid':uid,'image_path':image.relative_to(output).as_posix()}
                                   for uid,image in sorted(images.items())]})
    return summary


def filter_identity(config):
    """Seeds/ratios/workers do not change the single stored image pool."""
    identity={'rules_version':FILTER_RULES_VERSION,
            'input':str(source_directory(config)),'original_output':str(original_directory(config)),'classes':config['classes'],
            'hard_filter':{'min_side':200,'tiny_model_side':4,'max_box_fraction':.9,'imgsz':640,**config.get('hard_filter',{})},
            'near_duplicates':config.get('near_duplicates',{'enabled':False,'action':'record','max_candidates':200}),
            'storage':config.get('storage',{'original_images':'link'})}
    if config.get('input_layout')=='named': identity['input_layout']='named'
    return identity


def _filter_reports(db,output,config):
    counts=Counter();reasons=Counter();details=[]
    statuses={'keep':'保留','reject':'筛除'}
    for row in db.execute('SELECT uid,image_path,label_path,classes_json,reject_reasons_json,clean_status,boxes_json,clean_boxes_json FROM samples NOT INDEXED'):
        classes=json.loads(row['classes_json']);problem=json.loads(row['reject_reasons_json'])
        kept_classes={b['class'] for b in json.loads(row['clean_boxes_json'])}
        state=statuses[row['clean_status']]
        details.append([row['uid'],row['image_path'],row['label_path'],';'.join(classes),';'.join(sorted(kept_classes)),state,
                        ';'.join(REASON_NAMES.get(r,r) for r in problem),len(json.loads(row['boxes_json'])),
                        len(json.loads(row['clean_boxes_json']))])
        for category in ['总体',*classes]:
            category_state='筛除' if state=='保留' and category!='总体' and category not in kept_classes else state
            counts[category,'总数']+=1;counts[category,category_state]+=1
            for reason in set(problem): reasons[category,reason]+=1
    _csv(output/'逐图筛选.csv',['样本ID','原图片路径','原标签路径','原类别','保留类别','筛选状态','筛选原因','原可转换框数','保留框数'],details)
    _csv(output/'类别筛选统计.csv',['类别','统计样本数','保留数','筛除数','筛除比例(%)'],
         [[category,counts[category,'总数'],counts[category,'保留'],counts[category,'筛除'],
           round(100*counts[category,'筛除']/counts[category,'总数'],2) if counts[category,'总数'] else 0]
          for category in ['总体',*config['classes']]])
    _csv(output/'筛选原因统计.csv',['类别','原因','涉及样本数','占该类别比例(%)'],
         [[category,REASON_NAMES.get(reason,reason),number,round(100*number/counts[category,'总数'],2)]
          for (category,reason),number in sorted(reasons.items())])
    archived=db.execute('SELECT COUNT(*) FROM occurrences').fetchone()[0]
    log(f'归档图片路径 {archived:,}，合并样本 {counts["总体","总数"]:,}；'
        f'保留 {counts["总体","保留"]:,}，筛除 {counts["总体","筛除"]:,}')


def filter_dataset(config):
    source,output=source_directory(config),resolve_path(config['output'])
    original=original_directory(config)
    if ((original==source and config.get('input_layout')!='named') or original==output
            or original.is_relative_to(output) or output.is_relative_to(original)):
        raise ValueError('original_output须与筛选output独立，并且不能直接使用原始图片根目录')
    if config.get('input_layout')=='named' and original!=source:
        raise ValueError('named输入直接引用原始副本，original_output须与input相同')
    if source==output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('筛选input与output须独立，不能相互包含')
    split_settings(config.get('split',{}))
    index=output/'数据路径.json';manifest=output/'_cache/experiment.sqlite'
    if index.is_file() and manifest.is_file():
        existing=json.loads(index.read_text(encoding='utf-8-sig'))
        if existing.get('layout_version')!=3:
            raise ValueError('已有结果使用旧布局，请使用新的output；新版数据目录包含images/labels')
        if existing.get('filter_config',{}).get('rules_version')!=FILTER_RULES_VERSION:
            raise ValueError('已有结果采用旧版类别筛选规则。请换新的output，并设置reuse_metrics_from复用当前结果的指标缓存')
        if existing['filter_config']!=filter_identity(config):
            raise ValueError('数据源/硬筛/类别/存储方式已改变，请用新的output，避免覆盖原实验')
        with open_manifest(manifest,readonly=True) as db:
            if dict(db.execute('SELECT key,value FROM metadata')).get('status')!='complete': raise ValueError('已有筛选未完成')
            summary=_export_datasets(db,config,output,existing)
        log('只更新划分列表，未读取原图、未复制图片。')
    else:
        if output.exists() and any(output.iterdir()) and not manifest.is_file():
            raise ValueError(f'已有输出目录没有可续跑的筛选缓存：{output}；请使用新的output')
        effective={**config,'source':str(source),'compute_scores':False,'scan':{'workers':config.get('workers',4),'ocr':False},
                   'annotation_scores_csv':None,'reweight':{},'weights':{},'scoring':{}}
        effective['hard_filter']=filter_identity(config)['hard_filter']
        if config.get('reuse_metrics_from'):
            rebuild_filter_from_cache(effective,resolve_path(config['reuse_metrics_from']),output/'_cache')
        else:
            scan_dataset(effective,output/'_cache',reports=False,resume=manifest.is_file())
        write_json(output/'筛选配置.json',config)
        with open_manifest(output/'_cache/experiment.sqlite') as db:
            _filter_reports(db,output,config)
            summary=_export_datasets(db,config,output)
    split_output=split_directory(config.get('split',{}),output)
    write_json(split_output/'划分参数.json',config.get('split',{}))
    log('筛选结果：'+str(output))
    log('筛选后训练输入：'+str(split_output/f"seed_{split_settings(config.get('split',{}))[0][0]}"/'data.yaml'))
    log('原始对照划分：'+str(resolve_path(config.get('original_split_output',f'datasets/{original.name}'))))
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/filter.json')
    args=parser.parse_args(argv)
    try:
        config=load_filter_config(args.config)
        with run_logging(resolve_path('logs'),'filter',3):
            try:
                filter_dataset(config)
            except KeyboardInterrupt:
                log('已停止筛选。保留输出目录，再次运行同一命令即可继续；修改workers不影响续跑。')
                return 130
        return 0
    except (ValueError,OSError,KeyError,sqlite3.Error) as exc:
        parser.exit(2,str(exc)+'\n')


if __name__=='__main__': raise SystemExit(main())
