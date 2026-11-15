"""Seed-specific YOLO lists; images are shared and never copied by this module."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

from simple_paths import resolve_path, write_json
from yolo.training_utils import split_groups, quality_resample


def split_settings(settings, holdout='val'):
    seeds=settings.get('seeds',[42])
    if not isinstance(seeds,list) or not seeds or any(not isinstance(s,int) or isinstance(s,bool) or s<0 for s in seeds):
        raise ValueError('split.seeds须为非空的非负整数列表，例如[42,43,44]')
    if len(seeds)!=len(set(seeds)): raise ValueError('split.seeds中不能重复')
    ratio=settings.get('test_fraction' if holdout=='test' else 'val_fraction',.2)
    if not isinstance(ratio,(int,float)) or isinstance(ratio,bool) or not math.isfinite(ratio) or not 0<ratio<1:
        raise ValueError('验证/测试比例须在0与1之间')
    return seeds,float(ratio)


def split_directory(settings, dataset):
    """Keep editable split metadata in the project, apart from the image pool."""
    value=settings.get('output',f'datasets/{Path(dataset).name}')
    if not isinstance(value,(str,Path)) or not str(value).strip():
        raise ValueError('split.output须为划分文件的保存目录')
    return resolve_path(value)


def _membership_digest(rows, quality_weights):
    stamp=hashlib.sha256()
    for row in sorted(rows,key=lambda r:r['uid']):
        boxes=row['boxes_json'] if row['baseline_eligible'] else row['clean_boxes_json']
        labels=sorted(({'class':b['class'],'coords':b['coords']} for b in json.loads(boxes)),
                      key=lambda b:json.dumps(b,sort_keys=True,separators=(',',':')))
        entry=[row['uid'],row['image_sha'],row.get('split_group'),labels,
               quality_weights.get(row['uid']) if quality_weights is not None else None]
        stamp.update(json.dumps(entry,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode())
        stamp.update(b'\n')
    return stamp.hexdigest()


def write_seed_split(dataset, rows, images, classes, seed, fraction=.2, holdout='val',
                     quality_weights=None, original=False, validation_cleanup=True,
                     split_output=None, sampling=None):
    """Create immutable lists for one seed, or verify and reuse existing lists."""
    if holdout not in ('val','test'): raise ValueError('holdout须为val或test')
    dataset=Path(dataset).resolve()
    root=split_directory({'output':split_output},dataset) if split_output is not None else split_directory({},dataset)
    folder=root/f'seed_{seed}'
    split_rows=rows if original else [{**r,'baseline_eligible':False} for r in rows]
    assignments=split_groups(split_rows,classes,seed,fraction,quality_weights=quality_weights)
    train=[r['uid'] for r in rows if assignments[r['uid']]=='train']
    held=[r['uid'] for r in rows if assignments[r['uid']]=='val']
    assigned_holdout=len(held)
    conflicts=set();merged=0
    if original and validation_cleanup:
        lookup={r['uid']:r for r in rows}
        signatures=defaultdict(set)
        for uid in held:
            r=lookup[uid]
            signature=json.dumps(sorted((b['class'],b['coords']) for b in json.loads(r['boxes_json'])),separators=(',',':'))
            signatures[r['image_sha']].add(signature)
        conflicts={sha for sha,values in signatures.items() if len(values)>1}
        seen=set();kept=[]
        for uid in held:
            row=lookup[uid]
            if row['image_sha'] in conflicts: continue
            key=row['image_sha'],next(iter(signatures[row['image_sha']]))
            if key in seen: merged+=1;continue
            seen.add(key);kept.append(uid)
        held=kept
    if not train or not held: raise ValueError(f'seed={seed}的训练/验证或测试集为空，请增加数据或换种子')
    draws=train
    if sampling is not None:
        training=set(train)
        draws,_=quality_resample([row for row in split_rows if row['uid'] in training],seed,
                                 min_weight=sampling['min_weight'],power=sampling['power'])
    digest_weights={row['uid']:row['quality_weighted_score'] for row in split_rows} if sampling is not None else quality_weights
    snapshot={'seed':seed,'holdout':holdout,'fraction':fraction,'classes':list(classes),'dataset':str(dataset),
              'membership_sha256':_membership_digest(split_rows,digest_weights),
              'method':'grouped_stratified_v2','quality_weighted':quality_weights is not None}
    if original and not validation_cleanup:
        snapshot['validation_cleanup']=False
    if sampling is not None:
        snapshot['training_sampling']=sampling
    summary={'seed':seed,'train':len(train),holdout:len(held),
             'assigned_holdout_before_cleanup':assigned_holdout,
             'conflicting_validation_groups':len(conflicts),
             'excluded_conflicting_validation_samples':assigned_holdout-len(held)-merged,
             'merged_validation_copies':merged}
    if sampling is not None:
        summary.update(train_draws=len(draws),train_unique_sampled=len(set(draws)))
    reused=False
    if folder.exists():
        file=folder/'划分配置.json'
        if not file.is_file() or json.loads(file.read_text(encoding='utf-8-sig'))!=snapshot:
            raise ValueError(f'已有划分的参数不同：{folder}；请用新的种子或新的输出目录，避免覆盖实验')
        if not all((folder/name).is_file() for name in ('train.txt',holdout+'.txt','data.yaml','划分统计.json')):
            raise ValueError(f'已有划分不完整：{folder}')
        reused=True
    else:
        folder.mkdir(parents=True,exist_ok=False)
        for name,uids in [('train',draws),(holdout,held)]:
            # Preserve exported image paths so YOLO finds their paired labels,
            # including when images are symlinks to another data volume.
            (folder/(name+'.txt')).write_text(''.join(Path(os.path.abspath(images[uid])).as_posix()+'\n' for uid in uids),encoding='utf-8')
        data={'train':'train.txt','val':'val.txt' if holdout=='val' else 'train.txt',
              'names':list(classes),'split_seed':seed}
        if holdout=='test': data['test']='test.txt'
        write_json(folder/'data.yaml',data)
        write_json(folder/'划分配置.json',snapshot)
        write_json(folder/'划分统计.json',summary)
    return {'folder':folder,'yaml':folder/'data.yaml','train_uids':train,'train_draws':draws,'holdout_uids':held,
            'summary':summary,'reused':reused}
