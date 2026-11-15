"""Restore source filenames in prepared YOLO datasets after training has stopped."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path

from simple_paths import load_json,resolve_path,write_json
from quality_experiment_pipeline import open_manifest
from yolo.training_utils import _atomic_bytes,image_names


def rewrite_lists(folder,mapping):
    for file in folder.rglob('*.txt'):
        lines=file.read_text(encoding='utf-8-sig').splitlines()
        updated=[mapping.get(os.path.abspath(file.parent/line),line) if line.strip() else line for line in lines]
        if updated!=lines:
            _atomic_bytes(file,('\n'.join(updated)+'\n').encode('utf-8'))
            file.with_suffix('.cache').unlink(missing_ok=True)


def restore_pool(dataset,index,records,source,project_datasets,group,dry_run=False):
    dataset=Path(dataset).resolve()
    names=image_names(records,source)
    marker=dataset/'图片命名恢复中.json'
    if marker.exists():
        plan=json.loads(marker.read_text(encoding='utf-8'))
    else:
        plan=[{'uid':row['uid'],'old':row['export_image'],'new':names[row['uid']].as_posix()}
              for row in records]
    moves=[]
    for item in plan:
        old=dataset/item['old'];new=dataset/item['new']
        for before,after in ((old,new),(dataset/'labels'/old.relative_to(dataset/'images').with_suffix('.txt'),
                                       dataset/'labels'/new.relative_to(dataset/'images').with_suffix('.txt'))):
            if not before.absolute().is_relative_to(dataset) or not after.absolute().is_relative_to(dataset):
                raise ValueError('恢复路径超出数据集目录')
            if before==after:
                continue
            if before.exists() and after.exists():
                raise ValueError(f'目标文件已存在：{after}')
            if not before.exists() and not after.exists():
                raise ValueError(f'待恢复文件缺失：{before}')
            if not marker.exists() and not before.exists():
                raise ValueError(f'待恢复文件缺失：{before}')
            moves.append((before,after))
    print(f'{group} 共{len(records):,}张，需移动图片或标签{len(moves):,}个',flush=True)
    if dry_run:
        for item in plan[:3]:
            print(item['old'],'->',item['new'])
        return
    write_json(marker,plan)
    for before,after in moves:
        if before.exists():
            after.parent.mkdir(parents=True,exist_ok=True)
            os.replace(before,after)
    mapping={str(dataset/item['old']):(dataset/item['new']).as_posix() for item in plan}
    current={item['uid']:item['new'] for item in plan}
    if group=='original_full':
        for row in index['samples']:
            row['export_image']=current[row['uid']]
        write_json(dataset/'图片索引.json',index)
    else:
        for row in index['filtered_samples']:
            row['image_path']=current[row['uid']]
        write_json(dataset/'数据路径.json',index)
    rewrite_lists(project_datasets,mapping)
    (dataset/'labels.cache').unlink(missing_ok=True)
    report=io.StringIO(newline='')
    writer=csv.writer(report)
    writer.writerow(['样本ID','原文件名','类别','原图片路径','原XML路径','训练图片路径','训练标签路径'])
    for row in records:
        relative=Path(current[row['uid']])
        boxes=json.loads(row['boxes_json'])
        writer.writerow([row['uid'],Path(row['image_path']).name,';'.join(sorted({box['class'] for box in boxes})),
                         row['image_path'],row['label_path'],str(dataset/relative),
                         str(dataset/'labels'/relative.relative_to('images').with_suffix('.txt'))])
    _atomic_bytes(project_datasets/group/'图片对应表.csv',report.getvalue().encode('utf-8-sig'))
    marker.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/filter.json')
    parser.add_argument('--dry-run',action='store_true',help='显示恢复计划')
    args=parser.parse_args()
    if not args.dry_run:
        from yolo.background import find_training_processes
        active=[run['pid'] for file in resolve_path('configs').glob('train*.json')
                for run in find_training_processes(file.resolve())]
        if active:
            raise ValueError(f'图片重命名需要先结束使用这些文件的训练，当前进程 {active}')
    config=load_json(args.config)
    source=resolve_path(config['input'])
    source=source/'by_label' if (source/'by_label').is_dir() else source
    original=resolve_path(config['original_output']);filtered=resolve_path(config['output'])
    raw=json.loads((original/'图片索引.json').read_text(encoding='utf-8-sig'))
    if (original/'复制配置.json').is_file() and (filtered/'复制配置.json').is_file():
        print(f'原始副本已使用原文件名：{original}',flush=True)
        print(f'硬筛副本已使用原文件名：{filtered}',flush=True)
        return
    if (filtered/'复制配置.json').is_file():
        restore_pool(original,raw,raw['samples'],source,resolve_path('datasets'),'original_full',args.dry_run)
        print(f'硬筛副本已使用原文件名，沿用当前train/val类别目录：{filtered}',flush=True)
        return
    clean=json.loads((filtered/'数据路径.json').read_text(encoding='utf-8-sig'))
    lookup={row['uid']:row for row in raw['samples']}
    with open_manifest(filtered/'_cache/experiment.sqlite',readonly=True) as db:
        boxes={row['uid']:row['clean_boxes_json'] for row in db.execute("SELECT uid,clean_boxes_json FROM samples WHERE clean_status='keep'")}
    records=[{**lookup[row['uid']],'export_image':row['image_path'],'boxes_json':boxes[row['uid']]}
             for row in clean['filtered_samples']]
    restore_pool(original,raw,raw['samples'],source,resolve_path('datasets'),'original_full',args.dry_run)
    restore_pool(filtered,clean,records,source,resolve_path('datasets'),'filtered_hard_v1',args.dry_run)


if __name__=='__main__':
    main()
