"""Copy an existing original/filtered training dataset using its exact split and original filenames."""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
import xml.etree.ElementTree as ET

import yaml
from simple_paths import load_json, resolve_path, write_json


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}',flush=True)


def duration(seconds):
    seconds=max(0,round(seconds))
    hours,remainder=divmod(seconds,3600)
    minutes,seconds=divmod(remainder,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def independent_copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    before=source.stat()
    if target.exists():
        if target.is_symlink() or target.stat().st_nlink!=1 or os.path.samefile(source,target):
            raise ValueError(f'目标并非独立文件：{target}')
        value=sha(source)
        if sha(target)!=value:
            raise ValueError(f'已有目标内容不一致：{target}')
    else:
        temporary=target.with_name(target.name+'.part')
        value=hashlib.sha256()
        with source.open('rb') as incoming, temporary.open('wb') as outgoing:
            for block in iter(lambda:incoming.read(1024*1024),b''):
                outgoing.write(block);value.update(block)
        value=value.hexdigest()
        if sha(temporary)!=value:
            raise ValueError(f'复制校验失败：{target}')
        os.replace(temporary,target)
    after=source.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError(f'复制期间源文件发生变化：{source}')
    if target.stat().st_nlink!=1 or os.path.samefile(source,target):
        raise ValueError(f'目标存在硬链接：{target}')
    return value


def write_csv(path,headers,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(headers);writer.writerows(rows)


def copy_scoring_support(source,output,image_paths):
    """Keep the new physical dataset usable by the existing score_dataset entry point."""
    index=load_json(source/'数据路径.json')
    if {entry['uid'] for entry in index['filtered_samples']}!=set(image_paths):
        raise ValueError('评分索引与新图片集合不一致')
    for entry in index['filtered_samples']:
        entry['image_path']=image_paths[entry['uid']]
    log('复制评分缓存和新路径索引')
    independent_copy(source/'_cache/experiment.sqlite',output/'_cache/experiment.sqlite')
    write_json(output/'数据路径.json',index)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',choices=('filtered','original'),default='filtered',
                        help='filtered复制硬筛组，original复制当前原始训练组')
    parser.add_argument('--filter-config',default='configs/filter.json')
    parser.add_argument('--train-config',help='默认读取所选组的现有训练配置')
    parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.workers<1:parser.error('workers必须大于0')
    config=load_json(args.filter_config)
    original_group=args.group=='original'
    training=load_json(args.train_config or ('configs/train_original.json' if original_group else 'configs/train.json'))
    source=resolve_path(config['original_output'] if original_group else config['output'])
    output=resolve_path(args.output)
    if source==output or source in output.parents or output in source.parents:
        raise ValueError('新数据集应使用独立的同级目录')
    data_path=resolve_path(training['data'])
    data=yaml.safe_load(data_path.read_text(encoding='utf-8-sig'))
    base=Path(data.get('path') or data_path.parent)
    if not base.is_absolute():base=data_path.parent/base
    classes=data['names']
    if isinstance(classes,dict):classes=[classes[i] for i in range(len(classes))]
    log('读取原文件名对应表和现有训练划分')
    original_index=load_json(resolve_path(config['original_output'])/'图片索引.json')
    origins={row['uid']:row for row in original_index['samples']}
    if original_group:
        index=[{'uid':row['uid'],'image_path':row['export_image']} for row in original_index['samples']]
    else:
        index=load_json(source/'数据路径.json')['filtered_samples']
    lookup={os.path.abspath(source/entry['image_path']):entry['uid'] for entry in index}
    if len(lookup)!=len(index):raise ValueError('图片路径索引重复')
    lists={};entries=[];seen=set();split_hashes={}
    holdout='val'
    for source_split in ('train','val'):
        split=holdout if source_split=='val' else 'train'
        listing=Path(data[source_split]);listing=listing if listing.is_absolute() else base/listing
        split_hashes[source_split]=sha(listing)
        values=[]
        for line in listing.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():continue
            image=Path(line);image=image if image.is_absolute() else listing.parent/image
            image=Path(os.path.abspath(image))
            uid=lookup[str(image)]
            if uid in seen:raise ValueError(f'训练及留出集划分有重复样本：{uid}')
            seen.add(uid);row=origins[uid]
            source_relative=image.relative_to(source/'images')
            parts=source_relative.parts
            if parts[0] in ('train','val','test'): parts=parts[1:]
            category=parts[0]
            relative=Path('images')/split/category/image.name
            label=source/'labels'/image.relative_to(source/'images').with_suffix('.txt')
            target_label=Path('labels')/relative.relative_to('images').with_suffix('.txt')
            entry={'uid':uid,'split':split,'source':str(image),'source_label':str(label),
                   'image':relative.as_posix(),'label':target_label.as_posix(),
                   'xml':(Path('annotations')/relative.relative_to('images').with_suffix('.xml')).as_posix(),
                   'category':category,'original':row['image_path'],'original_xml':row['label_path'],
                   'width':row['width'],'height':row['height']}
            entries.append(entry);values.append(entry)
        lists[split]=values if data.get('preserve_manifest_order') else sorted(values,key=lambda e:e['source'])
    if seen!={entry['uid'] for entry in index}:raise ValueError('划分清单与所选数据集图片集合不一致')
    if not entries:
        raise ValueError('所选数据集为空')
    names=Counter(e['label'].casefold() for e in entries)
    collisions=[name for name,count in names.items() if count>1]
    if collisions:
        raise ValueError(f'同一类别存在同名样本，需核对后才能使用原名：{collisions[:3]}')
    identity={'source':str(source),'data':str(data_path),'split_hashes':split_hashes,
              'classes':classes,'samples':len(entries),'version':2}
    if original_group:
        identity.update(group='original',source_holdout='val',exported_holdout='val',version=3)
    marker=output/'复制配置.json'
    migrate=False
    if output.exists() and any(output.iterdir()):
        if not marker.is_file():raise ValueError('目标目录已有不同任务内容')
        saved=load_json(marker)
        migrate=saved=={**identity,'version':1}
        if saved!=identity and not migrate:raise ValueError('目标目录已有不同任务内容')
    output.mkdir(parents=True,exist_ok=True)
    if migrate:
        log('复用上次中断的副本，将样本ID子目录整理为原类别目录')
        for entry in entries:
            for key,folder in [('image','images'),('label','labels'),('xml','annotations')]:
                target=output/entry[key]
                previous=output/folder/entry['split']/entry['uid']/target.name
                for old,new in [(previous,target),(previous.with_name(previous.name+'.part'),target.with_name(target.name+'.part'))]:
                    if old.exists():
                        old.resolve().relative_to(output)
                        new.resolve().relative_to(output)
                        if new.exists():raise ValueError(f'迁移目标已存在：{new}')
                        new.parent.mkdir(parents=True,exist_ok=True)
                        os.rename(old,new)
                if previous.parent.is_dir() and not any(previous.parent.iterdir()):previous.parent.rmdir()
        (output/'复制核对.json').unlink(missing_ok=True)
        write_json(marker,identity)
    for entry in entries:
        entry['bytes']=Path(entry['source']).stat().st_size
    total_bytes=sum(entry['bytes'] for entry in entries)
    required=sum(entry['bytes'] for entry in entries if not (output/entry['image']).exists())
    if not original_group and not (output/'_cache/experiment.sqlite').exists():
        required+=(source/'_cache/experiment.sqlite').stat().st_size
    if shutil.disk_usage(output).free<required+128*1024*1024:raise ValueError('剩余空间不足以创建独立图片副本')
    write_json(marker,identity)
    log(f'共 {len(entries):,} 张，待复制图片约 {required/1024**3:.2f} GiB，workers={args.workers}')
    directories={(output/entry[key]).parent for entry in entries for key in ('image','label','xml')}
    for directory in sorted(directories):
        directory.mkdir(parents=True,exist_ok=True)

    def copy(entry):
        entry=dict(entry)
        entry['image_sha256']=independent_copy(Path(entry['source']),output/entry['image'])
        entry['label_sha256']=independent_copy(Path(entry['source_label']),output/entry['label'])
        root=ET.Element('annotation');ET.SubElement(root,'filename').text=Path(entry['image']).name
        size=ET.SubElement(root,'size')
        for name,value in [('width',entry['width']),('height',entry['height']),('depth',3)]:ET.SubElement(size,name).text=str(value)
        categories=set()
        for line in (output/entry['label']).read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():continue
            label_id,xc,yc,bw,bh=line.split();category=classes[int(label_id)];categories.add(category)
            xc,yc,bw,bh=map(float,(xc,yc,bw,bh))
            obj=ET.SubElement(root,'object');ET.SubElement(obj,'name').text=category
            box=ET.SubElement(obj,'bndbox')
            coords=((xc-bw/2)*entry['width'],(yc-bh/2)*entry['height'],(xc+bw/2)*entry['width'],(yc+bh/2)*entry['height'])
            for key,value in zip(('xmin','ymin','xmax','ymax'),coords):ET.SubElement(box,key).text=str(round(value,6))
        xml=output/entry['xml'];xml.parent.mkdir(parents=True,exist_ok=True)
        xml.write_bytes(ET.tostring(root,encoding='utf-8',xml_declaration=True))
        entry['classes']=sorted(categories)
        return entry

    completed=[]
    started=last=time.monotonic()
    completed_bytes=last_bytes=0
    pending=set()
    iterator=iter(entries)
    executor=ThreadPoolExecutor(max_workers=args.workers)
    try:
        while True:
            while len(pending)<2*args.workers:
                entry=next(iterator,None)
                if entry is None:
                    break
                pending.add(executor.submit(copy,entry))
            if not pending:
                break
            ready,pending=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
            for future in ready:
                result=future.result()
                completed.append(result)
                completed_bytes+=result['bytes']
            now=time.monotonic()
            if now-last>=10:
                rate=(completed_bytes-last_bytes)/(now-last)
                eta=duration((total_bytes-completed_bytes)/rate) if rate>0 else '等待磁盘响应'
                log(f'复制并校验 {len(completed):,}/{len(entries):,} '
                    f'({len(completed)/len(entries):.2%}) | '
                    f'{completed_bytes/1024**3:.2f}/{total_bytes/1024**3:.2f} GiB | '
                    f'耗时 {duration(now-started)} | 速度 {rate/1024**2:.1f} MiB/秒 | 预计剩余 {eta}')
                last=now
                last_bytes=completed_bytes
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True,cancel_futures=True)
    completed.sort(key=lambda entry:(entry['split'],entry['source']))
    for split,values in lists.items():
        (output/(split+'.txt')).write_text(''.join('./'+e['image']+'\n' for e in values),encoding='utf-8')
    seed=data.get('split_seed',training['parameters']['seed'])
    dataset_config={'train':'train.txt','val':holdout+'.txt','names':classes,'split_seed':seed,
                    'preserve_manifest_order':True}
    write_json(output/'data.yaml',dataset_config)
    write_csv(output/'样本对应表.csv',
        ['sample_id','split','classes','image_path','label_path','yolo_label_path','previous_image','previous_label','original_image','original_xml','image_sha256','label_sha256'],
        [(e['uid'],e['split'],';'.join(e['classes']),e['image'],e['xml'],e['label'],e['source'],e['source_label'],e['original'],e['original_xml'],e['image_sha256'],e['label_sha256']) for e in completed])
    write_csv(output/'scores.csv',['sample_id','image_path','label_path','b1','b2'],
              [(e['uid'],e['image'],e['xml'],'','') for e in completed])
    for category in classes:
        folder=output/'by_label'/category;folder.mkdir(parents=True,exist_ok=True)
        for split in lists:
            write_csv(folder/(split+'.csv'),['sample_id','image_path','label_path'],
                      [(e['uid'],e['image'],e['xml']) for e in completed if e['split']==split and category in e['classes']])
    if original_group:
        renamed={entry['uid']:entry['image'] for entry in completed}
        updated_index={**original_index,'storage_counts':{'copy':len(completed)},
                       'samples':[{**row,'export_image':renamed[row['uid']]} for row in original_index['samples']]}
        write_json(output/'图片索引.json',updated_index)
    else:
        copy_scoring_support(source,output,{e['uid']:e['image'] for e in completed})
    write_json(output/'复制核对.json',{'status':'complete','samples':len(completed),'train':len(lists['train']),
               holdout:len(lists[holdout]),'seed':seed,'image_and_label_sha256_verified':True,
               'independent_files_verified':True,'loader_sample_order_verified':True,'split_hashes':split_hashes,
               'source_holdout':'val','exported_holdout':holdout,
               'copy_seconds':round(time.monotonic()-started,3)})
    log(f'完成，训练 {len(lists["train"]):,}，{holdout} {len(lists[holdout]):,}；数据配置 {output/"data.yaml"}')


if __name__=='__main__':main()
