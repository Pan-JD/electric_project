"""Stream the existing hard-filtered images, VOC labels and stable score IDs to tar."""
import argparse
from collections import Counter
import csv
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import xml.etree.ElementTree as ET

from simple_paths import load_json, resolve_path
from quality_experiment_pipeline import open_manifest


def add_bytes(archive, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def csv_bytes(headers, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(headers)
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8-sig')


def export_dataset(dataset, destination, dry_run=False):
    dataset = Path(dataset).resolve()
    print(f'正在读取硬筛样本对应信息：{dataset}',file=sys.stderr,flush=True)
    index = json.loads((dataset/'数据路径.json').read_text(encoding='utf-8-sig'))
    entries = index['filtered_samples']
    if len({entry['uid'] for entry in entries}) != len(entries):
        raise ValueError('数据路径.json 存在重复样本ID')
    with open_manifest(dataset/'_cache/experiment.sqlite', readonly=True) as db:
        config = json.loads(dict(db.execute('SELECT key,value FROM metadata'))['config'])
        records = {row['uid']:dict(row) for row in db.execute(
            "SELECT uid,image_path,label_path,width,height FROM samples WHERE clean_status='keep'")}
    classes = config['classes']
    named_copy = (dataset/'复制配置.json').is_file()
    print(f'已读取 {len(entries):,} 个导出样本，正在整理原文件名和XML',file=sys.stderr,flush=True)
    names = {}
    for entry in entries:
        row = records[entry['uid']]
        if named_copy:
            names[entry['uid']] = PurePosixPath(entry['image_path']).relative_to('images')
            continue
        original = PurePosixPath(row['image_path'].replace('\\','/'))
        if 'images' in original.parts:
            parts=original.parts[original.parts.index('images')+1:]
            if parts[0] in ('train','val','test'): parts=parts[1:]
            category=parts[0]
        else:
            position = original.parts.index('by_label')
            category = original.parts[position+1]
        if category not in classes:
            raise ValueError(f'原归档类别不在类别列表中：{original}')
        names[entry['uid']] = PurePosixPath(category)/original.name
    collisions = Counter(str(name.with_suffix('.xml')).casefold() for name in names.values())
    items = []
    for entry in entries:
        if dry_run and len(items) >= 3:
            continue
        uid = entry['uid']
        name = names[uid]
        if collisions[str(name.with_suffix('.xml')).casefold()] > 1:
            if named_copy:
                raise ValueError(f'原文件名存在冲突，请核对：{name}')
            name = name.parent/uid/name.name
        image = Path(os.path.abspath(dataset/entry['image_path']))
        image.relative_to(dataset/'images')
        label = dataset/'labels'/image.relative_to(dataset/'images').with_suffix('.txt')
        if not image.is_file() or not label.is_file():
            raise ValueError(f'图片或训练标签缺失：{image}')
        row = records[uid]
        root = ET.Element('annotation')
        ET.SubElement(root, 'filename').text = name.name
        size = ET.SubElement(root, 'size')
        for key, value in [('width',row['width']),('height',row['height']),('depth',3)]:
            ET.SubElement(size, key).text = str(value)
        for line in label.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f'YOLO检测标签格式异常：{label}')
            category_id = int(fields[0])
            if not 0 <= category_id < len(classes):
                raise ValueError(f'类别编号超出范围：{label}')
            xc,yc,width,height = map(float, fields[1:])
            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, 'name').text = classes[category_id]
            box = ET.SubElement(obj, 'bndbox')
            coords = [(xc-width/2)*row['width'], (yc-height/2)*row['height'],
                      (xc+width/2)*row['width'], (yc+height/2)*row['height']]
            for key,value in zip(('xmin','ymin','xmax','ymax'), coords):
                ET.SubElement(box, key).text = str(round(value,6))
        export_image = ('images/' if named_copy else 'by_label/')+name.as_posix()
        export_label = ('annotations/' if named_copy else 'by_label/')+name.with_suffix('.xml').as_posix()
        items.append((uid,image,label,export_image,export_label,
                      ET.tostring(root,encoding='utf-8',xml_declaration=True)))
        if len(items) % 1000 == 0:
            print(f'已整理标注 {len(items):,}/{len(entries):,}',file=sys.stderr,flush=True)
    print(f'评分导出 {len(entries):,} 张；来源 {dataset}',file=sys.stderr,flush=True)
    if dry_run:
        for uid,image,label,export_image,export_label,xml in items[:3]:
            print(f'{image.name} -> {export_image}',file=sys.stderr)
        return
    mapping = csv_bytes(['sample_id','image_path','label_path','training_image','training_label','original_image','original_xml'],
                        [(uid,ei,el,str(image),str(label),records[uid]['image_path'],records[uid]['label_path'])
                         for uid,image,label,ei,el,xml in items])
    scores = csv_bytes(['sample_id','image_path','label_path','b1','b2'],
                       [(uid,ei,el,'','') for uid,image,label,ei,el,xml in items])
    stream = sys.stdout.buffer if destination == '-' else open(destination, 'xb')
    try:
        with tarfile.open(fileobj=stream,mode='w|',dereference=True) as archive:
            add_bytes(archive,'样本对应表.csv',mapping)
            add_bytes(archive,'scores.csv',scores)
            for number,(uid,image,label,ei,el,xml) in enumerate(items,1):
                archive.add(image,arcname=ei,recursive=False)
                add_bytes(archive,el,xml)
                if number % 1000 == 0:
                    print(f'已导出 {number:,}/{len(items):,}',file=sys.stderr,flush=True)
    finally:
        if destination != '-':
            stream.close()
    print('导出完成',file=sys.stderr,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/filter.json')
    parser.add_argument('--dataset',help='现有硬筛结果目录，默认读取筛选配置output')
    parser.add_argument('--output',default='-',help='tar路径，默认向标准输出发送，可直接管道传输')
    parser.add_argument('--dry-run',action='store_true',help='核对全部名称映射及前三张标签，不读取图片内容')
    args = parser.parse_args()
    dataset = resolve_path(args.dataset or load_json(args.config)['output'])
    if args.output == '-' and not args.dry_run and sys.stdout.isatty():
        parser.error('请把输出管道连接到接收端，或用 --output 指定tar文件')
    export_dataset(dataset,args.output,args.dry_run)


if __name__ == '__main__':
    main()
