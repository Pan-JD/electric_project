"""Repair JPEG decoder warnings directly in the two exported training image pools."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time

CODE_DIRECTORY=Path(__file__).resolve().parents[1]
if str(CODE_DIRECTORY) not in sys.path:
    sys.path.insert(0,str(CODE_DIRECTORY))
from simple_paths import resolve_path
from yolo.training_utils import _atomic_bytes

JPEG_SUFFIXES={'.jpg','.jpeg','.jpe','.jfif','.mpo'}
IMAGE_SUFFIXES=JPEG_SUFFIXES|{'.png','.bmp','.tif','.tiff','.webp','.avif','.heic','.dng'}
_diagnostics=None


def _initialize_worker():
    import cv2
    cv2.setNumThreads(1)
    global _diagnostics
    _diagnostics=tempfile.TemporaryFile()
    # Each worker decodes one image at a time. Capture native libjpeg messages
    # locally so warnings can be associated with that exact image.
    os.dup2(_diagnostics.fileno(),2)


def _decode(data):
    import cv2
    import numpy as np
    _diagnostics.seek(0)
    _diagnostics.truncate()
    image=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
    _diagnostics.seek(0)
    message=_diagnostics.read().decode('utf-8',errors='replace').strip()
    return image,message


def _signature(path):
    try:
        stat=path.stat()
        return f'{stat.st_dev}/{stat.st_ino}/{stat.st_size}/{stat.st_mtime_ns}'
    except OSError:
        return 'missing'


def _check(path_string):
    import cv2
    path=Path(path_string)
    before=_signature(path)
    try:
        data=path.read_bytes()
        image,message=_decode(data)
        if image is not None and not message and data.endswith(b'\xff\xd9'):
            return path_string,'正常',''
        if image is None and not data.endswith(b'\xff\xd9'):
            image,message=_decode(data+b'\xff\xd9')
        if image is None:
            raise ValueError('JPEG无法解码')
        height,width=image.shape[:2]
        fixed=data+b'\xff\xd9' if not data.endswith(b'\xff\xd9') else data
        verified,remaining=_decode(fixed)
        if remaining or verified is None:
            success,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95])
            if not success:
                raise ValueError('JPEG重新编码失败')
            fixed=encoded.tobytes()
            verified,remaining=_decode(fixed)
        if remaining or verified is None or verified.shape[:2]!=(height,width):
            raise ValueError('修正后的JPEG检查失败')
        if _signature(path)!=before:
            raise ValueError('检查期间文件发生变化，请重新检查')
        # Replacing the exported directory entry breaks a source hardlink or
        # symlink. Opening the existing file for writing would modify the source.
        _atomic_bytes(path,fixed)
        return path_string,'已修正',''
    except (OSError,ValueError,cv2.error) as error:
        return path_string,'无法修正',str(error)[:500]


def _images(value):
    result=[]
    for entry in value if isinstance(value,list) else [value]:
        path=Path(entry)
        if path.is_dir():
            result.extend(str(file.absolute()) for file in sorted(path.rglob('*')) if file.suffix.lower() in IMAGE_SUFFIXES)
        elif path.suffix.lower()=='.txt':
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                if line.strip():
                    result.append(os.path.abspath(path.parent/line.strip()))
        else:
            result.append(str(path.absolute()))
    return result


def repair_jpegs(paths,workers=4):
    paths=sorted(set(paths))
    if any('images' not in Path(path).parts for path in paths):
        raise ValueError('JPEG修正请使用导出的YOLO images目录')
    print(f'JPEG修正开始 共{len(paths):,}张',flush=True)
    repaired=failed=0
    context=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=max(1,min(workers,4)),mp_context=context,initializer=_initialize_worker) as executor:
        last=time.monotonic()
        for number,(path,status,detail) in enumerate(executor.map(_check,paths,chunksize=8),1):
            repaired+=status=='已修正'
            failed+=status=='无法修正'
            if status=='无法修正':
                print(f'无法修正 {path} {detail}',flush=True)
            if time.monotonic()-last>=30:
                print(f'JPEG修正 {number:,}/{len(paths):,}张，已修正{repaired:,}张',flush=True)
                last=time.monotonic()
    print(f'JPEG修正完成 检查{len(paths):,}张，修正{repaired:,}张，失败{failed:,}张',flush=True)
    return failed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',action='append',help='训练配置，可重复指定，默认检查原始组和硬筛组')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    from yolo.train import load_training_config,_dataset_snapshot
    paths=set()
    for filename in args.config or ['configs/train_original.json','configs/train.json']:
        config=load_training_config(filename)
        data=resolve_path(config['data'])
        dataset=_dataset_snapshot(data)
        for name in ('train','val','test'):
            if dataset.get(name):
                paths.update(path for path in _images(dataset[name]) if Path(path).suffix.lower() in JPEG_SUFFIXES)
    return 1 if repair_jpegs(paths,args.workers) else 0


if __name__=='__main__':
    raise SystemExit(main())
