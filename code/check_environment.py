"""Check packaged code/models and conda dependencies without opening datasets."""
from __future__ import annotations

import argparse
import importlib.util
from importlib import metadata
from pathlib import Path
import sys


OCR_MODELS = (
    'ch_PP-OCRv4_det_infer.onnx',
    'ch_ppocr_mobile_v2.0_cls_infer.onnx',
    'ch_PP-OCRv4_rec_infer.onnx',
)


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def inspect_environment(project=None, with_ocr=True, gpu=False, module_available=None, version_lookup=None):
    """Return checks; only known resource paths and installed package metadata are read."""
    root = Path(project or Path(__file__).resolve().parents[1]).resolve()
    available = module_available or (lambda name: importlib.util.find_spec(name) is not None)
    version = version_lookup or _version
    checks = []
    resources = [('YOLO源码', root/'code/yolo/ultralytics-yolo26/ultralytics/__init__.py'),
                 ('YOLO初始权重', root/'models/yolo26n.pt'),
                 ('绘图字体', root/'assets/fonts/DejaVuSans.ttf')]
    if with_ocr:
        resources += [('OCR模型 '+name, root/'models/ocr'/name) for name in OCR_MODELS]
    for caption, path in resources:
        ok = path.is_file() and path.stat().st_size > 0
        checks.append({'name': caption, 'ok': ok, 'detail': str(path)})
    packages = [('numpy','numpy'), ('cv2','opencv-python'), ('PIL','Pillow'),
                ('yaml','PyYAML'), ('torch','torch'), ('torchvision','torchvision'),
                ('matplotlib','matplotlib'),('requests','requests'),('scipy','scipy'),
                ('psutil','psutil'),('polars','polars'),('thop','ultralytics-thop')]
    if with_ocr:
        runtime='onnxruntime-gpu' if gpu or (not version('onnxruntime') and version('onnxruntime-gpu')) else 'onnxruntime'
        packages += [('rapidocr_onnxruntime','rapidocr-onnxruntime'), ('onnxruntime',runtime)]
    for module, distribution in packages:
        ok = bool(available(module))
        value = version(distribution)
        if distribution in ('onnxruntime','onnxruntime-gpu'):
            ok = ok and value is not None
        if distribution=='ultralytics-thop' and value:
            import re
            numbers=tuple(int(x) for x in re.findall(r'\d+',value)[:3])
            ok=ok and numbers >= (2,0,18)
            if not ok: value+='（需要>=2.0.18）'
        checks.append({'name': distribution, 'ok': ok, 'detail': value or ('可导入' if ok else '未安装')})
    if with_ocr:
        cpu, cuda = version('onnxruntime'), version('onnxruntime-gpu')
        checks.append({'name':'ONNX Runtime发行包互斥', 'ok': not (cpu and cuda),
                       'detail':'同时安装CPU和GPU版本，请保留一个' if cpu and cuda else ('GPU' if cuda else 'CPU' if cpu else '未安装')})
    return {'root':str(root), 'ok':all(item['ok'] for item in checks), 'checks':checks,
            'ocr':with_ocr, 'gpu':gpu}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--without-ocr',action='store_true',help='只检查筛选/训练依赖')
    parser.add_argument('--gpu',action='store_true',help='检查GPU版ONNX Runtime安装状态')
    args=parser.parse_args(argv)
    result=inspect_environment(with_ocr=not args.without_ocr,gpu=args.gpu)
    print('项目目录：'+result['root'])
    print('Python：'+sys.version.split()[0])
    for item in result['checks']:
        print(('通过' if item['ok'] else '缺少/冲突')+' | '+item['name']+' | '+item['detail'])
    print('环境检查通过。' if result['ok'] else '请补齐以上资源或依赖后再运行。')
    if args.gpu:
        print('GPU检查覆盖安装包；CUDA设备与模型实际后端在评分启动时核对。')
    return 0 if result['ok'] else 1


if __name__=='__main__':
    raise SystemExit(main())
