"""Evaluate an existing detector and export per-class metrics and standard plots."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

CODE_DIRECTORY = Path(__file__).resolve().parents[1]
if str(CODE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CODE_DIRECTORY))

from simple_paths import configure_runtime, resolve_path, write_json


EVALUATION_FILES = {
    '各类指标.csv', 'BoxPR_curve.png', 'BoxF1_curve.png',
    'BoxP_curve.png', 'BoxR_curve.png',
    'confusion_matrix.png', 'confusion_matrix_normalized.png',
}


def load_detector(model):
    model = resolve_path(model)
    if not model.is_file() or model.suffix.lower() != '.pt':
        raise ValueError(f'请选择本地.pt权重文件：{model}')
    configure_runtime()
    repository = resolve_path('code/yolo/ultralytics-yolo26')
    sys.path.insert(0, str(repository))
    import ultralytics
    if not Path(ultralytics.__file__).resolve().is_relative_to(repository/'ultralytics'):
        raise ValueError(f'当前已加载其他Ultralytics，请启动独立Python进程：{ultralytics.__file__}')
    return ultralytics.YOLO(str(model))


def write_detection_report(metrics, output, classes=None):
    output = Path(output)
    positions = {int(category):position for position, category in enumerate(metrics.ap_class_index)}
    names = metrics.names
    names = names.items() if isinstance(names, dict) else enumerate(names)
    rows = []
    for category, name in names:
        if classes is not None and category not in classes:
            continue
        images = int(metrics.nt_per_image[category]) if metrics.nt_per_image is not None else 0
        instances = int(metrics.nt_per_class[category]) if metrics.nt_per_class is not None else 0
        position = positions.get(category)
        if position is None:
            precision = recall = f1 = ap50 = ap95 = None
        else:
            precision, recall, ap50, ap95 = map(float, metrics.class_result(position))
            f1 = float(metrics.box.f1[position])
        rows.append({'类别编号':int(category), '类别':name, '含该类图片数':images, '真实目标数':instances,
                     'Precision':precision, 'Recall':recall, 'F1':f1,
                     'mAP50':ap50, 'mAP50-95':ap95,
                     '评估状态':'已评估' if position is not None else '评估集无该类真实目标'})
    with (output/'各类指标.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            '类别编号', '类别', '含该类图片数', '真实目标数',
            'Precision', 'Recall', 'F1', 'mAP50', 'mAP50-95', '评估状态'])
        writer.writeheader()
        writer.writerows(rows)
    print('各类指标（0–1）', flush=True)
    for row in rows:
        if row['F1'] is not None:
            print(f"{row['类别']}  P={row['Precision']:.4f}  R={row['Recall']:.4f}  "
                  f"F1={row['F1']:.4f}  mAP50={row['mAP50']:.4f}  mAP50-95={row['mAP50-95']:.4f}", flush=True)
    return {'per_class':rows,
            'pr_f1_definition':'IoU=0.5，使用Ultralytics全类别平均F1平滑曲线最优置信度处的P、R和F1'}


def evaluate_model(model, data, output='outputs/evaluate', split='val',
                   imgsz=640, batch=16, device='auto', workers=2, conf=0.001, iou=0.7, source=None):
    from yolo.train import _dataset_snapshot

    if imgsz < 1 or batch < 1 or workers < 0 or not 0 <= conf <= 1 or not 0 <= iou <= 1:
        raise ValueError('imgsz、batch须为正数，workers须非负，conf、iou须介于0和1之间')
    overrides = {split:str(resolve_path(source))} if source else None
    dataset = _dataset_snapshot(resolve_path(data), overrides=overrides)
    if split not in dataset:
        raise ValueError(f'data.yaml缺少{split}划分')
    print(f'正在加载模型：{resolve_path(model)}', flush=True)
    detector = load_detector(model)
    if detector.task != 'detect':
        raise ValueError('当前评估脚本用于目标检测模型')
    classes = dataset.get('classes')
    single_cls = False
    model_names = detector.names
    source_names = dataset['names']
    model_names = list(model_names.values()) if isinstance(model_names, dict) else model_names
    if len(model_names) == 1 and len(source_names) > 1:
        source_names = dict(enumerate(source_names)) if isinstance(source_names, list) else source_names
        classes = [int(index) for index, name in source_names.items() if name == model_names[0]]
        if len(classes) != 1:
            raise ValueError(f'评估数据中无法唯一匹配模型类别：{model_names[0]}')
        single_cls = True
    from ultralytics.utils.files import increment_path

    output = increment_path(resolve_path(output), mkdir=True)
    dataset_path = output/'数据配置.yaml'
    write_json(dataset_path, dataset)
    print(f'评估划分 {split}；输出 {output}', flush=True)
    metrics = detector.val(data=str(dataset_path), split=split, imgsz=imgsz, batch=batch,
                           device=None if device == 'auto' else device, workers=workers,
                           conf=conf, iou=iou, plots=True, verbose=True,
                           project=str(output.parent), name=output.name, exist_ok=True,
                           classes=classes, single_cls=single_cls)
    details = write_detection_report(metrics, output, None if single_cls else classes)
    write_json(output/'最终指标.json', {
        'evaluation_weights':str(resolve_path(model)), 'data':str(resolve_path(data)), 'split':split,
        'evaluation_source':dataset[split],
        'metrics':{key:float(value) for key,value in metrics.results_dict.items()},
        'speed':{key:float(value) for key,value in metrics.speed.items()},
        'parameters':{'imgsz':imgsz, 'batch':batch, 'conf':conf, 'iou':iou, 'device':device,
                      'classes':classes, 'single_cls':single_cls}, **details})
    print(f'评估完成，结果保存在 {output}', flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='本地best.pt或last.pt路径')
    parser.add_argument('--data', required=True, help='带真实标注的data.yaml路径')
    parser.add_argument('--output', default='outputs/evaluate')
    parser.add_argument('--split', choices=('train','val','test'), default='val')
    parser.add_argument('--source', help='覆盖该划分的图片目录或图片路径TXT，默认使用data.yaml中的路径')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--conf', type=float, default=0.001, help='评估候选框最低置信度')
    parser.add_argument('--iou', type=float, default=0.7, help='NMS IoU阈值')
    args = parser.parse_args(argv)
    try:
        evaluate_model(**vars(args))
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc)+'\n')


if __name__ == '__main__':
    main()
