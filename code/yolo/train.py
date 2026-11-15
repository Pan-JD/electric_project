"""Train one local YOLO dataset using explicit input and output paths."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time


CODE_DIRECTORY = Path(__file__).resolve().parents[1]
if str(CODE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CODE_DIRECTORY))

from simple_paths import configure_runtime, load_json, resolve_path, write_json
from yolo.evaluate import EVALUATION_FILES, write_detection_report


DEFAULT_PARAMETERS = {
    'epochs': 100, 'imgsz': 640, 'batch': 16, 'device': 'auto',
    'workers': 6, 'seed': 42, 'optimizer': 'SGD', 'lr0': .01,
    'patience': 40, 'deterministic': True, 'amp': False, 'cache': False,
}
CONTROLLED_PARAMETERS = {
    'data', 'model', 'project', 'name', 'exist_ok', 'resume', 'pretrained',
    'task', 'mode', 'source',
}


def load_training_config(config_path='configs/train.json'):
    value = load_json(config_path)
    if not isinstance(value, dict):
        raise ValueError('训练配置必须是JSON对象')
    for key in ('data', 'model', 'output'):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f'训练配置缺少有效的 {key} 路径')
    parameters = value.get('parameters', {})
    if not isinstance(parameters, dict):
        raise ValueError('parameters 必须是JSON对象')
    return value


def _dataloader_settings(config):
    settings={'workers':config.get('parameters',{}).get('workers',DEFAULT_PARAMETERS['workers']),
              'prefetch_factor':config.get('prefetch_factor',2)}
    for name,minimum in (('workers',0),('prefetch_factor',1)):
        value=settings[name]
        if not isinstance(value,int) or isinstance(value,bool) or value<minimum:
            raise ValueError(f'{name}须为大于等于{minimum}的整数')
    return settings


def training_plan(config):
    """Resolve paths only; never opens images, weights or model libraries."""
    parameters = {**DEFAULT_PARAMETERS, **config.get('parameters', {})}
    blocked = CONTROLLED_PARAMETERS & parameters.keys()
    if blocked:
        raise ValueError('parameters含输入输出控制字段：' + ', '.join(sorted(blocked)))
    if parameters.get('save',True) is not True:
        raise ValueError('训练需要保存权重，请设置save=true或省略save')
    data, model, output = (resolve_path(config[key]) for key in ('data', 'model', 'output'))
    repository = resolve_path('code/yolo/ultralytics-yolo26')
    return {'data': str(data), 'model': str(model), 'output': str(output),
            'repository': str(repository), 'parameters': parameters,
            'runtime':{'log_interval_seconds':config.get('log_interval_seconds',30),
                       **_dataloader_settings(config),
                       'NCCL_P2P_DISABLE':str(config.get('environment',{}).get('NCCL_P2P_DISABLE','1')),
                       'TORCH_NCCL_BLOCKING_WAIT':str(config.get('environment',{}).get('TORCH_NCCL_BLOCKING_WAIT','0'))}}


def _training_environment(config, repository):
    interval=float(config.get('log_interval_seconds',30))
    if not math.isfinite(interval) or interval<=0:
        raise ValueError('log_interval_seconds须为正秒数')
    loader=_dataloader_settings(config)
    p2p=str(config.get('environment',{}).get('NCCL_P2P_DISABLE','1'))
    if p2p not in ('0','1'):raise ValueError('NCCL_P2P_DISABLE须为0或1')
    os.environ['NCCL_P2P_DISABLE']=p2p
    blocking=str(config.get('environment',{}).get('TORCH_NCCL_BLOCKING_WAIT','0'))
    if blocking not in ('0','1'):raise ValueError('TORCH_NCCL_BLOCKING_WAIT须为0或1')
    os.environ['TORCH_NCCL_BLOCKING_WAIT']=blocking
    os.environ['PYTHONUNBUFFERED']='1'
    os.environ['PYTHONIOENCODING']='utf-8'
    os.environ['ELECTRONIC_PROGRESS_INTERVAL_SECONDS']=str(interval)
    os.environ['ELECTRONIC_WORKERS']=str(loader['workers'])
    os.environ['ELECTRONIC_PREFETCH_FACTOR']=str(loader['prefetch_factor'])
    # torchrun children must import this project's bundled source as well.
    paths=[str(repository),*os.environ.get('PYTHONPATH','').split(os.pathsep)]
    os.environ['PYTHONPATH']=os.pathsep.join(dict.fromkeys(path for path in paths if path))
    return {'NCCL_P2P_DISABLE':p2p,'TORCH_NCCL_BLOCKING_WAIT':blocking,
            'log_interval_seconds':interval,**loader}


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _write_json(path, value):
    def scalar(item):
        if isinstance(item, Path):
            return str(item)
        if hasattr(item, 'item'):
            return item.item()
        raise TypeError(f'无法记录结果类型：{type(item).__name__}')
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False, default=scalar) + '\n', encoding='utf-8')


def _dataset_snapshot(path, overrides=None):
    """Validate and normalize a local data YAML, without opening any image/XML."""
    import yaml
    path = Path(path)
    value = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError('data必须指向本地YOLO data.yaml')
    if overrides:
        value.update(overrides)
    if 'download' in value:
        raise ValueError('请使用本地数据集配置，移除data.yaml中的download字段')
    root = Path(value.get('path') or path.parent)
    root = root.resolve() if root.is_absolute() else (path.parent / root).resolve()
    value['path'] = root.as_posix()
    if value.get('image_root') or value.get('label_root'):
        for key in ('image_root', 'label_root'):
            if not value.get(key):
                raise ValueError('data.yaml的image_root与label_root须同时提供')
            directory = Path(value[key])
            directory = directory.resolve() if directory.is_absolute() else (root / directory).resolve()
            if not directory.is_dir():
                raise ValueError(f'{key}目录不存在：{directory}')
            value[key] = directory.as_posix()
    for split in ('train', 'val', *(['test'] if value.get('test') else [])):
        entries = value.get(split)
        if isinstance(entries, str):
            sequence = [entries]
        elif isinstance(entries, list) and entries:
            sequence = entries
        else:
            raise ValueError(f'data.yaml缺少有效的{split}输入')
        resolved = []
        for entry in sequence:
            if not isinstance(entry, str) or not entry or '://' in entry:
                raise ValueError(f'{split}必须指向本地图片目录或路径列表')
            candidate = Path(entry)
            candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
            if not candidate.exists():
                raise ValueError(f'{split}输入不存在：{candidate}')
            resolved.append(candidate.as_posix())
        value[split] = resolved[0] if isinstance(entries, str) else resolved
    if not value.get('names'):
        raise ValueError('data.yaml缺少类别names')
    if value.get('classes') is not None:
        selected = value['classes']
        if not isinstance(selected, list) or not selected or any(
                not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(value['names'])
                for index in selected):
            raise ValueError('data.yaml中的classes须为有效的原始类别编号列表')
    return value


class _Tee:
    def __init__(self, console, stream):
        self.console, self.stream = console, stream

    def write(self, value):
        clean=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',value).replace('\r','')
        self.stream.write(clean)
        self.stream.flush()
        self.console.write(value if self.console.isatty() else clean)
        return len(value)

    def flush(self):
        self.stream.flush()
        self.console.flush()

    def __getattr__(self, key):
        return getattr(self.console, key)


@contextmanager
def _console_log(path):
    with Path(path).open('x', encoding='utf-8', buffering=1) as stream:
        with redirect_stdout(_Tee(sys.stdout, stream)), redirect_stderr(_Tee(sys.stderr, stream)):
            yield


def _timestamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def finalize_output(output):
    """Keep the evaluated checkpoint, training history and final detection reports."""
    output=Path(output).resolve()
    final=output/'最终指标.json'
    report=json.loads(final.read_text(encoding='utf-8'))
    selected=Path(report['evaluation_weights']).resolve()
    if not selected.is_relative_to(output) or not selected.is_file():
        raise ValueError('最终评估权重不存在或位于实验目录之外')
    environment=output/'环境记录.json'
    if environment.is_file():
        record=json.loads(environment.read_text(encoding='utf-8'))
        report.update(parameters=record.get('actual_parameters',{}),runtime=record.get('runtime',{}),
                      seconds=record.get('seconds'),initial_model=record.get('initial_model'),
                      python=record.get('python'),torch=record.get('torch_version'),
                      ultralytics=record.get('ultralytics_version'))
    write_json(final,report)
    retained={final,selected}
    retained.update(output/'weights'/name for name in ('best.pt','last.pt'))
    retained.update(output/name for name in ('results.csv','results.png'))
    retained.update(output/'evaluation'/name for name in EVALUATION_FILES)
    retained.update((output/'evaluation').glob('val_batch*.jpg'))
    for path in sorted(output.rglob('*'),key=lambda value:len(value.parts),reverse=True):
        if path in retained:
            continue
        if path.is_symlink() or not path.is_dir():
            path.unlink()
        elif not any(path.iterdir()):
            path.rmdir()


def run_training(config, dry_run=False):
    plan = training_plan(config)
    if dry_run:
        return plan
    print(f"开始训练准备，GPU={plan['parameters']['device']}，batch={plan['parameters']['batch']}",flush=True)
    data, model, output, repository = (Path(plan[key]) for key in ('data', 'model', 'output', 'repository'))
    runtime=_training_environment(config,repository)
    if model.suffix.lower() != '.pt' or not model.is_file():
        raise ValueError(f'本地预训练权重不存在：{model}；请放置项目models下的.pt权重')
    if not (repository/'ultralytics'/'__init__.py').is_file():
        raise ValueError(f'项目内YOLO源码不存在：{repository}')
    if output.exists():
        raise ValueError(f'输出目录已存在：{output}；请为新实验修改output')
    if not data.is_file():
        raise ValueError(f'数据配置不存在：{data}')
    dataset = _dataset_snapshot(data)
    configure_runtime()
    # Atomically own a new directory before recording configuration. Ultralytics
    # may reuse this owned directory, so it never silently changes output to output2.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    import yaml
    dataset_path = output/'数据配置.yaml'
    dataset_path.write_text(yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False), encoding='utf-8')
    _write_json(output/'实验配置.json', {'config': config, 'resolved': plan,
                                     'data_sha256': _sha256(data), 'model_sha256': _sha256(model)})
    record = {'status': 'running', 'started_at_utc': _timestamp(),
              'python': sys.version.split()[0], 'platform': platform.platform(),
              'source_data': str(data), 'initial_model': str(model), 'repository': str(repository),
              'data_split_seed':dataset.get('split_seed'), 'training_seed':plan['parameters']['seed']}
    record['runtime']=runtime
    started = time.monotonic()
    try:
        with _console_log(output/'训练日志.log'):
            print('正在加载YOLO和PyTorch依赖',flush=True)
            sys.path.insert(0, str(repository))
            ultralytics = importlib.import_module('ultralytics')
            if not Path(ultralytics.__file__).resolve().is_relative_to(repository/'ultralytics'):
                raise ValueError(f'当前进程已加载其他Ultralytics，请启动独立Python进程：{ultralytics.__file__}')
            if getattr(ultralytics, 'settings', None) is not None:
                ultralytics.settings.update({'weights_dir': str(resolve_path('models')),
                                             'datasets_dir': str(resolve_path('data')),
                                             'runs_dir': str(resolve_path('outputs'))})
            record['ultralytics_version'] = getattr(ultralytics, '__version__', None)
            record['ultralytics_path'] = str(Path(ultralytics.__file__).resolve())
            try:
                record['torch_version'] = metadata.version('torch')
            except metadata.PackageNotFoundError:
                record['torch_version'] = None
            parameters = dict(plan['parameters'])
            if dataset.get('classes') is not None:
                parameters['classes'] = dataset['classes']
            parameters['plots']=False
            evaluation_split='test' if dataset.get('test') else 'val'
            train_test_only=evaluation_split=='test' and dataset.get('val')==dataset.get('train')
            if train_test_only:
                # The independent test pool must never guide checkpoint selection.
                # The YAML monitor points to training data; final scoring uses last.pt.
                parameters['val']=False
                parameters['patience']=0
            if str(parameters.get('device')).lower() == 'auto':
                torch = importlib.import_module('torch')
                parameters['device'] = '0' if torch.cuda.is_available() else 'cpu'
            record['device'] = parameters['device']
            record['evaluation_split']=evaluation_split
            record['actual_parameters']=parameters
            record['early_stopping']={'criterion':'validation mAP50-95','patience':parameters['patience'],
                                      'strict_improvement':True}
            _write_json(output/'环境记录.json', record)
            arguments = {**parameters, 'data': str(dataset_path), 'project': str(output.parent),
                         'name': output.name, 'exist_ok': True, 'pretrained': True, 'resume': False}
            print(f'输入数据：{data}', flush=True)
            print(f'初始模型：{model}', flush=True)
            print(f'训练输出：{output}', flush=True)
            print(f"NCCL_P2P_DISABLE={runtime['NCCL_P2P_DISABLE']}；进度日志间隔 {runtime['log_interval_seconds']:g} 秒",flush=True)
            print(f"TORCH_NCCL_BLOCKING_WAIT={runtime['TORCH_NCCL_BLOCKING_WAIT']}",flush=True)
            print(f"数据加载 workers={runtime['workers']}，每个worker预取{runtime['prefetch_factor']}批",flush=True)
            if parameters['patience']:
                print(f"早停：验证集mAP50-95连续{parameters['patience']}轮无提升；最多{parameters['epochs']}轮。",flush=True)
            detector = ultralytics.YOLO(str(model))
            detector.train(**arguments)
            if (output/'results.csv').is_file():
                from ultralytics.utils.plotting import plot_results
                plot_results(file=str(output/'results.csv'))
            selected = output/'weights'/('last.pt' if train_test_only else 'best.pt')
            if not selected.is_file():
                raise ValueError(f'训练未生成评估权重：{selected}')
            metrics = ultralytics.YOLO(str(selected)).val(
                data=str(dataset_path), split=evaluation_split, imgsz=parameters['imgsz'],
                batch=parameters['batch'], device=parameters['device'], workers=parameters['workers'],
                seed=parameters['seed'], project=str(output), name='evaluation', exist_ok=False, plots=True,
                classes=dataset.get('classes'))
            values = dict(metrics.results_dict)
            details = write_detection_report(metrics, output/'evaluation', dataset.get('classes'))
            _write_json(output/'最终指标.json', {'metrics': values, 'evaluation_weights': str(selected),'split':evaluation_split,
                                              'data_split_seed':dataset.get('split_seed'),'training_seed':parameters['seed'],
                                              'data': str(data), 'output': str(output), **details})
            record.update(status='completed', finished_at_utc=_timestamp(),
                          seconds=round(time.monotonic()-started, 3), evaluation_weights=str(selected), metrics=values)
            _write_json(output/'环境记录.json', record)
        finalize_output(output)
        return {'output': str(output), 'evaluation_weights': str(selected),'split':evaluation_split, 'metrics': values}
    except BaseException as exc:
        record.update(status='stopped' if isinstance(exc,KeyboardInterrupt) else 'failed', finished_at_utc=_timestamp(),
                      seconds=round(time.monotonic()-started, 3), error=f'{type(exc).__name__}: {exc}')
        _write_json(output/'环境记录.json', record)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/train.json', help='训练配置，路径相对项目根目录')
    parser.add_argument('--dry-run', action='store_true', help='只显示路径和参数，不加载模型或读取训练数据')
    args = parser.parse_args(argv)
    try:
        config = load_training_config(args.config)
        result = run_training(config, dry_run=args.dry_run)
        if args.dry_run:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, str(exc)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())
