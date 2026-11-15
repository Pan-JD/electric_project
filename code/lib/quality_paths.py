"""本地/S5 的数据默认路径；代码资源和结果根目录随部署位置移动。"""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_SOURCES = {
    'local': str(PROJECT_ROOT/'data/source'),
    's5': str(PROJECT_ROOT/'data/source'),
}


def resolve_project_input(value):
    """Use an existing cwd-relative input, otherwise resolve under the project."""
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def environment_name(value='auto'):
    if value == 'auto':
        return 'local' if os.name == 'nt' else 's5'
    if value not in DATA_SOURCES:
        raise ValueError(f'未知运行环境：{value}')
    return value


def default_source(environment='auto'):
    return Path(DATA_SOURCES[environment_name(environment)])


def run_directory(name):
    # Run names are a single directory component on both platforms.
    if not name or name in ('.', '..') or any(c in name for c in '/\\:*?"<>|\0'):
        raise ValueError('--run-name 必须为单个目录名，不能包含路径分隔符')
    return PROJECT_ROOT/'run_results'/name


def stored_path(value):
    """解析缓存中的原平台路径，避免 Linux 将 G:\\... 当作相对文件名。"""
    value = str(value)
    windows = PureWindowsPath(value)
    if windows.drive:
        return windows
    if value.startswith('/'):
        return PurePosixPath(value)
    return stored_path(str(Path(value).resolve()))


def map_stored_path(value, mappings=()):
    original = stored_path(value)
    for old, new in mappings:
        try:
            relative = original.relative_to(stored_path(old))
        except ValueError:
            continue
        if '..' in relative.parts:
            raise ValueError('缓存路径含有越过映射根目录的 ..')
        return Path(new).resolve().joinpath(*relative.parts)
    if (isinstance(original, PureWindowsPath) and os.name != 'nt') or (
            isinstance(original, PurePosixPath) and os.name == 'nt'):
        raise FileNotFoundError(f'原平台路径需要 --path-map 映射：{value}')
    return Path(original).resolve()


def environment_path_maps(environment='auto'):
    destination = environment_name(environment)
    origin = 's5' if destination == 'local' else 'local'
    return [(DATA_SOURCES[origin], DATA_SOURCES[destination])]
