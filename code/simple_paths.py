"""Project-relative paths shared by the three public scripts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

PROJECT_ROOT=Path(__file__).resolve().parents[1]
_LIB=Path(__file__).resolve().parent/'lib'
if str(_LIB) not in sys.path:
    sys.path.insert(0,str(_LIB))


def project_root():
    return PROJECT_ROOT


def resolve_path(value):
    path=Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root()/path).resolve()


def load_json(value):
    with resolve_path(value).open(encoding='utf-8-sig') as stream:
        return json.load(stream)


def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                         prefix='.'+path.name+'.',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name)
            stream.write(content)
        os.replace(temporary,path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def relative_list_path(path,base):
    try:
        return './'+os.path.relpath(path,base).replace('\\','/')
    except ValueError:
        return Path(path).resolve().as_posix()


def configure_runtime():
    """Keep YOLO runtime settings/fonts inside this folder; never auto-install packages."""
    cache=project_root()/'.cache/ultralytics'
    cache.mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(cache)
    os.environ['YOLO_AUTOINSTALL']='false'
    font=project_root()/'assets/fonts/DejaVuSans.ttf'
    if not font.is_file():
        raise ValueError(f'项目字体缺失：{font}')
    for directory in (cache,cache/'Ultralytics'):
        directory.mkdir(exist_ok=True)
        for name in ('Arial.ttf','Arial.Unicode.ttf'):
            target=directory/name
            if not target.is_file(): shutil.copy2(font,target)
    return cache
