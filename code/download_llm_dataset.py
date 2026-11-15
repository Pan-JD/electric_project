"""Download the S5 scoring export as a binary tar stream without a server-side copy."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='本地tar保存路径')
    parser.add_argument('--host', default='diffusion_5880', help='本地SSH配置中的服务器别名')
    parser.add_argument('--remote-project', default='/home/panjinduo/code/electronic_project')
    parser.add_argument('--remote-python', default='/home/panjinduo/install/miniconda3/envs/Yolo/bin/python')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    partial = output.with_name(output.name+'.part')
    if output.exists() or partial.exists():
        parser.error(f'目标或未完成下载已存在，请换一个文件名：{output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ' '.join(shlex.quote(value) for value in
                       [args.remote_python, '-u', '-B', args.remote_project.rstrip('/')+'/code/export_llm_dataset.py'])
    try:
        with partial.open('xb') as stream:
            subprocess.run(['ssh','-T','-o','BatchMode=yes',args.host,command],stdout=stream,check=True)
        os.rename(partial,output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    print(f'交接包已下载：{output}',flush=True)


if __name__ == '__main__':
    main()
