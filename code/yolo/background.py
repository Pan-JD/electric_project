"""Start, view or stop an S5 training run using its recorded process ID."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE_DIRECTORY=Path(__file__).resolve().parents[1]
if str(CODE_DIRECTORY) not in sys.path:
    sys.path.insert(0,str(CODE_DIRECTORY))

from simple_paths import project_root, resolve_path, write_json
from yolo.train import load_training_config, training_plan


def process_info(pid):
    """Check owner and start time before inspecting or signaling a process."""
    folder=Path('/proc')/str(pid)
    try:
        if folder.stat().st_uid!=os.getuid():
            return None
        fields=(folder/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':
            return None
        return {'pid':pid,'ppid':int(fields[1]),'start_ticks':int(fields[19])}
    except (OSError,ValueError,IndexError):
        return None


def matches_training(info,config,pid_file=None):
    folder=Path('/proc')/str(info['pid'])
    try:
        arguments=(folder/'cmdline').read_bytes().decode().strip('\0').split('\0')
        cwd=(folder/'cwd').resolve()
        script=resolve_path('code/yolo/train.py')
        if not any((cwd/value).resolve()==script for value in arguments[1:]):
            return False
        position=arguments.index('--config')
        if (cwd/arguments[position+1]).resolve()!=config:
            return False
        if pid_file is None:
            return True
        boot_time=time.time()-float(Path('/proc/uptime').read_text().split()[0])
        started=boot_time+info['start_ticks']/os.sysconf('SC_CLK_TCK')
        return abs(started-pid_file.stat().st_mtime)<5
    except (OSError,ValueError,IndexError,UnicodeError):
        return False


def find_run(config,output,active_only=True):
    files=[file for file in (project_root()/'logs').glob('*.pid')
           if file.stem.rsplit('_',3)[0]==output.name]
    files.sort(key=lambda file:file.stat().st_mtime,reverse=True)
    for file in files:
        try:
            pid=int(file.read_text().strip())
        except (OSError,ValueError):
            continue
        log=file.with_suffix('.log')
        if not log.is_file():
            continue
        info=process_info(pid)
        if info and matches_training(info,config,file):
            return {'pid':pid,'log':log,'pid_file':file,'info':info,'output':output}
    if not active_only:
        for file in files:
            if file.with_suffix('.log').is_file():
                return {'pid':None,'log':file.with_suffix('.log'),'pid_file':file,'info':None,'output':output}
    return None


def find_training_processes(config):
    """Locate training roots by script/config even after outputs or PID files change."""
    matched={}
    for folder in Path('/proc').iterdir():
        if folder.name.isdigit():
            info=process_info(int(folder.name))
            if info and matches_training(info,config):
                matched[info['pid']]=info
    runs=[]
    for pid,info in matched.items():
        if info['ppid'] in matched:
            continue
        log=None
        output=None
        try:
            log=Path(os.readlink(f'/proc/{pid}/fd/1').removesuffix(' (deleted)'))
            for descriptor in Path(f'/proc/{pid}/fd').iterdir():
                target=Path(os.readlink(descriptor).removesuffix(' (deleted)'))
                if target.name=='训练日志.log':
                    output=target.parent
                    break
        except OSError:
            pass
        runs.append({'pid':pid,'info':info,'log':log,'output':output})
    return sorted(runs,key=lambda run:run['info']['start_ticks'],reverse=True)


def start_run(config,plan):
    output=Path(plan['output'])
    if output.exists():
        raise ValueError(f'输出目录已存在：{output}；请在训练配置中修改output')
    if not Path(plan['model']).is_file():
        raise ValueError(f'模型文件不存在：{plan["model"]}')
    if not (Path(plan['repository'])/'ultralytics/__init__.py').is_file():
        raise ValueError(f'YOLO源码目录不存在：{plan["repository"]}')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log=project_root()/'logs'/f'{output.name}_{stamp}.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    environment={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONIOENCODING':'utf-8'}
    with log.open('x',encoding='utf-8') as stream:
        process=subprocess.Popen([sys.executable,'-u','-B','code/yolo/train.py','--config',str(config)],
                                 cwd=project_root(),env=environment,stdin=subprocess.DEVNULL,
                                 stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    pid_file=log.with_suffix('.pid')
    pid_file.write_text(str(process.pid)+'\n',encoding='utf-8')
    return {'pid':process.pid,'log':log,'pid_file':pid_file,'info':process_info(process.pid),
            'output':output,'process':process}


def refresh_tree(tracked):
    """Track descendants across separate torchrun worker process groups."""
    children=defaultdict(list)
    table={}
    for folder in Path('/proc').iterdir():
        if folder.name.isdigit():
            info=process_info(int(folder.name))
            if info:
                table[info['pid']]=info
                children[info['ppid']].append(info['pid'])
    alive={pid:info for pid,info in table.items()
           if pid in tracked and info['start_ticks']==tracked[pid]['start_ticks']}
    pending=list(alive)
    while pending:
        for pid in children[pending.pop()]:
            if pid not in alive:
                tracked[pid]=table[pid]
                alive[pid]=table[pid]
                pending.append(pid)
    return alive


def send_signal(processes,number):
    for pid,expected in processes.items():
        current=process_info(pid)
        if current and current['start_ticks']==expected['start_ticks']:
            try:
                os.kill(pid,number)
            except ProcessLookupError:
                pass


def wait_for_exit(tracked,seconds):
    deadline=time.monotonic()+seconds
    alive=refresh_tree(tracked)
    while alive and time.monotonic()<deadline:
        time.sleep(.25)
        alive=refresh_tree(tracked)
    return alive


def stop_run(run):
    tracked={run['pid']:run['info']}
    refresh_tree(tracked)
    print(f'正在停止训练，进程号 {run["pid"]}',flush=True)
    send_signal({run['pid']:run['info']},signal.SIGINT)
    alive=wait_for_exit(tracked,8)
    if alive:
        send_signal(alive,signal.SIGTERM)
        alive=wait_for_exit(tracked,5)
    if alive:
        send_signal(alive,signal.SIGKILL)
        alive=wait_for_exit(tracked,3)
    if alive:
        raise ValueError('仍有训练进程未退出：'+','.join(str(pid) for pid in alive))
    print(f'训练已停止，日志 {run["log"]}',flush=True)
    if run.get('output') is None:
        return
    record=run['output']/'环境记录.json'
    if record.is_file():
        try:
            value=json.loads(record.read_text(encoding='utf-8'))
            if value.get('status')=='running':
                value.update(status='stopped',finished_at_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                             stopped_by='train_background.sh --stop')
                write_json(record,value)
        except (OSError,ValueError) as error:
            print(f'环境记录更新失败：{error}',file=sys.stderr)


def show_log(run,follow):
    print(f'日志文件 {run["log"]}',flush=True)
    arguments=['tail','-n','30']
    if not follow or not run['pid']:
        subprocess.run([*arguments,str(run['log'])],check=True)
        return
    print('正在显示日志，按Ctrl+C退出查看，训练继续在后台运行。',flush=True)
    follower=subprocess.Popen([*arguments,'-f',str(run['log'])])
    process=run.get('process')
    try:
        while follower.poll() is None:
            alive=process.poll() is None if process is not None else process_info(run['pid']) is not None
            if not alive:
                time.sleep(1)  # Let tail flush the final log lines.
                result=f'，退出码{process.returncode}' if process is not None else ''
                print(f'训练进程已结束{result}，日志 {run["log"]}',flush=True)
                break
            time.sleep(.5)
    except KeyboardInterrupt:
        print('\n已退出日志查看，训练继续在后台运行。',flush=True)
    finally:
        if follower.poll() is None:
            follower.terminate()
        follower.wait()


def main(argv=None):
    parser=argparse.ArgumentParser(description='原始组默认后台训练，支持查看日志和停止对应配置的训练')
    parser.add_argument('config',nargs='?',default='configs/train_original.json',help='训练配置路径')
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--stop',action='store_true',help='停止该配置正在运行的训练及其子进程')
    actions.add_argument('--logs',action='store_true',help='查看该配置已有训练的日志')
    parser.add_argument('--no-follow',action='store_true',help='启动后返回终端，或只显示最近30行日志')
    args=parser.parse_args(argv)
    try:
        if sys.platform!='linux':
            raise ValueError('请在S5的Linux终端运行此后台管理脚本')
        config=resolve_path(args.config)
        if args.stop:
            runs=find_training_processes(config)
            for run in runs:
                stop_run(run)
            if not runs:
                print('当前配置没有运行中的训练。')
            return 0
        plan=training_plan(load_training_config(config))
        import fcntl
        with config.open('rb') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX)
            active=find_training_processes(config)
            run=active[0] if active else find_run(config,Path(plan['output']),active_only=not args.logs)
            created=not args.logs and run is None
            if created:
                run=start_run(config,plan)
        if args.logs and run is None:
            raise ValueError('当前配置没有可查看的训练日志')
        if not args.logs:
            if created:
                print(f'已提交后台训练，进程号 {run["pid"]}',flush=True)
            else:
                print(f'已找到运行中的训练，进程号 {run["pid"]}',flush=True)
            print(f'训练配置 {config}',flush=True)
            print(f'训练输出 {run.get("output") or plan["output"]}',flush=True)
        if args.no_follow and not args.logs:
            print(f'日志文件 {run["log"]}',flush=True)
            return 0
        show_log(run,not args.no_follow)
    except (ValueError,OSError,KeyError) as error:
        parser.exit(2,str(error)+'\n')


if __name__=='__main__':
    raise SystemExit(main())
