"""Console/file logging and periodic stage progress, without worker-side SQLite access."""
from __future__ import annotations

import contextlib
import contextvars
import re
import sys
import threading
import time
from datetime import datetime


_OUTPUT_LOCK = threading.RLock()
_CURRENT = contextvars.ContextVar('quality_progress', default=None)


def log(message):
    with _OUTPUT_LOCK:
        print(time.strftime('[%H:%M:%S] ') + message, flush=True)


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class Tee:
    def __init__(self, console, stream):
        self.console, self.stream = console, stream

    def write(self, value):
        with _OUTPUT_LOCK:
            self.stream.write(value)
            self.stream.flush()
            try:
                return self.console.write(value)
            except UnicodeEncodeError:
                # Keep UTF-8 in the log even on consoles using a legacy codepage.
                encoding = getattr(self.console, 'encoding', None) or 'utf-8'
                return self.console.write(value.encode(encoding, 'replace').decode(encoding))

    def flush(self):
        with _OUTPUT_LOCK:
            self.stream.flush()
            self.console.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


class Progress:
    def __init__(self, interval=3.0, emit=log):
        self.interval, self.emit = interval, emit
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.name = None
        self.done, self.total, self.unit, self.note = 0, None, '张', ''
        self.rate_done = None
        self.started = time.monotonic()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._heartbeat, name='quality-progress', daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()

    def _heartbeat(self):
        while not self.stop_event.wait(self.interval):
            self.render()

    def begin(self, name, total=None, unit='张', note=''):
        with self.lock:
            self.name, self.total, self.unit, self.note = name, total, unit, note
            self.done, self.started = 0, time.monotonic()
            self.rate_done = None
        self.render()

    def update(self, done=None, total=None, note=None, rate_done=None):
        with self.lock:
            if done is not None:
                self.done = done
            if total is not None:
                self.total = total
            if note is not None:
                self.note = note
            if rate_done is not None:
                self.rate_done = rate_done

    def render(self, ending=''):
        with self.lock:
            if self.name is None:
                return
            elapsed = time.monotonic() - self.started
            if self.total is None:
                count = f'已发现 {self.done:,} {self.unit}'
            else:
                percentage = 100 * self.done / self.total if self.total else 100.0
                count = f'{self.done:,}/{self.total:,} {self.unit} ({percentage:.2f}%)'
            parts = [f'[{self.name}] {count}', f'耗时 {duration(elapsed)}']
            work = self.done if self.rate_done is None else self.rate_done
            if work and elapsed >= 1:
                rate = work / elapsed
                parts.append(f'速度 {rate:.2f} {self.unit}/秒')
                if self.total is not None and self.done < self.total:
                    parts.append(f'预计剩余 {duration((self.total-self.done)/rate)}')
            if self.note:
                parts.append(self.note)
            if ending:
                parts.append(ending)
            self.emit(' | '.join(parts))

    def end(self, success):
        with self.lock:
            self.render('阶段完成' if success else '阶段中断')
            self.name = None


@contextlib.contextmanager
def run_logging(directory, name, interval=3.0):
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name) or 'quality'
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    target = directory / f'{safe_name}_{stamp}.log'
    with target.open('x', encoding='utf-8', buffering=1) as stream:
        with contextlib.redirect_stdout(Tee(sys.stdout, stream)), contextlib.redirect_stderr(Tee(sys.stderr, stream)):
            reporter = Progress(interval)
            token = _CURRENT.set(reporter)
            reporter.start()
            try:
                log(f'日志文件：{target}')
                yield target
            finally:
                reporter.stop()
                _CURRENT.reset(token)


@contextlib.contextmanager
def progress_stage(name, total=None, unit='张', note=''):
    reporter = _CURRENT.get()
    if reporter:
        reporter.begin(name, total, unit, note)
    try:
        yield
    except BaseException:
        if reporter:
            reporter.end(False)
        raise
    else:
        if reporter:
            reporter.end(True)


def update_progress(done=None, total=None, note=None, rate_done=None):
    reporter = _CURRENT.get()
    if reporter:
        reporter.update(done, total, note, rate_done)


def progress_items(items):
    for number, item in enumerate(items, 1):
        yield item
        update_progress(number)
