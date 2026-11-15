#!/usr/bin/env bash
# Manage one background training run and follow its log on S5.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-/home/panjinduo/install/miniconda3/envs/Yolo/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python不存在：$PYTHON_BIN" >&2
    exit 2
fi
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
exec "$PYTHON_BIN" -u -B code/yolo/background.py "$@"
