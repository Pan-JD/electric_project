#!/usr/bin/env bash
# Manage one background training run using the active Python environment.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python不存在，请激活conda环境或设置PYTHON_BIN：$PYTHON_BIN" >&2
    exit 2
fi
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
exec "$PYTHON_BIN" -u -B code/yolo/background.py "$@"
