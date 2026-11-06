#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
command -v uv >/dev/null || { echo '请先安装 uv。' >&2; exit 3; }
export UV_CACHE_DIR="$PROJECT_DIR/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PROJECT_DIR/.python"
export PLAYWRIGHT_BROWSERS_PATH="$PROJECT_DIR/.browsers"
uv sync --frozen
.venv/bin/python -m playwright install chromium
echo '下载评测模型；来源与校验信息见 models/manifest.json。'
.venv/bin/python tools/fetch_models.py
echo '安装完成。运行 bash run.sh --help 查看命令。'
