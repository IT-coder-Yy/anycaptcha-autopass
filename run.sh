#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
export CAPTCHA_PROJECT_ROOT="$PROJECT_DIR"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$PROJECT_DIR/.browsers}"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -x .venv/bin/python ]]; then
  echo '缺少本地运行环境，请先运行 bash setup.sh。' >&2
  exit 3
fi
exec .venv/bin/python -m cli "$@"
