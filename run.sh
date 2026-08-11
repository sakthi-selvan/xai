#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -d .venv ]]; then
  uv venv --python 3.11 .venv
fi
uv pip install -r requirements.txt
exec .venv/bin/streamlit run app.py --server.headless true
