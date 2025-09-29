#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python3}
export PATH="$HOME/.local/bin:$PATH"

have_venv()     { $PY -m venv --help >/dev/null 2>&1; }
have_virtualenv(){ $PY -m virtualenv --version >/dev/null 2>&1; }

# Create .venv if missing, preferring stdlib venv, else virtualenv fallback
if [ ! -d ".venv" ]; then
  if have_venv; then
    echo "-> Creating venv with stdlib venv"
    $PY -m venv .venv
  else
    echo "-> stdlib venv not available; using virtualenv"
    if ! have_virtualenv; then
      echo "-> Installing virtualenv to user site"
      $PY -m pip install --user virtualenv
    fi
    $PY -m virtualenv .venv
  fi
fi

. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo "Environment ready. Activate with: source .venv/bin/activate"
