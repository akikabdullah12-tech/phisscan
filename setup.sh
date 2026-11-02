#!/usr/bin/env bash
# PhisScan - one-time setup on macOS / Linux
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
echo "Setup complete. Run:  source .venv/bin/activate && python app.py"
