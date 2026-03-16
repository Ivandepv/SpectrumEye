#!/usr/bin/env bash
# Run once after installing pyenv + Python 3.12
# Usage: cd ml && bash setup_env.sh

set -e

echo "Creating Python 3.12 virtual environment..."
python3.12 -m venv .venv

echo "Activating venv..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Done! To activate the environment:"
echo "  source ml/.venv/bin/activate"
