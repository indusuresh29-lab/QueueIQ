#!/bin/bash
set -e

echo "QueueIQ setup"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ""
echo "Setup complete."
echo "Run QueueIQ with:"
echo "source .venv/bin/activate"
echo "streamlit run app.py"
