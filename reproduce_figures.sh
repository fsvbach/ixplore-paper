#!/usr/bin/env bash
# Regenerate every figure and table of the paper from the stored results/.
# Run from anywhere:  ./reproduce_figures.sh
# Requires the Python environment from requirements.txt (see README.md).
set -euo pipefail
cd "$(dirname "$0")"
python -m src.figures
python -m src.tables
echo "== done: figures/ and tables/ regenerated"
