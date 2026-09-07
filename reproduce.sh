#!/usr/bin/env bash
# Regenerate every figure and table of the paper from the stored results.
# Run from the repository root:  ./reproduce.sh
# Requires the Python environment from requirements.txt (see README.md).
set -euo pipefail
cd "$(dirname "$0")"
for nb in notebooks/*/*.ipynb; do
  echo "== $nb"
  (cd "$(dirname "$nb")" && jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.kernel_name=python3 \
      --ExecutePreprocessor.timeout=1800 \
      "$(basename "$nb")")
done
echo "== done: figures/ and tables/ regenerated"
