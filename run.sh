#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT"

DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

# Auto-detect Python with required packages (test lightgbm — project's key dep)
# Priority: venv → bare 'python' → python3.10 → python3
_probe() { "$1" -c "import lightgbm" 2>/dev/null; }

if [ -f "$ROOT/.venv/bin/python" ] && _probe "$ROOT/.venv/bin/python"; then
    PYTHON="$ROOT/.venv/bin/python"
elif [ -f "$ROOT/.venv/Scripts/python" ] && _probe "$ROOT/.venv/Scripts/python"; then
    PYTHON="$ROOT/.venv/Scripts/python"
elif _probe python; then
    PYTHON="python"
elif _probe python3.10; then
    PYTHON="python3.10"
elif _probe python3; then
    PYTHON="python3"
else
    echo "ERROR: Could not find a Python with lightgbm installed." >&2
    echo "Run: pip install -r requirements.txt" >&2
    exit 1
fi


echo "Using Python: $PYTHON"

mkdir -p "$(dirname "$OUTPUT_PATH")"

"$PYTHON" src/generate_features.py \
  --data-dir "$DATA_DIR" \
  --out features.parquet

"$PYTHON" src/predict.py \
  --features features.parquet \
  --model "$MODEL_PATH" \
  --output "$OUTPUT_PATH"

echo "Done. Predictions written to $OUTPUT_PATH"
