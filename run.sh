#!/usr/bin/env bash
# Scoring entry point: data folder -> features -> predictions. Offline, no training.
#   ./run.sh <DATA_DIR> <MODEL_PATH> <OUTPUT_PATH>
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT"

DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

# Pick a Python that has the scoring dependencies installed.
# Priority: $PYTHON if set -> project venv -> python -> python3
_probe() { "$1" -c "import lightgbm, pandas, pyarrow" 2>/dev/null; }

if [ -n "${PYTHON:-}" ] && _probe "$PYTHON"; then
    :
elif [ -f "$ROOT/.venv/bin/python" ] && _probe "$ROOT/.venv/bin/python"; then
    PYTHON="$ROOT/.venv/bin/python"
elif [ -f "$ROOT/.venv/Scripts/python" ] && _probe "$ROOT/.venv/Scripts/python"; then
    PYTHON="$ROOT/.venv/Scripts/python"
elif _probe python; then
    PYTHON="python"
elif _probe python3; then
    PYTHON="python3"
else
    echo "ERROR: no Python with lightgbm, pandas and pyarrow found." >&2
    echo "Run: pip install -r requirements.txt" >&2
    exit 1
fi
echo "Using Python: $PYTHON"

mkdir -p "$(dirname "$OUTPUT_PATH")"
rm -f "$OUTPUT_PATH"          # a failed run must not leave an old file behind

# 1. Generate the features the model expects from whatever is in DATA_DIR
"$PYTHON" "$ROOT/src/generate_features.py" \
  --data-dir "$DATA_DIR" \
  --out features.parquet

# 2. Load the pickled model and produce predictions
"$PYTHON" "$ROOT/src/predict.py" \
  --features features.parquet \
  --model "$MODEL_PATH" \
  --output "$OUTPUT_PATH"

echo "Done. Predictions written to $OUTPUT_PATH"
