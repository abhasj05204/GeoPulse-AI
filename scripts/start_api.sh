#!/usr/bin/env bash
set -euo pipefail
STORAGE_ROOT="${GEOPULSE_STORAGE_DIR:-.}"
MODEL_PATH="${GEOPULSE_MODEL_PATH:-$STORAGE_ROOT/artifacts/recurrence_model.joblib}"
if [[ "${GEOPULSE_TRAIN_ON_START:-false}" == "true" && ! -f "$MODEL_PATH" ]]; then
  echo "GeoPulse model artifact not found; starting the configured training run."
  ARGS=(--start "${GEOPULSE_TRAIN_START:-2025-01-01}" --end "${GEOPULSE_TRAIN_END:-2025-07-01}"
        --limit "${GEOPULSE_TRAIN_LIMIT:-30000}" --cell-size-m "${GEOPULSE_CELL_SIZE_M:-1000}" --horizon-days 30)
  if [[ "${GEOPULSE_TUNE_ON_START:-false}" == "true" ]]; then ARGS+=(--tune --trials "${GEOPULSE_OPTUNA_TRIALS:-10}"); fi
  python scripts/train.py "${ARGS[@]}"
fi
exec uvicorn backend.main:app --host 0.0.0.0 --port "${PORT:-8000}"
