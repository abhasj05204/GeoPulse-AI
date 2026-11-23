"""Central paths and model feature contract; supports local and mounted storage."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STORAGE_ROOT=Path(os.getenv("GEOPULSE_STORAGE_DIR") or str(ROOT))
ARTIFACTS=STORAGE_ROOT/"artifacts"
DATA_RAW=STORAGE_ROOT/"data"/"raw"
DATA_PROCESSED=STORAGE_ROOT/"data"/"processed"
MODEL_PATH=Path(os.getenv("GEOPULSE_MODEL_PATH") or str(ARTIFACTS/"recurrence_model.joblib"))
FEATURE_COLUMNS=["count_7d","count_30d","count_90d","mean_daily_30d","days_since_last","neighbor_count_30d","day_of_week","month","x_km","y_km"]
for directory in (ARTIFACTS,DATA_RAW,DATA_PROCESSED): directory.mkdir(parents=True,exist_ok=True)
