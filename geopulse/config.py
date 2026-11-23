"""Central project paths and model feature contract."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ARTIFACTS=ROOT/"artifacts"
MODEL_PATH=Path(os.getenv("GEOPULSE_MODEL_PATH",str(ARTIFACTS/"recurrence_model.joblib")))
FEATURE_COLUMNS=["count_7d","count_30d","count_90d","mean_daily_30d","days_since_last","neighbor_count_30d","day_of_week","month","x_km","y_km"]
