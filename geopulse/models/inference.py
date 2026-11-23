"""Load the trained pipeline and return validated recurrence estimates."""
from pathlib import Path
import joblib
import pandas as pd

class RecurrencePredictor:
    def __init__(self, path="artifacts/recurrence_model.joblib"):
        path=Path(path)
        if not path.exists(): raise FileNotFoundError(f"Model artifact missing: {path}")
        artifact=joblib.load(path); self.model=artifact["model"]
        self.model_name=artifact["model_name"]; self.features=artifact["feature_columns"]
    def predict(self, rows: list[dict]) -> list[dict]:
        frame=pd.DataFrame(rows)
        missing=set(self.features)-set(frame.columns)
        if missing: raise ValueError(f"Missing features: {sorted(missing)}")
        p=self.model.predict_proba(frame[self.features])[:,1]
        return [{"recurrence_probability":float(v),"risk_score":round(float(v)*100,2),
          "risk_level":"High" if v>=.7 else "Medium" if v>=.4 else "Low",
          "model_name":self.model_name} for v in p]
