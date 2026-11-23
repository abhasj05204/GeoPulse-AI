"""Load the trained pipeline and return validation-derived recurrence bands."""
from pathlib import Path
import joblib
import pandas as pd

class RecurrencePredictor:
    def __init__(self,path="artifacts/recurrence_model.joblib"):
        path=Path(path)
        if not path.exists(): raise FileNotFoundError(f"Model artifact missing: {path}")
        artifact=joblib.load(path); self.model=artifact["model"]
        self.model_name=artifact["model_name"]; self.features=artifact["feature_columns"]
        self.decision_threshold=float(artifact.get("decision_threshold",.5))
        self.risk_thresholds=artifact.get("risk_thresholds",{"medium":.4,"high":.7})
    def predict(self,rows:list[dict])->list[dict]:
        frame=pd.DataFrame(rows); missing=set(self.features)-set(frame.columns)
        if missing: raise ValueError(f"Missing features: {sorted(missing)}")
        probabilities=self.model.predict_proba(frame[self.features])[:,1]
        output=[]
        for probability in probabilities:
            medium=self.risk_thresholds["medium"]; high=self.risk_thresholds["high"]
            level="High" if probability>=high else "Medium" if probability>=medium else "Low"
            output.append({"recurrence_probability":float(probability),"risk_score":round(float(probability)*100,2),
              "risk_level":level,"predicted_recurrence":bool(probability>=self.decision_threshold),
              "decision_threshold":self.decision_threshold,"model_name":self.model_name})
        return output
