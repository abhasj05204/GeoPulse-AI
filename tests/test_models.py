import numpy as np
import pandas as pd
import joblib
from geopulse.models.train import train_models
from geopulse.models.inference import RecurrencePredictor
from geopulse.features.panel import FEATURES

def test_train_save_and_infer(tmp_path):
    n=120
    frame=pd.DataFrame({"created_date":pd.date_range("2024-01-01",periods=n,freq="D"),
      "target":(np.arange(n)%3==0).astype(int)})
    rng=np.random.default_rng(42)
    for name in FEATURES: frame[name]=rng.normal(size=n)
    frame["count_30d"]=frame["target"]*5+rng.normal(size=n)
    report=train_models(frame,output_dir=tmp_path)
    assert report["best_model"] in {"logistic_regression","random_forest","hist_gradient_boosting"}
    assert report["selection_metric"]=="mean temporal cross-validation PR-AUC"
    assert report["baseline"]["pr_auc"]>=0
    artifact=joblib.load(tmp_path/"recurrence_model.joblib")
    assert artifact["risk_thresholds"]["high"]>artifact["risk_thresholds"]["medium"]
    assert (tmp_path/"evaluation.json").exists() and (tmp_path/"feature_importance.json").exists()
    predictor=RecurrencePredictor(tmp_path/"recurrence_model.joblib")
    result=predictor.predict([{name:float(frame.iloc[-1][name]) for name in FEATURES}])[0]
    assert 0<=result["recurrence_probability"]<=1
    assert result["risk_level"] in {"Low","Medium","High"}
    assert isinstance(result["predicted_recurrence"],bool)
