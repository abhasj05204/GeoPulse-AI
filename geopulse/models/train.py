"""Chronological benchmark with CV-only selection and validation-derived thresholds."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score,precision_score,recall_score,f1_score,brier_score_loss,roc_auc_score,precision_recall_curve
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from geopulse.config import ARTIFACTS
from geopulse.features.panel import FEATURES
from geopulse.models.validation import temporal_cv_scores,temporal_oof_predictions

def train_models(panel:pd.DataFrame,output_dir:str|Path=ARTIFACTS,tuned_params:dict|None=None)->dict:
    dates=pd.to_datetime(panel.created_date); cutoff=dates.quantile(.8)
    train,test=panel[dates<cutoff],panel[dates>=cutoff]
    if train.empty or test.empty or train.target.nunique()<2: raise ValueError("Insufficient temporal data or target classes")
    Xtr,ytr=train[FEATURES],train.target.astype(int); Xte,yte=test[FEATURES],test.target.astype(int)
    hgb_params={"max_iter":150,"learning_rate":.08,"random_state":42,**(tuned_params or {})}
    candidates={
      "logistic_regression":Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LogisticRegression(max_iter=1000,class_weight="balanced"))]),
      "random_forest":Pipeline([("impute",SimpleImputer(strategy="median")),("model",RandomForestClassifier(n_estimators=200,min_samples_leaf=3,class_weight="balanced",random_state=42,n_jobs=-1))]),
      "hist_gradient_boosting":Pipeline([("impute",SimpleImputer(strategy="median")),("model",HistGradientBoostingClassifier(**hgb_params))])}
    reports={}; fitted={}
    for name,model in candidates.items():
        cv=temporal_cv_scores(model,train,n_splits=3)
        model.fit(Xtr,ytr); prob=model.predict_proba(Xte)[:,1]; pred=prob>=.5
        metrics={"precision_at_0_5":float(precision_score(yte,pred,zero_division=0)),
          "recall_at_0_5":float(recall_score(yte,pred,zero_division=0)),"f1_at_0_5":float(f1_score(yte,pred,zero_division=0)),
          "pr_auc":float(average_precision_score(yte,prob)),"brier_score":float(brier_score_loss(yte,prob)),
          "cv_pr_auc_mean":float(np.mean(cv)) if cv else None,"cv_pr_auc_std":float(np.std(cv)) if cv else None,"cv_folds":len(cv)}
        if yte.nunique()==2: metrics["roc_auc"]=float(roc_auc_score(yte,prob))
        reports[name]=metrics; fitted[name]=model
    valid_models=[name for name,m in reports.items() if m["cv_pr_auc_mean"] is not None]
    if not valid_models: raise ValueError("Temporal CV produced no valid folds; increase the date range or sample size")
    # Select using training-period CV only; the final holdout remains untouched for evaluation.
    best=max(valid_models,key=lambda name:reports[name]["cv_pr_auc_mean"])
    selected=candidates[best]
    oof_y,oof_prob=temporal_oof_predictions(selected,train,n_splits=3)
    if len(oof_y)==0: raise ValueError("No out-of-fold predictions available for threshold selection")
    precision,recall,thresholds=precision_recall_curve(oof_y,oof_prob)
    f1_values=2*precision[:-1]*recall[:-1]/(precision[:-1]+recall[:-1]+1e-12)
    decision_threshold=float(thresholds[int(np.argmax(f1_values))]) if len(thresholds) else .5
    medium,high=np.quantile(oof_prob,[.5,.8]).tolist()
    if high<=medium: high=float(np.nextafter(medium,1.0))
    risk_thresholds={"medium":float(medium),"high":float(high),"method":"50th/80th percentile of temporal out-of-fold probabilities"}
    probability=selected.predict_proba(Xte)[:,1]; predicted=probability>=decision_threshold
    operating={"decision_threshold":decision_threshold,"precision":float(precision_score(yte,predicted,zero_division=0)),
      "recall":float(recall_score(yte,predicted,zero_division=0)),"f1":float(f1_score(yte,predicted,zero_division=0))}
    baseline_model=DummyClassifier(strategy="prior").fit(Xtr,ytr)
    baseline_prob=baseline_model.predict_proba(Xte)[:,1]
    baseline={"positive_rate_train":float(ytr.mean()),"positive_rate_test":float(yte.mean()),
      "pr_auc":float(average_precision_score(yte,baseline_prob)),"brier_score":float(brier_score_loss(yte,baseline_prob))}
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    joblib.dump({"model":selected,"model_name":best,"feature_columns":FEATURES,"cutoff":str(cutoff),
      "decision_threshold":decision_threshold,"risk_thresholds":risk_thresholds},out/"recurrence_model.joblib")
    perm=permutation_importance(selected,Xte,yte,n_repeats=5,random_state=42,scoring="average_precision",n_jobs=-1)
    importance=sorted([{"feature":name,"importance_mean":float(mean),"importance_std":float(std)} for name,mean,std in zip(FEATURES,perm.importances_mean,perm.importances_std)],key=lambda row:row["importance_mean"],reverse=True)
    (out/"feature_importance.json").write_text(json.dumps({"method":"permutation_importance","scoring":"average_precision","features":importance},indent=2),encoding="utf-8")
    report={"best_model":best,"selection_metric":"mean temporal cross-validation PR-AUC","temporal_cutoff":str(cutoff),
      "train_rows":len(train),"test_rows":len(test),"baseline":baseline,"metrics":reports,
      "selected_model_operating_metrics":operating,"risk_thresholds":risk_thresholds}
    (out/"evaluation.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); return report
