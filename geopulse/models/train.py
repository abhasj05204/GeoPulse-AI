"""Chronological benchmark, date-blocked CV, and permutation importance."""
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,precision_score,recall_score,f1_score,brier_score_loss,roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from geopulse.config import ARTIFACTS
from geopulse.features.panel import FEATURES
from geopulse.models.validation import temporal_cv_scores

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
        metrics={"precision":float(precision_score(yte,pred,zero_division=0)),"recall":float(recall_score(yte,pred,zero_division=0)),
          "f1":float(f1_score(yte,pred,zero_division=0)),"pr_auc":float(average_precision_score(yte,prob)),
          "brier_score":float(brier_score_loss(yte,prob)),"cv_pr_auc_mean":float(sum(cv)/len(cv)) if cv else None,
          "cv_pr_auc_std":float(pd.Series(cv).std(ddof=0)) if cv else None,"cv_folds":len(cv)}
        if yte.nunique()==2: metrics["roc_auc"]=float(roc_auc_score(yte,prob))
        reports[name]=metrics; fitted[name]=model
    best=max(reports,key=lambda name:reports[name]["pr_auc"]); selected=fitted[best]
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    joblib.dump({"model":selected,"model_name":best,"feature_columns":FEATURES,"cutoff":str(cutoff)},out/"recurrence_model.joblib")
    perm=permutation_importance(selected,Xte,yte,n_repeats=5,random_state=42,scoring="average_precision",n_jobs=-1)
    importance=sorted([{"feature":name,"importance_mean":float(mean),"importance_std":float(std)} for name,mean,std in zip(FEATURES,perm.importances_mean,perm.importances_std)],key=lambda row:row["importance_mean"],reverse=True)
    (out/"feature_importance.json").write_text(json.dumps({"method":"permutation_importance","scoring":"average_precision","features":importance},indent=2),encoding="utf-8")
    report={"best_model":best,"temporal_cutoff":str(cutoff),"train_rows":len(train),"test_rows":len(test),"metrics":reports}
    (out/"evaluation.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); return report
