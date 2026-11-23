"""Optuna tuning for histogram gradient boosting using chronological folds."""
from pathlib import Path
import json
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from geopulse.models.validation import temporal_cv_scores

def optimize_hist_gradient_boosting(panel,n_trials:int=15,output_dir:str|Path="artifacts")->dict:
    if n_trials<1: raise ValueError("n_trials must be positive")
    dates=panel.created_date
    cutoff=dates.quantile(.8)
    training=panel[dates<cutoff].copy()
    if training.empty or training.target.nunique()<2: raise ValueError("Insufficient training data for optimization")
    def objective(trial):
        params={
          "max_iter":trial.suggest_int("max_iter",80,240),
          "learning_rate":trial.suggest_float("learning_rate",.025,.18,log=True),
          "max_leaf_nodes":trial.suggest_int("max_leaf_nodes",15,63),
          "l2_regularization":trial.suggest_float("l2_regularization",1e-4,10,log=True),
          "random_state":42}
        estimator=Pipeline([("impute",SimpleImputer(strategy="median")),("model",HistGradientBoostingClassifier(**params))])
        scores=temporal_cv_scores(estimator,training,n_splits=3)
        return sum(scores)/len(scores) if scores else 0.0
    study=optuna.create_study(direction="maximize")
    study.optimize(objective,n_trials=n_trials,n_jobs=1)
    result={"best_params":{**study.best_params,"random_state":42},"best_cv_pr_auc":float(study.best_value),"trials":n_trials}
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"optimization.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result
