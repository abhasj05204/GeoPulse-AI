"""Date-blocked cross-validation for panel data with shared calendar dates."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score
from geopulse.features.panel import FEATURES

def _folds(frame:pd.DataFrame,n_splits:int):
    dates=pd.to_datetime(frame["created_date"]); unique_dates=np.sort(dates.unique())
    if len(unique_dates)<n_splits+2: return
    for train_idx,val_idx in TimeSeriesSplit(n_splits=n_splits).split(unique_dates):
        train_dates=unique_dates[train_idx]; val_dates=unique_dates[val_idx]
        train=frame[dates.isin(train_dates)]; valid=frame[dates.isin(val_dates)]
        if not train.empty and not valid.empty and train.target.nunique()>=2:
            yield train,valid

def temporal_cv_scores(estimator,frame:pd.DataFrame,n_splits:int=3)->list[float]:
    scores=[]
    for train,valid in _folds(frame,n_splits):
        model=clone(estimator); model.fit(train[FEATURES],train.target.astype(int))
        probability=model.predict_proba(valid[FEATURES])[:,1]
        scores.append(float(average_precision_score(valid.target.astype(int),probability)))
    return scores

def temporal_oof_predictions(estimator,frame:pd.DataFrame,n_splits:int=3)->tuple[np.ndarray,np.ndarray]:
    """Return out-of-fold labels and probabilities from expanding chronological folds."""
    labels=[]; probabilities=[]
    for train,valid in _folds(frame,n_splits):
        model=clone(estimator); model.fit(train[FEATURES],train.target.astype(int))
        labels.extend(valid.target.astype(int).tolist())
        probabilities.extend(model.predict_proba(valid[FEATURES])[:,1].tolist())
    return np.asarray(labels,dtype=int),np.asarray(probabilities,dtype=float)
