"""Date-blocked cross-validation for panel data with shared calendar dates."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score

def temporal_cv_scores(estimator,frame:pd.DataFrame,n_splits:int=3)->list[float]:
    dates=pd.to_datetime(frame["created_date"])
    unique_dates=np.sort(dates.unique())
    if len(unique_dates)<n_splits+2: return []
    scores=[]
    for train_idx,val_idx in TimeSeriesSplit(n_splits=n_splits).split(unique_dates):
        train_dates=unique_dates[train_idx]; val_dates=unique_dates[val_idx]
        train=frame[dates.isin(train_dates)]; valid=frame[dates.isin(val_dates)]
        if train.empty or valid.empty or train.target.nunique()<2: continue
        model=clone(estimator)
        model.fit(train[FEATURES],train.target.astype(int))
        probability=model.predict_proba(valid[FEATURES])[:,1]
        scores.append(float(average_precision_score(valid.target.astype(int),probability)))
    return scores

from geopulse.features.panel import FEATURES
