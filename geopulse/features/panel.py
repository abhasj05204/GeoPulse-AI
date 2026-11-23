"""Build cell/day rows with leakage-safe temporal and local spatial features."""
import numpy as np
import pandas as pd
from pyproj import Transformer
FEATURES=["count_7d","count_30d","count_90d","mean_daily_30d","days_since_last","neighbor_count_30d","day_of_week","month","x_km","y_km"]

def build_panel(complaints:pd.DataFrame,cell_size_m:int=500,horizon_days:int=30)->pd.DataFrame:
    if cell_size_m<=0 or horizon_days<=0: raise ValueError("Cell size and horizon must be positive")
    df=complaints.copy(); required={"created_date","latitude","longitude"}
    if not required.issubset(df): raise ValueError(f"Missing columns: {sorted(required-set(df))}")
    if df.empty: raise ValueError("No valid complaint records were supplied")
    parsed=pd.to_datetime(df.created_date,errors="coerce")
    if parsed.dt.tz is None: parsed=parsed.dt.tz_localize("America/New_York")
    else: parsed=parsed.dt.tz_convert("America/New_York")
    df["created_date"]=parsed.dt.floor("D").dt.tz_localize(None)
    tx=Transformer.from_crs("EPSG:4326","EPSG:32618",always_xy=True)
    x,y=tx.transform(df.longitude.to_numpy(),df.latitude.to_numpy())
    df["cx"]=np.floor(np.asarray(x)/cell_size_m).astype(int); df["cy"]=np.floor(np.asarray(y)/cell_size_m).astype(int)
    daily=df.groupby(["cx","cy","created_date"]).size().rename("count").reset_index()
    dates=pd.date_range(daily.created_date.min(),daily.created_date.max(),freq="D")
    cells=daily[["cx","cy"]].drop_duplicates()
    grid=cells.assign(_k=1).merge(pd.DataFrame({"created_date":dates,"_k":1}),on="_k").drop(columns="_k")
    panel=grid.merge(daily,on=["cx","cy","created_date"],how="left").fillna({"count":0})
    panel=panel.sort_values(["cx","cy","created_date"]).reset_index(drop=True)
    grouped=panel.groupby(["cx","cy"],sort=False)["count"]; prior=grouped.shift(1).fillna(0)
    for window in (7,30,90):
        panel[f"count_{window}d"]=prior.groupby([panel.cx,panel.cy]).transform(lambda s:s.rolling(window,min_periods=1).sum())
    panel["mean_daily_30d"]=panel.count_30d/30
    last=panel.created_date.where(panel["count"].gt(0)).groupby([panel.cx,panel.cy]).ffill()
    last=last.groupby([panel.cx,panel.cy]).shift(1)
    panel["days_since_last"]=(panel.created_date-last).dt.days.fillna(999).clip(upper=999)
    # Sum the prior-30-day counts in the eight immediately adjacent cells.
    keys=panel[["cx","cy","created_date"]]
    base=panel[["cx","cy","created_date","count_30d"]]
    nearby=np.zeros(len(panel),dtype=float)
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            if dx==0 and dy==0: continue
            neighbor=base.copy()
            neighbor["cx"]=neighbor.cx-dx; neighbor["cy"]=neighbor.cy-dy
            neighbor=neighbor.rename(columns={"count_30d":"neighbor_count"})
            aligned=keys.merge(neighbor,on=["cx","cy","created_date"],how="left")["neighbor_count"].fillna(0)
            nearby+=aligned.to_numpy()
    panel["neighbor_count_30d"]=nearby
    panel["day_of_week"]=panel.created_date.dt.dayofweek; panel["month"]=panel.created_date.dt.month
    panel["x_km"]=(panel.cx+.5)*cell_size_m/1000; panel["y_km"]=(panel.cy+.5)*cell_size_m/1000
    future=panel.groupby(["cx","cy"],sort=False)["count"].transform(lambda s:s.shift(-1).iloc[::-1].rolling(horizon_days,min_periods=1).sum().iloc[::-1])
    panel["target"]=future.gt(0).astype("int8")
    remaining=panel.groupby(["cx","cy"]).cumcount(ascending=False)
    return panel[remaining>=horizon_days].dropna(subset=FEATURES).reset_index(drop=True)
