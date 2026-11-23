"""DBSCAN complaint hotspot clustering in a projected metric CRS."""
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.cluster import DBSCAN

def detect_hotspots(frame: pd.DataFrame, eps_m: float=500, min_samples: int=10) -> pd.DataFrame:
    if not {"latitude","longitude"}.issubset(frame): raise ValueError("latitude and longitude are required")
    if eps_m<=0 or min_samples<2: raise ValueError("eps_m must be positive and min_samples >= 2")
    tx=Transformer.from_crs("EPSG:4326","EPSG:32618",always_xy=True)
    x,y=tx.transform(frame.longitude.to_numpy(),frame.latitude.to_numpy())
    result=frame.copy(); result["cluster"]=DBSCAN(eps=eps_m,min_samples=min_samples).fit_predict(np.column_stack([x,y]))
    return result
