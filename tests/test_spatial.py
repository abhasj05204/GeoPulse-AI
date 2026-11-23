import pandas as pd
from geopulse.spatial.hotspots import detect_hotspots

def test_dbscan_labels_nearby_points():
    df=pd.DataFrame({"latitude":[40.7,40.7001,40.7002,40.8],"longitude":[-73.9,-73.9001,-73.9002,-73.8]})
    out=detect_hotspots(df,eps_m=100,min_samples=2)
    assert out.cluster.iloc[0]==out.cluster.iloc[1]
    assert out.cluster.iloc[3]==-1
