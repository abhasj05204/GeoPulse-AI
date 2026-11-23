import pandas as pd
from geopulse.features.panel import build_panel

def test_panel_uses_lagged_counts_and_future_label():
    dates=pd.date_range("2024-01-01",periods=40,freq="D")
    frame=pd.DataFrame({"created_date":[dates[0],dates[10],dates[39]],"latitude":[40.7]*3,"longitude":[-73.9]*3})
    panel=build_panel(frame,cell_size_m=500,horizon_days=5)
    assert not panel.empty
    first=panel.sort_values("created_date").iloc[0]
    assert first.count_7d==0
    assert first.target==1
