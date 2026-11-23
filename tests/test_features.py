import pandas as pd
import pytest
from geopulse.features.panel import build_panel

def test_panel_uses_only_prior_counts_and_future_labels():
    dates=pd.date_range("2024-01-01",periods=40,freq="D")
    frame=pd.DataFrame({"created_date":[dates[0],dates[3],dates[39]],"latitude":[40.7]*3,"longitude":[-73.9]*3})
    panel=build_panel(frame,cell_size_m=500,horizon_days=5).sort_values("created_date")
    first=panel.iloc[0]
    assert first.created_date==dates[0]
    assert first.count_7d==0 and first.target==1
    assert panel.loc[panel.created_date.eq(dates[3]),"count_7d"].iloc[0]==1
    assert (panel.neighbor_count_30d>=0).all()

def test_panel_rejects_empty_input():
    with pytest.raises(ValueError):
        build_panel(pd.DataFrame(columns=["created_date","latitude","longitude"]))
