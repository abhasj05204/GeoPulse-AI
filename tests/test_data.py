import pandas as pd
import pytest
from datetime import date
from geopulse.data.nyc311 import validate_complaints,filter_infrastructure_complaints,download_nyc311

def test_validation_removes_invalid_rows():
    df=pd.DataFrame({"created_date":["2024-01-01","bad"],"complaint_type":["Street Condition","Noise"],
      "latitude":["40.7","999"],"longitude":["-73.9","-73.9"]})
    out=validate_complaints(df); assert len(out)==1

def test_validation_requires_columns():
    with pytest.raises(ValueError): validate_complaints(pd.DataFrame({"latitude":[1]}))

def test_infrastructure_filter_is_explicit():
    df=pd.DataFrame({"complaint_type":["Street Condition","Noise - Residential","Sewer"]})
    out=filter_infrastructure_complaints(df,["Street Condition","Sewer"])
    assert set(out.complaint_type)=={"Street Condition","Sewer"}

def test_infrastructure_filter_reports_unmatched_categories():
    with pytest.raises(ValueError): filter_infrastructure_complaints(pd.DataFrame({"complaint_type":["Noise"]}),["Street Condition"])

def test_downloader_samples_multiple_time_slices(tmp_path,monkeypatch):
    calls=[]
    class Response:
        def __init__(self,payload): self.payload=payload
        def raise_for_status(self): pass
        def json(self): return self.payload
    counter={"id":0}
    def fake_get(url,**kwargs):
        params=kwargs["params"]; calls.append(params); size=int(params["$limit"])
        rows=[]
        for _ in range(size):
            counter["id"]+=1
            rows.append({"unique_key":str(counter["id"]),"created_date":"2024-01-01","complaint_type":"Street Condition","latitude":"40.7","longitude":"-73.9"})
        return Response(rows)
    monkeypatch.setattr("geopulse.data.nyc311.requests.get",fake_get)
    path=download_nyc311(tmp_path/"raw.csv",date(2024,1,1),date(2024,2,1),limit=6)
    assert len(pd.read_csv(path))==6
    assert len(calls)==3
    assert len({c["$where"] for c in calls})==3
    assert all("$offset" not in c for c in calls)
