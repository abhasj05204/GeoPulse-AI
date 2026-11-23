import pandas as pd
import pytest
from geopulse.data.nyc311 import validate_complaints

def test_validation_removes_invalid_rows():
    df=pd.DataFrame({"created_date":["2024-01-01","bad"],"complaint_type":["Road","Noise"],"latitude":["40.7","999"],"longitude":["-73.9","-73.9"]})
    out=validate_complaints(df); assert len(out)==1

def test_validation_requires_columns():
    with pytest.raises(ValueError): validate_complaints(pd.DataFrame({"latitude":[1]}))
