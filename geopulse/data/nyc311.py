"""Monthly-stratified NYC 311 Socrata ingestion and source validation."""
import math,os
from datetime import date
from pathlib import Path
import pandas as pd
import requests

REQUIRED={"created_date","complaint_type","latitude","longitude"}
DEFAULT_INFRA_TYPES=["Street Condition","Street Light Condition","Sidewalk Condition","Sewer","Water System","Highway Condition"]

def download_nyc311(output:str|Path,start:date,end:date,limit:int=100000)->Path:
    """Download a bounded sample spread across months and within each month."""
    if start>=end or limit<1: raise ValueError("Require start < end and a positive limit")
    base=os.getenv("NYC311_API_BASE","https://data.cityofnewyork.us").rstrip("/")
    dataset=os.getenv("NYC311_DATASET_ID","erm2-nwe9")
    headers={"X-App-Token":os.environ["SOCRATA_APP_TOKEN"]} if os.getenv("SOCRATA_APP_TOKEN") else {}
    first=pd.Timestamp(start).replace(day=1); last=pd.Timestamp(end).replace(day=1)
    months=pd.date_range(first,last,freq="MS")
    budget=max(1,math.ceil(limit/len(months))); rows=[]
    for month in months:
        month_start=max(start,month.date())
        month_end=min(end,(month+pd.offsets.MonthBegin(1)).date())
        if month_start>=month_end: continue
        where=f"created_date >= '{month_start}T00:00:00' AND created_date < '{month_end}T00:00:00' AND latitude IS NOT NULL AND longitude IS NOT NULL"
        count_response=requests.get(f"{base}/resource/{dataset}.json",headers=headers,timeout=60,
            params={"$select":"count(*) as n","$where":where,"$limit":1})
        count_response.raise_for_status(); count=int(count_response.json()[0]["n"])
        if count==0: continue
        samples=min(3,budget,count); chunk=max(1,math.ceil(min(budget,count)/samples))
        if samples==1: offsets=[0]
        else: offsets=sorted({round(i*(count-chunk)/(samples-1)) for i in range(samples)})
        for offset in offsets:
            response=requests.get(f"{base}/resource/{dataset}.json",headers=headers,timeout=60,
              params={"$select":"unique_key,created_date,complaint_type,latitude,longitude",
                "$where":where,"$limit":chunk,"$offset":offset,"$order":"created_date ASC"})
            response.raise_for_status(); rows.extend(response.json())
    frame=pd.DataFrame(rows)
    if "unique_key" in frame: frame=frame.drop_duplicates("unique_key")
    frame=frame.head(limit)
    path=Path(output); path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False); return path

def validate_complaints(frame:pd.DataFrame)->pd.DataFrame:
    missing=REQUIRED-set(frame.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    df=frame.copy(); parsed=pd.to_datetime(df.created_date,errors="coerce")
    if parsed.dt.tz is None: parsed=parsed.dt.tz_localize("America/New_York")
    else: parsed=parsed.dt.tz_convert("America/New_York")
    df["created_date"]=parsed.dt.tz_convert("UTC")
    for col in ("latitude","longitude"): df[col]=pd.to_numeric(df[col],errors="coerce")
    df["complaint_type"]=df.complaint_type.astype("string").str.strip()
    df=df.dropna(subset=["created_date","latitude","longitude","complaint_type"])
    df=df[df.latitude.between(-90,90)&df.longitude.between(-180,180)]
    if "unique_key" in df: df=df.drop_duplicates("unique_key")
    return df.sort_values("created_date").reset_index(drop=True)

def filter_infrastructure_complaints(frame:pd.DataFrame,complaint_types:list[str]|None=None)->pd.DataFrame:
    """Restrict the modelling cohort to explicit infrastructure complaint classes."""
    configured=os.getenv("GEOPULSE_INFRA_TYPES")
    selected=complaint_types or ([x.strip() for x in configured.split(",") if x.strip()] if configured else DEFAULT_INFRA_TYPES)
    wanted={value.casefold() for value in selected}
    filtered=frame[frame.complaint_type.astype("string").str.casefold().isin(wanted)].copy()
    if filtered.empty:
        available=frame.complaint_type.value_counts().head(20).to_dict()
        raise ValueError(f"No records matched infrastructure categories {selected}. Inspect source categories first. Top available: {available}")
    return filtered.reset_index(drop=True)
