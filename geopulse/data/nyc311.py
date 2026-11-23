"""Bounded NYC 311 Socrata ingestion and source validation."""
import os
from datetime import date
from pathlib import Path
import pandas as pd
import requests

REQUIRED = {"created_date", "complaint_type", "latitude", "longitude"}


def download_nyc311(output: str | Path, start: date, end: date, limit: int = 100_000) -> Path:
    if start >= end or limit < 1:
        raise ValueError("Require start < end and a positive limit")
    base = os.getenv("NYC311_API_BASE", "https://data.cityofnewyork.us").rstrip("/")
    dataset = os.getenv("NYC311_DATASET_ID", "erm2-nwe9")
    headers = {"X-App-Token": os.environ["SOCRATA_APP_TOKEN"]} if os.getenv("SOCRATA_APP_TOKEN") else {}
    rows, offset, page_size = [], 0, 50_000
    while len(rows) < limit:
        size = min(page_size, limit - len(rows))
        response = requests.get(f"{base}/resource/{dataset}.json", headers=headers, timeout=60,
            params={"$select":"unique_key,created_date,complaint_type,latitude,longitude",
                    "$where":f"created_date >= '{start}T00:00:00' AND created_date < '{end}T00:00:00' AND latitude IS NOT NULL AND longitude IS NOT NULL",
                    "$limit":size,"$offset":offset,"$order":"created_date ASC"})
        response.raise_for_status()
        page = response.json()
        if not page: break
        rows.extend(page); offset += len(page)
        if len(page) < size: break
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def validate_complaints(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED - set(frame.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = frame.copy()
    df["created_date"] = pd.to_datetime(df.created_date, errors="coerce", utc=True)
    for col in ("latitude", "longitude"): df[col] = pd.to_numeric(df[col], errors="coerce")
    df["complaint_type"] = df.complaint_type.astype("string").str.strip()
    df = df.dropna(subset=["created_date","latitude","longitude","complaint_type"])
    df = df[df.latitude.between(-90,90) & df.longitude.between(-180,180)]
    if "unique_key" in df: df = df.drop_duplicates("unique_key")
    return df.sort_values("created_date").reset_index(drop=True)
