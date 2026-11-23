"""Train GeoPulse from a local CSV or a bounded NYC 311 API download."""
import argparse
from datetime import date
from pathlib import Path
import pandas as pd
from geopulse.data.nyc311 import download_nyc311,validate_complaints
from geopulse.features.panel import build_panel
from geopulse.models.train import train_models

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",help="Existing raw CSV; if omitted, download from NYC Open Data")
    p.add_argument("--start",default="2023-01-01"); p.add_argument("--end",default="2024-01-01")
    p.add_argument("--limit",type=int,default=100000); p.add_argument("--cell-size-m",type=int,default=500)
    p.add_argument("--horizon-days",type=int,default=30); a=p.parse_args()
    path=a.input or download_nyc311("data/raw/nyc311.csv",date.fromisoformat(a.start),date.fromisoformat(a.end),a.limit)
    clean=validate_complaints(pd.read_csv(path,low_memory=False))
    panel=build_panel(clean,a.cell_size_m,a.horizon_days)
    processed=Path("data/processed"); processed.mkdir(parents=True,exist_ok=True)
    panel.to_csv(processed/"cell_day_panel.csv",index=False)
    print(train_models(panel))
if __name__=="__main__": main()
