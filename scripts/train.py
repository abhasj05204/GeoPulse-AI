"""Train GeoPulse from a local CSV or stratified NYC 311 download."""
import argparse
from datetime import date
from pathlib import Path
import pandas as pd
from geopulse.config import DATA_PROCESSED
from geopulse.data.nyc311 import download_nyc311,validate_complaints,filter_infrastructure_complaints
from geopulse.features.panel import build_panel
from geopulse.models.train import train_models
from geopulse.models.optimize import optimize_hist_gradient_boosting

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",help="Existing raw CSV; otherwise download from NYC Open Data")
    p.add_argument("--start",default="2023-01-01"); p.add_argument("--end",default="2024-01-01")
    p.add_argument("--limit",type=int,default=100000); p.add_argument("--cell-size-m",type=int,default=500)
    p.add_argument("--horizon-days",type=int,default=30); p.add_argument("--complaint-types",help="Comma-separated source complaint_type values")
    p.add_argument("--tune",action="store_true"); p.add_argument("--trials",type=int,default=15); a=p.parse_args()
    path=a.input or download_nyc311("data/raw/nyc311.csv",date.fromisoformat(a.start),date.fromisoformat(a.end),a.limit)
    clean=validate_complaints(pd.read_csv(path,low_memory=False))
    categories=[x.strip() for x in a.complaint_types.split(",")] if a.complaint_types else None
    clean=filter_infrastructure_complaints(clean,categories)
    panel=build_panel(clean,a.cell_size_m,a.horizon_days)
    DATA_PROCESSED.mkdir(parents=True,exist_ok=True)
    panel.to_csv(DATA_PROCESSED/"cell_day_panel.csv",index=False)
    tuned=optimize_hist_gradient_boosting(panel,a.trials)["best_params"] if a.tune else None
    print(train_models(panel,tuned_params=tuned))
if __name__=="__main__": main()
