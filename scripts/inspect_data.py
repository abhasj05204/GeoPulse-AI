"""Print a compact source-quality and complaint-category audit for a raw CSV."""
import argparse,json
from pathlib import Path
import pandas as pd
from geopulse.data.nyc311 import validate_complaints

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("input"); parser.add_argument("--output",default="data/processed/source_audit.json")
    args=parser.parse_args(); raw=pd.read_csv(args.input,low_memory=False); clean=validate_complaints(raw)
    report={"raw_rows":int(len(raw)),"valid_rows":int(len(clean)),"dropped_rows":int(len(raw)-len(clean)),
      "unique_complaint_types":int(clean.complaint_type.nunique()),"date_min":str(clean.created_date.min()),
      "date_max":str(clean.created_date.max()),"latitude_coverage":float(clean.latitude.notna().mean()),
      "longitude_coverage":float(clean.longitude.notna().mean()),"top_complaint_types":clean.complaint_type.value_counts().head(50).to_dict()}
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
