"""FastAPI service for inference, hotspot clustering, model reports, and grounded GenAI."""
import json
from pathlib import Path
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from geopulse.config import FEATURE_COLUMNS,MODEL_PATH,ROOT
from geopulse.models.inference import RecurrencePredictor
from geopulse.spatial.hotspots import detect_hotspots
from geopulse.genai.analyst import summarize

app=FastAPI(title="GeoPulse AI API",version="0.2.0",description="Geospatial complaint recurrence analytics")
class PredictionRequest(BaseModel): features:dict[str,float]=Field(...)
class HotspotRequest(BaseModel): points:list[dict]=Field(...,min_length=2,max_length=10000); eps_m:float=500; min_samples:int=10
class AnalystRequest(BaseModel): question:str=Field(...,min_length=1,max_length=1000); evidence:dict=Field(...)
@app.get("/api/health")
def health(): return {"status":"ok","model_available":MODEL_PATH.exists()}
@app.get("/api/model")
def model_info():
    if not MODEL_PATH.exists(): raise HTTPException(503,"Train a model first")
    p=RecurrencePredictor(); return {"model_name":p.model_name,"features":p.features}
@app.get("/api/performance")
def performance():
    path=ROOT/"artifacts"/"evaluation.json"
    if not path.exists(): raise HTTPException(503,"Evaluation report not found; train a model first")
    return json.loads(path.read_text(encoding="utf-8"))
@app.get("/api/explain")
def explain():
    path=ROOT/"artifacts"/"feature_importance.json"
    if not path.exists(): raise HTTPException(503,"Feature importance report not found; train a model first")
    return json.loads(path.read_text(encoding="utf-8"))
@app.post("/api/predict")
def predict(payload:PredictionRequest):
    try: return RecurrencePredictor().predict([payload.features])[0]
    except FileNotFoundError as exc: raise HTTPException(503,str(exc)) from exc
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc
@app.post("/api/hotspots")
def hotspots(payload:HotspotRequest):
    try:
        import pandas as pd
        result=detect_hotspots(pd.DataFrame(payload.points),payload.eps_m,payload.min_samples)
        return {"points":result.to_dict(orient="records"),"cluster_count":int(result.loc[result.cluster>=0,"cluster"].nunique()),"noise_count":int((result.cluster==-1).sum())}
    except (ValueError,KeyError) as exc: raise HTTPException(422,str(exc)) from exc
@app.post("/api/ai/summary")
def ai_summary(payload:AnalystRequest):
    try: return {"summary":summarize(payload.question,payload.evidence)}
    except RuntimeError as exc: raise HTTPException(503,str(exc)) from exc
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc
