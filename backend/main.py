"""FastAPI service for inference, risk mapping, hotspots, and grounded GenAI."""
import json
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from geopulse.config import MODEL_PATH,ARTIFACTS,DATA_PROCESSED
from geopulse.models.inference import RecurrencePredictor
from geopulse.spatial.hotspots import detect_hotspots
from geopulse.genai.analyst import summarize
app=FastAPI(title="GeoPulse AI API",version="0.4.2",description="Geospatial complaint recurrence analytics")
class PredictionRequest(BaseModel): features:dict[str,float]=Field(...)
class HotspotRequest(BaseModel): points:list[dict]=Field(...,min_length=2,max_length=10000); eps_m:float=500; min_samples:int=10
class AnalystRequest(BaseModel): question:str=Field(...,min_length=1,max_length=1000); evidence:dict=Field(...)
class AnalystQuestionRequest(BaseModel): question:str=Field(...,min_length=1,max_length=1000)
@app.get("/api/health")
def health(): return {"status":"ok","model_available":MODEL_PATH.exists()}
@app.get("/api/model")
def model_info():
    if not MODEL_PATH.exists(): raise HTTPException(503,"Train a model first")
    p=RecurrencePredictor(); return {"model_name":p.model_name,"features":p.features}
@app.get("/api/performance")
def performance():
    path=ARTIFACTS/"evaluation.json"
    if not path.exists(): raise HTTPException(503,"Evaluation report not found; train a model first")
    return json.loads(path.read_text(encoding="utf-8"))
@app.get("/api/explain")
def explain():
    path=ARTIFACTS/"feature_importance.json"
    if not path.exists(): raise HTTPException(503,"Feature importance report not found; train a model first")
    return json.loads(path.read_text(encoding="utf-8"))
@app.get("/api/analytics/summary")
def analytics_summary():
    import pandas as pd
    path=DATA_PROCESSED/"cell_day_panel.csv"
    if not path.exists(): raise HTTPException(503,"Processed panel not found; run the training pipeline first")
    df=pd.read_csv(path)
    return {"panel_rows":len(df),"spatial_cells":int(df[["cx","cy"]].drop_duplicates().shape[0]),
      "date_start":str(df.created_date.min()),"date_end":str(df.created_date.max()),"complaint_events":int(df["count"].sum())}
@app.get("/api/risks")
def latest_risks():
    import pandas as pd
    from pyproj import Transformer
    path=DATA_PROCESSED/"cell_day_panel.csv"
    if not path.exists(): raise HTTPException(503,"Processed panel not found; run the training pipeline first")
    if not MODEL_PATH.exists(): raise HTTPException(503,"Trained model artifact not found")
    df=pd.read_csv(path); df["created_date"]=pd.to_datetime(df.created_date)
    latest=df.sort_values("created_date").groupby(["cx","cy"],as_index=False).tail(1).copy()
    try: predictions=RecurrencePredictor().predict(latest.to_dict(orient="records"))
    except (ValueError,FileNotFoundError) as exc: raise HTTPException(422,str(exc)) from exc
    tx=Transformer.from_crs("EPSG:32618","EPSG:4326",always_xy=True)
    lon,lat=tx.transform(latest.x_km.to_numpy()*1000,latest.y_km.to_numpy()*1000)
    return {"as_of":str(latest.created_date.max()),"cells":[
      {"cx":int(row.cx),"cy":int(row.cy),"latitude":float(la),"longitude":float(lo),**pred}
      for (_,row),la,lo,pred in zip(latest.iterrows(),lat,lon,predictions)]}
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
    if len(str(payload.evidence))>12000: raise HTTPException(413,"Evidence context exceeds 12,000 characters")
    try: return {"summary":summarize(payload.question,payload.evidence)}
    except RuntimeError as exc: raise HTTPException(503,str(exc)) from exc
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc
@app.post("/api/ai/ask")
def ai_ask(payload:AnalystQuestionRequest):
    try:
        risk_data=latest_risks(); risks=risk_data["cells"]; report=performance(); importance=explain()
        counts={level:sum(1 for row in risks if row["risk_level"]==level) for level in ("High","Medium","Low")}
        top=sorted(risks,key=lambda row:row["recurrence_probability"],reverse=True)[:10]
        evidence={"as_of":risk_data["as_of"],"cell_count":len(risks),"risk_band_counts":counts,
          "top_10_recurrence_cells":top,"model_evaluation":report,"top_feature_importance":importance["features"][:8]}
        if len(str(evidence))>12000: evidence["top_10_recurrence_cells"]=top[:5]
        if len(str(evidence))>12000: raise HTTPException(413,"Generated evidence context exceeds 12,000 characters")
        return {"summary":summarize(payload.question,evidence),"evidence_used":evidence}
    except HTTPException: raise
    except RuntimeError as exc: raise HTTPException(503,str(exc)) from exc
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc
