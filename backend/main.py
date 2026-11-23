"""FastAPI endpoints for health, model metadata, and live recurrence inference."""
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from geopulse.config import FEATURE_COLUMNS,MODEL_PATH
from geopulse.models.inference import RecurrencePredictor

app=FastAPI(title="GeoPulse AI API",version="0.1.0",description="Geospatial complaint recurrence analytics")
class PredictionRequest(BaseModel):
    features: dict[str,float]=Field(...,description="Engineered features from the GeoPulse pipeline")
@app.get("/api/health")
def health(): return {"status":"ok","model_available":MODEL_PATH.exists()}
@app.get("/api/model")
def model_info():
    if not MODEL_PATH.exists(): raise HTTPException(503,"Train a model before requesting metadata")
    p=RecurrencePredictor(); return {"model_name":p.model_name,"features":p.features}
@app.post("/api/predict")
def predict(payload:PredictionRequest):
    try: return RecurrencePredictor().predict([payload.features])[0]
    except FileNotFoundError as exc: raise HTTPException(503,str(exc)) from exc
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc
