# API reference

Run locally with `uvicorn backend.main:app --reload`. Interactive OpenAPI docs are available at `/docs`.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Service and model-artifact status |
| GET | `/api/model` | Loaded model name and required feature contract |
| GET | `/api/performance` | Chronological holdout evaluation report |
| GET | `/api/explain` | Holdout permutation feature importance |
| POST | `/api/predict` | Recurrence probability for engineered cell features |
| POST | `/api/hotspots` | DBSCAN cluster labels for supplied coordinates |
| POST | `/api/ai/summary` | Gemini summary grounded in supplied structured evidence |

Prediction request body:

```json
{
  "features": {
    "count_7d": 2, "count_30d": 8, "count_90d": 18,
    "mean_daily_30d": 0.267, "days_since_last": 4,
    "day_of_week": 1, "month": 6, "x_km": 580, "y_km": 4500
  }
}
```

The feature example is illustrative only. Use the same cell size, CRS, and feature pipeline as model training. A 503 response indicates that a required model/report or external credential is unavailable; 422 indicates invalid input.
