# API reference

Run locally with `uvicorn backend.main:app --reload`. Interactive OpenAPI documentation is available at `/docs`.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Service and model-artifact status |
| GET | `/api/model` | Loaded model name and feature contract |
| GET | `/api/performance` | Chronological holdout and temporal CV evaluation |
| GET | `/api/explain` | Holdout permutation feature importance |
| GET | `/api/analytics/summary` | Processed panel and spatial-cell summary |
| GET | `/api/risks` | Latest cell-level recurrence predictions and map coordinates |
| POST | `/api/predict` | Recurrence probability for engineered cell features |
| POST | `/api/hotspots` | DBSCAN clusters for supplied coordinates |
| POST | `/api/ai/ask` | Natural-language question grounded in current model and spatial evidence |
| POST | `/api/ai/summary` | Gemini summary grounded in caller-supplied evidence |

Prediction request example (values are illustrative):

```json
{"features":{"count_7d":2,"count_30d":8,"count_90d":18,"mean_daily_30d":0.267,"days_since_last":4,"neighbor_count_30d":12,"day_of_week":1,"month":6,"x_km":580,"y_km":4500}}
```

Use the same grid size, projection, and feature pipeline as model training. A 503 response indicates a missing artifact, data panel, or external credential; 422 indicates invalid input.
