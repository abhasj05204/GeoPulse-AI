# Deployment

## Render Blueprint

The repository includes `render.yaml` with two lightweight Python web services:

1. `geopulse-api`: FastAPI backend. On first startup it can acquire a bounded six-month NYC 311 sample, filter configured infrastructure categories, train the model, save artifacts, and start Uvicorn.
2. `geopulse-dashboard`: Streamlit analytical dashboard. It obtains the API's internal host/port from the Render service reference.

### Deploy

1. Push the repository to GitHub.
2. In Render, create a new Blueprint and connect this repository.
3. Review the two services and their free-plan limits, then apply the Blueprint.
4. Optionally add `SOCRATA_APP_TOKEN` to the API service to improve Socrata request limits.
5. Add `GEMINI_API_KEY` to the API service to enable the AI analyst. ML and spatial APIs work without this key.
6. Wait for the API's initial data acquisition/training startup to complete, then open the dashboard service URL.

The API health endpoint is `/api/health`; Swagger documentation is `/docs`. The dashboard health endpoint is `/_stcore/health`.

## Training on startup

The Render API service is configured with `GEOPULSE_TRAIN_ON_START=true`, a 2025-01-01 through 2025-07-01 date window, a 30,000-record cap, and 1,000 m cells. This is a bounded first-deploy configuration, not a validated production training configuration. Startup fails visibly if the public source is unavailable, configured infrastructure categories do not match the source, or the sample is insufficient.

For a local/manual run, set `GEOPULSE_TRAIN_ON_START=false` and use `python scripts/train.py`. Optuna tuning can be enabled with `--tune --trials 15`; avoid tuning during every web-service restart.

## Storage and service limits

The free Render plan uses an ephemeral filesystem. Model and processed-data artifacts may need to be regenerated after a fresh instance or redeploy. For persistent artifacts, upgrade the API service and attach a persistent disk, then set `GEOPULSE_STORAGE_DIR` to the disk mount path (for example `/var/data`). The API stores raw data, processed panels, and model artifacts beneath that directory. The dashboard does not need direct filesystem access; it reads results through FastAPI.

Free services may sleep when idle and cold-start latency is expected. The configured training run can exceed typical interactive startup time; if that occurs, disable train-on-start and run training as a separate one-off job or use a persistent paid service. Actual service URLs are assigned by Render and are not pre-generated in this repository.

## Secrets

Never commit `.env`, API keys, or provider tokens. Configure `GEMINI_API_KEY` and optional `SOCRATA_APP_TOKEN` through Render environment settings. Only the API service needs the Gemini key.
