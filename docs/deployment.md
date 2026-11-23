# Deployment

## Render Blueprint

The repository includes `render.yaml` with two lightweight Python web services:

1. `geopulse-api`: FastAPI backend. On first startup it can acquire a bounded six-month NYC 311 sample, filter configured infrastructure categories, train the model, save artifacts, and start Uvicorn.
2. `geopulse-dashboard`: Streamlit analytical dashboard. Its API URL is resolved from the API service's public `RENDER_EXTERNAL_URL`, so the two Free web services do not depend on private-network ingress.

### Deploy

1. Push the repository to GitHub.
2. In Render, create a new Blueprint and connect this repository.
3. Review the two services and their free-plan limitations, then apply the Blueprint.
4. Optionally add `SOCRATA_APP_TOKEN` to the API service to improve Socrata request limits.
5. Add `GEMINI_API_KEY` to the API service to enable the AI analyst. ML and spatial APIs work without this key.
6. Wait for the API's initial data acquisition/training startup to complete, then open the dashboard service URL.

The API health endpoint is `/api/health`; Swagger documentation is `/docs`. The dashboard health endpoint is `/_stcore/health`.

## Training on startup

The Render API service is configured with `GEOPULSE_TRAIN_ON_START=true`, a 2025-01-01 through 2025-07-01 date window, a 30,000-record cap, and 1,000 m cells. This is a bounded first-deploy configuration, not a validated production training configuration. Startup fails visibly if the public source is unavailable, configured infrastructure categories do not match the source, or the sample is insufficient.

For a local/manual run, set `GEOPULSE_TRAIN_ON_START=false` and use `python scripts/train.py`. Optuna tuning can be enabled with `--tune --trials 15`; avoid tuning during every web-service restart.

## Storage and service limits

Render Free web services spin down after 15 minutes without traffic and can take about a minute to spin back up. Their filesystem is ephemeral: files are lost on restart, redeploy, or spin-down. The configured API therefore retrains from the bounded public sample when the model artifact is missing. This increases cold-start time and external API traffic. Free services cannot attach persistent disks and should be treated as a demo environment, not production hosting.

For persistent artifacts, upgrade the API service to a paid compute plan, attach a persistent disk, and set `GEOPULSE_STORAGE_DIR` to its mount path (for example `/var/data`). The API stores raw data, processed panels, and model artifacts beneath that directory. The dashboard does not need direct filesystem access; it reads results through FastAPI. A disk is accessible only to its attached service.

Actual service URLs are assigned by Render and are not pre-generated in this repository. The dashboard uses the API's public service URL through the Blueprint reference.

## Secrets

Never commit `.env`, API keys, or provider tokens. Configure `GEMINI_API_KEY` and optional `SOCRATA_APP_TOKEN` through Render environment settings. Only the API service needs the Gemini key.
