# GeoPulse AI
### Predict. Prioritize. Prevent.

GeoPulse AI is a focused geospatial machine-learning project for identifying spatial complaint hotspots, estimating 30-day complaint recurrence risk, exposing model outputs through FastAPI, and exploring results in a compact Streamlit dashboard.

## Strict project scope

- **ML pipelines:** public complaint data ingestion, validation, projected spatial aggregation, leakage-aware temporal features, recurrence modelling, evaluation, and permutation-based explainability.
- **Backend API:** FastAPI endpoints for inference, hotspot clustering, model reports, and optional evidence-grounded GenAI summaries.
- **Analytical dashboard:** Streamlit interface for model metrics, feature importance, CSV-based hotspot exploration, predictions, and AI summaries.

No marketing site, authentication, profiles, subscriptions, or unrelated CRUD application is included. GenAI is an optional explanatory layer and does not replace the ML model.

## Architecture

See [architecture](docs/architecture.md), [methodology](docs/methodology.md), [dataset notes](docs/datasets.md), and [API reference](docs/api.md).

## Local setup

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Copy `.env.example` to `.env`. A Socrata app token is optional for small downloads; a Gemini API key is needed only for the AI analyst endpoint. Never commit secrets.

## Acquire data and train

Download a bounded NYC 311 date range and train the initial benchmark:

```bash
python scripts/train.py --start 2023-01-01 --end 2024-01-01 --limit 100000
```

Or provide a previously downloaded CSV:

```bash
python scripts/train.py --input data/raw/nyc311.csv
```

The pipeline validates source columns, projects NYC coordinates to EPSG:32618, aggregates records to configurable 500 m cells, builds lagged 7/30/90-day features, labels whether a complaint occurs in the next 30 days, and compares logistic regression, random forest, and histogram gradient boosting on a chronological holdout. It writes model and evaluation artifacts locally under `artifacts/`; raw data and model binaries are excluded from Git.

## Run the API and dashboard

Terminal 1:

```bash
uvicorn backend.main:app --reload
```

Terminal 2:

```streamlit run dashboard/app.py
```

Set `GEOPULSE_API_URL` when the API is not running at `http://127.0.0.1:8000`. FastAPI Swagger docs: `/docs`.

## Tests

```bash
pytest
```

GitHub Actions runs the Python test suite on pushes and pull requests to `main` and `develop`.

## Interpretation and limitations

The initial projection is NYC-specific. Complaint recurrence is not infrastructure severity, and risk bands are provisional display categories. The initial temporal holdout does not fully remove spatial dependence between nearby cells. Model metrics must be generated from a real, documented data run before being used in a portfolio or operational claim. OSM enrichment, spatially blocked validation, calibrated thresholds, and deployment remain future work.
