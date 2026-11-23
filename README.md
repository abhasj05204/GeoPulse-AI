# GeoPulse AI
### Predict. Prioritize. Prevent.

GeoPulse AI is a geospatial machine-learning project for analysing recurring urban infrastructure complaints, identifying spatial hotspots, estimating short-term recurrence risk, and presenting evidence-grounded analytical explanations.

## Project scope

This repository contains only three product components:

1. **Machine-learning pipelines** — reproducible data validation, geospatial feature engineering, hotspot analysis, recurrence modelling, evaluation, and explainability.
2. **Backend API** — a lightweight FastAPI service exposing trained models and spatial analytics.
3. **Analytical dashboard** — a focused Streamlit interface consuming the API and visualising real analytical outputs.

Generative AI is an explanatory layer grounded in computed project evidence. It does not replace the predictive model, autonomously operate systems, or provide official engineering diagnoses.

## Current implementation status

The repository is being initialized. Model performance, supported datasets, and deployment URLs will be reported only after implementation and verification.

## Planned workflow

Public civic complaint data → validation and cleaning → spatial aggregation and temporal feature engineering → hotspot analysis and recurrence model → evaluation and SHAP explanations → FastAPI → Streamlit dashboard.

The prediction target is intended to estimate whether a spatial cell will receive one or more new qualifying complaints in a defined future window. Features must be computed strictly from information available at prediction time. Chronological evaluation and leakage checks are required.

## Repository layout

```
geopulse/
  config.py
  data/
  features/
  spatial/
  models/
  explainability/
  genai/
backend/
dashboard/
tests/
configs/
docs/
scripts/
```

## Local development

Python 3.11 is the initial target. Install dependencies with:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Copy `.env.example` to `.env` and configure only the integrations you intend to use. Never commit secrets or raw datasets.

## Data

The initial candidate source is NYC 311 public service-request data. Its schema, licensing, availability, infrastructure-related categories, coordinate coverage, and temporal coverage must be checked before model training. OpenStreetMap may be used for contextual features subject to its usage policy and attribution requirements.

See [Dataset notes](docs/datasets.md) and [Methodology](docs/methodology.md).

## Quality principles

- No fabricated or placeholder model metrics presented as results.
- Chronological splits; no future-derived features in training inputs.
- Baselines before complex models.
- Reproducible configuration and fixed random seeds.
- Explicit CRS handling and metric-distance calculations.
- Tests for data contracts, spatial operations, inference, and API behaviour.
- LLM explanations must cite or quote structured model evidence and disclose uncertainty.

## License

License selection is pending. Dataset terms and attribution requirements remain applicable independently of this repository's license.
