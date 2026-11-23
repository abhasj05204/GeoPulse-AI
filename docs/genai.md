# Generative AI analyst

GeoPulse uses Gemini 3.8 Flash through Google's Interactions API. The model is an optional explanation layer over deterministic ML and spatial results; it does not train models, calculate risk scores, or make official engineering diagnoses.

## Grounding

`POST /api/ai/ask` assembles a compact evidence object from the current processed cell panel, latest model predictions, model evaluation report, and permutation feature importance. The LLM is instructed to use only these values, state when evidence is insufficient, and distinguish recurrence probability from infrastructure severity. The API returns both the narrative and the evidence object supplied to the model so the answer can be audited.

`POST /api/ai/summary` is available for callers that already have a structured evidence object. Evidence context is size-limited.

## Configuration

Set `GEMINI_API_KEY` in the server environment and use `GEMINI_MODEL=gemini-3.8-flash` (or another currently supported model ID). Keys are sent in the HTTP header and are never stored in the repository. The request sets `store=false`; no conversation history is retained by this integration.

The endpoint returns HTTP 503 when the key is absent or the provider is unavailable. GenAI tests mock the provider response and do not require a real key.
