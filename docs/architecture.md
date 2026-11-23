# Architecture

```mermaid
flowchart LR
 A[Public 311 source] --> B[Validation and cleaning]
 B --> C[Projected spatial cells]
 C --> D[Lagged features and forward labels]
 D --> E[Model benchmark and evaluation]
 E --> F[Persisted model artifact]
 F --> G[FastAPI inference]
 G --> H[Streamlit analytical dashboard]
 E --> I[Explainability and reports]
 I --> G
 G --> J[Optional grounded GenAI]
```

Training runs offline. The API loads the saved model; dashboard requests never retrain models. This is an analytical tool, not an infrastructure severity detector or engineering diagnosis.
