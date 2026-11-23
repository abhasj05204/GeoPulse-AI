# Real-data experiment: NYC 311, January–June 2025

**Status: Completed successfully in GitHub Actions, including 10 Optuna trials.**

## Dataset and cohort

- Source: NYC Open Data, 311 Service Requests from 2020 to Present (dataset ID `erm2-nwe9`).
- Query window: 2025-01-01 through 2025-07-01 (end exclusive).
- Bounded sample: 30,000 records; 146 complaint categories observed.
- Coordinate and timestamp validation: 30,000 valid records; 100% latitude and longitude coverage.
- Infrastructure cohort: configured road, streetlight, sidewalk, sewer, water-system, water-leak, and traffic-signal complaint types.
- Spatial aggregation: 1,000 m grid cells, EPSG:32618.
- Prediction target: at least one new complaint in the next 30 days.
- Retained panel: 79,662 cell-day rows; chronological cutoff 2025-04-24.
- Training rows: 63,393. Final holdout rows: 16,269.
- Observed sample dates: 2025-01-01 through 2025-06-21.

This is a bounded, time-slice-stratified sample, not the complete NYC 311 population.

## Model selection

The selected model is Random Forest. Selection used mean PR-AUC across three expanding-window temporal CV folds within the training period. The final chronological holdout was not used for model selection.

| Model | Mean CV PR-AUC | Holdout PR-AUC | Holdout ROC-AUC | Brier score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.4911 | 0.4561 | 0.6240 | 0.2240 |
| Random Forest | **0.6405** | **0.5581** | **0.7297** | **0.1972** |
| Histogram Gradient Boosting (Optuna-tuned) | 0.5988 | 0.5167 | 0.7095 | 0.2053 |
| Prior-probability baseline | — | 0.3308 | — | 0.2270 |

Random Forest holdout PR-AUC is 0.2273 absolute points (about 68.7% relative) above the prior baseline. The holdout positive rate is 0.3308, which is also the PR-AUC of the constant prior baseline.

### Optuna result

Ten trials tuned histogram gradient boosting using expanding-window temporal CV. Best HGB CV PR-AUC was 0.5988; it did not exceed Random Forest's 0.6405 CV PR-AUC, so Random Forest remained selected.

Best HGB parameters:

- `max_iter=187`
- `learning_rate=0.1161`
- `max_leaf_nodes=49`
- `l2_regularization=0.00152`

## Operating point and risk bands

- F1-maximizing operating threshold from temporal out-of-fold predictions: **0.2773**.
- Holdout precision at that threshold: 0.4364.
- Holdout recall at that threshold: **0.8328**.
- Holdout F1 at that threshold: 0.5727.
- Cohort-relative Medium boundary (50th percentile): 0.3814.
- Cohort-relative High boundary (80th percentile): 0.6610.

The operating threshold is for binary recurrence classification. Low/Medium/High are relative probability bands for visualization; neither is an engineering severity classification or calibrated absolute risk.

## Permutation importance

Top holdout features by mean decrease in average precision:

| Feature | Importance |
|---|---:|
| Projected easting (`x_km`) | 0.1057 |
| Projected northing (`y_km`) | 0.0985 |
| Prior 90-day cell count | 0.0776 |
| Adjacent-cell 30-day count | 0.0691 |
| Days since previous complaint | 0.0688 |

The strong coordinate contribution suggests the model captures spatial structure. It also highlights the need for spatially blocked validation before interpreting these results as transferable to unseen neighbourhoods or another city.

## Reproducibility artifact

[Download the trained model, processed panel, evaluation report, feature-importance report, Optuna result, and source audit from the successful GitHub Actions run](https://github.com/abhasj05204/GeoPulse-AI/actions/runs/45249192051/artifacts/14855006814).

The artifact is tied to the exact workflow run and source revision. Re-run the workflow after changing the dataset window, complaint cohort, grid size, or model code.

## Limitations

- This is one bounded sample and one chronological holdout, not a production validation.
- Spatial dependence may inflate performance when nearby cells appear across folds.
- The Brier score is reported, but probability calibration has not yet been optimized.
- The NYC projection and complaint taxonomy are not portable to other cities without adaptation.
- Recurrence is not defect severity; maintenance decisions require domain review.
