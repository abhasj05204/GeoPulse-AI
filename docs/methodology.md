# Modelling methodology

The prediction unit is a projected square grid cell observed daily. The label is whether the cell receives at least one new infrastructure complaint during the next 30 days. Grid size and horizon are configurable and should be sensitivity-tested.

## Leakage controls
- NYC timestamps are interpreted in America/New_York local time; coordinates are transformed from EPSG:4326 to EPSG:32618 before grid assignment.
- Rolling predictors are shifted by one day and exclude the prediction date.
- Neighbour-cell counts aggregate already-lagged 30-day counts from the eight adjacent cells.
- Labels use only future daily counts; rows without a fully observed horizon are removed.
- Preprocessors are fitted inside each estimator pipeline on training folds only.
- The final 20% chronological holdout is not used for model selection or threshold selection.

## Validation and model selection
The training period is evaluated with expanding-window `TimeSeriesSplit` folds built from unique calendar dates, so all cells from one date remain in the same fold. Logistic regression, random forest, and histogram gradient boosting are compared using mean temporal-CV PR-AUC. The model with the highest mean CV PR-AUC is selected; the final chronological holdout is then used once for reporting generalization metrics.

A prior-probability `DummyClassifier` is reported as a baseline. Reports include precision, recall, F1 at 0.5, PR-AUC, ROC-AUC where defined, Brier score, and CV mean/std. A separate operating threshold is selected from the chosen model's out-of-fold training-period precision-recall curve by maximizing F1, and its holdout precision/recall/F1 are reported separately.

## Risk bands and explainability
The API's Low/Medium/High labels are cohort-relative bands derived from the 50th and 80th percentiles of temporal out-of-fold training probabilities. They are not absolute calibrated risk thresholds and are not infrastructure severity ratings. The numeric recurrence probability remains the model output. Permutation importance is computed on the untouched holdout using average precision; negative importance means shuffling a feature did not harm (and may improve) holdout score.

## Known limitations
The initial projection and source are NYC-specific. Complaint recurrence is not defect severity. The initial chronological evaluation does not fully remove spatial dependence between nearby cells; spatially blocked validation and probability calibration are important next steps. Model performance and risk bands should not be used for operational maintenance decisions without local engineering review.
