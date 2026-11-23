# Modelling methodology

The initial prediction unit is a projected square grid cell observed daily. The label is whether the cell receives at least one new complaint during the next 30 days. Grid size and horizon are configurable and must be sensitivity-tested.

## Leakage controls
- NYC longitude/latitude are transformed from EPSG:4326 to EPSG:32618 before cell assignment.
- Rolling predictors are shifted by one day and therefore exclude the prediction date.
- Labels use only future daily counts; rows without a fully observed horizon are removed.
- A chronological 80/20 holdout is used; no random row split.
- Preprocessing is fitted inside each model pipeline on training data only.

## Initial models and metrics
Balanced logistic regression, random forest, and histogram gradient boosting are compared. Precision, recall, F1, PR-AUC, Brier score, and ROC-AUC (when defined) are reported. The highest holdout PR-AUC selects the persisted model. A persistence baseline, rolling-origin validation, calibration checks, spatially blocked robustness tests, and tuned thresholds are planned before operational interpretation.

## Known limitations
The current projection is NYC-specific. Complaint rows within adjacent cells and dates are dependent; the initial chronological split does not eliminate spatial leakage. Complaint recurrence is not defect severity. Risk bands are provisional display categories and must not be interpreted as calibrated maintenance priority.
