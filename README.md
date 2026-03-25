# QCL ROI Level: IR Hypoxia Exploration (TMA + Random Forest)

This repository provides ROI-level pipelines to explore hypoxia-related signals in prostate cancer using IR-derived features and pathology annotations.

## Key Hypoxia Proxy: `_common`
A continuous hypoxia proxy named `_common` is computed from two ROI-level measurements:
- `PIMO_HS_Tum` (configurable via `pimo_col`)
- `GLUT1_HS_Tum` (configurable via `glut1_col`)

In this codebase, `_common` uses the **geometric mean** formulation:

\[
_common = \sqrt{(PIMO\_HS\_Tum + 1)(GLUT1\_HS\_Tum + 1)} - 1
\]

## Contents (Scripts)
- `src/build_roi_raw_data.py`
  - Builds padded ROI cubes and polygon ROI masks from a manifest + core master.
- `src/roi_hypoxia_rf_classifier_full_vis.py`
  - ROI-level binary classification using quantile thresholds on `_common`.
  - Supports **multi-sheet Excel** (sheet name = stratum label).
- `src/roi_hypoxia_rf_regressor_train.py`
  - Trains a regression model on the continuous `_common` score.
  - Supports **multi-sheet Excel** (each sheet becomes the stratum label for that subset).
- `src/hs_distribution.py`
  - Distribution/summary analysis for H-score and related proxies.

## Configs
Example configs are provided in `configs/`:
- `configs/build_roi_raw_data_pimo_roi_level.example.yaml`
- `configs/roi_hypoxia_rf_classifier_full_vis_roi_level.example.yaml`
- `configs/roi_hypoxia_rf_regressor_train_roi_level.example.yaml`

Both classifier and regressor configs include optional filtering:
- `filters.excluded_patient_ids`
- `filters.excluded_core_ids`

## How to Run
Run commands from `QCL_ROI_level/src`:

### 1) Build ROI raw data
```bash
python build_roi_raw_data.py --config ../configs/build_roi_raw_data_pimo_roi_level.example.yaml

### 2) Train + visualize ROI hypoxia classifier
```bash
python roi_hypoxia_rf_classifier_full_vis.py --config ../configs/roi_hypoxia_rf_classifier_full_vis_roi_level.example.yaml

### 3) Train ROI hypoxia regressor
python roi_hypoxia_rf_regressor_train.py --config ../configs/roi_hypoxia_rf_regressor_train_roi_level.example.yaml

