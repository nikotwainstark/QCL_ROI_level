# -*- coding: utf-8 -*-
# roi_hypoxia_rf_classifier_full_vis.py
#
# ROI-level hypoxia binary classifier (RF) + full visualization suite (fig1-fig8).
#
# Modes (controlled by MULTI_SHEET_EXCEL):
#   false (default) — read the first sheet of MASTER_EXCEL_PATH only.
#       Output → OUTPUT_DIR/<YYYY-mm-dd_HHMMSS>/
#   true — iterate all non-empty sheets in MASTER_EXCEL_PATH; each sheet name
#       is treated as a *stratum* label.  The full classification + visualisation
#       pipeline runs independently per sheet.
#       Output → OUTPUT_DIR/<sanitized_sheet_name>/<YYYY-mm-dd_HHMMSS>/
#
# Per-sheet pipeline:
# 1) Load/filter rows, compute hypoxia proxy `_common`, derive binary label.
# 2) Pixel sampling → X (n_pixels, n_bands) + y (n_pixels,).
# 3) Repeated patient-level hold-out → RF → pixel & ROI AUC.
# 4) Permutation null distribution + confusion matrices + ROC curves.
# 5) Figures fig1..fig8 (png + pdf) + summary artefacts.
#
# Config:
#   - paths + hyper-parameters are loaded by `experiment_config.activate_hs_models_experiment()`
#   - CLI: `--config ../configs/<...>.yaml` (YAML takes precedence via resolver logic)
#
# Dependencies:  pip install pandas numpy openpyxl
import sys
from pathlib import Path

# Ensure this script can import sibling helpers when run from arbitrary cwd.
sys.path.append(str(Path(__file__).parent))

import datetime
import logging
import os
import pathlib
import re

import joblib
import matplotlib
try:
    from IPython import get_ipython
    if get_ipython() is None:
        matplotlib.use("Agg")
except (ImportError, NameError):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                              roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Configuration
# =============================================================================

MASTER_EXCEL_PATH = r"E:\temp project code\PIMO\ROI_level\data\roi_manifest_P4_filtered_GGs_stratum_edited.xlsx"

PIMO_COL     = "PIMO_HS_Tum"
GLUT1_COL    = "GLUT1_HS_Tum"
ROI_DATA_COL    = "ROI_DATA"
ROI_MASK_COL    = "ROI_EPI_MASK"
CORE_ID_COL     = "CORE_ID"
PATIENT_ID_COL  = "PATIENT_ID"
PATH_REPORT_COL = "PO Path report"  # pathology report column (raw string values)
EXCLUDED_PATIENT_IDS = [210, 224]
EXCLUDED_CORE_IDS = ["202H", "202J"]

N_PIXELS_PER_ROI    = 500   # pixels randomly sampled per ROI
RANDOM_SEED         = 42

WAVENUMBERS_PATH = r"E:\temp project code\PIMO\total_data_P4\wavenumbers.npy"

# Repeated patient-level hold-out
N_REPEATS           = 50    # number of independent train/test splits
TEST_PATIENT_RATIO  = 0.25   # fraction of patients held out as test each repeat

# Permutation test
N_PERMUTATIONS      = 200   # number of label-shuffle permutations

# Quantile split:
#   rows <= QUANTILE_LOW  percentile → label 0  (low hypoxia)
#   rows >= QUANTILE_HIGH percentile → label 1  (high hypoxia)
#   rows in between are excluded from the output
# Set both to 0.5 for a simple median split with no exclusions.
QUANTILE_LOW  = 0.15
QUANTILE_HIGH = 0.85

OUTPUT_DIR = r"E:\temp project code\PIMO\ROI_level\data\outputs\P4\random forest classifier outputs\GG_stratum_geometric_mean"

MULTI_SHEET_EXCEL = True

# -----------------------------------------------------------------------------
# Config activation (YAML + env/local_overrides priority)
# -----------------------------------------------------------------------------
try:
    from experiment_config import activate_hs_models_experiment

    activate_hs_models_experiment(sys.modules[__name__])
    print("[config] roi_hypoxia_rf_classifier_full_vis: config loaded / overrides applied.")
except Exception as e:
    print(f"[config] roi_hypoxia_rf_classifier_full_vis: skip config loading ({e}).")


# =============================================================================
# Helpers
# =============================================================================

_SANITIZE_RE = re.compile(r'[\\/:*?"<>|]+')


def _sanitize_sheet_name(name: str) -> str:
    """Replace Windows-illegal path characters with underscores."""
    sanitized = _SANITIZE_RE.sub("_", str(name)).strip().strip("_")
    return sanitized or "unnamed"


def _setup_logger(run_dir: pathlib.Path) -> logging.Logger:
    """Return the module logger with a fresh FileHandler pointing to *run_dir*."""
    lgr = logging.getLogger("hs_models")
    lgr.setLevel(logging.INFO)
    # Remove previous FileHandlers (keep at most one StreamHandler)
    lgr.handlers = [h for h in lgr.handlers if isinstance(h, logging.StreamHandler)
                    and not isinstance(h, logging.FileHandler)]
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in lgr.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
        lgr.addHandler(ch)
    fh = logging.FileHandler(run_dir / "run_log.txt", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    lgr.addHandler(fh)
    return lgr


# =============================================================================
# Core pipeline — operates on a single DataFrame (one sheet / one stratum)
# =============================================================================

def run_full_pipeline(df_raw: pd.DataFrame, *, run_dir: pathlib.Path,
                      stratum_label: str) -> None:
    """Execute the full classify-and-visualise pipeline for *df_raw*.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw manifest rows (one sheet / one stratum).
    run_dir : pathlib.Path
        Directory into which all outputs will be written.
    stratum_label : str
        Human-readable name for this stratum (used in log lines).
    """

    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(run_dir)

    rng = np.random.default_rng(RANDOM_SEED)

    logger.info(f"{'=' * 60}")
    logger.info(f"Stratum: {stratum_label}")
    logger.info(f"Run directory : {run_dir}")
    logger.info(f"RANDOM_SEED={RANDOM_SEED}  N_REPEATS={N_REPEATS}  "
                f"N_PIXELS_PER_ROI={N_PIXELS_PER_ROI}  N_PERMUTATIONS={N_PERMUTATIONS}")

    # -----------------------------------------------------------------
    # Load and clean
    # -----------------------------------------------------------------
    df = df_raw.copy()
    logger.info(f"Rows loaded: {len(df)}")

    # Optional row-level filters from YAML
    if EXCLUDED_PATIENT_IDS and PATIENT_ID_COL in df.columns:
        before = len(df)
        excluded_patients = set(str(v) for v in EXCLUDED_PATIENT_IDS)
        pid_series = df[PATIENT_ID_COL].apply(lambda x: str(x) if pd.notna(x) else "")
        df = df[~pid_series.isin(excluded_patients)].copy()
        logger.info(
            f"Filter excluded_patient_ids={sorted(excluded_patients)}: "
            f"removed {before - len(df)} rows, kept {len(df)}."
        )

    if EXCLUDED_CORE_IDS and CORE_ID_COL in df.columns:
        before = len(df)
        excluded_cores = set(str(v) for v in EXCLUDED_CORE_IDS)
        core_series = df[CORE_ID_COL].apply(lambda x: str(x) if pd.notna(x) else "")
        df = df[~core_series.isin(excluded_cores)].copy()
        logger.info(
            f"Filter excluded_core_ids={sorted(excluded_cores)}: "
            f"removed {before - len(df)} rows, kept {len(df)}."
        )

    for col in [PIMO_COL, GLUT1_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df[PIMO_COL].notna() & df[GLUT1_COL].notna()].copy()
    logger.info(f"Valid rows (both H-scores present): {len(df)}")

    valid_path_mask = df[ROI_DATA_COL].apply(
        lambda p: isinstance(p, str) and os.path.exists(p)
    )
    n_missing = int((~valid_path_mask).sum())
    df = df[valid_path_mask].copy()
    logger.info(f"Rows with valid ROI_DATA path: {len(df)}  (skipped {n_missing} missing)")

    if len(df) == 0:
        logger.warning("No valid rows remain — skipping this stratum.")
        return

    # -----------------------------------------------------------------
    # Compute _common  (geometric mean: √((P+1)(G+1)) - 1)
    # -----------------------------------------------------------------
    p, g = df[PIMO_COL].to_numpy(float), df[GLUT1_COL].to_numpy(float)
    with np.errstate(invalid="ignore"):
        df["_common"] = np.sqrt((p + 1) * (g + 1)) - 1

    _q25 = df["_common"].quantile(0.25)
    _q75 = df["_common"].quantile(0.75)
    logger.info(f"_common  min={df['_common'].min():.2f}  max={df['_common'].max():.2f}  "
                f"mean={df['_common'].mean():.2f}  median={df['_common'].median():.2f}  "
                f"IQR=[{_q25:.2f}, {_q75:.2f}]  (Q75-Q25={_q75-_q25:.2f})")

    # -----------------------------------------------------------------
    # Binary label
    # -----------------------------------------------------------------
    lo = float(df["_common"].quantile(QUANTILE_LOW))
    hi = float(df["_common"].quantile(QUANTILE_HIGH))

    df["_common_binary"] = np.where(
        df["_common"] <= lo, 0,
        np.where(df["_common"] >= hi, 1, np.nan)
    )

    df_labeled = df[df["_common_binary"].notna()].copy()
    df_labeled["_common_binary"] = df_labeled["_common_binary"].astype(int)

    n_low  = int((df_labeled["_common_binary"] == 0).sum())
    n_high = int((df_labeled["_common_binary"] == 1).sum())
    n_mid  = len(df) - len(df_labeled)

    if len(df_labeled) == 0 or n_low == 0 or n_high == 0:
        logger.warning(f"Insufficient labelled data (low={n_low}, high={n_high}) — skipping this stratum.")
        return

    def _print_ids(subset_df, label):
        logger.info(f"  {label} (n={len(subset_df)}):")
        for col in [CORE_ID_COL, PATIENT_ID_COL]:
            if col in subset_df.columns:
                ids = np.unique(subset_df[col].dropna().astype(str))
                logger.info(f"    {col}: {list(ids)}")

    logger.info(f"Split:  low ≤ {lo:.2f} (Q{QUANTILE_LOW:.2f})  |  high ≥ {hi:.2f} (Q{QUANTILE_HIGH:.2f})")
    logger.info(f"  Low  (0): {n_low}")
    logger.info(f"  High (1): {n_high}")
    if n_mid:
        logger.info(f"  Excluded (middle): {n_mid}")

    _print_ids(df_labeled[df_labeled["_common_binary"] == 0], "Low  (0)")
    _print_ids(df_labeled[df_labeled["_common_binary"] == 1], "High (1)")
    if n_mid:
        df_mid = df[df["_common_binary"].isna()]
        _print_ids(df_mid, "Middle (excluded)")

    # Layer 1 — Pathology report distribution in low / high hypoxia groups
    if PATH_REPORT_COL in df_labeled.columns:
        logger.info("Pathology report distribution by hypoxia group (confounder check):")
        for lbl, name in [(0, "Low  (0)"), (1, "High (1)")]:
            _path_dist = (df_labeled[df_labeled["_common_binary"] == lbl][PATH_REPORT_COL]
                          .value_counts().sort_index().to_dict())
            logger.info(f"  {name}: {_path_dist}")
    else:
        logger.info(f"WARNING: column '{PATH_REPORT_COL}' not found — skipping pathology report distribution logging.")

    # -----------------------------------------------------------------
    # Pixel sampling  →  X  (n_pixels, n_bands)  +  y  (n_pixels,)
    # -----------------------------------------------------------------
    def _sample_pixels(row, n):
        cube_path = row.get(ROI_DATA_COL)
        mask_path = row.get(ROI_MASK_COL)
        if not (isinstance(mask_path, str) and os.path.exists(mask_path)):
            return None
        try:
            cube = np.load(cube_path)
            mask = np.load(mask_path)
        except Exception:
            return None
        if cube.ndim != 3 or mask.shape != cube.shape[:2]:
            return None
        pixels = cube[mask > 0].astype(np.float32)
        if len(pixels) == 0:
            return None
        idx = rng.choice(len(pixels), size=min(n, len(pixels)), replace=False)
        return pixels[idx]

    all_pixels      = []
    all_labels      = []
    all_patient_ids = []
    all_roi_ids     = []
    skipped = 0

    for roi_idx, row in df_labeled.iterrows():
        pixels = _sample_pixels(row, N_PIXELS_PER_ROI)
        if pixels is None:
            skipped += 1
            continue
        n = len(pixels)
        all_pixels.append(pixels)
        all_labels.extend([int(row["_common_binary"])] * n)
        pid = str(row[PATIENT_ID_COL]) if PATIENT_ID_COL in df_labeled.columns else "unknown"
        all_patient_ids.extend([pid] * n)
        all_roi_ids.extend([roi_idx] * n)

    if len(all_pixels) == 0:
        logger.warning("No valid pixels sampled — skipping this stratum.")
        return

    X = np.vstack(all_pixels)
    y = np.array(all_labels,      dtype=np.int32)
    patient_ids = np.array(all_patient_ids)
    roi_ids     = np.array(all_roi_ids)

    logger.info(f"Pixel sampling complete  (skipped {skipped} ROIs — no valid mask/cube)")
    logger.info(f"  X shape : {X.shape}  (dtype={X.dtype})")
    logger.info(f"  y shape : {y.shape}  (0={int((y==0).sum())}, 1={int((y==1).sum())})")
    logger.info(f"  Unique patients in pool: {len(np.unique(patient_ids))}")
    logger.info(f"  Unique ROIs    in pool: {len(np.unique(roi_ids))}")

    # -----------------------------------------------------------------
    # Repeated patient-level hold-out splits
    # -----------------------------------------------------------------
    unique_patients = np.unique(patient_ids)

    _pat_labels = {
        p: set(y[patient_ids == p].tolist()) for p in unique_patients
    }
    _low_only  = np.array([p for p, ls in _pat_labels.items() if ls == {0}])
    _high_only = np.array([p for p, ls in _pat_labels.items() if ls == {1}])
    _both      = np.array([p for p, ls in _pat_labels.items() if ls == {0, 1}])

    logger.info(f"Patient strata:  low-only={len(_low_only)}  "
                f"high-only={len(_high_only)}  both={len(_both)}")

    def _stratified_test_patients(rng_s, ratio):
        chosen = []
        for stratum in [_low_only, _high_only, _both]:
            if len(stratum) == 0:
                continue
            k = max(1, round(len(stratum) * ratio))
            k = min(k, len(stratum))
            chosen.extend(rng_s.choice(stratum, k, replace=False).tolist())
        return set(str(p) for p in chosen)

    splits = []

    for repeat in range(N_REPEATS):
        rng_split  = np.random.default_rng(repeat)
        test_pats  = _stratified_test_patients(rng_split, TEST_PATIENT_RATIO)
        train_pats = set(str(p) for p in unique_patients) - test_pats

        train_mask = np.isin(patient_ids, list(train_pats))
        test_mask  = np.isin(patient_ids, list(test_pats))

        X_train_full = X[train_mask]
        y_train_full = y[train_mask]
        X_test        = X[test_mask]
        y_test        = y[test_mask]

        # Balance train set by downsampling the majority class
        idx0 = np.where(y_train_full == 0)[0]
        idx1 = np.where(y_train_full == 1)[0]
        n_bal = min(len(idx0), len(idx1))
        rng_bal = np.random.default_rng(repeat + 1000)
        idx0_bal = rng_bal.choice(idx0, n_bal, replace=False)
        idx1_bal = rng_bal.choice(idx1, n_bal, replace=False)
        bal_idx  = np.concatenate([idx0_bal, idx1_bal])
        rng_bal.shuffle(bal_idx)

        X_train = X_train_full[bal_idx]
        y_train = y_train_full[bal_idx]

        # Layer 2 — Pathology report distribution in this split's train / test sets
        _train_PATH_counts, _test_PATH_counts = {}, {}
        if PATH_REPORT_COL in df_labeled.columns:
            _train_roi_ids = np.unique(roi_ids[train_mask])
            _test_roi_ids  = np.unique(roi_ids[test_mask])
            _train_PATH_counts = (df_labeled.loc[_train_roi_ids, PATH_REPORT_COL]
                                  .value_counts().sort_index().to_dict())
            _test_PATH_counts  = (df_labeled.loc[_test_roi_ids,  PATH_REPORT_COL]
                                  .value_counts().sort_index().to_dict())

        splits.append({
            "repeat":              repeat,
            "test_pats":           test_pats,
            "X_train":             X_train,
            "y_train":             y_train,
            "X_test":              X_test,
            "y_test":              y_test,
            "train_PATH_counts":   _train_PATH_counts,
            "test_PATH_counts":    _test_PATH_counts,
        })

        logger.info(f"  Repeat {repeat:02d}:  "
                    f"train {X_train.shape[0]} px ({len(train_pats)} pts)  |  "
                    f"test  {X_test.shape[0]} px ({len(test_pats)} pts, "
                    f"0={int((y_test==0).sum())} 1={int((y_test==1).sum())})")
        def _pat_px_summary(pats, mask):
            return {str(p): int((patient_ids[mask] == p).sum()) for p in sorted(pats)}

        logger.info(f"    train patients: {_pat_px_summary(train_pats, train_mask)}")
        logger.info(f"    test  patients: {_pat_px_summary(test_pats,  test_mask)}")
        if PATH_REPORT_COL in df_labeled.columns:
            logger.info(f"    train PATH: {_train_PATH_counts}  |  test PATH: {_test_PATH_counts}")

    logger.info(f"{N_REPEATS} splits ready.")

    # -----------------------------------------------------------------
    # Random Forest Classifier — repeated patient-level hold-out eval
    # -----------------------------------------------------------------
    RF_N_ESTIMATORS     = 500
    RF_MAX_FEATURES     = "sqrt"
    RF_MIN_SAMPLES_LEAF = 5

    results = []

    for split in splits:
        repeat   = split["repeat"]
        X_tr_raw = split["X_train"]
        y_tr     = split["y_train"]
        X_te_raw = split["X_test"]
        y_te     = split["y_test"]

        scaler   = StandardScaler()
        X_tr     = scaler.fit_transform(X_tr_raw)
        X_te     = scaler.transform(X_te_raw)

        rf = RandomForestClassifier(
            n_estimators     = RF_N_ESTIMATORS,
            max_features     = RF_MAX_FEATURES,
            min_samples_leaf = RF_MIN_SAMPLES_LEAF,
            class_weight     = "balanced",
            n_jobs           = -1,
            random_state     = repeat,
        )
        rf.fit(X_tr, y_tr)

        prob_te   = rf.predict_proba(X_te)[:, 1]
        auc_pixel = roc_auc_score(y_te, prob_te)

        test_mask   = np.isin(patient_ids, list(split["test_pats"]))
        roi_ids_te  = roi_ids[test_mask]
        unique_rois = np.unique(roi_ids_te)

        roi_probs  = []
        roi_labels = []
        for rid in unique_rois:
            px_mask = roi_ids_te == rid
            roi_probs.append(prob_te[px_mask].mean())
            roi_labels.append(int(y_te[px_mask][0]))

        auc_roi = roc_auc_score(roi_labels, roi_probs) if len(np.unique(roi_labels)) > 1 else float("nan")

        results.append({
            "repeat":             repeat,
            "auc_pixel":          auc_pixel,
            "auc_roi":            auc_roi,
            "feature_importance": rf.feature_importances_.copy(),
            "n_test_rois":        len(unique_rois),
            "y_te":               y_te.copy(),
            "prob_te":            prob_te.copy(),
            "roi_labels":         list(roi_labels),
            "roi_probs":          list(roi_probs),
            "test_PATH_counts":   split.get("test_PATH_counts",  {}),
            "train_PATH_counts":  split.get("train_PATH_counts", {}),
        })

        logger.info(f"  Repeat {repeat:02d}:  pixel AUC={auc_pixel:.4f}  |  ROI AUC={auc_roi:.4f}  "
                    f"(test ROIs={len(unique_rois)}, px={len(y_te)})")

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    auc_pixels = np.array([r["auc_pixel"] for r in results])
    auc_rois   = np.array([r["auc_roi"]   for r in results if not np.isnan(r["auc_roi"])])
    fi_mean    = np.mean([r["feature_importance"] for r in results], axis=0)
    fi_std     = np.std( [r["feature_importance"] for r in results], axis=0)

    logger.info("=" * 60)
    logger.info(f"Pixel-level AUC:  {auc_pixels.mean():.4f} ± {auc_pixels.std():.4f}  "
                f"(min={auc_pixels.min():.4f}, max={auc_pixels.max():.4f})")
    logger.info(f"ROI-level   AUC:  {auc_rois.mean():.4f} ± {auc_rois.std():.4f}  "
                f"(min={auc_rois.min():.4f}, max={auc_rois.max():.4f})")
    logger.info("=" * 60)
    logger.info(f"Feature importance shape: {fi_mean.shape}")

    # -----------------------------------------------------------------
    # Permutation test — aggregated: each split × K permutations
    # -----------------------------------------------------------------
    _K_PER_SPLIT = max(1, N_PERMUTATIONS // len(splits))
    _total_perm  = len(splits) * _K_PER_SPLIT

    logger.info("=" * 60)
    logger.info(f"Permutation test  (aggregated: {len(splits)} splits × {_K_PER_SPLIT} = {_total_perm} permutations)")
    logger.info("=" * 60)

    null_auc_pixel = []
    null_auc_roi   = []
    _perm_counter  = 0

    for _sp in splits:
        _X_tr_raw_p  = _sp["X_train"]
        _y_tr_p      = _sp["y_train"]
        _X_te_raw_p  = _sp["X_test"]
        _y_te_p      = _sp["y_test"]
        _test_pats_p = _sp["test_pats"]

        _scaler_p = StandardScaler()
        _X_tr_p   = _scaler_p.fit_transform(_X_tr_raw_p)
        _X_te_p   = _scaler_p.transform(_X_te_raw_p)

        _test_mask_p   = np.isin(patient_ids, list(_test_pats_p))
        _roi_ids_te_p  = roi_ids[_test_mask_p]
        _unique_rois_p = np.unique(_roi_ids_te_p)

        for _k in range(_K_PER_SPLIT):
            _seed_p    = _perm_counter + 9999
            _rng_p     = np.random.default_rng(_seed_p)
            _y_tr_shuf = _rng_p.permutation(_y_tr_p)

            _rf_p = RandomForestClassifier(
                n_estimators     = RF_N_ESTIMATORS,
                max_features     = RF_MAX_FEATURES,
                min_samples_leaf = RF_MIN_SAMPLES_LEAF,
                class_weight     = "balanced",
                n_jobs           = -1,
                random_state     = _seed_p,
            )
            _rf_p.fit(_X_tr_p, _y_tr_shuf)
            _prob_p = _rf_p.predict_proba(_X_te_p)[:, 1]

            if len(np.unique(_y_te_p)) > 1:
                null_auc_pixel.append(roc_auc_score(_y_te_p, _prob_p))

            _roi_probs_p, _roi_labels_p = [], []
            for _rid in _unique_rois_p:
                _px = _roi_ids_te_p == _rid
                _roi_probs_p.append(_prob_p[_px].mean())
                _roi_labels_p.append(int(_y_te_p[_px][0]))
            if len(np.unique(_roi_labels_p)) > 1:
                null_auc_roi.append(roc_auc_score(_roi_labels_p, _roi_probs_p))

            _perm_counter += 1
            if _perm_counter % 50 == 0:
                logger.info(f"  {_perm_counter}/{_total_perm} permutations done ...")

    null_auc_pixel = np.array(null_auc_pixel)
    null_auc_roi   = np.array(null_auc_roi)

    _real_px  = auc_pixels.mean()
    _real_roi = auc_rois.mean()
    _p_pixel  = float((null_auc_pixel >= _real_px).sum()) / len(null_auc_pixel) if len(null_auc_pixel) else float("nan")
    _p_roi    = float((null_auc_roi   >= _real_roi).sum()) / len(null_auc_roi)  if len(null_auc_roi)   else float("nan")

    logger.info(f"  Real pixel AUC : {_real_px:.4f}  |  Null: {null_auc_pixel.mean():.4f} ± {null_auc_pixel.std():.4f}")
    logger.info(f"  Real ROI   AUC : {_real_roi:.4f}  |  Null: {null_auc_roi.mean():.4f} ± {null_auc_roi.std():.4f}")
    logger.info(f"  p-value pixel  : {_p_pixel:.4f}  ({int((null_auc_pixel >= _real_px).sum())}/{len(null_auc_pixel)} permutations >= real)")
    logger.info(f"  p-value ROI    : {_p_roi:.4f}  ({int((null_auc_roi >= _real_roi).sum())}/{len(null_auc_roi)} permutations >= real)")

    # -----------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------

    # Wavenumber axis
    if os.path.exists(WAVENUMBERS_PATH):
        _wn    = np.load(WAVENUMBERS_PATH)
        _bands = _wn[:fi_mean.shape[0]]
        _x_label  = "Wavenumber (cm⁻¹)"
        _invert_x = True
    else:
        _bands    = np.arange(fi_mean.shape[0])
        _x_label  = "Band index"
        _invert_x = False

    # results_summary.csv
    _all_PATH_keys = sorted({
        k for r in results
        for k in list(r.get("test_PATH_counts",  {}).keys()) +
                 list(r.get("train_PATH_counts", {}).keys())
    })
    _df_summary = pd.DataFrame([{
        "repeat":      r["repeat"],
        "auc_pixel":   r["auc_pixel"],
        "auc_roi":     r["auc_roi"],
        "n_test_rois": r["n_test_rois"],
        **{f"train_{k}": r.get("train_PATH_counts", {}).get(k, 0) for k in _all_PATH_keys},
        **{f"test_{k}":  r.get("test_PATH_counts",  {}).get(k, 0) for k in _all_PATH_keys},
    } for r in results])
    _df_summary.to_csv(run_dir / "results_summary.csv", index=False)
    logger.info(f"Saved  results_summary.csv  ({len(_df_summary)} rows, PATH report keys: {_all_PATH_keys})")

    np.savez(run_dir / "feature_importance.npz",
             fi_mean=fi_mean, fi_std=fi_std, bands=_bands)
    logger.info("Saved  feature_importance.npz")

    np.savez(run_dir / "null_auc.npz",
             null_auc_pixel=null_auc_pixel, null_auc_roi=null_auc_roi,
             p_pixel=np.array([_p_pixel]), p_roi=np.array([_p_roi]))
    logger.info("Saved  null_auc.npz")

    joblib.dump(results, run_dir / "results_full.joblib")
    logger.info(f"Saved  results_full.joblib  ({len(results)} repeats)")

    logger.info(f"All outputs → {run_dir}")

    # -----------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------
    sns.set_theme(style="whitegrid", font_scale=1.05)

    _repeats    = np.array([r["repeat"]      for r in results])
    _auc_px_all = np.array([r["auc_pixel"]   for r in results])
    _auc_roi_all= np.array([r["auc_roi"]     for r in results])
    _n_roi_all  = np.array([r["n_test_rois"] for r in results])

    # Figure 1 — AUC distribution
    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, vals, title, color in zip(
        axes1,
        [_auc_px_all, _auc_roi_all],
        ["Pixel-level AUC", "ROI-level AUC"],
        ["#1976D2", "#E53935"],
    ):
        mu, sd = vals[~np.isnan(vals)].mean(), vals[~np.isnan(vals)].std()
        ax.scatter(_repeats, vals, color=color, s=30, alpha=0.7, zorder=3)
        ax.axhline(mu,      color=color, linewidth=1.8, linestyle="-",  label=f"mean={mu:.3f}")
        ax.axhline(mu + sd, color=color, linewidth=1.0, linestyle="--", label=f"±std={sd:.3f}")
        ax.axhline(mu - sd, color=color, linewidth=1.0, linestyle="--")
        ax.fill_between([-0.5, len(results)-0.5], mu-sd, mu+sd,
                        color=color, alpha=0.08)
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", label="chance (0.5)")
        ax.set_xlim(-0.5, len(results)-0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Repeat")
        ax.set_ylabel("AUC")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig1.suptitle(f"Figure 1 — AUC Distribution across Repeats  [{stratum_label}]", fontweight="bold")
    fig1.tight_layout()
    fig1.savefig(run_dir / "fig1_auc_distribution.png", dpi=300, bbox_inches="tight")
    fig1.savefig(run_dir / "fig1_auc_distribution.pdf", bbox_inches="tight")
    plt.close(fig1)

    # Figure 2 — Feature importance spectrum
    _mean_spectrum = X.mean(axis=0)

    fig2, ax2 = plt.subplots(figsize=(13, 4))

    ax2b = ax2.twinx()
    ax2b.plot(_bands, _mean_spectrum, color="#BDBDBD", linewidth=1.2,
              alpha=0.7, label="mean spectrum")
    ax2b.fill_between(_bands, _mean_spectrum, alpha=0.12, color="#BDBDBD")
    ax2b.set_ylabel("Mean absorbance (a.u.)", color="#9E9E9E", fontsize=9)
    ax2b.tick_params(axis="y", labelcolor="#9E9E9E")
    ax2b.set_zorder(1)

    ax2.set_zorder(2)
    ax2.patch.set_visible(False)
    ax2.plot(_bands, fi_mean, color="#388E3C", linewidth=1.6, label="mean importance")
    ax2.fill_between(_bands, fi_mean - fi_std, fi_mean + fi_std,
                     color="#388E3C", alpha=0.25, label="±std")
    _top10 = np.argsort(fi_mean)[::-1][:10]
    _top5  = _top10[:5]
    ax2.scatter(_bands[_top10], fi_mean[_top10], color="#F57F17", s=45, zorder=5,
                label="top-10 bands")

    for _idx in _top5:
        _wn_val = _bands[_idx]
        _fi_val = fi_mean[_idx]
        ax2.axvline(_wn_val, color="#F57F17", linewidth=0.8, linestyle="--", alpha=0.6, zorder=4)
        ax2.text(_wn_val, 0,
                 f"{_wn_val:.0f}", ha="center", va="bottom",
                 fontsize=3, color="#E65100", rotation=90, zorder=6)

    ax2.set_xlabel(_x_label)
    ax2.set_ylabel("MDI Feature Importance", color="#388E3C")
    ax2.tick_params(axis="y", labelcolor="#388E3C")
    ax2.set_xlim(_bands.min(), _bands.max())
    ax2.set_title(f"Figure 2 — RF Feature Importance Spectrum  [{stratum_label}]",
                  fontweight="bold")

    lines2, labels2 = ax2.get_legend_handles_labels()
    lines2b, labels2b = ax2b.get_legend_handles_labels()
    ax2.legend(lines2 + lines2b, labels2 + labels2b, fontsize=9, loc="upper left")

    fig2.tight_layout()
    fig2.savefig(run_dir / "fig2_feature_importance.png", dpi=300, bbox_inches="tight")
    fig2.savefig(run_dir / "fig2_feature_importance.pdf", bbox_inches="tight")
    plt.close(fig2)

    # Figure 3 — Per-repeat AUC line plot
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(_repeats, _auc_px_all,  "o-", color="#1976D2", linewidth=1.4,
             markersize=5, label="Pixel AUC")
    ax3.plot(_repeats, _auc_roi_all, "s-", color="#E53935", linewidth=1.4,
             markersize=5, label="ROI AUC")
    for mu, color, ls in [
        (_auc_px_all.mean(),  "#1976D2", "--"),
        (_auc_roi_all[~np.isnan(_auc_roi_all)].mean(), "#E53935", "--"),
    ]:
        ax3.axhline(mu, color=color, linewidth=1.0, linestyle=ls)
    ax3.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", label="chance")
    ax3.set_xlabel("Repeat")
    ax3.set_ylabel("AUC")
    ax3.set_title(f"Figure 3 — AUC per Repeat  [{stratum_label}]", fontweight="bold")
    ax3.set_xticks(_repeats)
    ax3.legend(fontsize=9)
    fig3.tight_layout()
    fig3.savefig(run_dir / "fig3_auc_per_repeat.png", dpi=300, bbox_inches="tight")
    fig3.savefig(run_dir / "fig3_auc_per_repeat.pdf", bbox_inches="tight")
    plt.close(fig3)

    # Figure 4 — Test ROI count vs ROI AUC
    fig4, ax4 = plt.subplots(figsize=(6, 5))
    _valid = ~np.isnan(_auc_roi_all)
    ax4.scatter(_n_roi_all[_valid], _auc_roi_all[_valid],
                color="#7B1FA2", s=40, alpha=0.75)
    ax4.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
    ax4.set_xlabel("Number of test ROIs")
    ax4.set_ylabel("ROI-level AUC")
    ax4.set_title(f"Figure 4 — Test ROI Count vs ROI AUC  [{stratum_label}]", fontweight="bold")
    fig4.tight_layout()
    fig4.savefig(run_dir / "fig4_roi_count_vs_auc.png", dpi=300, bbox_inches="tight")
    fig4.savefig(run_dir / "fig4_roi_count_vs_auc.pdf", bbox_inches="tight")
    plt.close(fig4)

    # Figure 5 — Confusion matrices (pixel & ROI, threshold=0.5 vs Youden's J)
    _yt_px  = np.concatenate([r["y_te"]      for r in results])
    _yp_px  = np.concatenate([r["prob_te"]   for r in results])
    _yt_roi = np.concatenate([r["roi_labels"] for r in results])
    _yp_roi = np.concatenate([r["roi_probs"]  for r in results])
    _fpr_px, _tpr_px, _thr_px  = roc_curve(_yt_px,  _yp_px)
    _fpr_roi, _tpr_roi, _thr_roi = roc_curve(_yt_roi, _yp_roi)
    _youden_thr_pixel = float(_thr_px[np.argmax(_tpr_px - _fpr_px)])
    _youden_thr_roi   = float(_thr_roi[np.argmax(_tpr_roi - _fpr_roi)])

    _all_y_true_px   = _yt_px
    _all_y_true_roi  = _yt_roi
    _all_y_pred_px_05   = (_yp_px >= 0.5).astype(int)
    _all_y_pred_roi_05  = np.array(_yp_roi >= 0.5, dtype=int)
    _all_y_pred_px_j    = (_yp_px >= _youden_thr_pixel).astype(int)
    _all_y_pred_roi_j   = np.array(_yp_roi >= _youden_thr_roi, dtype=int)

    _acc_px_05  = (np.array(_all_y_true_px) == _all_y_pred_px_05).mean()
    _acc_roi_05 = (np.array(_all_y_true_roi) == _all_y_pred_roi_05).mean()
    _acc_px_j   = (np.array(_all_y_true_px) == _all_y_pred_px_j).mean()
    _acc_roi_j  = (np.array(_all_y_true_roi) == _all_y_pred_roi_j).mean()
    logger.info("Confusion matrix accuracy (pooled test, all repeats):")
    logger.info(f"  Threshold=0.5:    Pixel {_acc_px_05:.4f}  |  ROI {_acc_roi_05:.4f}")
    logger.info(f"  Youden's J:      Pixel {_acc_px_j:.4f}  |  ROI {_acc_roi_j:.4f}")

    fig5, axes5 = plt.subplots(2, 2, figsize=(10, 8))
    for row, (y_pred_px, y_pred_roi, thr_label, acc_px, acc_roi) in enumerate([
        (_all_y_pred_px_05,  _all_y_pred_roi_05,  "Threshold = 0.5",     _acc_px_05,  _acc_roi_05),
        (_all_y_pred_px_j,   _all_y_pred_roi_j,   f"Youden's J (px={_youden_thr_pixel:.2f}, roi={_youden_thr_roi:.2f})", _acc_px_j,   _acc_roi_j),
    ]):
        for col, (y_true, y_pred, title, acc) in enumerate([
            (_all_y_true_px,  y_pred_px,  "Pixel-level", acc_px),
            (_all_y_true_roi, y_pred_roi, "ROI-level",   acc_roi),
        ]):
            ax = axes5[row, col]
            cm = confusion_matrix(y_true, y_pred, normalize="true")
            disp = ConfusionMatrixDisplay(cm, display_labels=["Low (0)", "High (1)"])
            disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
            ax.set_title(f"{title}  |  {thr_label}  |  Acc={acc:.3f}")

    fig5.suptitle(f"Figure 5 — Normalised Confusion Matrix  [{stratum_label}]", fontweight="bold")
    fig5.tight_layout()
    fig5.savefig(run_dir / "fig5_confusion_matrix.png", dpi=300, bbox_inches="tight")
    fig5.savefig(run_dir / "fig5_confusion_matrix.pdf", bbox_inches="tight")
    plt.close(fig5)

    # Figure 6 — Permutation test: null AUC distribution vs real AUC
    fig6, axes6 = plt.subplots(1, 2, figsize=(11, 4))

    for ax, null_vals, real_val, p_val, title, color in zip(
        axes6,
        [null_auc_pixel,    null_auc_roi],
        [_real_px,          _real_roi],
        [_p_pixel,          _p_roi],
        ["Pixel-level AUC", "ROI-level AUC"],
        ["#1976D2",         "#E53935"],
    ):
        ax.hist(null_vals, bins=30, color="#BDBDBD", edgecolor="white",
                linewidth=0.5, label=f"Null  (N={len(null_vals)})\n"
                                      f"mean={null_vals.mean():.3f} ± {null_vals.std():.3f}")
        ax.axvline(real_val, color=color, linewidth=2.0, linestyle="-",
                   label=f"Real mean = {real_val:.3f}\np = {p_val:.4f}")
        ax.axvline(0.5, color="gray", linewidth=0.8, linestyle=":",
                   label="chance (0.5)")
        ax.set_xlabel("AUC")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig6.suptitle(f"Figure 6 — Permutation Test  [{stratum_label}]",
                  fontweight="bold")
    fig6.tight_layout()
    fig6.savefig(run_dir / "fig6_permutation_test.png", dpi=300, bbox_inches="tight")
    fig6.savefig(run_dir / "fig6_permutation_test.pdf", bbox_inches="tight")
    plt.close(fig6)

    # Figure 7 — Aggregated ROC curves
    _common_fpr = np.linspace(0, 1, 200)

    fig7, axes7 = plt.subplots(1, 2, figsize=(12, 5))

    for ax, y_true_key, prob_key, label_key, prob_key_roi, title, color in zip(
        axes7,
        ["y_te",       None],
        ["prob_te",    None],
        [None,         "roi_labels"],
        [None,         "roi_probs"],
        ["Pixel-level ROC", "ROI-level ROC"],
        ["#1976D2",    "#E53935"],
    ):
        tpr_list = []
        all_fpr_raw, all_tpr_raw = [], []

        for r in results:
            if title.startswith("Pixel"):
                _yt = r["y_te"]
                _yp = r["prob_te"]
            else:
                _yt = np.array(r["roi_labels"])
                _yp = np.array(r["roi_probs"])

            if len(np.unique(_yt)) < 2:
                continue

            fpr_r, tpr_r, _ = roc_curve(_yt, _yp)
            interp_fn = interp1d(fpr_r, tpr_r, kind="linear",
                                 bounds_error=False, fill_value=(0.0, 1.0))
            tpr_list.append(interp_fn(_common_fpr))
            ax.plot(fpr_r, tpr_r, color=color, alpha=0.12, linewidth=0.8)
            all_fpr_raw.append(fpr_r)
            all_tpr_raw.append(tpr_r)

        tpr_arr  = np.vstack(tpr_list)
        mean_tpr = tpr_arr.mean(axis=0)
        std_tpr  = tpr_arr.std(axis=0)

        ax.plot(_common_fpr, mean_tpr, color=color, linewidth=2.2,
                label=f"Mean ROC  (AUC={np.trapezoid(mean_tpr, _common_fpr):.3f})")
        ax.fill_between(_common_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                        color=color, alpha=0.15, label="±std")
        ax.plot([0, 1], [0, 1], "k:", linewidth=0.8, label="Chance")

        if title.startswith("Pixel"):
            _yt_all = np.concatenate([r["y_te"]    for r in results])
            _yp_all = np.concatenate([r["prob_te"] for r in results])
        else:
            _yt_all = np.concatenate([r["roi_labels"] for r in results])
            _yp_all = np.concatenate([r["roi_probs"]  for r in results])

        fpr_pool, tpr_pool, thresholds_pool = roc_curve(_yt_all, _yp_all)
        _idx_05 = np.argmin(np.abs(thresholds_pool - 0.5))
        _fpr_05, _tpr_05 = fpr_pool[_idx_05], tpr_pool[_idx_05]
        ax.scatter(_fpr_05, _tpr_05, marker="o", s=90, color="#FF6F00", zorder=6,
                   label=f"Thr=0.5  (Sens={_tpr_05:.2f}, Spec={1-_fpr_05:.2f})")

        _j_scores = tpr_pool - fpr_pool
        _idx_j    = np.argmax(_j_scores)
        _fpr_j, _tpr_j = fpr_pool[_idx_j], tpr_pool[_idx_j]
        _thr_j = thresholds_pool[_idx_j]
        ax.scatter(_fpr_j, _tpr_j, marker="*", s=160, color="#6A1B9A", zorder=6,
                   label=f"Youden's J  (Sens={_tpr_j:.2f}, Spec={1-_fpr_j:.2f}, thr={_thr_j:.2f})")

        if title.startswith("Pixel"):
            _youden_thr_pixel = float(_thr_j)
        else:
            _youden_thr_roi = float(_thr_j)

        ax.set_xlabel("False Positive Rate  (1 − Specificity)")
        ax.set_ylabel("True Positive Rate  (Sensitivity)")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)

    fig7.suptitle(f"Figure 7 — ROC Curves across Repeats  [{stratum_label}]", fontweight="bold")
    fig7.tight_layout()
    fig7.savefig(run_dir / "fig7_roc_curves.png", dpi=300, bbox_inches="tight")
    fig7.savefig(run_dir / "fig7_roc_curves.pdf", bbox_inches="tight")
    plt.close(fig7)

    # Figure 8 — Predicted probability violin plots
    logger.info("NOTE: Youden's J thresholds used in Figure 8 row 2 are derived from the "
                "pooled ROC curve across all repeats (test predictions concatenated). "
                "This introduces mild data leakage — thresholds should be treated as "
                "illustrative, not as independently validated operating points.")
    logger.info(f"  Youden's J threshold — Pixel: {_youden_thr_pixel:.3f}  |  ROI: {_youden_thr_roi:.3f}")

    _prob_df_rows = []
    for r in results:
        for prob, label in zip(r["prob_te"], r["y_te"]):
            _prob_df_rows.append({"level": "Pixel", "prob": float(prob),
                                   "label": "High (1)" if label == 1 else "Low (0)"})
        for prob, label in zip(r["roi_probs"], r["roi_labels"]):
            _prob_df_rows.append({"level": "ROI", "prob": float(prob),
                                   "label": "High (1)" if label == 1 else "Low (0)"})

    _prob_df = pd.DataFrame(_prob_df_rows)

    fig8, axes8 = plt.subplots(2, 2, figsize=(11, 9), sharey=True)

    _row_configs = [
        (0, 0.5,                "Threshold = 0.5  (default)",         "gray"),
        (1, None,               "Threshold = Youden's J (pooled ROC, mild leakage)", "#6A1B9A"),
    ]

    for row_idx, _base_thr, _thr_label, _thr_color in _row_configs:
        for col_idx, (level, youden_thr) in enumerate(
            [("Pixel", _youden_thr_pixel), ("ROI", _youden_thr_roi)]
        ):
            ax = axes8[row_idx, col_idx]
            thr = _base_thr if _base_thr is not None else youden_thr
            _color_pal = {"Low (0)": "#1976D2", "High (1)": "#E53935"}
            _sub = _prob_df[_prob_df["level"] == level]

            sns.violinplot(data=_sub, x="label", y="prob", hue="label", palette=_color_pal,
                           inner=None, alpha=0.55, ax=ax, order=["Low (0)", "High (1)"],
                           legend=False)
            sns.stripplot(data=_sub, x="label", y="prob", hue="label", palette=_color_pal,
                          size=2.0, alpha=0.18, jitter=True, ax=ax,
                          order=["Low (0)", "High (1)"], legend=False)

            ax.axhline(thr, color=_thr_color, linewidth=1.4, linestyle="--",
                       label=f"Threshold = {thr:.3f}")

            for i, lbl in enumerate(["Low (0)", "High (1)"]):
                _grp = _sub[_sub["label"] == lbl]["prob"]
                _pct_above = (_grp >= thr).mean() * 100
                _med = _grp.median()
                ax.text(i, _med + 0.02, f"med={_med:.2f}", ha="center",
                        fontsize=7.5, color="black")
                ax.text(i, thr + 0.03, f"{_pct_above:.0f}% ≥ thr",
                        ha="center", fontsize=7, color=_thr_color)

            ax.set_title(f"{level}-level  |  {_thr_label}", fontsize=9)
            ax.set_xlabel("")
            ax.set_ylabel("Predicted P(High)")
            ax.set_ylim(0, 1.1)
            ax.legend(fontsize=8, loc="upper right")

    fig8.suptitle(f"Figure 8 — Predicted Probability Distribution  [{stratum_label}]\n"
                  "Row 1: default threshold (0.5)  |  Row 2: Youden's J optimal threshold",
                  fontweight="bold")
    fig8.tight_layout()
    fig8.savefig(run_dir / "fig8_probability_violin.png", dpi=300, bbox_inches="tight")
    fig8.savefig(run_dir / "fig8_probability_violin.pdf", bbox_inches="tight")
    plt.close(fig8)

    logger.info(f"All figures saved → {run_dir}")
    logger.info(f"Pipeline complete for [{stratum_label}].")


# =============================================================================
# Dispatch: multi-sheet loop vs single-sheet (backward compatible)
# =============================================================================

_run_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
_output_base = pathlib.Path(OUTPUT_DIR)

if MULTI_SHEET_EXCEL:
    xls = pd.ExcelFile(MASTER_EXCEL_PATH)
    sheet_names = xls.sheet_names
    print(f"[multi-sheet] Found {len(sheet_names)} sheets: {sheet_names}")
    for sheet_name in sheet_names:
        df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
        if df_sheet.empty:
            print(f"[multi-sheet] Sheet '{sheet_name}' is empty — skipping.")
            continue
        _safe_name = _sanitize_sheet_name(sheet_name)
        _sheet_run_dir = _output_base / _safe_name / _run_ts
        print(f"\n{'#' * 70}")
        print(f"# Sheet: {sheet_name}  →  {_sheet_run_dir}")
        print(f"{'#' * 70}\n")
        run_full_pipeline(df_sheet, run_dir=_sheet_run_dir, stratum_label=sheet_name)
    xls.close()
    print(f"\n[multi-sheet] All sheets processed.  Session timestamp: {_run_ts}")
else:
    df = pd.read_excel(MASTER_EXCEL_PATH)
    _run_dir = _output_base / _run_ts
    run_full_pipeline(df, run_dir=_run_dir, stratum_label="single-sheet")
