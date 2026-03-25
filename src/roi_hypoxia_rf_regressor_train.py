# -*- coding: utf-8 -*-
# roi_hypoxia_rf_regressor_train.py
#
# ROI-level hypoxia **regression** model (RandomForestRegressor) — training only.
#
# Differences from the binary classifier (`roi_hypoxia_rf_classifier_full_vis.py`):
#   - Target is the continuous `_common` score, NOT a binary 0/1 label.
#     No quantile thresholding or middle-row exclusion is applied.
#   - Input Excel is multi-sheet: every sheet is treated as a GG stratum.
#     All sheets are concatenated into a single DataFrame; a STRATUM column
#     records the originating sheet name for downstream stratified analysis.
#   - No class-balance down-sampling (unnecessary for regression).
#   - Optional OOF (out-of-fold): per repeat, predict test patients only; aggregate
#     per-ROI mean prediction across repeats → pred_oof; ROI-level Spearman + MAE;
#     stratified scatter + LOWESS under oof_eval/.
#
# Outputs (inside OUTPUT_DIR/<run_timestamp>/):
#   repeat_NN_model.joblib   — trained RandomForestRegressor for repeat NN
#   repeat_NN_scaler.joblib  — StandardScaler fitted on that repeat's training set
#   splits_meta.csv          — one row per repeat (n_train_px, n_test_px, ...)
#   oof_eval/roi_oof_predictions.csv — OOF ROI-level table (if run_oof_eval)
#   oof_eval/oof_metrics.csv         — Spearman + MAE overall and per stratum
#   oof_eval/oof_scatter_lowess_by_stratum.png|pdf
#   full_cohort_model.joblib — (optional) model trained on ALL pixels
#   full_cohort_scaler.joblib
#   run_log.txt              — human-readable log
#
# Config:
#   paths / columns / hyper-parameters loaded via
#   `experiment_config.activate_roi_hypoxia_rf_regressor_train()`
#   CLI: `--config ../configs/<...>.yaml`
#
# Dependencies: pip install pandas numpy openpyxl scikit-learn joblib pyyaml scipy statsmodels matplotlib

import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import datetime
import logging
import os
import pathlib

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as lowess_smooth
except ImportError:
    lowess_smooth = None

# =============================================================================
# Default configuration (overridden by YAML via experiment_config)
# =============================================================================

MASTER_EXCEL_PATH = r"E:\temp project code\PIMO\ROI_level\data\roi_manifest_P4_filtered_GGs_stratum_edited.xlsx"

PIMO_COL       = "PIMO_HS_Tum"
GLUT1_COL      = "GLUT1_HS_Tum"
ROI_DATA_COL   = "ROI_DATA"
ROI_MASK_COL   = "ROI_EPI_MASK"
CORE_ID_COL    = "CORE_ID"
PATIENT_ID_COL = "PATIENT_ID"
PATH_REPORT_COL = "PO Path report"
STRATUM_COL    = "STRATUM"

N_PIXELS_PER_ROI = 500
RANDOM_SEED      = 42

WAVENUMBERS_PATH = r"E:\temp project code\PIMO\total_data_P4\wavenumbers.npy"

N_REPEATS          = 50
TEST_PATIENT_RATIO = 0.25

RF_N_ESTIMATORS     = 500
RF_MAX_FEATURES     = "sqrt"
RF_MIN_SAMPLES_LEAF = 5

TRAIN_FULL_COHORT = True
EXCLUDED_PATIENT_IDS = [210, 224]
EXCLUDED_CORE_IDS = ["202H", "202J"]

RUN_OOF_EVAL = True
LOWESS_FRAC = 0.3

OUTPUT_DIR = r"E:\temp project code\PIMO\ROI_level\data\outputs\P4\rf_regressor_train\all_strata"

# =============================================================================
# Config activation (YAML + env/local_overrides priority)
# =============================================================================
try:
    from experiment_config import activate_roi_hypoxia_rf_regressor_train

    activate_roi_hypoxia_rf_regressor_train(sys.modules[__name__])
    print("[config] roi_hypoxia_rf_regressor_train: config loaded / overrides applied.")
except Exception as e:
    print(f"[config] roi_hypoxia_rf_regressor_train: skip config loading ({e}).")

# =============================================================================
# Reproducibility + Output directory + Logging
# =============================================================================
rng = np.random.default_rng(RANDOM_SEED)

_run_ts  = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
_run_dir = pathlib.Path(OUTPUT_DIR) / _run_ts
_run_dir.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("rf_regressor_train")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
_fh  = logging.FileHandler(_run_dir / "run_log.txt", encoding="utf-8")
_fh.setFormatter(_fmt)
_ch  = logging.StreamHandler()
_ch.setFormatter(_fmt)
logger.addHandler(_fh)
logger.addHandler(_ch)

logger.info(f"Run directory : {_run_dir}")
logger.info(f"RANDOM_SEED={RANDOM_SEED}  N_REPEATS={N_REPEATS}  "
            f"N_PIXELS_PER_ROI={N_PIXELS_PER_ROI}")
logger.info(f"RF: n_estimators={RF_N_ESTIMATORS}  max_features={RF_MAX_FEATURES}  "
            f"min_samples_leaf={RF_MIN_SAMPLES_LEAF}")
logger.info(f"OOF eval: RUN_OOF_EVAL={RUN_OOF_EVAL}  LOWESS_FRAC={LOWESS_FRAC}")

# =============================================================================
# Load multi-sheet Excel and concatenate
# =============================================================================

xls = pd.ExcelFile(MASTER_EXCEL_PATH)
sheet_names = xls.sheet_names
logger.info(f"Excel file: {MASTER_EXCEL_PATH}")
logger.info(f"Sheets found ({len(sheet_names)}): {sheet_names}")

frames = []
for sname in sheet_names:
    sdf = pd.read_excel(xls, sheet_name=sname)
    sdf[STRATUM_COL] = sname
    frames.append(sdf)
    logger.info(f"  Sheet '{sname}': {len(sdf)} rows")

df = pd.concat(frames, ignore_index=True)
logger.info(f"Total rows after concat: {len(df)}")

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

# =============================================================================
# Clean: valid H-scores and ROI paths
# =============================================================================

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

# =============================================================================
# Compute _common  (geometric mean: sqrt((P+1)*(G+1)) - 1)
# =============================================================================

p, g = df[PIMO_COL].to_numpy(float), df[GLUT1_COL].to_numpy(float)
with np.errstate(invalid="ignore"):
    df["_common"] = np.sqrt((p + 1) * (g + 1)) - 1

_q25 = df["_common"].quantile(0.25)
_q75 = df["_common"].quantile(0.75)
logger.info(f"_common  min={df['_common'].min():.2f}  max={df['_common'].max():.2f}  "
            f"mean={df['_common'].mean():.2f}  median={df['_common'].median():.2f}  "
            f"IQR=[{_q25:.2f}, {_q75:.2f}]")

# Per-stratum summary
for sname in sheet_names:
    sub = df[df[STRATUM_COL] == sname]["_common"]
    if len(sub):
        logger.info(f"  {sname}: n={len(sub)}  mean={sub.mean():.2f}  "
                    f"median={sub.median():.2f}  min={sub.min():.2f}  max={sub.max():.2f}")

# All rows with valid _common are used (no quantile exclusion)
df_labeled = df[df["_common"].notna()].copy()
logger.info(f"ROIs for regression: {len(df_labeled)}")

# =============================================================================
# Pixel sampling  ->  X (n_pixels, n_bands)  +  y_common (n_pixels,)
# =============================================================================

def _sample_pixels(row, n, rng_local):
    """Load cube + mask for one ROI, return up to n epithelial pixels."""
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
    idx = rng_local.choice(len(pixels), size=min(n, len(pixels)), replace=False)
    return pixels[idx]

all_pixels      = []
all_targets     = []
all_patient_ids = []
all_roi_ids     = []
all_strata      = []
skipped = 0

for roi_idx, row in df_labeled.iterrows():
    pixels = _sample_pixels(row, N_PIXELS_PER_ROI, rng)
    if pixels is None:
        skipped += 1
        continue
    n = len(pixels)
    all_pixels.append(pixels)
    target_val = float(row["_common"])
    all_targets.extend([target_val] * n)
    pid = str(row[PATIENT_ID_COL]) if PATIENT_ID_COL in df_labeled.columns else "unknown"
    all_patient_ids.extend([pid] * n)
    all_roi_ids.extend([roi_idx] * n)
    all_strata.extend([str(row[STRATUM_COL])] * n)

X           = np.vstack(all_pixels)
y           = np.array(all_targets, dtype=np.float32)
patient_ids = np.array(all_patient_ids)
roi_ids     = np.array(all_roi_ids)
strata      = np.array(all_strata)

logger.info(f"Pixel sampling complete  (skipped {skipped} ROIs — no valid mask/cube)")
logger.info(f"  X shape : {X.shape}  (dtype={X.dtype})")
logger.info(f"  y shape : {y.shape}  min={y.min():.2f}  max={y.max():.2f}  mean={y.mean():.2f}")
logger.info(f"  Unique patients in pool: {len(np.unique(patient_ids))}")
logger.info(f"  Unique ROIs    in pool: {len(np.unique(roi_ids))}")

# =============================================================================
# Patient-level hold-out splits
# =============================================================================

unique_patients = np.unique(patient_ids)
logger.info(f"Total unique patients: {len(unique_patients)}")

def _random_test_patients(rng_local, ratio):
    """Sample ~ratio fraction of patients for the test set."""
    k = max(1, round(len(unique_patients) * ratio))
    k = min(k, len(unique_patients) - 1)
    return set(rng_local.choice(unique_patients, k, replace=False).tolist())

splits_meta = []
oof_pred_by_roi = defaultdict(list)  # roi_row_index -> list of per-repeat mean test pred

for repeat in range(N_REPEATS):
    rng_split = np.random.default_rng(repeat)
    test_pats = _random_test_patients(rng_split, TEST_PATIENT_RATIO)
    train_pats = set(unique_patients.tolist()) - test_pats

    train_mask = np.isin(patient_ids, list(train_pats))
    test_mask  = np.isin(patient_ids, list(test_pats))

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test  = X[test_mask]
    y_test  = y[test_mask]

    if len(X_train) == 0:
        logger.warning(f"  Repeat {repeat:02d}: empty training set — skipping.")
        continue

    # ------------------------------------------------------------------
    # Fit scaler on training pixels only
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test) if len(X_test) > 0 else np.empty((0, X.shape[1]))

    # ------------------------------------------------------------------
    # Train RF Regressor
    # ------------------------------------------------------------------
    rf = RandomForestRegressor(
        n_estimators     = RF_N_ESTIMATORS,
        max_features     = RF_MAX_FEATURES,
        min_samples_leaf = RF_MIN_SAMPLES_LEAF,
        n_jobs           = -1,
        random_state     = repeat,
    )
    rf.fit(X_tr, y_train)

    # ------------------------------------------------------------------
    # OOF: test-set predictions only (per repeat), ROI-level mean over pixels
    # ------------------------------------------------------------------
    if RUN_OOF_EVAL and len(X_test) > 0:
        pred_te = rf.predict(X_te)
        roi_te = roi_ids[test_mask]
        for rid in np.unique(roi_te):
            px_mask = roi_te == rid
            oof_pred_by_roi[rid].append(float(np.mean(pred_te[px_mask])))

    # ------------------------------------------------------------------
    # Persist model + scaler
    # ------------------------------------------------------------------
    joblib.dump(rf,     _run_dir / f"repeat_{repeat:02d}_model.joblib")
    joblib.dump(scaler, _run_dir / f"repeat_{repeat:02d}_scaler.joblib")

    splits_meta.append({
        "repeat":           repeat,
        "n_train_px":       len(X_train),
        "n_test_px":        len(X_test),
        "n_train_patients": len(train_pats),
        "n_test_patients":  len(test_pats),
    })

    logger.info(f"  Repeat {repeat:02d}:  train {len(X_train)} px ({len(train_pats)} pts)  |  "
                f"test {len(X_test)} px ({len(test_pats)} pts)  — saved.")

# Save splits metadata
meta_df = pd.DataFrame(splits_meta)
meta_df.to_csv(_run_dir / "splits_meta.csv", index=False)
logger.info(f"Saved  splits_meta.csv  ({len(meta_df)} rows)")

# =============================================================================
# OOF aggregation, metrics (Spearman + MAE), stratified scatter + LOWESS
# =============================================================================

def _roi_metrics(true_arr, pred_arr):
    true_arr = np.asarray(true_arr, dtype=float)
    pred_arr = np.asarray(pred_arr, dtype=float)
    mask = np.isfinite(true_arr) & np.isfinite(pred_arr)
    t, p = true_arr[mask], pred_arr[mask]
    if len(t) < 2:
        return float("nan"), float("nan"), float("nan")
    rho, pval = spearmanr(t, p)
    mae = mean_absolute_error(t, p)
    return float(rho), float(pval), float(mae)


if RUN_OOF_EVAL:
    _oof_dir = _run_dir / "oof_eval"
    _oof_dir.mkdir(parents=True, exist_ok=True)

    oof_rows = []
    for rid, preds in oof_pred_by_roi.items():
        if rid not in df_labeled.index:
            continue
        row = df_labeled.loc[rid]
        pred_oof = float(np.mean(preds))
        oof_rows.append({
            "roi_row_index": rid,
            "patient_id": str(row[PATIENT_ID_COL]) if PATIENT_ID_COL in row.index else "",
            "core_id": str(row[CORE_ID_COL]) if CORE_ID_COL in row.index else "",
            "stratum": str(row[STRATUM_COL]) if STRATUM_COL in row.index else "",
            "true_common": float(row["_common"]),
            "pred_oof": pred_oof,
            "n_repeats_contributing": len(preds),
        })

    oof_df = pd.DataFrame(oof_rows)
    if len(oof_df) == 0:
        logger.warning("OOF: no ROI predictions collected — check N_REPEATS and test splits.")
    else:
        oof_df.to_csv(_oof_dir / "roi_oof_predictions.csv", index=False)
        logger.info(f"Saved  oof_eval/roi_oof_predictions.csv  (n_ROIs={len(oof_df)})")

        _all_roi_idx = set(df_labeled.index)
        _missing = _all_roi_idx - set(oof_pred_by_roi.keys())
        if _missing:
            logger.warning(
                f"OOF: {len(_missing)} ROIs never appeared in a test fold "
                f"(no pred_oof); example indices: {list(_missing)[:10]}"
            )

        tc = oof_df["true_common"].to_numpy()
        po = oof_df["pred_oof"].to_numpy()
        rho_all, p_all, mae_all = _roi_metrics(tc, po)
        metric_rows = [{
            "stratum": "overall",
            "n_rois": len(oof_df),
            "spearman_rho": rho_all,
            "spearman_p": p_all,
            "mae": mae_all,
        }]

        for sname in sheet_names:
            sub = oof_df[oof_df["stratum"] == sname]
            if len(sub) == 0:
                continue
            r_rho, r_p, r_mae = _roi_metrics(sub["true_common"].to_numpy(), sub["pred_oof"].to_numpy())
            metric_rows.append({
                "stratum": sname,
                "n_rois": len(sub),
                "spearman_rho": r_rho,
                "spearman_p": r_p,
                "mae": r_mae,
            })

        pd.DataFrame(metric_rows).to_csv(_oof_dir / "oof_metrics.csv", index=False)
        logger.info(
            f"OOF overall: Spearman rho={rho_all:.4f}  p={p_all:.4g}  MAE={mae_all:.4f}  "
            f"(n_ROIs={len(oof_df)})"
        )

        # Stratified scatter + LOWESS (order: sheet order first, then any extras)
        _seen = set()
        strata_present = []
        for s in list(sheet_names) + sorted(oof_df["stratum"].astype(str).unique(), key=str):
            if s in _seen:
                continue
            if s in set(oof_df["stratum"].astype(str)):
                strata_present.append(s)
                _seen.add(s)
        if not strata_present:
            strata_present = ["(all ROIs)"]
        n_p = max(1, len(strata_present))
        n_cols = min(3, n_p)
        n_rows = int(np.ceil(n_p / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows), squeeze=False)
        for i, sname in enumerate(strata_present):
            r, c = divmod(i, n_cols)
            ax = axes[r][c]
            if sname == "(all ROIs)":
                sub = oof_df
            else:
                sub = oof_df[oof_df["stratum"].astype(str) == str(sname)]
            xs = sub["true_common"].to_numpy(dtype=float)
            ys = sub["pred_oof"].to_numpy(dtype=float)
            ax.scatter(xs, ys, s=12, alpha=0.5, color="#1976D2", edgecolors="none")
            lim_lo = float(min(xs.min() if len(xs) else 0, ys.min() if len(ys) else 0))
            lim_hi = float(max(xs.max() if len(xs) else 1, ys.max() if len(ys) else 1))
            pad = 0.05 * (lim_hi - lim_lo + 1e-9)
            ax.plot([lim_lo - pad, lim_hi + pad], [lim_lo - pad, lim_hi + pad], "k--", lw=1.0, alpha=0.6)
            ax.set_xlim(lim_lo - pad, lim_hi + pad)
            ax.set_ylim(lim_lo - pad, lim_hi + pad)
            ax.set_aspect("equal", adjustable="box")

            if lowess_smooth is not None and len(xs) >= 3:
                order = np.argsort(xs)
                x_s = xs[order]
                y_s = ys[order]
                try:
                    smoothed = lowess_smooth(y_s, x_s, frac=LOWESS_FRAC, return_sorted=True)
                    ax.plot(smoothed[:, 0], smoothed[:, 1], color="#E53935", lw=2.0, label="LOWESS")
                except Exception as ex:
                    logger.warning(f"LOWESS failed for stratum {sname}: {ex}")
            elif lowess_smooth is None:
                logger.warning("statsmodels not installed — LOWESS curve skipped (scatter + y=x only).")

            r_rho, _, r_mae = _roi_metrics(xs, ys)
            ax.set_title(f"{sname}  n={len(sub)}  rho={r_rho:.3f}  MAE={r_mae:.3f}", fontsize=10)
            ax.set_xlabel("true _common")
            ax.set_ylabel("pred_oof")
            ax.grid(True, alpha=0.3)

        for j in range(len(strata_present), n_rows * n_cols):
            r, c = divmod(j, n_cols)
            axes[r][c].set_visible(False)

        fig.suptitle("OOF: true vs pred (per stratum)", fontsize=12, y=1.02)
        fig.tight_layout()
        _png = _oof_dir / "oof_scatter_lowess_by_stratum.png"
        _pdf = _oof_dir / "oof_scatter_lowess_by_stratum.pdf"
        fig.savefig(_png, dpi=200, bbox_inches="tight")
        fig.savefig(_pdf, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved  {_png.name}  /  {_pdf.name}")

# =============================================================================
# (Optional) Full-cohort model — trained on ALL pixels, for external deployment
# =============================================================================

if TRAIN_FULL_COHORT:
    logger.info("=" * 60)
    logger.info("Training full-cohort model (all pixels, no hold-out) ...")

    scaler_full = StandardScaler()
    X_full = scaler_full.fit_transform(X)

    rf_full = RandomForestRegressor(
        n_estimators     = RF_N_ESTIMATORS,
        max_features     = RF_MAX_FEATURES,
        min_samples_leaf = RF_MIN_SAMPLES_LEAF,
        n_jobs           = -1,
        random_state     = RANDOM_SEED,
    )
    rf_full.fit(X_full, y)

    joblib.dump(rf_full,     _run_dir / "full_cohort_model.joblib")
    joblib.dump(scaler_full, _run_dir / "full_cohort_scaler.joblib")

    logger.info(f"  Full-cohort model saved  (n_pixels={len(X)}, n_patients={len(unique_patients)})")

# =============================================================================
# Feature importance snapshot (wavenumber-aligned)
# =============================================================================

fi_arrays = []
for repeat in range(N_REPEATS):
    mpath = _run_dir / f"repeat_{repeat:02d}_model.joblib"
    if mpath.exists():
        fi_arrays.append(joblib.load(mpath).feature_importances_)

if fi_arrays:
    fi_mean = np.mean(fi_arrays, axis=0)
    fi_std  = np.std(fi_arrays, axis=0)

    if os.path.exists(WAVENUMBERS_PATH):
        wn = np.load(WAVENUMBERS_PATH)
        bands = wn[:fi_mean.shape[0]]
    else:
        bands = np.arange(fi_mean.shape[0])

    np.savez(_run_dir / "feature_importance.npz",
             fi_mean=fi_mean, fi_std=fi_std, bands=bands)
    logger.info(f"Saved  feature_importance.npz  (n_bands={len(bands)})")

# =============================================================================
# Done
# =============================================================================

logger.info("=" * 60)
logger.info(f"All outputs -> {_run_dir}")
logger.info("Training complete.")
