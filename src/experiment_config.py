"""
experiment_config.py

Config entry helpers for ROI-level scripts.

This module mirrors the patch-level approach:
  - Resolve experiment YAML via:
      CLI --config path > configs/<primary>.yaml > configs/<example>.yaml
  - Load YAML (PyYAML) and "activate" fields into a target python module.

ROI scripts are not torch training loops, so the injected variables are
primarily constants like MANIFEST_PATH / OUTPUT_DIR / hyper-parameters.
"""

from __future__ import annotations

import sys
from pathlib import Path
import os
from typing import Any, Dict

from local_overrides_loader import load_local_overrides_dict


def repo_root() -> Path:
    # QCL_ROI_level/src -> QCL_ROI_level
    return Path(__file__).resolve().parent.parent


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError("PyYAML is required: pip install pyyaml") from e

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Experiment config must be a YAML mapping: {path}")
    return data


def _argv_config_path() -> Path | None:
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            p = Path(argv[i + 1].strip().strip('"'))
            return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()
        if a.startswith("--config="):
            p = Path(a.split("=", 1)[1].strip().strip('"'))
            return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()
    return None


def resolve_yaml_path(primary: str, example: str) -> Path:
    """
    Resolve experiment YAML path.
    Order: --config CLI > configs/primary > configs/example.
    """
    cli = _argv_config_path()
    if cli is not None:
        if not cli.is_file():
            raise FileNotFoundError(f"--config file not found: {cli}")
        return cli

    root = repo_root()
    for name in (primary, example):
        p = root / "configs" / name
        if p.is_file():
            return p.resolve()

    raise FileNotFoundError(
        f"Experiment YAML not found. Place {primary} or {example} under configs/, or use --config."
    )


def _set_if_present(module: Any, key: str, value: Any) -> None:
    if value is None:
        return
    setattr(module, key, value)


def _effective_data_root() -> str | None:
    """
    Effective base root for resolving relative paths in experiment YAML.

    Priority:
      env QCL_DATA_PATH > configs/local_overrides.yaml.data_path
    """
    if os.environ.get("QCL_DATA_PATH", "").strip():
        return os.environ["QCL_DATA_PATH"].strip()
    ov = load_local_overrides_dict()
    v = ov.get("data_path")
    if v is not None and str(v).strip() != "":
        return str(v).strip()
    return None


def _resolve_cfg_path(v: Any) -> Any:
    """
    If `v` is a string path:
      - absolute -> resolve to absolute and return
      - relative -> join with effective data root if available
      - else -> return as-is
    """
    if not isinstance(v, str):
        return v

    s = v.strip()
    if s == "":
        return v

    try:
        p = Path(s)
        if p.is_absolute():
            return str(p.resolve())
    except Exception:
        return v

    data_root = _effective_data_root()
    if data_root:
        try:
            return str((Path(data_root) / s).resolve())
        except Exception:
            return v

    # If we cannot resolve a relative path to an absolute location, avoid
    # overwriting script defaults (keeps Phase 3 "fallback" behavior).
    return None


def activate_roi_hypoxia_rf_regressor_train(module: Any) -> None:
    """Inject YAML config into roi_hypoxia_rf_regressor_train.py."""
    path = resolve_yaml_path(
        "roi_hypoxia_rf_regressor_train_roi_level.yaml",
        "roi_hypoxia_rf_regressor_train_roi_level.example.yaml",
    )
    cfg = _load_yaml(path)

    paths = cfg.get("paths") or {}
    cols = cfg.get("columns") or {}
    sampling = cfg.get("sampling") or {}
    rf = cfg.get("rf") or {}
    training = cfg.get("training") or {}
    evaluation = cfg.get("evaluation") or {}
    filters = cfg.get("filters") or {}

    _set_if_present(module, "MASTER_EXCEL_PATH", _resolve_cfg_path(paths.get("master_excel_path")))
    _set_if_present(module, "WAVENUMBERS_PATH", _resolve_cfg_path(paths.get("wavenumbers_path")))
    _set_if_present(module, "OUTPUT_DIR", _resolve_cfg_path(paths.get("output_dir")))

    _set_if_present(module, "PIMO_COL", cols.get("pimo_col"))
    _set_if_present(module, "GLUT1_COL", cols.get("glut1_col"))
    _set_if_present(module, "ROI_DATA_COL", cols.get("roi_data_col"))
    _set_if_present(module, "ROI_MASK_COL", cols.get("roi_mask_col"))
    _set_if_present(module, "CORE_ID_COL", cols.get("core_id_col"))
    _set_if_present(module, "PATIENT_ID_COL", cols.get("patient_id_col"))
    _set_if_present(module, "PATH_REPORT_COL", cols.get("path_report_col"))
    _set_if_present(module, "STRATUM_COL", cols.get("stratum_col"))

    _set_if_present(module, "N_PIXELS_PER_ROI", sampling.get("n_pixels_per_roi"))
    _set_if_present(module, "RANDOM_SEED", sampling.get("random_seed"))

    _set_if_present(module, "RF_N_ESTIMATORS", rf.get("n_estimators"))
    _set_if_present(module, "RF_MAX_FEATURES", rf.get("max_features"))
    _set_if_present(module, "RF_MIN_SAMPLES_LEAF", rf.get("min_samples_leaf"))
    _set_if_present(module, "N_REPEATS", rf.get("n_repeats"))
    _set_if_present(module, "TEST_PATIENT_RATIO", rf.get("test_patient_ratio"))

    _set_if_present(module, "TRAIN_FULL_COHORT", training.get("train_full_cohort"))

    _set_if_present(module, "RUN_OOF_EVAL", evaluation.get("run_oof_eval"))
    _set_if_present(module, "LOWESS_FRAC", evaluation.get("lowess_frac"))

    excluded_patient_ids = filters.get("excluded_patient_ids")
    if isinstance(excluded_patient_ids, list):
        _set_if_present(module, "EXCLUDED_PATIENT_IDS", excluded_patient_ids)

    excluded_core_ids = filters.get("excluded_core_ids")
    if isinstance(excluded_core_ids, list):
        _set_if_present(module, "EXCLUDED_CORE_IDS", excluded_core_ids)


def activate_build_roi_raw_data_experiment(module: Any) -> None:
    path = resolve_yaml_path(
        "build_roi_raw_data_pimo_roi_level.yaml",
        "build_roi_raw_data_pimo_roi_level.example.yaml",
    )
    cfg = _load_yaml(path)

    paths = cfg.get("paths") or {}
    cols = cfg.get("columns") or {}
    exclude = cfg.get("exclude") or {}
    core_ids = exclude.get("core_ids")

    _set_if_present(module, "MANIFEST_PATH", _resolve_cfg_path(paths.get("manifest_path")))
    _set_if_present(module, "CORE_MASTER_PATH", _resolve_cfg_path(paths.get("core_master_path")))
    _set_if_present(module, "OUTPUT_RAW_DIR", _resolve_cfg_path(paths.get("output_raw_dir")))

    _set_if_present(module, "ROI_CLICKS_COL", cols.get("roi_clicks_col"))
    _set_if_present(module, "CORE_ID_COL", cols.get("core_id_col"))
    _set_if_present(module, "HS_PATH_COL", cols.get("hs_path_col"))
    _set_if_present(module, "EPI_MASK_PATH_COL", cols.get("epi_mask_path_col"))

    if isinstance(core_ids, list):
        _set_if_present(module, "EXCLUDE_CORE_IDS", core_ids)


def activate_roi_distribution_plots_experiment(module: Any) -> None:
    path = resolve_yaml_path(
        "roi_distribution_plots_pimo_roi_level.yaml",
        "roi_distribution_plots_pimo_roi_level.example.yaml",
    )
    cfg = _load_yaml(path)

    paths = cfg.get("paths") or {}
    targets = cfg.get("targets") or {}
    roi = cfg.get("roi") or {}
    sampling = cfg.get("sampling") or {}
    plot = cfg.get("plot") or {}
    comparison = cfg.get("comparison_sets") or {}
    spectral = cfg.get("spectral_test") or {}

    _set_if_present(module, "MANIFEST_PATH", _resolve_cfg_path(paths.get("manifest_path")))
    _set_if_present(module, "WAVENUMBERS_PATH", _resolve_cfg_path(paths.get("wavenumbers_path")))
    _set_if_present(module, "OUTPUT_VISUAL_DIR", _resolve_cfg_path(paths.get("output_visual_dir")))

    _set_if_present(module, "TARGET_COL", targets.get("target_col"))
    _set_if_present(module, "GG_COL", targets.get("gg_col"))
    _set_if_present(module, "HIGH_RISK_COL", targets.get("high_risk_col"))

    _set_if_present(module, "ROI_DATA_COL", roi.get("roi_data_col"))
    _set_if_present(module, "ROI_EPI_MASK_COL", roi.get("roi_epi_mask_col"))

    _set_if_present(module, "N_PIXELS_MEDIAN_SPECTRUM", sampling.get("n_pixels_median_spectrum"))
    _set_if_present(module, "RANDOM_STATE", sampling.get("random_state"))

    _set_if_present(module, "FIG_FORMAT", plot.get("fig_format"))
    exclude_core_ids = plot.get("exclude_core_ids")
    if isinstance(exclude_core_ids, list):
        _set_if_present(module, "EXCLUDE_CORE_IDS", exclude_core_ids)

    if isinstance(comparison, dict):
        _set_if_present(module, "COMPARISON_SETS", comparison)

    # spectral test + quantile controls
    _set_if_present(module, "SPECTRAL_TEST_SET_HIGH", spectral.get("high"))
    _set_if_present(module, "SPECTRAL_TEST_SET_REST", spectral.get("rest"))

    _set_if_present(module, "HS_LOW_QUANTILE", spectral.get("hs_low_quantile"))
    _set_if_present(module, "HS_HIGH_QUANTILE", spectral.get("hs_high_quantile"))
    _set_if_present(module, "N_PERMUTATION", spectral.get("n_permutation"))
    _set_if_present(module, "FDR_ALPHA", spectral.get("fdr_alpha"))

    _set_if_present(module, "N_PIXELS_PCA", sampling.get("n_pixels_pca"))
    _set_if_present(module, "N_PCA_COMPONENTS_SEARCH", sampling.get("n_pca_components_search"))
    _set_if_present(module, "WAVENUMBER_MAX_CM", sampling.get("wavenumber_max_cm"))


def activate_hs_models_experiment(module: Any) -> None:
    path = resolve_yaml_path(
        "roi_hypoxia_rf_classifier_full_vis_roi_level.yaml",
        "roi_hypoxia_rf_classifier_full_vis_roi_level.example.yaml",
    )
    cfg = _load_yaml(path)

    paths = cfg.get("paths") or {}
    cols = cfg.get("columns") or {}
    sampling = cfg.get("sampling") or {}
    rf = cfg.get("rf") or {}
    filters = cfg.get("filters") or {}

    _set_if_present(module, "MASTER_EXCEL_PATH", _resolve_cfg_path(paths.get("master_excel_path")))
    _set_if_present(module, "WAVENUMBERS_PATH", _resolve_cfg_path(paths.get("wavenumbers_path")))
    _set_if_present(module, "OUTPUT_DIR", _resolve_cfg_path(paths.get("output_dir")))
    _set_if_present(module, "MULTI_SHEET_EXCEL", paths.get("multi_sheet_excel"))

    _set_if_present(module, "PIMO_COL", cols.get("pimo_col"))
    _set_if_present(module, "GLUT1_COL", cols.get("glut1_col"))
    _set_if_present(module, "ROI_DATA_COL", cols.get("roi_data_col"))
    _set_if_present(module, "ROI_MASK_COL", cols.get("roi_mask_col"))
    _set_if_present(module, "CORE_ID_COL", cols.get("core_id_col"))
    _set_if_present(module, "PATIENT_ID_COL", cols.get("patient_id_col"))
    _set_if_present(module, "PATH_REPORT_COL", cols.get("path_report_col"))

    _set_if_present(module, "N_PIXELS_PER_ROI", sampling.get("n_pixels_per_roi"))
    _set_if_present(module, "RANDOM_SEED", sampling.get("random_seed"))
    _set_if_present(module, "QUANTILE_LOW", sampling.get("quantile_low"))
    _set_if_present(module, "QUANTILE_HIGH", sampling.get("quantile_high"))

    _set_if_present(module, "N_REPEATS", rf.get("n_repeats"))
    _set_if_present(module, "TEST_PATIENT_RATIO", rf.get("test_patient_ratio"))
    _set_if_present(module, "N_PERMUTATIONS", rf.get("n_permutations"))

    excluded_patient_ids = filters.get("excluded_patient_ids")
    if isinstance(excluded_patient_ids, list):
        _set_if_present(module, "EXCLUDED_PATIENT_IDS", excluded_patient_ids)

    excluded_core_ids = filters.get("excluded_core_ids")
    if isinstance(excluded_core_ids, list):
        _set_if_present(module, "EXCLUDED_CORE_IDS", excluded_core_ids)

