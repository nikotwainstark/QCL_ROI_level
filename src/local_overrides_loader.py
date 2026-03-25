"""
local_overrides_loader.py

Purpose:
  - Merge machine-local paths / device from:
      1) Environment variables (highest priority)
      2) configs/local_overrides.yaml at repo root (if present)
      3) Entrypoint script defaults passed in by caller

This module is intentionally aligned with the patch-level implementation:
  - env: QCL_DATA_PATH / QCL_OUTPUT_DIR / QCL_DEVICE
  - yaml: configs/local_overrides.yaml keys: data_path / output_dir / device / output_folder_name

Note:
  ROI scripts are mostly classical ML/plotting and may not use `device`.
  However, keeping the same API keeps future extensions easy.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple


def _repo_root() -> Path:
    # QCL_ROI_level/src -> QCL_ROI_level
    return Path(__file__).resolve().parent.parent


def _local_overrides_path() -> Path:
    return _repo_root() / "configs" / "local_overrides.yaml"


def load_local_overrides_dict() -> Dict[str, Any]:
    """Return parsed YAML dict, or {} if file missing / empty / parse error."""
    path = _local_overrides_path()
    if not path.is_file():
        return {}

    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError("configs/local_overrides.yaml exists but cannot be parsed: please install PyYAML (pip install pyyaml)") from e

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}
    return data


def _resolve_path_str(p: str, repo_root: Path) -> str:
    p = os.path.expandvars(os.path.expanduser(str(p).strip()))
    path = Path(p)
    if path.is_absolute():
        return str(path.resolve())
    return str((repo_root / path).resolve())


def resolve_training_paths(
    *,
    default_data_path: str,
    records_subdir: str,
    default_folder_name: str,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Compute effective DATA_PATH and OUTPUT_DIR.

    Priority for data_path:
      QCL_DATA_PATH > yaml.data_path > default_data_path

    Priority for output_dir:
      QCL_OUTPUT_DIR > yaml.output_dir >
        join(dirname(data_path), records_subdir, folder_name_effective)

    Optional yaml keys:
      - output_folder_name: overrides default_folder_name when deriving OUTPUT_DIR
    """
    repo_root = _repo_root()
    yaml_path = _local_overrides_path()
    ov = load_local_overrides_dict()

    meta: Dict[str, Any] = {
        "local_overrides_path": str(yaml_path),
        "yaml_file_exists": yaml_path.is_file(),
        "yaml_keys_used": [],
        "env_keys_used": [],
    }

    folder_eff = ov.get("output_folder_name")
    if folder_eff is not None and str(folder_eff).strip() != "":
        folder_name_effective = str(folder_eff).strip()
        meta["yaml_keys_used"].append("output_folder_name")
    else:
        folder_name_effective = default_folder_name
    meta["folder_name_effective"] = folder_name_effective

    if os.environ.get("QCL_DATA_PATH", "").strip():
        data_path = os.environ["QCL_DATA_PATH"].strip()
        meta["env_keys_used"].append("QCL_DATA_PATH")
    elif ov.get("data_path") is not None and str(ov.get("data_path")).strip() != "":
        data_path = str(ov["data_path"]).strip()
        meta["yaml_keys_used"].append("data_path")
    else:
        data_path = default_data_path

    data_path = _resolve_path_str(data_path, repo_root)
    meta["data_path_effective"] = data_path

    if os.environ.get("QCL_OUTPUT_DIR", "").strip():
        output_dir = os.environ["QCL_OUTPUT_DIR"].strip()
        meta["env_keys_used"].append("QCL_OUTPUT_DIR")
    elif ov.get("output_dir") is not None and str(ov.get("output_dir")).strip() != "":
        output_dir = str(ov["output_dir"]).strip()
        meta["yaml_keys_used"].append("output_dir")
    else:
        output_dir = os.path.join(
            os.path.dirname(data_path), records_subdir, folder_name_effective
        )

    output_dir = _resolve_path_str(output_dir, repo_root)
    meta["output_dir_effective"] = output_dir
    return data_path, output_dir, meta


def apply_device_override(train_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override train_config["device"] if QCL_DEVICE env or yaml device is set.

    Priority: QCL_DEVICE > yaml.device > (leave train_config unchanged).
    """
    meta: Dict[str, Any] = {"source": "train_script_default"}
    ov = load_local_overrides_dict()

    if os.environ.get("QCL_DEVICE", "").strip():
        train_config["device"] = os.environ["QCL_DEVICE"].strip()
        meta["source"] = "env:QCL_DEVICE"
        meta["value"] = train_config["device"]
        return meta

    if ov.get("device") is not None and str(ov.get("device")).strip() != "":
        train_config["device"] = str(ov["device"]).strip()
        meta["source"] = "local_overrides.yaml:device"
        meta["value"] = train_config["device"]
        return meta

    meta["value"] = train_config.get("device")
    return meta

