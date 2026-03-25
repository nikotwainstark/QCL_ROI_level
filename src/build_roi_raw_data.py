# -*- coding: utf-8 -*-
"""
ROI-level TMA data generation: extract polygon ROI regions from core hyperspectral
cubes via ROI_CLICKS, pad to axis-aligned rectangles, and save to a raw output directory.

Requirements:
  - A manifest Excel with columns ROI_CLICKS (path to .npz containing coords_xy,
    N vertices (x,y), arbitrary polygon) and CORE_ID (matching the core master).
  - A core master Excel providing hs_path and epi_mask_path per CORE_ID.
  - If the core master points to pre-processed data, note the pre-processing stage.

Pipeline:
  - For each manifest row: load core (H,W,C) and epi_mask (H,W) via CORE_ID,
    load polygon vertices from ROI_CLICKS .npz.
  - Crop to the polygon bounding box; zero-fill outside the polygon for the cube,
    boolean polygon mask, and epithelial mask (all share (H_crop, W_crop) shape).
  - Output to OUTPUT_RAW_DIR; single-pass linear processing, no modelling.
"""

import sys
from pathlib import Path

# Ensure this script can import sibling helpers when run from arbitrary cwd.
sys.path.append(str(Path(__file__).parent))

import os
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths and column names
# -----------------------------------------------------------------------------
MANIFEST_PATH = r"E:\temp project code\PIMO\ROI_level\data\roi_master_file_training_manifest_global_benign_v2.xlsx"
CORE_MASTER_PATH = r"E:\temp project code\PIMO\pimo_core_master_P4_2nd_derivative_windows_edited_epi_mask.xlsx"
OUTPUT_RAW_DIR = r"E:\temp project code\PIMO\ROI_level\data\P4_2nd_derivative"

ROI_CLICKS_COL = "ROI_CLICKS"       # from manifest; path to .npz with coords_xy (N vertices)
CORE_ID_COL = "CORE_ID"             # from manifest; links to core master
HS_PATH_COL = "hs_path"             # from core master; hyperspectral cube .npy path
EPI_MASK_PATH_COL = "epi_mask_path" # from core master; epithelial mask .npy path

# CORE_IDs to exclude (e.g. problematic cores); empty list means no exclusion
EXCLUDE_CORE_IDS = []

# -----------------------------------------------------------------------------
# Config activation (YAML + env/local_overrides priority)
# -----------------------------------------------------------------------------
try:
    from experiment_config import activate_build_roi_raw_data_experiment

    activate_build_roi_raw_data_experiment(sys.modules[__name__])
    print("[config] build_roi_raw_data: config loaded / overrides applied.")
except Exception as e:
    print(f"[config] build_roi_raw_data: skip config loading ({e}).")


def _save_epi_mask_png(roi_epi_mask, png_path):
    """Save ROI epithelial mask as a grayscale PNG for visual inspection (white=epithelium, black=background)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arr = np.asarray(roi_epi_mask, dtype=np.float64)
    if arr.max() > 1 or (arr.min() < 0 and arr.max() > 0):
        arr = np.clip(arr, 0, None) / max(float(arr.max()), 1e-6)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 1) Load polygon vertices (x, y) in pixel coordinates from ROI_CLICKS .npz
# -----------------------------------------------------------------------------
def load_roi_clicks(roi_clicks_path):
    """
    roi_clicks_path: path to .npz containing 'coords_xy' (N, 2) float, (x, y) order, N>=3
    Returns: (N, 2) float64, each row is (x, y)
    """
    if not isinstance(roi_clicks_path, str) or not os.path.exists(roi_clicks_path):
        return None
    data = np.load(roi_clicks_path)
    if "coords_xy" not in data:
        return None
    return np.asarray(data["coords_xy"], dtype=np.float64)


# -----------------------------------------------------------------------------
# 2) Extract polygon region from core cube, pad to axis-aligned bounding box;
#    optionally crop the pixel-aligned epithelial mask with the same bbox
# -----------------------------------------------------------------------------
def extract_roi_from_core(core, coords_xy, epi_mask=None):
    """
    core: (H, W, C) hyperspectral cube
    coords_xy: (N, 2) polygon vertices in (x, y) i.e. (col, row), N>=3
    epi_mask: (H, W) optional epithelial mask, pixel-aligned with core

    Crops to the polygon bounding box; pixels inside the polygon keep their
    original values, pixels outside are zero-filled.
    Returns:
      roi_cube: (H_crop, W_crop, C), zero-filled outside polygon
      roi_polygon_mask: (H_crop, W_crop) bool, True inside polygon
      roi_epi_mask: (H_crop, W_crop) if epi_mask provided, zero-filled outside; else None
    """
    from matplotlib.path import Path

    H, W, C = core.shape
    x = np.clip(coords_xy[:, 0], 0, W - 1e-6)
    y = np.clip(coords_xy[:, 1], 0, H - 1e-6)
    x_min, x_max = int(np.floor(x.min())), int(np.ceil(x.max())) + 1
    y_min, y_max = int(np.floor(y.min())), int(np.ceil(y.max())) + 1
    x_min, x_max = max(0, x_min), min(W, x_max)
    y_min, y_max = max(0, y_min), min(H, y_max)
    if x_max <= x_min or y_max <= y_min:
        return None, None, None

    path = Path(coords_xy)
    h_crop, w_crop = y_max - y_min, x_max - x_min
    jj, ii = np.meshgrid(np.arange(h_crop), np.arange(w_crop), indexing="ij")
    pts_global = np.column_stack([x_min + ii.ravel() + 0.5, y_min + jj.ravel() + 0.5])
    mask_crop = path.contains_points(pts_global).reshape(h_crop, w_crop)

    roi_cube = np.zeros((h_crop, w_crop, C), dtype=core.dtype)
    roi_cube[:] = core[y_min:y_max, x_min:x_max, :]
    roi_cube[~mask_crop, :] = 0

    roi_epi_mask = None
    if epi_mask is not None and epi_mask.shape[:2] == (H, W):
        roi_epi_mask = np.zeros((h_crop, w_crop), dtype=epi_mask.dtype)
        roi_epi_mask[:] = epi_mask[y_min:y_max, x_min:x_max]
        roi_epi_mask[~mask_crop] = 0

    return roi_cube, mask_crop, roi_epi_mask


# -----------------------------------------------------------------------------
# 3) Per-row extraction: look up hs_path / epi_mask_path by CORE_ID, load core
#    and epi mask, load ROI clicks, and extract the ROI (cube + polygon mask + epi mask)
# -----------------------------------------------------------------------------
def extract_roi_for_row(row, core_lut_hs, core_lut_epi):
    """
    row: one manifest row; must contain CORE_ID and ROI_CLICKS columns
    core_lut_hs: dict CORE_ID -> hs_path
    core_lut_epi: dict CORE_ID -> epi_mask_path (pixel-aligned with core)

    Returns:
      (roi_cube, roi_polygon_mask, roi_epi_mask, fail_reason)
      On success: (cube, polygon_mask, epi_mask or None, None)
      On failure: (None, None, None, reason_string)
      roi_epi_mask shares the first two dims with roi_cube; if the epi file is
      missing or shape-mismatched, roi_epi_mask is None but cube/mask are still returned.
    """
    cid = row.get(CORE_ID_COL)
    if cid is None or pd.isna(cid):
        return None, None, None, "CORE_ID_empty"
    cid = str(cid).strip()
    hs_path = core_lut_hs.get(cid)
    if hs_path is None or not hs_path:
        return None, None, None, f"hs_path_not_in_core_master(CORE_ID={cid})"
    if not os.path.exists(hs_path):
        return None, None, None, f"hs_path_file_not_found: {hs_path}"
    clicks_path = row.get(ROI_CLICKS_COL)
    if clicks_path is None or pd.isna(clicks_path):
        return None, None, None, "ROI_CLICKS_empty"
    clicks_path = str(clicks_path).strip()
    if not os.path.exists(clicks_path):
        return None, None, None, f"roi_clicks_file_not_found: {clicks_path}"
    coords = load_roi_clicks(clicks_path)
    if coords is None:
        return None, None, None, f"roi_clicks_load_failed_or_no_coords_xy: {clicks_path}"
    if len(coords) < 3:
        return None, None, None, f"roi_clicks_too_few_points(n={len(coords)}): {clicks_path}"
    core = np.load(hs_path)
    if core.ndim != 3:
        return None, None, None, f"core_not_3d(ndim={core.ndim}): {hs_path}"
    epi_mask = None
    epi_path = core_lut_epi.get(cid) if core_lut_epi else None
    if epi_path and os.path.exists(epi_path):
        em = np.load(epi_path)
        if em.ndim == 2 and em.shape == core.shape[:2]:
            epi_mask = em
    roi_cube, roi_polygon_mask, roi_epi_mask = extract_roi_from_core(core, coords, epi_mask=epi_mask)
    if roi_cube is None or roi_polygon_mask is None:
        return None, None, None, "extract_roi_from_core_returned_None"
    return roi_cube, roi_polygon_mask, roi_epi_mask, None


# -----------------------------------------------------------------------------
# 4) Main pipeline: read manifest + core master, extract ROIs row by row,
#    save padded arrays to the raw output directory
# -----------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_RAW_DIR, exist_ok=True)

    manifest = pd.read_excel(MANIFEST_PATH, sheet_name="all_rois")
    core_master = pd.read_excel(CORE_MASTER_PATH)
    core_lut_hs = core_master.set_index(CORE_ID_COL)[HS_PATH_COL].to_dict()
    core_lut_epi = core_master.set_index(CORE_ID_COL)[EPI_MASK_PATH_COL].to_dict() if EPI_MASK_PATH_COL in core_master.columns else {}

    # Keep only rows with valid ROI_CLICKS and CORE_ID
    manifest = manifest[manifest[ROI_CLICKS_COL].notna()].copy()
    manifest = manifest[manifest[CORE_ID_COL].notna()].copy()

    if EXCLUDE_CORE_IDS:
        exclude_set = {str(c).strip() for c in EXCLUDE_CORE_IDS}
        n_before = len(manifest)
        manifest = manifest[~manifest[CORE_ID_COL].astype(str).str.strip().isin(exclude_set)].copy()
        if len(manifest) < n_before:
            print(f"Excluded {n_before - len(manifest)} rows by EXCLUDE_CORE_IDS (cores: {EXCLUDE_CORE_IDS})")

    saved_paths = []
    failed = []
    for idx, row in manifest.iterrows():
        cid = str(row[CORE_ID_COL]).strip()
        rid = row.get("ROI_ID")
        if pd.isna(rid):
            rid = idx
        roi_cube, roi_polygon_mask, roi_epi_mask, fail_reason = extract_roi_for_row(row, core_lut_hs, core_lut_epi)
        if fail_reason is not None:
            failed.append({"idx": idx, "CORE_ID": cid, "ROI_ID": rid, "reason": fail_reason})
            print(f"  [FAIL] idx={idx} CORE_ID={cid} ROI_ID={rid} -> {fail_reason}")
            continue
        # Filename: CORE_ID_roi_ROI_ID; epi_mask shares (H, W) with the cube
        base = f"{cid}_roi_{rid}"
        cube_path = os.path.join(OUTPUT_RAW_DIR, f"{base}_padded.npy")
        mask_path = os.path.join(OUTPUT_RAW_DIR, f"{base}_mask.npy")
        np.save(cube_path, roi_cube)
        np.save(mask_path, roi_polygon_mask)
        row_saved = {
            "manifest_idx": idx,
            "CORE_ID": cid,
            "ROI_ID": rid,
            "roi_padded_path": cube_path,
            "roi_mask_path": mask_path,
            "shape_H": roi_cube.shape[0],
            "shape_W": roi_cube.shape[1],
            "shape_C": roi_cube.shape[2],
        }
        if roi_epi_mask is not None:
            epi_path = os.path.join(OUTPUT_RAW_DIR, f"{base}_epi_mask.npy")
            np.save(epi_path, roi_epi_mask)
            row_saved["roi_epi_mask_path"] = epi_path
            # Also export a grayscale PNG for quick visual inspection
            epi_png_path = os.path.join(OUTPUT_RAW_DIR, f"{base}_epi_mask.png")
            _save_epi_mask_png(roi_epi_mask, epi_png_path)
            row_saved["roi_epi_mask_png_path"] = epi_png_path
        else:
            row_saved["roi_epi_mask_path"] = ""
            row_saved["roi_epi_mask_png_path"] = ""
        saved_paths.append(row_saved)

    print(f"Output directory: {OUTPUT_RAW_DIR}")
    print(f"Succeeded: {len(saved_paths)} ROIs")
    print(f"Failed: {len(failed)} ROIs")
    if failed:
        from collections import Counter
        reason_counts = Counter(f["reason"].split("(")[0].split(":")[0] for f in failed)
        print("Failure reason summary:", dict(reason_counts))
    if saved_paths:
        paths_df = pd.DataFrame(saved_paths)
        manifest_out = os.path.join(OUTPUT_RAW_DIR, "manifest_saved_paths.csv")
        paths_df.to_csv(manifest_out, index=False, encoding="utf-8-sig")
        print(f"Saved path manifest: {manifest_out}")


if __name__ == "__main__":
    main()
