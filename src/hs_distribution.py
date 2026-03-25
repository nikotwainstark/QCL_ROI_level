# hs_distribution.py
# Step 1.1: H-score distribution visualisation
# - Overall distribution: PIMO_HS_Tum and GLUT1_HS_Tum histogram (density) + KDE, same axes, no kernel clipping
# - Per-pathology-group distribution: x = user-defined GG group, y = H-score, violin

import sys
from pathlib import Path

# Ensure this script can import sibling helpers when run from arbitrary cwd.
sys.path.append(str(Path(__file__).parent))

import os
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MASTER_EXCEL_PATH = r"E:\temp project code\PIMO\ROI_level\data\roi_manifest_P4_filtered_copy(1).xlsx"
OUTPUT_DIR = r"E:\temp project code\PIMO\ROI_level\data\visual\hs_distribution"

PIMO_COL = "PIMO_HS_Tum"
GLUT1_COL = "GLUT1_HS_Tum"
GG_COL = "gg_simplified"
HIGH_RISK_COL = "high_risk_pattern"
PATIENT_COL = "PATIENT_ID"  # used in 1.3c; falls back to CORE_ID if absent

GROUP_ORDER = [
    "B", "GG1", "GG2", "GG3",
    "GG4-non high risk",
    "GG4-high risk",
]

FIG_FORMAT = "pdf"
EXCLUDE_CORE_IDS = ["202H","202J"]
EXCLUDE_PATIENT_IDS = [210, 224]

# 1.1a histogram
N_HIST_BINS = 35

# 1.2 joint distribution 2D KDE
KDE_LEVELS = 6
KDE_THRESH = 0.05

# 1.3a / 1.3b quantile thresholds (percentile, e.g. 20 = Q20)
COMMON_LOW_PCT = 20   # low-hypoxia quantile (shared by 1.3a cohort-wide and 1.3b within-subset)
COMMON_HIGH_PCT = 90  # high-hypoxia quantile

# 1.4 common high vs low stats and visualisation
ALPHA_1_4 = 0.05

# 2.1 ROI-level spectral test
WAVENUMBERS_PATH = r"E:\temp project code\PIMO\total_data_P4\wavenumbers.npy"
ROI_DATA_COL = "ROI_DATA"
ROI_EPI_MASK_COL = "ROI_EPI_MASK"
FDR_ALPHA_2_1 = 0.05        # significance threshold (FDR q-value)
N_PERMUTATIONS_2_1 = 2000  # permutation rounds for full-spectrum L2 distance test
WAVENUMBER_MAX_CM_2_1 = None  # upper wavenumber cutoff (cm^-1); None = full spectrum


def _grade_group_risk_profile(row):
    gg_val = str(row[GG_COL]).strip()
    if gg_val == "GG4":
        hr = row.get(HIGH_RISK_COL)
        return "GG4-high risk" if (
            hr is True or hr == 1 or (isinstance(hr, str) and hr.lower() in ("true", "1"))
        ) else "GG4-non high risk"
    return gg_val


def _set_backend():
    import matplotlib
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            matplotlib.use("Agg")
    except (ImportError, NameError):
        matplotlib.use("Agg")


def calc_common(pimo_hs, glut1_hs):
    """
    Compute common hypoxia proxy from PIMO and GLUT1 H-scores.
    Formula: geometric mean = sqrt((P+1)*(G+1)) - 1.
    Accepts scalars or arrays; returns ndarray of same shape.
    """
    p = np.asarray(pimo_hs, dtype=np.float64)
    g = np.asarray(glut1_hs, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.sqrt((p + 1) * (g + 1)) - 1


# -----------------------------------------------------------------------------
# Load and clean data
# -----------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
xl = pd.ExcelFile(MASTER_EXCEL_PATH)
sheet = "all_rois" if "all_rois" in xl.sheet_names else xl.sheet_names[0]
df = pd.read_excel(MASTER_EXCEL_PATH, sheet_name=sheet)

if HIGH_RISK_COL not in df.columns:
    df[HIGH_RISK_COL] = False
df["_group"] = df.apply(_grade_group_risk_profile, axis=1)
df = df[df["_group"].isin(GROUP_ORDER)].copy()
if EXCLUDE_CORE_IDS and "CORE_ID" in df.columns:
    exclude_set = {str(c).strip() for c in EXCLUDE_CORE_IDS}
    df = df[~df["CORE_ID"].astype(str).str.strip().isin(exclude_set)].copy()
if EXCLUDE_PATIENT_IDS and PATIENT_COL in df.columns:
    exclude_pats = {str(p).strip() for p in EXCLUDE_PATIENT_IDS}
    n_before = len(df)
    df = df[~df[PATIENT_COL].apply(lambda x: str(x).strip() if pd.notna(x) else "").isin(exclude_pats)].copy()
    print(f"Excluded patient IDs {sorted(exclude_pats)}: removed {n_before - len(df)} rows, kept {len(df)}")

for col in (PIMO_COL, GLUT1_COL):
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df[df[PIMO_COL].notna() & df[GLUT1_COL].notna()].copy()
print(f"Valid ROI count: {len(df)}")

# compute _common via helper without modifying the original Excel
df["_common"] = calc_common(df[PIMO_COL], df[GLUT1_COL])

for col, name in [(PIMO_COL, "PIMO_HS_Tum"), (GLUT1_COL, "GLUT1_HS_Tum")]:
    v = df[col].dropna()
    if len(v) == 0:
        continue
    n_neg = (v < 0).sum()
    print(f"  {name}: min={v.min():.4f}, max={v.max():.4f}, negative_count={n_neg}")


# -----------------------------------------------------------------------------
# Section 1.1a: Overall distribution — PIMO and GLUT1 histogram + KDE (same axes, no clipping)
# -----------------------------------------------------------------------------
_set_backend()
import matplotlib.pyplot as plt
import seaborn as sns

pimo = df[PIMO_COL].dropna().values
glut1 = df[GLUT1_COL].dropna().values
x_min = min(pimo.min() if len(pimo) else 0, glut1.min() if len(glut1) else 0)
x_max = max(pimo.max() if len(pimo) else 0, glut1.max() if len(glut1) else 0)
bins = np.linspace(x_min, x_max, N_HIST_BINS)

fig1, ax = plt.subplots(figsize=(7, 4))
if len(pimo) > 0:
    ax.hist(pimo, bins=bins, density=True, alpha=0.5, color="steelblue", edgecolor="none")
    sns.kdeplot(pimo, ax=ax, color="steelblue", linewidth=2, label="PIMO H-score")
if len(glut1) > 0:
    ax.hist(glut1, bins=bins, density=True, alpha=0.5, color="coral", edgecolor="none")
    sns.kdeplot(glut1, ax=ax, color="coral", linewidth=2, label="GLUT1 H-score")
ax.set_xlabel("H-score")
ax.set_ylabel("Density")
ax.set_title("Step 1.1a  Overall H-score distribution (PIMO vs GLUT1)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
path1 = os.path.join(OUTPUT_DIR, f"step1_1a_overall_hist_kde_pimo_glut1.{FIG_FORMAT}")
save_kw = dict(format=FIG_FORMAT, bbox_inches="tight")
if FIG_FORMAT in ("png", "jpg", "jpeg", "tiff"):
    save_kw["dpi"] = 150
fig1.savefig(path1, **save_kw)
plt.close(fig1)
print(f"Saved overall distribution: {path1}")


# -----------------------------------------------------------------------------
# Section 1.1b: Per-pathology-group distribution — COMMON / PIMO / GLUT1
# -----------------------------------------------------------------------------
present_groups = [g for g in GROUP_ORDER if (df["_group"] == g).any()]
if not present_groups:
    print("No valid groups found; skipping per-group violin plot.")
else:
    import seaborn as sns

    # reshape to long form: each row is (group, metric, value)
    long_rows = []
    for _, row in df.iterrows():
        g = row["_group"]
        try:
            common_v = float(row["_common"])
        except (TypeError, ValueError):
            common_v = np.nan
        pimo_v = float(row[PIMO_COL]) if pd.notna(row[PIMO_COL]) else np.nan
        glut1_v = float(row[GLUT1_COL]) if pd.notna(row[GLUT1_COL]) else np.nan
        if np.isfinite(common_v):
            long_rows.append({"group": g, "metric": "COMMON", "value": common_v})
        if np.isfinite(pimo_v):
            long_rows.append({"group": g, "metric": "PIMO_HS", "value": pimo_v})
        if np.isfinite(glut1_v):
            long_rows.append({"group": g, "metric": "GLUT1_HS", "value": glut1_v})

    if long_rows:
        long_df = pd.DataFrame(long_rows)
        present_groups = [g for g in GROUP_ORDER if (long_df["group"] == g).any()]
        n_cat = len(present_groups)
        fig_w = max(20, 3.8 * n_cat)

        palette = dict(zip(present_groups, sns.color_palette("husl", n_colors=len(present_groups))))
        fig2, axes = plt.subplots(1, 3, figsize=(fig_w, 5))
        for ax, (metric, title) in zip(
            axes,
            [
                ("COMMON", "Step 1.1b  COMMON (common hypoxia)"),
                ("PIMO_HS", "Step 1.1b  PIMO H-score"),
                ("GLUT1_HS", "Step 1.1b  GLUT1 H-score"),
            ],
        ):
            sub = long_df[long_df["metric"] == metric].dropna(subset=["value"])
            if len(sub) > 0:
                sns.violinplot(
                    data=sub,
                    x="group",
                    y="value",
                    order=present_groups,
                    inner=None,
                    linewidth=1.2,
                    color=".3",
                    width=0.95,
                    ax=ax,
                )
                for c in ax.collections:
                    c.set_facecolor("none")
                sns.stripplot(
                    data=sub,
                    x="group",
                    y="value",
                    order=present_groups,
                    hue="group",
                    palette=palette,
                    alpha=0.5,
                    size=4,
                    jitter=0.2,
                    dodge=False,
                    legend=False,
                    ax=ax,
                )
                bp = sns.boxplot(
                    data=sub,
                    x="group",
                    y="value",
                    order=present_groups,
                    width=0.18,
                    showfliers=False,
                    linewidth=1.2,
                    patch_artist=True,
                    ax=ax,
                )
                for patch in bp.patches:
                    patch.set_facecolor(".95")
                    patch.set_edgecolor(".3")
                means = sub.groupby("group", observed=True)["value"].mean().reindex(present_groups).dropna()
                for i, gr in enumerate(present_groups):
                    if gr not in means.index:
                        continue
                    mu = means[gr]
                    ax.scatter(i, mu, s=80, c="darkred", zorder=5, edgecolors="none")
                    ax.hlines(mu, i, i + 0.38, colors="black", linestyle="--", linewidth=1, zorder=4)
                    ax.text(
                        i + 0.42,
                        mu,
                        f"mean = {mu:.2f}",
                        va="center",
                        fontsize=6,
                        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="gray", alpha=0.9),
                    )
            ax.set_title(title)
            ax.set_ylabel("Value")
            ax.set_xlabel("Grade group / risk")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        path2 = os.path.join(OUTPUT_DIR, f"step1_1b_violin_by_group.{FIG_FORMAT}")
        fig2.savefig(path2, **save_kw)
        plt.close(fig2)
        print(f"Saved per-group violin: {path2}")


# -----------------------------------------------------------------------------
# Section 1.2: PIMO vs GLUT1 joint distribution (scatter + 2D KDE, coloured by group)
# -----------------------------------------------------------------------------
present_groups_12 = [g for g in GROUP_ORDER if (df["_group"] == g).any()]
if not present_groups_12:
    print("No valid groups found; skipping joint distribution plot.")
else:
    import seaborn as sns

    fig_scatter, ax_sc = plt.subplots(figsize=(6, 6))
    # grey overall 2D KDE contours showing total density
    sns.kdeplot(
        x=df[PIMO_COL],
        y=df[GLUT1_COL],
        ax=ax_sc,
        levels=KDE_LEVELS,
        color="0.5",
        linewidths=1.0,
        fill=False,
        thresh=KDE_THRESH,
    )

    # per-group scatter, colours consistent with 1.1b violin (husl palette)
    palette_12 = dict(zip(present_groups_12, sns.color_palette("husl", n_colors=len(present_groups_12))))
    for g in present_groups_12:
        sub = df[df["_group"] == g]
        if len(sub) == 0:
            continue
        ax_sc.scatter(
            sub[PIMO_COL],
            sub[GLUT1_COL],
            s=18,
            alpha=0.6,
            color=palette_12[g],
            label=g,
            edgecolors="none",
        )

    ax_sc.set_xlabel("PIMO H-score")
    ax_sc.set_ylabel("GLUT1 H-score")
    ax_sc.set_title("Step 1.2  Joint distribution: PIMO vs GLUT1 (colored by group)")
    ax_sc.grid(True, alpha=0.3)
    ax_sc.legend(title="Group", fontsize=8)
    fig_scatter.tight_layout()
    path_joint = os.path.join(OUTPUT_DIR, f"step1_2_joint_pimo_vs_glut1.{FIG_FORMAT}")
    fig_scatter.savefig(path_joint, **save_kw)
    plt.close(fig_scatter)
    print(f"Saved joint distribution plot: {path_joint}")


# -----------------------------------------------------------------------------
# Section 1.3a: Cohort-wide quantile thresholds -> common_high / common_low dataframes
# -----------------------------------------------------------------------------
q_low = float(df["_common"].quantile(COMMON_LOW_PCT / 100.0))
q_high = float(df["_common"].quantile(COMMON_HIGH_PCT / 100.0))

df_common_high = df[df["_common"] >= q_high].copy()
df_common_low = df[df["_common"] <= q_low].copy()

print(f"1.3a cohort common hypoxia: Q{COMMON_LOW_PCT}={q_low:.4f}, Q{COMMON_HIGH_PCT}={q_high:.4f}")
print(f"  common_high (>=Q{COMMON_HIGH_PCT}): {len(df_common_high)} ROIs -> df_common_high")
print(f"  common_low  (<=Q{COMMON_LOW_PCT}):  {len(df_common_low)} ROIs -> df_common_low")


# -----------------------------------------------------------------------------
# Section 1.3b: Within-subset quantile thresholds -> both_high / both_low
#   (compared within GG to avoid GG-grade confounding)
# -----------------------------------------------------------------------------
subset_both_high_low = {}  # group -> {"both_high": df, "both_low": df}

for group in GROUP_ORDER:
    sub = df[df["_group"] == group].copy()
    if len(sub) < 2:
        subset_both_high_low[group] = {"both_high": sub.iloc[0:0].copy(), "both_low": sub.iloc[0:0].copy()}
        continue
    q_low_sub = float(sub["_common"].quantile(COMMON_LOW_PCT / 100.0))
    q_high_sub = float(sub["_common"].quantile(COMMON_HIGH_PCT / 100.0))
    mask_high = sub["_common"] >= q_high_sub
    mask_low = sub["_common"] <= q_low_sub
    subset_both_high_low[group] = {
        "both_high": sub.loc[mask_high].copy(),
        "both_low": sub.loc[mask_low].copy(),
    }
    n_high = mask_high.sum()
    n_low = mask_low.sum()
    print(f"1.3b [{group}] within-subset common Q{COMMON_LOW_PCT}/Q{COMMON_HIGH_PCT}: both_high={n_high} ROIs, both_low={n_low} ROIs -> subset_both_high_low['{group}']")


# -----------------------------------------------------------------------------
# Section 1.3c: Print patient IDs covered by each 1.3a / 1.3b dataframe
# -----------------------------------------------------------------------------
_id_col = PATIENT_COL if PATIENT_COL in df.columns else "CORE_ID"
if _id_col not in df.columns:
    print("1.3c skipped: no PATIENT_ID or CORE_ID column found.")
else:
    print("\n--- 1.3c patient IDs covered by each df ---")
    print(f"  Using column: {_id_col}")
    # 1.3a
    p_high = sorted(df_common_high[_id_col].dropna().astype(str).str.strip().unique().tolist())
    p_low = sorted(df_common_low[_id_col].dropna().astype(str).str.strip().unique().tolist())
    print(f"  1.3a common_high (df_common_high): {len(p_high)} patients -> {p_high}")
    print(f"  1.3a common_low  (df_common_low):  {len(p_low)} patients -> {p_low}")
    # 1.3b
    for group in GROUP_ORDER:
        if group not in subset_both_high_low:
            continue
        d = subset_both_high_low[group]
        ph = sorted(d["both_high"][_id_col].dropna().astype(str).str.strip().unique().tolist())
        pl = sorted(d["both_low"][_id_col].dropna().astype(str).str.strip().unique().tolist())
        print(f"  1.3b [{group}] both_high: {len(ph)} patients -> {ph}")
        print(f"  1.3b [{group}] both_low:  {len(pl)} patients -> {pl}")


# -----------------------------------------------------------------------------
# Section 1.4: common high vs low — stats / tests / GG composition (3-panel figure)
# -----------------------------------------------------------------------------
from scipy import stats

# label both groups and concatenate into a long table
df_high = df_common_high.copy()
df_low = df_common_low.copy()
df_high["_1_4_group"] = "common_high"
df_low["_1_4_group"] = "common_low"
df_14 = pd.concat([df_high, df_low], axis=0, ignore_index=True)
n_high, n_low = len(df_high), len(df_low)

# Mann-Whitney U test (two-sided)
def _mw_p(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return p

p_pimo = _mw_p(df_common_high[PIMO_COL].values, df_common_low[PIMO_COL].values)
p_glut1 = _mw_p(df_common_high[GLUT1_COL].values, df_common_low[GLUT1_COL].values)
p_common = _mw_p(df_common_high["_common"].values, df_common_low["_common"].values)

# GG contingency table: common_high/low x _group
ct = pd.crosstab(df_14["_1_4_group"], df_14["_group"])
try:
    chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)
except Exception:
    p_chi2 = np.nan
if ct.size < 2 or ct.shape[0] < 2 or ct.shape[1] < 2:
    p_chi2 = np.nan

# summary table
rows = [
    ["variable", "common_high_median", "common_high_IQR", "common_low_median", "common_low_IQR", "p_MannWhitney"],
    [PIMO_COL, df_common_high[PIMO_COL].median(), f"{df_common_high[PIMO_COL].quantile(0.25):.2f}-{df_common_high[PIMO_COL].quantile(0.75):.2f}",
     df_common_low[PIMO_COL].median(), f"{df_common_low[PIMO_COL].quantile(0.25):.2f}-{df_common_low[PIMO_COL].quantile(0.75):.2f}", f"{p_pimo:.4f}"],
    [GLUT1_COL, df_common_high[GLUT1_COL].median(), f"{df_common_high[GLUT1_COL].quantile(0.25):.2f}-{df_common_high[GLUT1_COL].quantile(0.75):.2f}",
     df_common_low[GLUT1_COL].median(), f"{df_common_low[GLUT1_COL].quantile(0.25):.2f}-{df_common_low[GLUT1_COL].quantile(0.75):.2f}", f"{p_glut1:.4f}"],
    ["_common", df_common_high["_common"].median(), f"{df_common_high['_common'].quantile(0.25):.2f}-{df_common_high['_common'].quantile(0.75):.2f}",
     df_common_low["_common"].median(), f"{df_common_low['_common'].quantile(0.25):.2f}-{df_common_low['_common'].quantile(0.75):.2f}", f"{p_common:.4f}"],
]
tbl = pd.DataFrame(rows[1:], columns=rows[0])
print("\n--- 1.4 common high vs low summary table ---")
print(tbl.to_string(index=False))
print(f"\nGG contingency table (common_high vs common_low x GG):\n{ct.to_string()}")
print(f"  Chi2/Fisher p = {p_chi2:.4f}" if not np.isnan(p_chi2) else "  GG contingency table p not computed")
out_csv = os.path.join(OUTPUT_DIR, "step1_4_common_high_low_summary.csv")
tbl.to_csv(out_csv, index=False, encoding="utf-8-sig")
ct.to_csv(os.path.join(OUTPUT_DIR, "step1_4_gg_crosstab.csv"), encoding="utf-8-sig")
print(f"  Saved: {out_csv}, step1_4_gg_crosstab.csv")

# 3-panel figure: PIMO / GLUT1 group comparison + GG composition
fig_14, axes = plt.subplots(1, 3, figsize=(12, 4))
ax1, ax2, ax3 = axes

# left: PIMO
sns.boxplot(data=df_14, x="_1_4_group", y=PIMO_COL, order=["common_high", "common_low"], ax=ax1)
ax1.set_title(f"Step 1.4  PIMO H-score\nn_high={n_high}, n_low={n_low}; p={p_pimo:.4f}")
ax1.set_ylabel("H-score")
ax1.set_xlabel("")
ax1.tick_params(axis="x", rotation=15)

# centre: GLUT1
sns.boxplot(data=df_14, x="_1_4_group", y=GLUT1_COL, order=["common_high", "common_low"], ax=ax2)
ax2.set_title(f"Step 1.4  GLUT1 H-score\nn_high={n_high}, n_low={n_low}; p={p_glut1:.4f}")
ax2.set_ylabel("H-score")
ax2.set_xlabel("")
ax2.tick_params(axis="x", rotation=15)

# right: GG composition (ROI count per GG group, grouped bar chart)
ct_plot = ct.reindex(columns=[g for g in GROUP_ORDER if g in ct.columns]).fillna(0).astype(int)
x = np.arange(ct_plot.shape[1])
w = 0.35
vals_high = ct_plot.loc["common_high"].values if "common_high" in ct_plot.index else np.zeros(len(x))
vals_low = ct_plot.loc["common_low"].values if "common_low" in ct_plot.index else np.zeros(len(x))
ax3.bar(x - w/2, vals_high, width=w, label="common_high", color="coral")
ax3.bar(x + w/2, vals_low, width=w, label="common_low", color="steelblue")
ax3.set_xticks(x)
ax3.set_xticklabels(ct_plot.columns, rotation=25, ha="right")
ax3.set_ylabel("ROI count")
ax3.set_title(f"Step 1.4  GG composition\nchi2 p={p_chi2:.4f}" if not np.isnan(p_chi2) else "Step 1.4  GG composition")
ax3.legend()

plt.tight_layout()
path_14 = os.path.join(OUTPUT_DIR, f"step1_4_common_high_vs_low.{FIG_FORMAT}")
fig_14.savefig(path_14, **save_kw)
plt.close(fig_14)
print(f"Saved 1.4 figure: {path_14}")


# =============================================================================
# Section 2.1: ROI-level spectral test — common high vs low
# Strategy: cohort-wide candidate bands (A/B/C) + stratified validation per 1.3b subsets (D)
# =============================================================================

def _load_roi_median_spectrum(cube_path, mask_path):
    """Load epithelial-region pixels for a ROI; return median spectrum (1D ndarray) or None on failure."""
    if not isinstance(cube_path, str) or not os.path.exists(cube_path):
        return None
    if not isinstance(mask_path, str) or not os.path.exists(mask_path):
        return None
    try:
        cube = np.load(cube_path)
        mask = np.load(mask_path)
    except Exception:
        return None
    if cube.ndim != 3 or mask.ndim != 2 or cube.shape[:2] != mask.shape:
        return None
    epi = mask.astype(bool) if mask.dtype == bool else (mask > 0)
    pix = cube[epi].astype(np.float64)
    return np.nanmedian(pix, axis=0) if len(pix) > 0 else None


def _extract_spectra(df_subset, band_mask=None):
    """
    Extract median spectrum for each ROI in df_subset.
    Returns (X, valid_idx_list): X.shape = (n_valid, bands_used); valid_idx_list = original df row indices.
    """
    spectra, valid_idx = [], []
    for idx, row in df_subset.iterrows():
        sp = _load_roi_median_spectrum(row.get(ROI_DATA_COL), row.get(ROI_EPI_MASK_COL))
        if sp is None:
            continue
        if band_mask is not None:
            sp = sp[band_mask]
        spectra.append(sp)
        valid_idx.append(idx)
    if not spectra:
        return np.empty((0, 0)), []
    return np.vstack(spectra), valid_idx


def _fdr_bh(p_values):
    """Benjamini-Hochberg FDR correction; returns q-value array (no external package)."""
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    rank = np.empty(n, dtype=np.float64)
    rank[order] = np.arange(1, n + 1)
    q = np.minimum(1.0, p * n / rank)
    # enforce monotonicity
    q_mono = q.copy()
    for i in range(n - 2, -1, -1):
        q_mono[order[i]] = min(q_mono[order[i]], q_mono[order[i + 1]])
    return q_mono


# --- load wavenumbers ---
if os.path.exists(WAVENUMBERS_PATH):
    wn = np.load(WAVENUMBERS_PATH)
    if WAVENUMBER_MAX_CM_2_1 is not None:
        band_mask_21 = wn < WAVENUMBER_MAX_CM_2_1
        if not np.any(band_mask_21):
            band_mask_21 = None
    else:
        band_mask_21 = None
    x_axis_21 = wn[band_mask_21] if band_mask_21 is not None else wn
    x_label_21 = "Wavenumber (cm\u207b\u00b9)"
else:
    wn = None
    band_mask_21 = None
    x_axis_21 = None
    x_label_21 = "Band index"
    print("2.1 Warning: wavenumbers.npy not found; x-axis will use Band index")

# --------------------------------------------------------------------------
# Step A: extract ROI median spectra for cohort-wide common_high / common_low
# --------------------------------------------------------------------------
print("\n--- 2.1 Step A: extract cohort-wide spectra ---")
X_high_all, idx_high_all = _extract_spectra(df_common_high, band_mask_21)
X_low_all, idx_low_all = _extract_spectra(df_common_low, band_mask_21)
n_high_sp, n_low_sp = X_high_all.shape[0], X_low_all.shape[0]
print(f"  common_high: {n_high_sp}/{len(df_common_high)} ROIs valid")
print(f"  common_low:  {n_low_sp}/{len(df_common_low)} ROIs valid")

if n_high_sp < 2 or n_low_sp < 2 or X_high_all.shape[1] == 0:
    print("  Insufficient valid ROIs; skipping Section 2.1.")
else:
    n_bands = X_high_all.shape[1]
    x_axis_21 = x_axis_21 if (x_axis_21 is not None and len(x_axis_21) == n_bands) else np.arange(n_bands)

    # --------------------------------------------------------------------------
    # Step B: full-spectrum L2 distance permutation test
    # --------------------------------------------------------------------------
    print(f"\n--- 2.1 Step B: full-spectrum permutation test (n_perm={N_PERMUTATIONS_2_1}) ---")
    mean_high_all = np.nanmean(X_high_all, axis=0)
    mean_low_all = np.nanmean(X_low_all, axis=0)
    obs_l2 = np.sqrt(np.nansum((mean_high_all - mean_low_all) ** 2))

    pooled = np.vstack([X_high_all, X_low_all])
    labels_perm = np.array([1] * n_high_sp + [0] * n_low_sp)
    rng_perm = np.random.default_rng(42)
    perm_dists = []
    for _ in range(N_PERMUTATIONS_2_1):
        perm = rng_perm.permutation(labels_perm)
        m1 = np.nanmean(pooled[perm == 1], axis=0)
        m0 = np.nanmean(pooled[perm == 0], axis=0)
        perm_dists.append(np.sqrt(np.nansum((m1 - m0) ** 2)))
    p_perm_21 = (np.sum(np.array(perm_dists) >= obs_l2) + 1) / (N_PERMUTATIONS_2_1 + 1)
    print(f"  Full-spectrum L2 distance = {obs_l2:.6f}, permutation p = {p_perm_21:.4f}")

    # --------------------------------------------------------------------------
    # Step C: per-band Mann-Whitney U + FDR, Cohen's d, output CSV
    # --------------------------------------------------------------------------
    print(f"\n--- 2.1 Step C: per-band test + FDR (alpha={FDR_ALPHA_2_1}) ---")
    from scipy import stats as _stats

    p_bands, d_bands, diff_bands = [], [], []
    for j in range(n_bands):
        a, b = X_high_all[:, j], X_low_all[:, j]
        _, p = _stats.mannwhitneyu(a, b, alternative="two-sided")
        p_bands.append(p)
        mh, ml = np.nanmean(a), np.nanmean(b)
        sh = np.nanstd(a, ddof=1) if len(a) > 1 else 0.0
        sl = np.nanstd(b, ddof=1) if len(b) > 1 else 0.0
        sp_d = np.sqrt(((len(a) - 1) * sh**2 + (len(b) - 1) * sl**2) / (len(a) + len(b) - 2)) if (len(a) + len(b)) > 2 else np.nan
        d_bands.append((mh - ml) / sp_d if (sp_d and sp_d > 0) else np.nan)
        diff_bands.append(mh - ml)

    p_bands = np.array(p_bands)
    q_bands = _fdr_bh(p_bands)
    d_bands = np.array(d_bands)
    diff_bands = np.array(diff_bands)
    n_sig = int(np.sum(q_bands <= FDR_ALPHA_2_1))
    print(f"  FDR-significant bands: {n_sig} / {n_bands}")

    band_result = pd.DataFrame({
        "band_idx": np.arange(n_bands),
        "wavenumber": x_axis_21,
        "mean_high": np.nanmean(X_high_all, axis=0),
        "mean_low": np.nanmean(X_low_all, axis=0),
        "mean_diff": diff_bands,
        "p_mw": p_bands,
        "q_fdr": q_bands,
        "cohens_d": d_bands,
        "significant": q_bands <= FDR_ALPHA_2_1,
    })
    csv_21 = os.path.join(OUTPUT_DIR, "step2_1c_spectral_test.csv")
    band_result.to_csv(csv_21, index=False, encoding="utf-8-sig")
    print(f"  Saved per-band results: {csv_21}")

    # --------------------------------------------------------------------------
    # Step C figure: mean spectrum comparison (+-1 std) + FDR colorbar + Cohen's d
    # --------------------------------------------------------------------------
    # colour definitions
    C_HIGH = (247/255, 105/255, 130/255)   # coral   — common_high
    C_LOW  = (62/255,  176/255, 167/255)   # teal    — common_low
    C_SIG  = (220/255, 50/255,  50/255)    # deep red — significant band colorbar

    fig_21 = plt.figure(figsize=(12, 7.5))
    # 3 rows: spectrum main (3 parts) / FDR colorbar (0.18) / Cohen's d (1 part)
    gs = fig_21.add_gridspec(3, 1, height_ratios=[3, 0.18, 1], hspace=0.05)
    ax_sp  = fig_21.add_subplot(gs[0])
    ax_bar = fig_21.add_subplot(gs[1], sharex=ax_sp)
    ax_d   = fig_21.add_subplot(gs[2], sharex=ax_sp)

    std_high = np.nanstd(X_high_all, axis=0, ddof=1)
    std_low  = np.nanstd(X_low_all,  axis=0, ddof=1)
    med_high = np.nanmedian(X_high_all, axis=0)
    med_low  = np.nanmedian(X_low_all,  axis=0)

    # spectrum plot
    ax_sp.fill_between(x_axis_21, med_high - std_high, med_high + std_high, alpha=0.22, color=C_HIGH)
    ax_sp.plot(x_axis_21, med_high, color=C_HIGH, linewidth=1.8, label=f"common_high (n={n_high_sp})")
    ax_sp.fill_between(x_axis_21, med_low - std_low, med_low + std_low, alpha=0.22, color=C_LOW)
    ax_sp.plot(x_axis_21, med_low,  color=C_LOW,  linewidth=1.8, label=f"common_low (n={n_low_sp})")
    ax_sp.set_ylabel("Median intensity")
    ax_sp.set_title(f"Step 2.1C  ROI median spectrum: common_high vs common_low\n"
                    f"(perm p={p_perm_21:.4f}, {n_sig} bands FDR<={FDR_ALPHA_2_1})")
    ax_sp.legend(fontsize=9)
    ax_sp.grid(True, alpha=0.3)
    plt.setp(ax_sp.get_xticklabels(), visible=False)

    # FDR colorbar: significant bands filled dark red, others light grey
    sig_mask = q_bands <= FDR_ALPHA_2_1
    bar_colors = [C_SIG if s else (0.88, 0.88, 0.88) for s in sig_mask]
    bw = (x_axis_21[1] - x_axis_21[0]) if len(x_axis_21) > 1 else 1
    ax_bar.bar(x_axis_21, np.ones(len(x_axis_21)), width=bw,
               color=bar_colors, edgecolor="none", linewidth=0)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_yticks([])
    ax_bar.set_ylabel(f"FDR\n<={FDR_ALPHA_2_1}", fontsize=7, rotation=0, labelpad=28, va="center")
    ax_bar.spines[["top", "left", "right"]].set_visible(False)
    plt.setp(ax_bar.get_xticklabels(), visible=False)

    # Cohen's d (significant bands opaque, non-significant translucent)
    d_sig_high   = np.where(sig_mask & (d_bands > 0),  d_bands, np.nan)
    d_sig_low    = np.where(sig_mask & (d_bands < 0),  d_bands, np.nan)
    d_nosig_high = np.where(~sig_mask & (d_bands > 0), d_bands, np.nan)
    d_nosig_low  = np.where(~sig_mask & (d_bands < 0), d_bands, np.nan)
    ax_d.bar(x_axis_21, d_sig_high,   width=bw, color=[C_HIGH], alpha=0.9, edgecolor="none")
    ax_d.bar(x_axis_21, d_sig_low,    width=bw, color=[C_LOW],  alpha=0.9, edgecolor="none")
    ax_d.bar(x_axis_21, d_nosig_high, width=bw, color=[C_HIGH], alpha=0.3, edgecolor="none")
    ax_d.bar(x_axis_21, d_nosig_low,  width=bw, color=[C_LOW],  alpha=0.3, edgecolor="none")
    ax_d.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_d.set_xlabel(x_label_21)
    ax_d.set_ylabel("Cohen's d")
    ax_d.grid(True, alpha=0.3)

    fig_21.tight_layout()
    path_21 = os.path.join(OUTPUT_DIR, f"step2_1c_spectral_common_high_vs_low.{FIG_FORMAT}")
    fig_21.savefig(path_21, **save_kw)
    plt.close(fig_21)
    print(f"  Saved spectrum comparison figure: {path_21}")

    # --------------------------------------------------------------------------
    # Step D: stratified validation per 1.3b subsets — directional consistency
    #   within each GG subset: both_high vs both_low, check direction agreement
    # --------------------------------------------------------------------------
    print(f"\n--- 2.1 Step D: stratified validation (directional consistency) ---")
    direction_rows = []
    heatmap_groups, heatmap_diffs_full = [], []   # full-wavenumber diff arrays for plotting
    for group in GROUP_ORDER:
        if group not in subset_both_high_low:
            continue
        d_sub = subset_both_high_low[group]
        X_sh, _ = _extract_spectra(d_sub["both_high"], band_mask_21)
        X_sl, _ = _extract_spectra(d_sub["both_low"], band_mask_21)
        if X_sh.shape[0] < 2 or X_sl.shape[0] < 2:
            print(f"  [{group}] skipped (both_high={X_sh.shape[0]}, both_low={X_sl.shape[0]}, insufficient ROIs)")
            continue
        mean_sh = np.nanmean(X_sh, axis=0)
        mean_sl = np.nanmean(X_sl, axis=0)
        diff_sub = mean_sh - mean_sl   # positive = high stronger; length = n_bands (full wavenumber range)
        if n_sig > 0:
            agree = np.mean(np.sign(diff_sub[sig_mask]) == np.sign(diff_bands[sig_mask]))
        else:
            agree = np.nan
        l2_sub = np.sqrt(np.nansum(diff_sub ** 2))
        direction_rows.append({
            "group": group,
            "n_both_high": X_sh.shape[0],
            "n_both_low": X_sl.shape[0],
            "L2_distance": round(l2_sub, 6),
            "directional_consistency_on_global_sig_bands": round(agree, 4) if not np.isnan(agree) else "N/A (no sig bands)",
        })
        heatmap_groups.append(group)
        heatmap_diffs_full.append(diff_sub)
        if not np.isnan(agree):
            print(f"  [{group}] n_high={X_sh.shape[0]}, n_low={X_sl.shape[0]}, "
                  f"L2={l2_sub:.4f}, directional_consistency={agree:.2%}")
        else:
            print(f"  [{group}] n_high={X_sh.shape[0]}, n_low={X_sl.shape[0]}, L2={l2_sub:.4f}, no significant bands")

    if direction_rows:
        dir_df = pd.DataFrame(direction_rows)
        dir_csv = os.path.join(OUTPUT_DIR, "step2_1d_direction_consistency.csv")
        dir_df.to_csv(dir_csv, index=False, encoding="utf-8-sig")
        print(f"  Saved stratified directional consistency: {dir_csv}")

    # --------------------------------------------------------------------------
    # Step D figure (scheme A): 4 rows sharing x-axis
    #   Row 1: global reference spectrum +/- std
    #   Row 2: FDR colorbar (anchor)
    #   Row 3: per-subset diff spectra overlaid (FDR-sig region shaded)
    #   Row 4: heatmap (full wavenumber range, pcolormesh)
    # --------------------------------------------------------------------------
    if not heatmap_groups or not np.any(sig_mask):
        print("  Step D combined figure skipped: no valid subsets or no significant bands.")
    else:
        from matplotlib.colors import LinearSegmentedColormap

        n_hm_rows = len(heatmap_groups)
        hm_height = max(1.5, n_hm_rows * 0.55)

        fig_D = plt.figure(figsize=(14, 7.5 + hm_height))
        gs_D = fig_D.add_gridspec(
            4, 1,
            height_ratios=[3, 0.15, 2.5, hm_height],
            hspace=0.06,
        )
        ax_ref  = fig_D.add_subplot(gs_D[0])
        ax_fdr  = fig_D.add_subplot(gs_D[1], sharex=ax_ref)
        ax_diff = fig_D.add_subplot(gs_D[2], sharex=ax_ref)
        ax_hm2  = fig_D.add_subplot(gs_D[3], sharex=ax_ref)

        # row 1: global reference spectrum +/- std
        ax_ref.fill_between(x_axis_21, med_high - std_high, med_high + std_high, alpha=0.22, color=C_HIGH)
        ax_ref.plot(x_axis_21, med_high, color=C_HIGH, linewidth=1.8, label=f"common_high (n={n_high_sp})")
        ax_ref.fill_between(x_axis_21, med_low - std_low, med_low + std_low, alpha=0.22, color=C_LOW)
        ax_ref.plot(x_axis_21, med_low,  color=C_LOW,  linewidth=1.8, label=f"common_low (n={n_low_sp})")
        ax_ref.set_ylabel("Median intensity")
        ax_ref.set_title(
            f"Step 2.1D  Stratified validation - GG-subset directional consistency\n"
            f"Row 1: Global reference  |  Row 2: FDR colorbar  |  Row 3: Per-subset diff  |  Row 4: Heatmap\n"
            f"(global FDR-sig bands: {n_sig}  |  subgroups shown: {n_hm_rows})"
        )
        ax_ref.legend(fontsize=9, loc="upper right")
        ax_ref.grid(True, alpha=0.3)
        plt.setp(ax_ref.get_xticklabels(), visible=False)

        # row 2: FDR colorbar (reuse Step C style)
        ax_fdr.bar(x_axis_21, np.ones(len(x_axis_21)), width=bw,
                   color=bar_colors, edgecolor="none", linewidth=0)
        ax_fdr.set_ylim(0, 1)
        ax_fdr.set_yticks([])
        ax_fdr.set_ylabel(f"FDR\n<={FDR_ALPHA_2_1}", fontsize=7, rotation=0, labelpad=32, va="center")
        ax_fdr.spines[["top", "left", "right"]].set_visible(False)
        plt.setp(ax_fdr.get_xticklabels(), visible=False)

        # row 3: per-subset diff spectra overlaid
        group_palette = dict(zip(GROUP_ORDER, sns.color_palette("husl", len(GROUP_ORDER))))
        # shade FDR-significant regions as continuous blocks (minimise axvspan calls)
        sig_idx = np.where(sig_mask)[0]
        if len(sig_idx) > 0:
            blocks, s = [], sig_idx[0]
            for k in range(1, len(sig_idx)):
                if sig_idx[k] != sig_idx[k - 1] + 1:
                    blocks.append((s, sig_idx[k - 1]))
                    s = sig_idx[k]
            blocks.append((s, sig_idx[-1]))
            for s_i, e_i in blocks:
                ax_diff.axvspan(x_axis_21[s_i] - bw / 2, x_axis_21[e_i] + bw / 2,
                                color=C_SIG, alpha=0.10, linewidth=0, zorder=0)
        ax_diff.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=2)
        for group, diff_full in zip(heatmap_groups, heatmap_diffs_full):
            c = group_palette.get(group, "gray")
            ax_diff.plot(x_axis_21, diff_full, color=c, linewidth=1.3, alpha=0.85,
                         label=group, zorder=3)
        ax_diff.set_ylabel("\u0394 intensity\n(both_high\u2212both_low)")
        ax_diff.legend(fontsize=8, ncol=min(n_hm_rows, 4), loc="upper right")
        ax_diff.grid(True, alpha=0.2)
        plt.setp(ax_diff.get_xticklabels(), visible=False)

        # row 4: heatmap (full wavenumber range, pcolormesh)
        cmap_div = LinearSegmentedColormap.from_list(
            "teal_white_coral", [C_LOW, (1.0, 1.0, 1.0), C_HIGH], N=256
        )
        H_full = np.vstack(heatmap_diffs_full)   # (n_hm_rows, n_bands)
        vmax2 = np.nanmax(np.abs(H_full))
        vmax2 = vmax2 if vmax2 > 0 else 1.0

        # pcolormesh needs (n_bands+1,) edge vector
        wn_edges = np.concatenate([
            [x_axis_21[0] - bw / 2],
            (x_axis_21[:-1] + x_axis_21[1:]) / 2,
            [x_axis_21[-1] + bw / 2],
        ])
        pcm = ax_hm2.pcolormesh(
            wn_edges, np.arange(n_hm_rows + 1),
            H_full, cmap=cmap_div, vmin=-vmax2, vmax=vmax2, shading="flat"
        )
        # y-axis labels: GG name + ROI counts (avoid overlap with colorbar)
        ytick_labels = []
        for group in heatmap_groups:
            d_sub_tmp = subset_both_high_low[group]
            X_sh_tmp, _ = _extract_spectra(d_sub_tmp["both_high"], band_mask_21)
            X_sl_tmp, _ = _extract_spectra(d_sub_tmp["both_low"],  band_mask_21)
            ytick_labels.append(f"{group}  (H={X_sh_tmp.shape[0]}, L={X_sl_tmp.shape[0]})")
        ax_hm2.set_yticks(np.arange(n_hm_rows) + 0.5)
        ax_hm2.set_yticklabels(ytick_labels, fontsize=9)
        ax_hm2.set_xlabel(x_label_21)
        ax_hm2.set_ylabel("GG subgroup")
        cbar2 = fig_D.colorbar(pcm, ax=ax_hm2, fraction=0.02, pad=0.01)
        cbar2.set_label("mean diff (both_high\u2212both_low)", fontsize=8)

        fig_D.tight_layout()
        path_D = os.path.join(OUTPUT_DIR, f"step2_1d_direction_by_group.{FIG_FORMAT}")
        fig_D.savefig(path_D, **save_kw)
        plt.close(fig_D)
        print(f"  Saved Step D combined figure: {path_D}")

print("\nSection 2.1 complete.")

# ==============================================================================
# Section 2.2: Hypoxia analysis within GG4 — high risk vs non-high risk
#   Step A: _common / PIMO / GLUT1 distributions within GG4 (coloured by risk pattern)
#   Step B: Statistical comparison (Mann-Whitney U) + visualisation
# ==============================================================================
print("\n" + "=" * 70)
print("Section 2.2: GG4 hypoxia analysis - high risk vs non-high risk")
print("=" * 70)

_df_gg4 = df[df["_group"].isin(["GG4-high risk", "GG4-non high risk"])].copy()
_n_gg4  = len(_df_gg4)
print(f"  GG4 ROI total: {_n_gg4}")
if _n_gg4 == 0:
    print("  No GG4 data; skipping Section 2.2.")
else:
    import matplotlib.pyplot as _plt22
    from scipy import stats as _stats22

    _HR_COLOR  = (220/255, 80/255,  60/255)   # deep red-orange — GG4-high risk
    _NHR_COLOR = (62/255, 130/255, 180/255)   # steel blue      — GG4-non high risk
    _palette22 = {
        "GG4-high risk":     _HR_COLOR,
        "GG4-non high risk": _NHR_COLOR,
    }
    _order22 = ["GG4-non high risk", "GG4-high risk"]

    _metrics22 = [
        (PIMO_COL,  "PIMO H-score"),
        (GLUT1_COL, "GLUT1 H-score"),
        ("_common",  "Common hypoxia index\n(geometric mean)"),
    ]

    # --------------------------------------------------------------------------
    # Step A: distribution plot (histogram + KDE, 3 columns, coloured by risk pattern)
    # --------------------------------------------------------------------------
    print("\n--- 2.2 Step A: GG4 hypoxia distribution plot ---")

    import seaborn as _sns22
    fig_22a, axes_22a = _plt22.subplots(1, 3, figsize=(13, 4))

    for _ax22, (_col22, _lbl22) in zip(axes_22a, _metrics22):
        for _grp22 in _order22:
            _vals22 = _df_gg4.loc[_df_gg4["_group"] == _grp22, _col22].dropna().values
            _c22    = _palette22[_grp22]
            if len(_vals22) == 0:
                continue
            _ax22.hist(_vals22, bins=18, density=True, alpha=0.38,
                       color=_c22, edgecolor="none", label=None)
            _sns22.kdeplot(_vals22, ax=_ax22, color=_c22, linewidth=2,
                           label=f"{_grp22} (n={len(_vals22)})")
        _ax22.set_xlabel(_lbl22)
        _ax22.set_ylabel("Density")
        _ax22.legend(fontsize=8)
        _ax22.grid(True, alpha=0.3)

    axes_22a[1].set_title("Step 2.2A  GG4: hypoxia distribution by risk pattern", pad=8)
    fig_22a.tight_layout()
    _path_22a = os.path.join(OUTPUT_DIR, f"step2_2a_gg4_hypoxia_distribution.{FIG_FORMAT}")
    fig_22a.savefig(_path_22a, **save_kw)
    _plt22.close(fig_22a)
    print(f"  Saved distribution plot: {_path_22a}")

    # --------------------------------------------------------------------------
    # Step B: statistical comparison (Mann-Whitney) + violin/box + stripplot
    # --------------------------------------------------------------------------
    print("\n--- 2.2 Step B: GG4 high risk vs non-high risk statistical comparison ---")

    _stat_rows22 = []
    for _col22, _lbl22 in _metrics22:
        _a22 = _df_gg4.loc[_df_gg4["_group"] == "GG4-high risk",     _col22].dropna().values
        _b22 = _df_gg4.loc[_df_gg4["_group"] == "GG4-non high risk", _col22].dropna().values
        if len(_a22) < 2 or len(_b22) < 2:
            _p22 = np.nan
        else:
            _, _p22 = _stats22.mannwhitneyu(_a22, _b22, alternative="two-sided")
        _stat_rows22.append({
            "metric":        _lbl22.replace("\n", " "),
            "n_high_risk":   len(_a22),
            "median_high":   float(np.nanmedian(_a22)) if len(_a22) else np.nan,
            "n_non_high":    len(_b22),
            "median_non":    float(np.nanmedian(_b22)) if len(_b22) else np.nan,
            "p_mannwhitney": round(_p22, 6) if not np.isnan(_p22) else "N/A",
        })
        _tag = f"p={_p22:.4f}" if not np.isnan(_p22) else "N/A"
        print(f"  {_lbl22.split(chr(10))[0]:30s}  "
              f"HR median={np.nanmedian(_a22):.2f}  NHR median={np.nanmedian(_b22):.2f}  {_tag}")

    pd.DataFrame(_stat_rows22).to_csv(
        os.path.join(OUTPUT_DIR, "step2_2b_gg4_hypoxia_stats.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("  Saved stats: step2_2b_gg4_hypoxia_stats.csv")

    # visualisation: violin + stripplot, 3 columns
    fig_22b, axes_22b = _plt22.subplots(1, 3, figsize=(13, 5))
    for _ax22, (_col22, _lbl22), _row22 in zip(axes_22b, _metrics22, _stat_rows22):
        _sns22.violinplot(
            data=_df_gg4, x="_group", y=_col22,
            order=_order22, palette=_palette22,
            inner=None, linewidth=1.2, ax=_ax22
        )
        _sns22.stripplot(
            data=_df_gg4, x="_group", y=_col22,
            order=_order22, palette=_palette22,
            size=4, alpha=0.7, jitter=True, ax=_ax22
        )
        # median line
        for _xi, _grp22 in enumerate(_order22):
            _med22 = _df_gg4.loc[_df_gg4["_group"] == _grp22, _col22].median()
            if not np.isnan(_med22):
                _ax22.hlines(_med22, _xi - 0.35, _xi + 0.35,
                             colors="black", linewidth=1.5, linestyle="--")
        _p_val22 = _row22["p_mannwhitney"]
        _p_str22 = f"p={_p_val22:.4f}" if isinstance(_p_val22, float) else f"p={_p_val22}"
        _ax22.set_title(f"Step 2.2B  {_lbl22.split(chr(10))[0]}\n{_p_str22}", fontsize=9)
        _ax22.set_xlabel("")
        _ax22.set_ylabel(_lbl22)
        _ax22.tick_params(axis="x", rotation=15)
        _ax22.set_ylim(bottom=0)
        _ax22.grid(True, axis="y", alpha=0.3)

    fig_22b.tight_layout()
    _path_22b = os.path.join(OUTPUT_DIR, f"step2_2b_gg4_hypoxia_comparison.{FIG_FORMAT}")
    fig_22b.savefig(_path_22b, **save_kw)
    _plt22.close(fig_22b)
    print(f"  Saved comparison figure: {_path_22b}")

    # --------------------------------------------------------------------------
    # Step C: baseline spectrum comparison across all GG4 — high risk vs non-high risk
    #   All GG4 ROIs (not restricted to both_high/low); per-band MWU + FDR + Cohen's d
    # --------------------------------------------------------------------------
    _prereqs_22c = (
        "band_mask_21" in dir()
        and "x_axis_21" in dir()
        and x_axis_21 is not None
    )
    if not _prereqs_22c:
        print("\n  Step C/D skipped: Section 2.1 must run first (wavenumber array not loaded).")
    else:
        print("\n--- 2.2 Step C: GG4 baseline spectrum HR vs NHR ---")
        _df_hr  = _df_gg4[_df_gg4["_group"] == "GG4-high risk"]
        _df_nhr = _df_gg4[_df_gg4["_group"] == "GG4-non high risk"]

        _X_hr,  _ = _extract_spectra(_df_hr,  band_mask_21)
        _X_nhr, _ = _extract_spectra(_df_nhr, band_mask_21)
        _n_hr, _n_nhr = _X_hr.shape[0], _X_nhr.shape[0]
        print(f"  GG4-high risk ROIs: {_n_hr}  |  GG4-non high risk ROIs: {_n_nhr}")

        if _n_hr < 2 or _n_nhr < 2:
            print("  Insufficient valid ROIs; skipping Step C.")
        else:
            _nb_c = _X_hr.shape[1]
            _bw_c = (x_axis_21[1] - x_axis_21[0]) if _nb_c > 1 else 1

            _p_c, _d_c, _diff_c = [], [], []
            for _j in range(_nb_c):
                _a, _b = _X_hr[:, _j], _X_nhr[:, _j]
                _, _pv = _stats22.mannwhitneyu(_a, _b, alternative="two-sided")
                _p_c.append(_pv)
                _mh, _ml = np.nanmean(_a), np.nanmean(_b)
                _sh = np.nanstd(_a, ddof=1) if _n_hr  > 1 else 0.0
                _sl = np.nanstd(_b, ddof=1) if _n_nhr > 1 else 0.0
                _sp = np.sqrt(((_n_hr-1)*_sh**2 + (_n_nhr-1)*_sl**2) / (_n_hr+_n_nhr-2))
                _d_c.append((_mh - _ml) / _sp if _sp > 0 else np.nan)
                _diff_c.append(_mh - _ml)

            _p_c    = np.array(_p_c)
            _q_c    = _fdr_bh(_p_c)
            _d_c    = np.array(_d_c)
            _diff_c = np.array(_diff_c)
            _sig_c  = _q_c <= FDR_ALPHA_2_1
            _nsig_c = int(np.sum(_sig_c))
            print(f"  FDR-sig bands: {_nsig_c} / {_nb_c}")

            # save CSV
            pd.DataFrame({
                "wavenumber": x_axis_21, "mean_diff": _diff_c,
                "cohens_d": _d_c, "p_mw": _p_c,
                "q_fdr": _q_c, "significant": _sig_c,
            }).to_csv(os.path.join(OUTPUT_DIR, "step2_2c_gg4_hr_vs_nhr_spectral.csv"),
                      index=False, encoding="utf-8-sig")

            # figure: median spectrum +/- std / FDR colorbar / Cohen's d (3 rows, same style as 2.1C)
            _med_hr  = np.nanmedian(_X_hr,  axis=0)
            _med_nhr = np.nanmedian(_X_nhr, axis=0)
            _std_hr  = np.nanstd(_X_hr,  axis=0, ddof=1)
            _std_nhr = np.nanstd(_X_nhr, axis=0, ddof=1)

            fig_c = _plt22.figure(figsize=(12, 7.5))
            _gs_c = fig_c.add_gridspec(3, 1, height_ratios=[3, 0.18, 1], hspace=0.05)
            _ax_sp_c  = fig_c.add_subplot(_gs_c[0])
            _ax_bar_c = fig_c.add_subplot(_gs_c[1], sharex=_ax_sp_c)
            _ax_d_c   = fig_c.add_subplot(_gs_c[2], sharex=_ax_sp_c)

            _ax_sp_c.fill_between(x_axis_21, _med_hr  - _std_hr,  _med_hr  + _std_hr,
                                   alpha=0.22, color=_HR_COLOR)
            _ax_sp_c.plot(x_axis_21, _med_hr,  color=_HR_COLOR,  linewidth=1.8,
                          label=f"GG4-high risk (n={_n_hr})")
            _ax_sp_c.fill_between(x_axis_21, _med_nhr - _std_nhr, _med_nhr + _std_nhr,
                                   alpha=0.22, color=_NHR_COLOR)
            _ax_sp_c.plot(x_axis_21, _med_nhr, color=_NHR_COLOR, linewidth=1.8,
                          label=f"GG4-non high risk (n={_n_nhr})")
            _ax_sp_c.set_ylabel("Median intensity")
            _ax_sp_c.set_title(
                f"Step 2.2C  GG4 baseline spectrum: high risk vs non-high risk\n"
                f"(FDR-sig bands: {_nsig_c}/{_nb_c})"
            )
            _ax_sp_c.legend(fontsize=9)
            _ax_sp_c.grid(True, alpha=0.3)
            _plt22.setp(_ax_sp_c.get_xticklabels(), visible=False)

            _bc_c = [_HR_COLOR if _s else (0.88, 0.88, 0.88) for _s in _sig_c]
            _ax_bar_c.bar(x_axis_21, np.ones(_nb_c), width=_bw_c,
                          color=_bc_c, edgecolor="none", linewidth=0)
            _ax_bar_c.set_ylim(0, 1)
            _ax_bar_c.set_yticks([])
            _ax_bar_c.set_ylabel(f"FDR\n<={FDR_ALPHA_2_1}", fontsize=7,
                                  rotation=0, labelpad=28, va="center")
            _ax_bar_c.spines[["top", "left", "right"]].set_visible(False)
            _plt22.setp(_ax_bar_c.get_xticklabels(), visible=False)

            _dsh_c = np.where(_sig_c  & (_d_c > 0), _d_c, np.nan)
            _dsl_c = np.where(_sig_c  & (_d_c < 0), _d_c, np.nan)
            _dnh_c = np.where(~_sig_c & (_d_c > 0), _d_c, np.nan)
            _dnl_c = np.where(~_sig_c & (_d_c < 0), _d_c, np.nan)
            _ax_d_c.bar(x_axis_21, _dsh_c, width=_bw_c, color=[_HR_COLOR],  alpha=0.9, edgecolor="none")
            _ax_d_c.bar(x_axis_21, _dsl_c, width=_bw_c, color=[_NHR_COLOR], alpha=0.9, edgecolor="none")
            _ax_d_c.bar(x_axis_21, _dnh_c, width=_bw_c, color=[_HR_COLOR],  alpha=0.3, edgecolor="none")
            _ax_d_c.bar(x_axis_21, _dnl_c, width=_bw_c, color=[_NHR_COLOR], alpha=0.3, edgecolor="none")
            _ax_d_c.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            _ax_d_c.set_xlabel(x_label_21)
            _ax_d_c.set_ylabel("Cohen's d\n(HR \u2212 NHR)")
            _ax_d_c.grid(True, alpha=0.3)

            fig_c.tight_layout()
            _path_c = os.path.join(OUTPUT_DIR, f"step2_2c_gg4_hr_vs_nhr_spectrum.{FIG_FORMAT}")
            fig_c.savefig(_path_c, **save_kw)
            _plt22.close(fig_c)
            print(f"  Saved baseline spectrum comparison: {_path_c}")

        # --------------------------------------------------------------------------
        # Step D: GG4 subgroup hypoxia diff spectra overlaid
        #   Two diff spectrum lines (HR / NHR) + global diff_bands reference, shared x-axis
        # --------------------------------------------------------------------------
        print("\n--- 2.2 Step D: GG4 subgroup hypoxia diff spectra overlaid ---")

        _has_global_diff = "diff_bands" in dir() and diff_bands is not None
        _diff_lines = {}   # group -> diff array (both_high mean - both_low mean)

        # per-subgroup diff spectra
        _Xsh_pool, _Xsl_pool = [], []   # used to combine full GG4
        for _grp_d in ["GG4-high risk", "GG4-non high risk"]:
            if _grp_d not in subset_both_high_low:
                print(f"  [{_grp_d}] not in subset_both_high_low; skipping")
                continue
            _Xsh_d, _ = _extract_spectra(subset_both_high_low[_grp_d]["both_high"], band_mask_21)
            _Xsl_d, _ = _extract_spectra(subset_both_high_low[_grp_d]["both_low"],  band_mask_21)
            if _Xsh_d.shape[0] < 2 or _Xsl_d.shape[0] < 2:
                print(f"  [{_grp_d}] insufficient ROIs; skipping")
                continue
            _diff_lines[_grp_d] = np.nanmean(_Xsh_d, axis=0) - np.nanmean(_Xsl_d, axis=0)
            _Xsh_pool.append(_Xsh_d)
            _Xsl_pool.append(_Xsl_d)

        # GG4 combined diff (HR + NHR both_high / both_low pooled, then mean difference)
        if len(_Xsh_pool) > 0 and len(_Xsl_pool) > 0:
            _Xsh_gg4 = np.vstack(_Xsh_pool)
            _Xsl_gg4 = np.vstack(_Xsl_pool)
            _diff_lines["GG4 combined"] = (
                np.nanmean(_Xsh_gg4, axis=0) - np.nanmean(_Xsl_gg4, axis=0)
            )
            print(f"  GG4 combined: n_high={_Xsh_gg4.shape[0]}, n_low={_Xsl_gg4.shape[0]}")

        if len(_diff_lines) == 0:
            print("  No valid subgroups; skipping Step D.")
        else:
            _bw_d = (x_axis_21[1] - x_axis_21[0]) if len(x_axis_21) > 1 else 1

            fig_d = _plt22.figure(figsize=(12, 5))
            _gs_d = fig_d.add_gridspec(2, 1, height_ratios=[3, 0.15], hspace=0.05)
            _ax_diff_d = fig_d.add_subplot(_gs_d[0])
            _ax_bar_d  = fig_d.add_subplot(_gs_d[1], sharex=_ax_diff_d)

            # global diff_bands reference line (grey dashed)
            if _has_global_diff:
                _ax_diff_d.plot(x_axis_21, diff_bands, color="0.65", linewidth=1.2,
                                linestyle="--", label="Global diff (2.1C)", zorder=2)

            # shade FDR-significant regions (using global sig_mask)
            if _has_global_diff and "sig_mask" in dir():
                _sig_idx_d = np.where(sig_mask)[0]
                if len(_sig_idx_d) > 0:
                    _blk_d, _s_d = [], _sig_idx_d[0]
                    for _k in range(1, len(_sig_idx_d)):
                        if _sig_idx_d[_k] != _sig_idx_d[_k-1] + 1:
                            _blk_d.append((_s_d, _sig_idx_d[_k-1]))
                            _s_d = _sig_idx_d[_k]
                    _blk_d.append((_s_d, _sig_idx_d[-1]))
                    for _si, _ei in _blk_d:
                        _ax_diff_d.axvspan(x_axis_21[_si] - _bw_d/2,
                                           x_axis_21[_ei] + _bw_d/2,
                                           color=(220/255, 50/255, 50/255), alpha=0.08,
                                           linewidth=0, zorder=0)

            _ax_diff_d.axhline(0, color="gray", linewidth=0.8, linestyle=":", zorder=1)
            _colors_d = {
                "GG4-high risk":     _HR_COLOR,
                "GG4-non high risk": _NHR_COLOR,
                "GG4 combined":      (0.4, 0.4, 0.4),   # dark grey — combined GG4
            }
            _styles_d = {
                "GG4-high risk":     (1.8, "solid"),
                "GG4-non high risk": (1.8, "solid"),
                "GG4 combined":      (2.2, "dashdot"),   # thicker dash-dot to highlight combined
            }
            for _grp_d, _dline in _diff_lines.items():
                _lw_d, _ls_d = _styles_d.get(_grp_d, (1.8, "solid"))
                _ax_diff_d.plot(x_axis_21, _dline, color=_colors_d[_grp_d],
                                linewidth=_lw_d, linestyle=_ls_d,
                                label=f"{_grp_d} (both_high\u2212both_low)",
                                zorder=3)

            _ax_diff_d.set_ylabel("\u0394 intensity (both_high \u2212 both_low)")
            _ax_diff_d.set_title(
                "Step 2.2D  GG4 subgroup: hypoxia diff spectrum\n"
                "Shaded = global FDR-sig bands (2.1C reference)"
            )
            _ax_diff_d.legend(fontsize=9, loc="upper right")
            _ax_diff_d.grid(True, alpha=0.3)
            _plt22.setp(_ax_diff_d.get_xticklabels(), visible=False)

            # FDR colorbar (global significant bands reference)
            if _has_global_diff and "sig_mask" in dir():
                _bc_d = [(220/255, 50/255, 50/255) if _s else (0.88, 0.88, 0.88)
                         for _s in sig_mask]
                _ax_bar_d.bar(x_axis_21, np.ones(len(x_axis_21)), width=_bw_d,
                              color=_bc_d, edgecolor="none", linewidth=0)
            _ax_bar_d.set_ylim(0, 1)
            _ax_bar_d.set_yticks([])
            _ax_bar_d.set_ylabel("Global\nFDR sig", fontsize=7,
                                  rotation=0, labelpad=36, va="center")
            _ax_bar_d.spines[["top", "left", "right"]].set_visible(False)
            _ax_bar_d.set_xlabel(x_label_21)

            fig_d.tight_layout()
            _path_d = os.path.join(OUTPUT_DIR, f"step2_2d_gg4_hypoxia_diff_spectra.{FIG_FORMAT}")
            fig_d.savefig(_path_d, **save_kw)
            _plt22.close(fig_d)
            print(f"  Saved diff spectra overlay figure: {_path_d}")

print("\nSection 2.2 complete.")
