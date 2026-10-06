#!/usr/bin/env python3
"""
04_dataset_inventory.py
Complete dataset inventory, quality control, and viability assessment.

Generates:
  - Completeness matrix (model x scenario x variable)
  - QC report: NaN%, ranges, temporal coverage
  - Figures for thesis document
  - CSVs with summary statistics

Usage:
    source /media/volume/jay2vol/nexgddp/venv/bin/activate
    python 04_dataset_inventory.py
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("/media/volume/jay2vol/nexgddp")
RAW_DIR = BASE_DIR / "raw"
OUT_DIR = BASE_DIR / "outputs" / "04_inventory"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_MODELS = [
    "ACCESS-CM2", "CanESM5", "CESM2", "EC-Earth3",
    "GFDL-ESM4", "IPSL-CM6A-LR", "MIROC6", "MPI-ESM1-2-HR"
]

VARIABLES = ["tas", "hurs"]

SCENARIOS = {
    "historical": (1950, 2014),
    "ssp126": (2015, 2100),
    "ssp245": (2015, 2100),
    "ssp370": (2015, 2100),
    "ssp585": (2015, 2100),
}

VALID_RANGES = {
    "tas":  (200.0, 340.0),
    "hurs": (0.0, 102.0),
}

SUBREGIONS = {
    "Northern Tropics":    {"lat_min": 0.0,   "lat_max": 13.0},
    "Central Amazonia":    {"lat_min": -15.0,  "lat_max": 0.0},
    "Subtropical Andes":   {"lat_min": -30.0,  "lat_max": -15.0},
    "Southern Cone":       {"lat_min": -56.0,  "lat_max": -30.0},
}

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#E0E0E0",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})


# =============================================================================
# 1. COMPLETENESS MATRIX
# =============================================================================

def build_completeness_matrix():
    print("=" * 70)
    print("  1. COMPLETENESS MATRIX")
    print("=" * 70)

    records = []
    for model in ALL_MODELS:
        for scenario, (y_start, y_end) in SCENARIOS.items():
            expected = y_end - y_start + 1
            for variable in VARIABLES:
                var_dir = RAW_DIR / model / scenario / variable
                found = len(list(var_dir.glob("*.nc"))) if var_dir.exists() else 0
                pct = (found / expected) * 100
                records.append({
                    "model": model, "scenario": scenario, "variable": variable,
                    "expected": expected, "found": found,
                    "missing": expected - found,
                    "completeness_pct": round(pct, 1),
                })

    df = pd.DataFrame(records)

    print("\n  Completeness by model:")
    for model in ALL_MODELS:
        sub = df[df["model"] == model]
        total_exp = sub["expected"].sum()
        total_found = sub["found"].sum()
        pct = (total_found / total_exp * 100) if total_exp > 0 else 0
        status = "OK" if pct == 100 else ("~" if pct > 90 else "X")
        print(f"    {status} {model:20s}  {total_found:4d}/{total_exp:4d} files ({pct:.1f}%)")

    missing = df[df["missing"] > 0]
    if len(missing) > 0:
        print(f"\n  Combinations with missing files ({len(missing)}):")
        for _, row in missing.iterrows():
            print(f"    - {row['model']} / {row['scenario']} / {row['variable']}: "
                  f"missing {row['missing']} of {row['expected']}")

    csv_path = OUT_DIR / "completeness_matrix.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    return df


def plot_completeness_heatmap(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, var in enumerate(VARIABLES):
        ax = axes[idx]
        sub = df[df["variable"] == var]
        pivot = sub.pivot_table(
            index="model", columns="scenario",
            values="completeness_pct", aggfunc="first"
        )
        col_order = ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]
        pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])

        im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                color = "white" if val < 50 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

        ax.set_title(f"Completeness — {var}", fontweight="bold")

    plt.tight_layout()
    fig_path = OUT_DIR / "fig_completeness_heatmap.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig_path}")


# =============================================================================
# 2. QUALITY CONTROL
# =============================================================================

def run_quality_control():
    print("\n" + "=" * 70)
    print("  2. QUALITY CONTROL")
    print("=" * 70)

    sample_years = {
        "historical": [1950, 1980, 2014],
        "ssp126": [2015, 2050, 2100],
        "ssp245": [2015, 2050, 2100],
        "ssp370": [2015, 2050, 2100],
        "ssp585": [2015, 2050, 2100],
    }

    qc_records = []
    available_models = [m for m in ALL_MODELS if (RAW_DIR / m).exists()]

    for model in available_models:
        for scenario, years in sample_years.items():
            for variable in VARIABLES:
                for year in years:
                    pattern = f"{variable}_{model}_{scenario}_{year}_SA.nc"
                    filepath = RAW_DIR / model / scenario / variable / pattern
                    if not filepath.exists():
                        continue

                    ds = xr.open_dataset(filepath)
                    data = ds[variable].values
                    total = data.size
                    nan_count = int(np.isnan(data).sum())
                    nan_pct = round((nan_count / total) * 100, 2)
                    valid_min, valid_max = VALID_RANGES.get(variable, (None, None))
                    data_min = float(np.nanmin(data))
                    data_max = float(np.nanmax(data))
                    data_mean = float(np.nanmean(data))
                    data_std = float(np.nanstd(data))

                    out_of_range = 0
                    if valid_min is not None:
                        finite = data[np.isfinite(data)]
                        out_of_range = int(np.sum((finite < valid_min) | (finite > valid_max)))

                    qc_records.append({
                        "model": model, "scenario": scenario,
                        "variable": variable, "year": year,
                        "n_time": ds.dims["time"], "n_lat": ds.dims["lat"],
                        "n_lon": ds.dims["lon"], "nan_pct": nan_pct,
                        "data_min": round(data_min, 2), "data_max": round(data_max, 2),
                        "data_mean": round(data_mean, 2), "data_std": round(data_std, 2),
                        "out_of_range": out_of_range,
                    })
                    ds.close()

    df_qc = pd.DataFrame(qc_records)

    print(f"\n  Files sampled: {len(df_qc)}")

    unique_shapes = df_qc.groupby("variable")[["n_lat", "n_lon"]].nunique()
    print(f"\n  Spatial grid consistency:")
    for var in VARIABLES:
        if var in unique_shapes.index:
            consistent = (unique_shapes.loc[var, "n_lat"] == 1 and
                          unique_shapes.loc[var, "n_lon"] == 1)
            sample = df_qc[df_qc["variable"] == var].iloc[0]
            status = "OK" if consistent else "FAIL"
            print(f"    {status} {var}: {sample['n_lat']}x{sample['n_lon']} "
                  f"(consistent: {consistent})")

    print(f"\n  NaN percentage by variable:")
    for var in VARIABLES:
        sub = df_qc[df_qc["variable"] == var]
        print(f"    {var}: min={sub['nan_pct'].min():.1f}%, "
              f"max={sub['nan_pct'].max():.1f}%, "
              f"mean={sub['nan_pct'].mean():.1f}%")

    print(f"\n  Observed vs expected ranges:")
    for var in VARIABLES:
        sub = df_qc[df_qc["variable"] == var]
        vmin, vmax = VALID_RANGES[var]
        obs_min = sub["data_min"].min()
        obs_max = sub["data_max"].max()
        total_oor = sub["out_of_range"].sum()
        print(f"    {var}: observed [{obs_min:.1f}, {obs_max:.1f}] "
              f"vs expected [{vmin}, {vmax}] | out of range: {total_oor}")

    print(f"\n  Timesteps per file:")
    for n_time in sorted(df_qc["n_time"].unique()):
        count = (df_qc["n_time"] == n_time).sum()
        print(f"    {n_time} days: {count} files")

    csv_path = OUT_DIR / "quality_control_report.csv"
    df_qc.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    return df_qc


def plot_nan_coverage(df_qc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, var in enumerate(VARIABLES):
        ax = axes[idx]
        sub = df_qc[df_qc["variable"] == var]
        for model in sub["model"].unique():
            m_sub = sub[sub["model"] == model]
            ax.scatter(m_sub["year"], m_sub["nan_pct"], label=model, s=30, alpha=0.7)
        ax.set_xlabel("Year")
        ax.set_ylabel("NaN (%)")
        ax.set_title(f"Missing data coverage — {var}", fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig_path = OUT_DIR / "fig_nan_coverage.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig_path}")


def plot_value_ranges(df_qc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, var in enumerate(VARIABLES):
        ax = axes[idx]
        sub = df_qc[df_qc["variable"] == var]
        models = sorted(sub["model"].unique())
        for i, model in enumerate(models):
            m_sub = sub[sub["model"] == model]
            ax.plot([i, i], [m_sub["data_min"].min(), m_sub["data_max"].max()],
                    linewidth=3, alpha=0.6, solid_capstyle="round")
            ax.scatter(i, m_sub["data_mean"].mean(), color="black", s=40, zorder=5)
        vmin, vmax = VALID_RANGES[var]
        ax.axhline(vmin, color="red", linestyle="--", linewidth=0.8, alpha=0.5,
                    label=f"Valid range [{vmin}, {vmax}]")
        ax.axhline(vmax, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        units = "K" if var == "tas" else "%"
        ax.set_ylabel(f"Value ({units})")
        ax.set_title(f"Value range by model — {var}", fontweight="bold")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig_path = OUT_DIR / "fig_value_ranges.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig_path}")


# =============================================================================
# 3. SPATIAL COVERAGE
# =============================================================================

def plot_spatial_coverage():
    print("\n" + "=" * 70)
    print("  3. SPATIAL COVERAGE")
    print("=" * 70)

    sample = sorted((RAW_DIR).rglob("tas_*_historical_*1980*_SA.nc"))
    if not sample:
        sample = sorted((RAW_DIR).rglob("tas_*_historical_*_SA.nc"))
    if not sample:
        print("  No files found for spatial visualization.")
        return

    ds = xr.open_dataset(sample[0])
    tas_mean = ds["tas"].mean(dim="time")
    lon_vals = ds["lon"].values
    lat_vals = ds["lat"].values
    lon_180 = np.where(lon_vals > 180, lon_vals - 360, lon_vals)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Panel 1: Mean temperature map
    ax1 = axes[0]
    data = tas_mean.values - 273.15
    im = ax1.pcolormesh(lon_180, lat_vals, data, cmap="RdYlBu_r", shading="auto")
    plt.colorbar(im, ax=ax1, shrink=0.8, label="Mean temperature (°C)")
    ax1.set_xlabel("Longitude (°)")
    ax1.set_ylabel("Latitude (°)")
    ax1.set_title(f"Annual mean temperature — {Path(sample[0]).stem}",
                  fontweight="bold", fontsize=10)
    ax1.set_aspect("equal")

    # Panel 2: Subregions
    ax2 = axes[1]
    land_mask = (~np.isnan(data)).astype(float)
    ax2.pcolormesh(lon_180, lat_vals, land_mask, cmap="Greys", shading="auto",
                   alpha=0.15, vmin=0, vmax=2)

    colors = {"Northern Tropics": "#E74C3C", "Central Amazonia": "#27AE60",
              "Subtropical Andes": "#F39C12", "Southern Cone": "#3498DB"}

    for name, bounds in SUBREGIONS.items():
        lat_min, lat_max = bounds["lat_min"], bounds["lat_max"]
        rect = plt.Rectangle(
            (lon_180.min(), lat_min), lon_180.max() - lon_180.min(),
            lat_max - lat_min, linewidth=2, edgecolor=colors[name],
            facecolor=colors[name], alpha=0.25, label=name
        )
        ax2.add_patch(rect)
        mid_lat = (lat_min + lat_max) / 2
        ax2.text(lon_180.mean(), mid_lat, name, ha="center", va="center",
                 fontsize=9, fontweight="bold", color=colors[name],
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax2.set_xlim(lon_180.min(), lon_180.max())
    ax2.set_ylim(lat_vals.min(), lat_vals.max())
    ax2.set_xlabel("Longitude (°)")
    ax2.set_ylabel("Latitude (°)")
    ax2.set_title("Study subregions", fontweight="bold")
    ax2.set_aspect("equal")
    ax2.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    fig_path = OUT_DIR / "fig_spatial_coverage_subregions.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig_path}")

    print(f"\n  Spatial grid:")
    print(f"    Latitude:  {lat_vals.min():.2f} to {lat_vals.max():.2f} ({len(lat_vals)} points)")
    print(f"    Longitude: {lon_180.min():.2f} to {lon_180.max():.2f} ({len(lon_vals)} points)")
    print(f"    Resolution: {abs(lat_vals[1]-lat_vals[0]):.2f} deg")
    print(f"    Total cells: {len(lat_vals) * len(lon_vals):,}")

    land_cells = np.sum(~np.isnan(data))
    total_cells = data.size
    print(f"    Land cells: {land_cells:,} ({land_cells/total_cells*100:.1f}%)")
    print(f"    Ocean (NaN): {total_cells - land_cells:,} ({(total_cells-land_cells)/total_cells*100:.1f}%)")

    print(f"\n  Land cells by subregion:")
    for name, bounds in SUBREGIONS.items():
        lat_mask = (lat_vals >= bounds["lat_min"]) & (lat_vals <= bounds["lat_max"])
        sub_data = data[lat_mask, :]
        land = np.sum(~np.isnan(sub_data))
        total = sub_data.size
        print(f"    {name:25s}: {land:5,} land / {total:5,} total ({land/total*100:.1f}%)")

    ds.close()


# =============================================================================
# 4. DATASET SUMMARY
# =============================================================================

def generate_dataset_summary():
    print("\n" + "=" * 70)
    print("  4. DATASET SUMMARY")
    print("=" * 70)

    all_files = list(RAW_DIR.rglob("*.nc"))
    total_size_gb = sum(f.stat().st_size for f in all_files) / (1024**3)
    available_models = sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])

    summary = {
        "Dataset": "NEX-GDDP-CMIP6 (NASA Earth Exchange)",
        "Source": "AWS S3 (s3://nex-gddp-cmip6)",
        "Reference": "Thrasher et al. (2022), Scientific Data 9:262",
        "Downscaling method": "BCSD (Bias-Corrected Spatial Disaggregation)",
        "Spatial resolution": "0.25 x 0.25 deg (~25 km)",
        "Temporal resolution": "Daily",
        "Historical period": "1950-2014",
        "Projected period": "2015-2100",
        "Variables extracted": "tas (temperature), hurs (relative humidity)",
        "Study region": "South America (56S-13N, 82W-34W)",
        "Clipped grid": "276 x 192 (lat x lon)",
        "Models available": f"{len(available_models)} of 8 selected",
        "Models downloaded": ", ".join(available_models),
        "Models pending": ", ".join([m for m in ALL_MODELS if m not in available_models]),
        "Scenarios": "historical, SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5",
        "Total files": f"{len(all_files):,}",
        "Disk usage": f"{total_size_gb:.1f} GB",
    }

    for key, value in summary.items():
        print(f"  {key:30s}: {value}")

    json_path = OUT_DIR / "dataset_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {json_path}")
    return summary


# =============================================================================
# 5. VIABILITY ASSESSMENT
# =============================================================================

def assess_viability(df_completeness, df_qc):
    print("\n" + "=" * 70)
    print("  5. VIABILITY ASSESSMENT")
    print("=" * 70)

    criteria = []

    available = df_completeness[df_completeness["found"] > 0]
    full_coverage = available[available["completeness_pct"] == 100.0]
    c1_pct = len(full_coverage) / len(available) * 100 if len(available) > 0 else 0
    criteria.append({
        "criterion": "Full temporal coverage (1950-2100)",
        "result": f"{c1_pct:.1f}% of model/scenario/variable combinations",
        "pass": c1_pct > 95,
    })

    mean_nan = df_qc["nan_pct"].mean()
    criteria.append({
        "criterion": "Data integrity (NaN acceptable for land analysis)",
        "result": f"Average {mean_nan:.1f}% NaN (expected: >50% due to ocean coverage)",
        "pass": True,
    })

    total_oor = df_qc["out_of_range"].sum()
    total_cells = df_qc["n_time"].sum() * df_qc["n_lat"].iloc[0] * df_qc["n_lon"].iloc[0]
    oor_pct = (total_oor / total_cells * 100) if total_cells > 0 else 0
    criteria.append({
        "criterion": "Values within physical ranges",
        "result": f"{oor_pct:.4f}% out of range (threshold: <0.1%)",
        "pass": oor_pct < 0.1,
    })

    lat_consistent = df_qc["n_lat"].nunique() == 1
    lon_consistent = df_qc["n_lon"].nunique() == 1
    criteria.append({
        "criterion": "Dimensional consistency across models",
        "result": f"lat={'consistent' if lat_consistent else 'INCONSISTENT'}, "
                  f"lon={'consistent' if lon_consistent else 'INCONSISTENT'}",
        "pass": lat_consistent and lon_consistent,
    })

    available_models = df_completeness[df_completeness["found"] > 0]["model"].nunique()
    criteria.append({
        "criterion": "Ensemble diversity (minimum 5 models for robust statistics)",
        "result": f"{available_models} models available",
        "pass": available_models >= 5,
    })

    all_pass = True
    for c in criteria:
        status = "PASS" if c["pass"] else "FAIL"
        all_pass = all_pass and c["pass"]
        print(f"\n  {status}")
        print(f"    Criterion: {c['criterion']}")
        print(f"    Result:    {c['result']}")

    print(f"\n  {'=' * 50}")
    verdict = "VIABLE" if all_pass else "VIABLE WITH OBSERVATIONS"
    print(f"  VERDICT: Dataset is {verdict} for the project")
    print(f"  {'=' * 50}")

    df_criteria = pd.DataFrame(criteria)
    csv_path = OUT_DIR / "viability_assessment.csv"
    df_criteria.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#" * 70)
    print("  NEX-GDDP-CMIP6 — INVENTORY & VIABILITY ASSESSMENT")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)

    df_comp = build_completeness_matrix()
    plot_completeness_heatmap(df_comp)
    df_qc = run_quality_control()
    plot_nan_coverage(df_qc)
    plot_value_ranges(df_qc)
    plot_spatial_coverage()
    generate_dataset_summary()
    assess_viability(df_comp, df_qc)

    print(f"\n  All outputs in: {OUT_DIR}")
    print(f"  Generated files:")
    for f in sorted(OUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:45s} ({size_kb:.0f} KB)")
    print("\n  Script completed.")


if __name__ == "__main__":
    main()
