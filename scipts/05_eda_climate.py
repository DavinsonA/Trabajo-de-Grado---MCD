#!/usr/bin/env python3
"""
05_eda_climate.py
Exploratory Data Analysis (EDA) for tas and hurs.

Generates figures and statistics for thesis deliverables:
  - Annual time series by model and scenario
  - Distributions by subregion and period (shared x-axis)
  - Seasonal cycle by subregion
  - Inter-model comparison
  - tas-hurs correlation
  - Spatial maps (historical, future, change)
  - Feature summary statistics

Usage:
    source /media/volume/jay2vol/nexgddp/venv/bin/activate
    python 05_eda_climate.py
"""

import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("/media/volume/jay2vol/nexgddp")
RAW_DIR = BASE_DIR / "raw"
OUT_DIR = BASE_DIR / "outputs" / "05_eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBREGIONS = {
    "Northern Tropics":    {"lat_min": 0.0,   "lat_max": 13.0},
    "Central Amazonia":    {"lat_min": -15.0,  "lat_max": 0.0},
    "Subtropical Andes":   {"lat_min": -30.0,  "lat_max": -15.0},
    "Southern Cone":       {"lat_min": -56.0,  "lat_max": -30.0},
}

SUBREGION_COLORS = {
    "Northern Tropics": "#E74C3C",
    "Central Amazonia": "#27AE60",
    "Subtropical Andes": "#F39C12",
    "Southern Cone": "#3498DB",
}

MODEL_COLORS = {
    "ACCESS-CM2": "#1f77b4",
    "CanESM5": "#ff7f0e",
    "CESM2": "#2ca02c",
    "MIROC6": "#d62728",
    "MPI-ESM1-2-HR": "#9467bd",
    "EC-Earth3": "#8c564b",
    "GFDL-ESM4": "#e377c2",
    "IPSL-CM6A-LR": "#7f7f7f",
}

SCENARIO_COLORS = {
    "historical": "#333333",
    "ssp126": "#2196F3",
    "ssp245": "#FFC107",
    "ssp370": "#FF9800",
    "ssp585": "#F44336",
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
# HELPER FUNCTIONS
# =============================================================================

def get_available_models():
    models = sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])
    print(f"  Available models: {models}")
    return models


def subset_lat(ds, lat_min, lat_max):
    mask = (ds.lat >= lat_min) & (ds.lat <= lat_max)
    return ds.sel(lat=mask)


def extract_year_from_filename(filepath):
    parts = Path(filepath).stem.split("_")
    for p in reversed(parts):
        if p.isdigit() and len(p) == 4:
            return int(p)
    return None


def load_annual_series(model, scenario, variable, subregion_bounds=None):
    var_dir = RAW_DIR / model / scenario / variable
    if not var_dir.exists():
        return None
    files = sorted(var_dir.glob("*.nc"))
    if not files:
        return None

    annual_values = []
    for f in files:
        ds = xr.open_dataset(f)
        if subregion_bounds:
            ds = subset_lat(ds, subregion_bounds["lat_min"],
                            subregion_bounds["lat_max"])
        val = float(ds[variable].mean().values)
        year = extract_year_from_filename(f)
        if year is None:
            year = int(ds["time"].dt.year[0].values)
        annual_values.append({"year": year, "value": val})
        ds.close()

    df = pd.DataFrame(annual_values).set_index("year").sort_index()
    return df["value"]


def load_monthly_series(model, scenario, variable, subregion_bounds=None,
                        year_start=None, year_end=None):
    var_dir = RAW_DIR / model / scenario / variable
    if not var_dir.exists():
        return None
    files = sorted(var_dir.glob("*.nc"))
    if not files:
        return None

    monthly_records = []
    for f in files:
        file_year = extract_year_from_filename(f)
        if year_start and file_year is not None and file_year < year_start:
            continue
        if year_end and file_year is not None and file_year > year_end:
            continue

        ds = xr.open_dataset(f)
        if subregion_bounds:
            ds = subset_lat(ds, subregion_bounds["lat_min"],
                            subregion_bounds["lat_max"])

        spatial_mean = ds[variable].mean(dim=["lat", "lon"])
        monthly = spatial_mean.resample(time="1ME").mean()

        for t, val in zip(monthly["time"].values, monthly.values):
            if hasattr(t, 'year'):
                y, m = t.year, t.month
            else:
                t_dt = pd.Timestamp(t)
                y, m = t_dt.year, t_dt.month
            monthly_records.append({"year": y, "month": m, "value": float(val)})

        ds.close()

    if not monthly_records:
        return None
    return pd.DataFrame(monthly_records)


def load_spatial_mean(model, scenario, variable, year_start, year_end):
    var_dir = RAW_DIR / model / scenario / variable
    if not var_dir.exists():
        return None

    selected = []
    for f in sorted(var_dir.glob("*.nc")):
        file_year = extract_year_from_filename(f)
        if file_year is not None and year_start <= file_year <= year_end:
            selected.append(f)
    if not selected:
        return None

    datasets = [xr.open_dataset(f) for f in selected]
    combined = xr.concat(datasets, dim="time")
    mean_map = combined[variable].mean(dim="time")
    for ds in datasets:
        ds.close()
    return mean_map


def format_timeseries_axes(ax, ylabel):
    """Ensure time series axes show proper tick labels."""
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.tick_params(axis='x', rotation=0)


# =============================================================================
# EDA 1: ANNUAL TIME SERIES
# =============================================================================

def eda_annual_timeseries(models):
    print("\n" + "=" * 70)
    print("  EDA 1: Annual time series — tas")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (region_name, bounds) in enumerate(SUBREGIONS.items()):
        ax = axes[idx]
        ax.set_title(region_name, fontweight="bold")

        for model in models:
            for scenario in ["historical", "ssp245", "ssp585"]:
                series = load_annual_series(model, scenario, "tas", bounds)
                if series is None:
                    continue
                series_c = series - 273.15
                color = SCENARIO_COLORS[scenario]
                alpha = 0.5 if scenario == "historical" else 0.35
                lw = 1.2 if scenario == "historical" else 0.8
                ax.plot(series_c.index, series_c.values, color=color,
                        alpha=alpha, linewidth=lw)

        format_timeseries_axes(ax, "Temperature (°C)")

    legend_patches = [
        plt.Line2D([0], [0], color=SCENARIO_COLORS["historical"], lw=2, label="Historical"),
        plt.Line2D([0], [0], color=SCENARIO_COLORS["ssp245"], lw=2, label="SSP2-4.5"),
        plt.Line2D([0], [0], color=SCENARIO_COLORS["ssp585"], lw=2, label="SSP5-8.5"),
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, 1.02))
    plt.suptitle("Annual mean temperature by subregion (all models)",
                 fontweight="bold", y=1.05)
    plt.tight_layout()
    fig_path = OUT_DIR / "fig01_annual_timeseries_tas.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")


def eda_annual_timeseries_hurs(models):
    print("\n" + "=" * 70)
    print("  EDA 1b: Annual time series — hurs")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (region_name, bounds) in enumerate(SUBREGIONS.items()):
        ax = axes[idx]
        ax.set_title(region_name, fontweight="bold")

        for model in models:
            for scenario in ["historical", "ssp245", "ssp585"]:
                series = load_annual_series(model, scenario, "hurs", bounds)
                if series is None:
                    continue
                color = SCENARIO_COLORS[scenario]
                alpha = 0.5 if scenario == "historical" else 0.35
                ax.plot(series.index, series.values, color=color,
                        alpha=alpha, linewidth=0.8)

        format_timeseries_axes(ax, "Relative humidity (%)")

    legend_patches = [
        plt.Line2D([0], [0], color=SCENARIO_COLORS["historical"], lw=2, label="Historical"),
        plt.Line2D([0], [0], color=SCENARIO_COLORS["ssp245"], lw=2, label="SSP2-4.5"),
        plt.Line2D([0], [0], color=SCENARIO_COLORS["ssp585"], lw=2, label="SSP5-8.5"),
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, 1.02))
    plt.suptitle("Annual mean relative humidity by subregion (all models)",
                 fontweight="bold", y=1.05)
    plt.tight_layout()
    fig_path = OUT_DIR / "fig02_annual_timeseries_hurs.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")


# =============================================================================
# EDA 2: DISTRIBUTIONS BY PERIOD (shared x-axis across regions)
# =============================================================================

def eda_distributions(models):
    print("\n" + "=" * 70)
    print("  EDA 2: Distributions by period")
    print("=" * 70)

    periods = {
        "Baseline\n(1961-1990)": ("historical", 1961, 1990),
        "Recent historical\n(1991-2014)": ("historical", 1991, 2014),
        "Transition\n(2015-2030, SSP2-4.5)": ("ssp245", 2015, 2030),
        "Mid-century\n(2041-2070, SSP2-4.5)": ("ssp245", 2041, 2070),
        "End-of-century\n(2071-2100, SSP5-8.5)": ("ssp585", 2071, 2100),
    }

    sample_models = models[:3]

    for variable, units, title in [("tas", "°C", "Temperature"),
                                    ("hurs", "%", "Relative humidity")]:

        # FIRST PASS: collect all values to determine global x-limits
        global_min = np.inf
        global_max = -np.inf

        all_data = {}
        for region_name, bounds in SUBREGIONS.items():
            all_data[region_name] = {}
            for period_name, (scenario, y_start, y_end) in periods.items():
                all_values = []
                for model in sample_models:
                    series = load_monthly_series(model, scenario, variable,
                                                 bounds, y_start, y_end)
                    if series is not None:
                        vals = series["value"].values.copy()
                        if variable == "tas":
                            vals = vals - 273.15
                        all_values.extend(vals)

                arr = np.array(all_values)
                arr = arr[np.isfinite(arr)] if len(arr) > 0 else arr
                all_data[region_name][period_name] = arr

                if len(arr) > 0:
                    global_min = min(global_min, np.percentile(arr, 0.5))
                    global_max = max(global_max, np.percentile(arr, 99.5))

        # Add padding
        padding = (global_max - global_min) * 0.05
        xlim = (global_min - padding, global_max + padding)

        # SECOND PASS: plot with shared x-limits
        fig, axes = plt.subplots(len(SUBREGIONS), len(periods),
                                 figsize=(18, 12))

        for row, (region_name, bounds) in enumerate(SUBREGIONS.items()):
            for col, (period_name, _) in enumerate(periods.items()):
                ax = axes[row, col]
                arr = all_data[region_name][period_name]

                if len(arr) > 0:
                    ax.hist(arr, bins=40, alpha=0.7,
                            color=SUBREGION_COLORS[region_name],
                            edgecolor="white", linewidth=0.3)
                    mu = np.mean(arr)
                    ax.axvline(mu, color="black", linestyle="--",
                               linewidth=1, alpha=0.7)
                    ax.text(0.95, 0.95, f"μ={mu:.1f}\nσ={np.std(arr):.1f}",
                            transform=ax.transAxes, ha="right",
                            va="top", fontsize=7,
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

                ax.set_xlim(xlim)
                ax.tick_params(axis='both', labelsize=8)

                if row == 0:
                    ax.set_title(period_name, fontsize=8, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(region_name, fontsize=9, fontweight="bold")
                if row == len(SUBREGIONS) - 1:
                    ax.set_xlabel(f"{variable} ({units})", fontsize=8)

        plt.suptitle(f"Monthly {title} distribution by subregion and period",
                     fontweight="bold", fontsize=13)
        plt.tight_layout()
        fig_path = OUT_DIR / f"fig03_distributions_{variable}.png"
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")


# =============================================================================
# EDA 3: SEASONAL CYCLE
# =============================================================================

def eda_seasonal_cycle(models):
    print("\n" + "=" * 70)
    print("  EDA 3: Seasonal cycle")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    month_names = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    sample_models = models[:3]

    for var_idx, (variable, ylabel) in enumerate([
        ("tas", "Temperature (°C)"),
        ("hurs", "Relative humidity (%)")
    ]):
        ax = axes[var_idx]
        for region_name, bounds in SUBREGIONS.items():
            monthly_all = []
            for model in sample_models:
                series = load_monthly_series(model, "historical", variable,
                                             bounds, 1961, 1990)
                if series is not None:
                    monthly_all.append(series)
            if not monthly_all:
                continue

            combined = pd.concat(monthly_all)
            if variable == "tas":
                combined["value"] = combined["value"] - 273.15

            climatology = combined.groupby("month")["value"].agg(["mean", "std"])
            ax.plot(range(1, 13), climatology["mean"],
                    color=SUBREGION_COLORS[region_name],
                    linewidth=2, marker="o", markersize=4, label=region_name)
            ax.fill_between(range(1, 13),
                            climatology["mean"] - climatology["std"],
                            climatology["mean"] + climatology["std"],
                            color=SUBREGION_COLORS[region_name], alpha=0.15)

        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_names)
        ax.set_xlabel("Month")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Seasonal cycle — {variable} (1961-1990)", fontweight="bold")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = OUT_DIR / "fig04_seasonal_cycle.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")


# =============================================================================
# EDA 4: INTER-MODEL COMPARISON
# =============================================================================

def eda_model_comparison(models):
    print("\n" + "=" * 70)
    print("  EDA 4: Inter-model comparison")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (region_name, bounds) in enumerate(SUBREGIONS.items()):
        ax = axes[idx]
        ax.set_title(region_name, fontweight="bold")

        model_means = {}
        for model in models:
            series = load_annual_series(model, "historical", "tas", bounds)
            if series is not None:
                model_means[model] = series - 273.15

        if not model_means:
            continue

        for model, series in model_means.items():
            ax.plot(series.index, series.values,
                    color=MODEL_COLORS.get(model, "#888"),
                    linewidth=1.2, alpha=0.7, label=model)

        df_all = pd.DataFrame(model_means)
        ens_mean = df_all.mean(axis=1)
        ens_std = df_all.std(axis=1)
        ax.plot(ens_mean.index, ens_mean.values, color="black",
                linewidth=2, label="Ensemble mean", zorder=10)
        ax.fill_between(ens_mean.index,
                        ens_mean - 2 * ens_std, ens_mean + 2 * ens_std,
                        color="gray", alpha=0.2, label="±2σ")

        format_timeseries_axes(ax, "Temperature (°C)")
        ax.legend(fontsize=7, loc="upper left")

        spread = df_all.max(axis=1) - df_all.min(axis=1)
        ax.text(0.98, 0.05, f"Mean spread: {spread.mean():.2f}°C",
                transform=ax.transAxes, ha="right", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.suptitle("Inter-model comparison — tas historical (1950-2014)",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig_path = OUT_DIR / "fig05_model_comparison_tas.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")


# =============================================================================
# EDA 5: TAS vs HURS CORRELATION
# =============================================================================

def eda_correlation_tas_hurs(models):
    print("\n" + "=" * 70)
    print("  EDA 5: tas vs hurs correlation")
    print("=" * 70)

    decades = {
        "1960-1980": ("historical", 1960, 1980),
        "1990-2014": ("historical", 1990, 2014),
        "2040-2060 (SSP2-4.5)": ("ssp245", 2040, 2060),
        "2080-2100 (SSP5-8.5)": ("ssp585", 2080, 2100),
    }

    fig, axes = plt.subplots(len(SUBREGIONS), len(decades), figsize=(16, 14))
    sample_models = models[:3]
    corr_records = []

    for row, (region_name, bounds) in enumerate(SUBREGIONS.items()):
        for col, (decade_name, (scenario, y_start, y_end)) in enumerate(decades.items()):
            ax = axes[row, col]
            tas_vals, hurs_vals = [], []

            for model in sample_models:
                tas_s = load_monthly_series(model, scenario, "tas",
                                            bounds, y_start, y_end)
                hurs_s = load_monthly_series(model, scenario, "hurs",
                                              bounds, y_start, y_end)
                if tas_s is not None and hurs_s is not None:
                    tas_s["key"] = tas_s["year"] * 100 + tas_s["month"]
                    hurs_s["key"] = hurs_s["year"] * 100 + hurs_s["month"]
                    merged = pd.merge(tas_s, hurs_s, on="key",
                                      suffixes=("_tas", "_hurs"))
                    tas_vals.extend((merged["value_tas"] - 273.15).values)
                    hurs_vals.extend(merged["value_hurs"].values)

            if tas_vals and hurs_vals:
                tas_arr = np.array(tas_vals)
                hurs_arr = np.array(hurs_vals)
                valid = np.isfinite(tas_arr) & np.isfinite(hurs_arr)
                tx, hx = tas_arr[valid], hurs_arr[valid]

                ax.scatter(tx, hx, s=3, alpha=0.3,
                           color=SUBREGION_COLORS[region_name])

                if len(tx) > 10:
                    z = np.polyfit(tx, hx, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(tx.min(), tx.max(), 50)
                    ax.plot(x_line, p(x_line), "k--", linewidth=1, alpha=0.7)
                    r = np.corrcoef(tx, hx)[0, 1]
                    ax.text(0.05, 0.95, f"r = {r:.3f}\nn = {len(tx)}",
                            transform=ax.transAxes, fontsize=8, va="top",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
                    corr_records.append({
                        "region": region_name, "period": decade_name,
                        "scenario": scenario, "r": round(r, 4),
                        "slope": round(z[0], 4), "n": len(tx),
                    })

            ax.tick_params(axis='both', labelsize=8)
            if row == 0:
                ax.set_title(decade_name, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{region_name}\nhurs (%)", fontsize=8)
            if row == len(SUBREGIONS) - 1:
                ax.set_xlabel("tas (°C)", fontsize=8)

    plt.suptitle("Temperature-humidity correlation by subregion and period",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig_path = OUT_DIR / "fig06_correlation_tas_hurs.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")

    if corr_records:
        df_corr = pd.DataFrame(corr_records)
        csv_path = OUT_DIR / "correlation_tas_hurs.csv"
        df_corr.to_csv(csv_path, index=False)
        print(f"  Correlation table: {csv_path}")
        print("\n  Summary of tas-hurs correlations:")
        print(df_corr.to_string(index=False))


# =============================================================================
# EDA 6: SPATIAL MAPS
# =============================================================================

def eda_spatial_maps(models):
    print("\n" + "=" * 70)
    print("  EDA 6: Spatial maps")
    print("=" * 70)

    sample_models = models[:3]

    periods = {
        "Historical (1961-1990)": ("historical", 1961, 1990),
        "Future (2071-2100, SSP5-8.5)": ("ssp585", 2071, 2100),
    }

    for variable, cmap, units, label in [
        ("tas", "RdYlBu_r", "°C", "Mean temperature"),
        ("hurs", "BrBG", "%", "Mean relative humidity"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for col, (period_name, (scenario, y_start, y_end)) in enumerate(periods.items()):
            ax = axes[col]
            maps = []
            for model in sample_models:
                m = load_spatial_mean(model, scenario, variable, y_start, y_end)
                if m is not None:
                    maps.append(m)
            if not maps:
                ax.set_title(f"{period_name}\n(no data)")
                continue

            ensemble = xr.concat(maps, dim="model").mean(dim="model")
            lon_vals = ensemble.lon.values
            lat_vals = ensemble.lat.values
            lon_180 = np.where(lon_vals > 180, lon_vals - 360, lon_vals)
            data = ensemble.values
            if variable == "tas":
                data = data - 273.15

            im = ax.pcolormesh(lon_180, lat_vals, data, cmap=cmap, shading="auto")
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label(f"{label} ({units})")
            for reg_bounds in SUBREGIONS.values():
                ax.axhline(reg_bounds["lat_min"], color="gray",
                           linewidth=0.5, linestyle=":")
            ax.set_xlabel("Longitude (°)")
            ax.set_ylabel("Latitude (°)")
            ax.set_title(period_name, fontweight="bold")
            ax.set_aspect("equal")

        plt.suptitle(f"Spatial map — {label}", fontweight="bold", fontsize=13)
        plt.tight_layout()
        fig_path = OUT_DIR / f"fig07_spatial_map_{variable}.png"
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")

    # Change map
    print("\n  Generating projected change map...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for var_idx, (variable, cmap, units, label) in enumerate([
        ("tas", "RdBu_r", "°C", "Temperature change"),
        ("hurs", "BrBG", "%", "Relative humidity change"),
    ]):
        ax = axes[var_idx]
        hist_maps, future_maps = [], []
        for model in sample_models:
            h = load_spatial_mean(model, "historical", variable, 1961, 1990)
            f_map = load_spatial_mean(model, "ssp585", variable, 2071, 2100)
            if h is not None and f_map is not None:
                hist_maps.append(h)
                future_maps.append(f_map)

        if hist_maps and future_maps:
            hist_ens = xr.concat(hist_maps, dim="model").mean(dim="model")
            fut_ens = xr.concat(future_maps, dim="model").mean(dim="model")
            diff = fut_ens - hist_ens
            lon_vals = diff.lon.values
            lat_vals = diff.lat.values
            lon_180 = np.where(lon_vals > 180, lon_vals - 360, lon_vals)
            data = diff.values
            vmax = np.nanpercentile(np.abs(data), 95)
            im = ax.pcolormesh(lon_180, lat_vals, data, cmap=cmap,
                               shading="auto", vmin=-vmax, vmax=vmax)
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label(f"Δ{variable} ({units})")
            ax.set_xlabel("Longitude (°)")
            ax.set_ylabel("Latitude (°)")
            ax.set_title(f"{label}\n(SSP5-8.5 2071-2100 vs Hist 1961-1990)",
                         fontweight="bold", fontsize=10)
            ax.set_aspect("equal")

    plt.tight_layout()
    fig_path = OUT_DIR / "fig08_spatial_change_map.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")


# =============================================================================
# EDA 7: FEATURE SUMMARY STATISTICS
# =============================================================================

def eda_feature_summary(models):
    print("\n" + "=" * 70)
    print("  EDA 7: Feature summary statistics")
    print("=" * 70)

    sample_models = models[:3]
    records = []

    analysis_periods = [
        ("Baseline 1961-1990", "historical", 1961, 1990),
        ("Recent historical 1991-2014", "historical", 1991, 2014),
        ("Transition 2015-2030", "ssp245", 2015, 2030),
        ("SSP2-4.5 2041-2070", "ssp245", 2041, 2070),
        ("SSP5-8.5 2071-2100", "ssp585", 2071, 2100),
    ]

    for variable in ["tas", "hurs"]:
        for region_name, bounds in SUBREGIONS.items():
            for period_name, scenario, y_start, y_end in analysis_periods:
                all_vals = []
                for model in sample_models:
                    series = load_monthly_series(model, scenario, variable,
                                                 bounds, y_start, y_end)
                    if series is not None:
                        vals = series["value"].values.copy()
                        if variable == "tas":
                            vals = vals - 273.15
                        all_vals.extend(vals)

                if all_vals:
                    arr = np.array(all_vals)
                    arr = arr[np.isfinite(arr)]
                    if len(arr) > 0:
                        records.append({
                            "variable": variable, "region": region_name,
                            "period": period_name,
                            "mean": round(float(np.mean(arr)), 2),
                            "std": round(float(np.std(arr)), 2),
                            "min": round(float(np.min(arr)), 2),
                            "p25": round(float(np.percentile(arr, 25)), 2),
                            "median": round(float(np.median(arr)), 2),
                            "p75": round(float(np.percentile(arr, 75)), 2),
                            "max": round(float(np.max(arr)), 2),
                            "iqr": round(float(np.percentile(arr, 75) -
                                               np.percentile(arr, 25)), 2),
                            "n_samples": len(arr),
                        })

    df_features = pd.DataFrame(records)
    csv_path = OUT_DIR / "feature_summary_stats.csv"
    df_features.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print(f"\n  Summary (first 12 rows):")
    print(df_features.head(12).to_string(index=False))
    return df_features


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#" * 70)
    print("  NEX-GDDP-CMIP6 — CLIMATE EDA")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)

    models = get_available_models()
    t0 = datetime.now()

    eda_annual_timeseries(models)
    eda_annual_timeseries_hurs(models)
    eda_distributions(models)
    eda_seasonal_cycle(models)
    eda_model_comparison(models)
    eda_correlation_tas_hurs(models)
    eda_spatial_maps(models)
    eda_feature_summary(models)

    elapsed = (datetime.now() - t0).total_seconds()

    print(f"\n" + "=" * 70)
    print(f"  EDA completed in {elapsed / 60:.1f} minutes")
    print(f"  All outputs in: {OUT_DIR}")
    print(f"\n  Generated files:")
    for f in sorted(OUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:45s} ({size_kb:.0f} KB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
