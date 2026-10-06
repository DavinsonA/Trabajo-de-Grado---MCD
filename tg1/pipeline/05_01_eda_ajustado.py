#!/usr/bin/env python3
"""
05_01_eda_ajustado.py
Adjusted EDA figures for tas, hurs, and tas-hurs correlation.

Generates only:
  - Annual tas time series with a shared y-axis across subregions
  - Annual hurs time series with a shared y-axis across subregions
  - tas-hurs correlation without the 1960-1980 period

Usage:
    source /media/volume/jay2vol/nexgddp/venv/bin/activate
    python 05_01_eda_ajustado.py
"""

import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("/media/volume/jay2vol/nexgddp")
RAW_DIR = BASE_DIR / "raw"
OUT_DIR = BASE_DIR / "outputs" / "05_01_eda_ajustado"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS_TO_PLOT = ["historical", "ssp245", "ssp585"]

CORRELATION_PERIODS = {
    "Present\n(1990-2014)": ("historical", 1990, 2014),
    "Mid-century\n(2040-2060, SSP2-4.5)": ("ssp245", 2040, 2060),
    "End-century\n(2080-2100, SSP5-8.5)": ("ssp585", 2080, 2100),
}

MAX_CORRELATION_MODELS = 3

SUBREGIONS = {
    "Northern Tropics": {"lat_min": 0.0, "lat_max": 13.0},
    "Central Amazonia": {"lat_min": -15.0, "lat_max": 0.0},
    "Subtropical Andes": {"lat_min": -30.0, "lat_max": -15.0},
    "Southern Cone": {"lat_min": -56.0, "lat_max": -30.0},
}

SUBREGION_COLORS = {
    "Northern Tropics": "#E74C3C",
    "Central Amazonia": "#27AE60",
    "Subtropical Andes": "#F39C12",
    "Southern Cone": "#3498DB",
}

SCENARIO_COLORS = {
    "historical": "#333333",
    "ssp245": "#FFC107",
    "ssp585": "#F44336",
}

SCENARIO_LABELS = {
    "historical": "Historical",
    "ssp245": "SSP2-4.5",
    "ssp585": "SSP5-8.5",
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
# HELPERS
# =============================================================================

def get_available_models():
    if not RAW_DIR.exists():
        print(f"  Raw directory not found: {RAW_DIR}")
        return []

    models = sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])
    print(f"  Available models: {models}")
    return models


def subset_lat(ds, lat_min, lat_max):
    mask = (ds.lat >= lat_min) & (ds.lat <= lat_max)
    return ds.sel(lat=mask)


def extract_year_from_filename(filepath):
    parts = Path(filepath).stem.split("_")
    for part in reversed(parts):
        if part.isdigit() and len(part) == 4:
            return int(part)
    return None


def load_annual_series(model, scenario, variable, subregion_bounds):
    var_dir = RAW_DIR / model / scenario / variable
    if not var_dir.exists():
        return None

    files = sorted(var_dir.glob("*.nc"))
    if not files:
        return None

    annual_values = []
    for filepath in files:
        ds = xr.open_dataset(filepath)
        ds = subset_lat(ds, subregion_bounds["lat_min"],
                        subregion_bounds["lat_max"])

        value = float(ds[variable].mean().values)
        year = extract_year_from_filename(filepath)
        if year is None:
            year = int(ds["time"].dt.year[0].values)

        annual_values.append({"year": year, "value": value})
        ds.close()

    if not annual_values:
        return None

    df = pd.DataFrame(annual_values).set_index("year").sort_index()
    return df["value"]


def load_monthly_series(model, scenario, variable, subregion_bounds,
                        year_start, year_end):
    var_dir = RAW_DIR / model / scenario / variable
    if not var_dir.exists():
        return None

    files = sorted(var_dir.glob("*.nc"))
    if not files:
        return None

    monthly_records = []
    for filepath in files:
        file_year = extract_year_from_filename(filepath)
        if file_year is not None and file_year < year_start:
            continue
        if file_year is not None and file_year > year_end:
            continue

        ds = xr.open_dataset(filepath)
        ds = subset_lat(ds, subregion_bounds["lat_min"],
                        subregion_bounds["lat_max"])

        spatial_mean = ds[variable].mean(dim=["lat", "lon"])
        monthly = spatial_mean.resample(time="1ME").mean()

        for time_value, value in zip(monthly["time"].values, monthly.values):
            if hasattr(time_value, "year"):
                year, month = time_value.year, time_value.month
            else:
                timestamp = pd.Timestamp(time_value)
                year, month = timestamp.year, timestamp.month

            monthly_records.append({
                "year": year,
                "month": month,
                "value": float(value),
            })

        ds.close()

    if not monthly_records:
        return None

    return pd.DataFrame(monthly_records)


def format_timeseries_axes(ax, ylabel):
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.tick_params(axis="x", rotation=0)


def compute_common_axis(values, tick_count=6):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None, None

    data_min = float(values.min())
    data_max = float(values.max())
    span = data_max - data_min
    padding = span * 0.05 if span > 0 else 1.0

    axis_min = data_min - padding
    axis_max = data_max + padding
    ticks = np.linspace(axis_min, axis_max, tick_count)

    return (axis_min, axis_max), ticks


def collect_annual_series(models, variable):
    series_by_region = {}
    all_values = []

    for region_name, bounds in SUBREGIONS.items():
        region_series = []

        for model in models:
            for scenario in SCENARIOS_TO_PLOT:
                series = load_annual_series(model, scenario, variable, bounds)
                if series is None:
                    continue

                if variable == "tas":
                    series = series - 273.15

                finite_values = series.to_numpy()
                finite_values = finite_values[np.isfinite(finite_values)]
                all_values.extend(finite_values)

                region_series.append({
                    "model": model,
                    "scenario": scenario,
                    "series": series,
                })

        series_by_region[region_name] = region_series

    return series_by_region, all_values


def merge_tas_hurs(tas_series, hurs_series):
    tas = tas_series.copy()
    hurs = hurs_series.copy()

    tas["key"] = tas["year"] * 100 + tas["month"]
    hurs["key"] = hurs["year"] * 100 + hurs["month"]

    merged = pd.merge(tas, hurs, on="key", suffixes=("_tas", "_hurs"))
    if merged.empty:
        return None

    tas_c = merged["value_tas"].to_numpy() - 273.15
    hurs_pct = merged["value_hurs"].to_numpy()
    valid = np.isfinite(tas_c) & np.isfinite(hurs_pct)

    if not np.any(valid):
        return None

    return tas_c[valid], hurs_pct[valid]


# =============================================================================
# ADJUSTED FIGURES
# =============================================================================

def plot_annual_timeseries(models, variable, ylabel, title, output_name):
    print("\n" + "=" * 70)
    print(f"  Annual time series - {variable}")
    print("=" * 70)

    series_by_region, all_values = collect_annual_series(models, variable)
    ylim, yticks = compute_common_axis(all_values)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (region_name, _) in enumerate(SUBREGIONS.items()):
        ax = axes[idx]
        ax.set_title(region_name, fontweight="bold")

        for item in series_by_region[region_name]:
            scenario = item["scenario"]
            series = item["series"]
            alpha = 0.5 if scenario == "historical" else 0.35
            linewidth = 1.2 if scenario == "historical" else 0.8

            ax.plot(
                series.index,
                series.values,
                color=SCENARIO_COLORS[scenario],
                alpha=alpha,
                linewidth=linewidth,
            )

        if ylim is not None:
            ax.set_ylim(*ylim)
            ax.set_yticks(yticks)

        format_timeseries_axes(ax, ylabel)

    legend_patches = [
        plt.Line2D(
            [0], [0],
            color=SCENARIO_COLORS[scenario],
            lw=2,
            label=SCENARIO_LABELS[scenario],
        )
        for scenario in SCENARIOS_TO_PLOT
    ]

    fig.legend(
        handles=legend_patches,
        loc="upper center",
        ncol=3,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.02),
    )
    plt.suptitle(title, fontweight="bold", y=1.05)
    plt.tight_layout()

    fig_path = OUT_DIR / output_name
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig_path}")


def plot_correlation_tas_hurs(models):
    print("\n" + "=" * 70)
    print("  tas vs hurs correlation")
    print("=" * 70)

    selected_models = models[:MAX_CORRELATION_MODELS]
    n_rows = len(SUBREGIONS)
    n_cols = len(CORRELATION_PERIODS)

    correlation_data = {}
    all_tas_values = []
    all_hurs_values = []

    for region_name, bounds in SUBREGIONS.items():
        for period_name, (scenario, year_start, year_end) in CORRELATION_PERIODS.items():
            tas_values = []
            hurs_values = []

            for model in selected_models:
                tas_series = load_monthly_series(
                    model, scenario, "tas", bounds, year_start, year_end
                )
                hurs_series = load_monthly_series(
                    model, scenario, "hurs", bounds, year_start, year_end
                )

                if tas_series is None or hurs_series is None:
                    continue

                merged = merge_tas_hurs(tas_series, hurs_series)
                if merged is None:
                    continue

                tas_model, hurs_model = merged
                tas_values.extend(tas_model)
                hurs_values.extend(hurs_model)

            tas_arr = np.array(tas_values)
            hurs_arr = np.array(hurs_values)

            if len(tas_arr) > 0 and len(hurs_arr) > 0:
                valid = np.isfinite(tas_arr) & np.isfinite(hurs_arr)
                tas_arr = tas_arr[valid]
                hurs_arr = hurs_arr[valid]

            if len(tas_arr) > 0 and len(hurs_arr) > 0:
                correlation_data[(region_name, period_name)] = (tas_arr, hurs_arr)
                all_tas_values.extend(tas_arr)
                all_hurs_values.extend(hurs_arr)
            else:
                correlation_data[(region_name, period_name)] = None

    xlim, xticks = compute_common_axis(all_tas_values)
    ylim, yticks = compute_common_axis(all_hurs_values)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 3.4 * n_rows),
        squeeze=False,
    )

    corr_records = []

    for row, (region_name, bounds) in enumerate(SUBREGIONS.items()):
        for col, (period_name, (scenario, year_start, year_end)) in enumerate(
            CORRELATION_PERIODS.items()
        ):
            ax = axes[row, col]
            panel_data = correlation_data[(region_name, period_name)]

            if panel_data is not None:
                tas_arr, hurs_arr = panel_data
                ax.scatter(
                    tas_arr,
                    hurs_arr,
                    s=3,
                    alpha=0.3,
                    color=SUBREGION_COLORS[region_name],
                )

                if len(tas_arr) > 10:
                    slope, intercept = np.polyfit(tas_arr, hurs_arr, 1)
                    line = np.poly1d([slope, intercept])
                    x_line = np.linspace(tas_arr.min(), tas_arr.max(), 50)
                    ax.plot(x_line, line(x_line), "k--", linewidth=1, alpha=0.7)

                    r_value = np.corrcoef(tas_arr, hurs_arr)[0, 1]
                    ax.text(
                        0.05,
                        0.95,
                        f"r = {r_value:.3f}\nn = {len(tas_arr)}",
                        transform=ax.transAxes,
                        fontsize=8,
                        va="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )

                    corr_records.append({
                        "region": region_name,
                        "period": period_name.replace("\n", " "),
                        "scenario": scenario,
                        "year_start": year_start,
                        "year_end": year_end,
                        "r": round(float(r_value), 4),
                        "slope": round(float(slope), 4),
                        "n": len(tas_arr),
                        "models_used": ", ".join(selected_models),
                    })
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                )

            if xlim is not None:
                ax.set_xlim(*xlim)
                ax.set_xticks(xticks)
            if ylim is not None:
                ax.set_ylim(*ylim)
                ax.set_yticks(yticks)

            ax.tick_params(axis="both", labelsize=8)
            if row == 0:
                ax.set_title(period_name, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{region_name}\nhurs (%)", fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel("tas (deg C)", fontsize=8)

    plt.suptitle(
        "Temperature-humidity correlation by subregion and period",
        fontweight="bold",
        fontsize=13,
    )
    plt.tight_layout()

    fig_path = OUT_DIR / "fig03_correlation_tas_hurs_adjusted.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig_path}")

    if corr_records:
        df_corr = pd.DataFrame(corr_records)
        csv_path = OUT_DIR / "correlation_tas_hurs_adjusted.csv"
        df_corr.to_csv(csv_path, index=False)
        print(f"  Correlation table saved: {csv_path}")
    else:
        print("  No correlation records generated.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#" * 70)
    print("  NEX-GDDP-CMIP6 - ADJUSTED EDA FIGURES")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)

    models = get_available_models()
    if not models:
        return

    plot_annual_timeseries(
        models=models,
        variable="tas",
        ylabel="Temperature (deg C)",
        title="Annual mean temperature by subregion (shared y-axis)",
        output_name="fig01_annual_timeseries_tas_adjusted.png",
    )

    plot_annual_timeseries(
        models=models,
        variable="hurs",
        ylabel="Relative humidity (%)",
        title="Annual mean relative humidity by subregion (shared y-axis)",
        output_name="fig02_annual_timeseries_hurs_adjusted.png",
    )

    plot_correlation_tas_hurs(models)

    print(f"\n  All outputs in: {OUT_DIR}")
    print("  Script completed.")


if __name__ == "__main__":
    main()
