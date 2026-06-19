#!/usr/bin/env python3
"""
06_preprocessing.py
Preprocessing pipeline for NEX-GDDP-CMIP6 South America data.

Transforms daily raw data into analysis-ready datasets:
  1. Monthly spatial means by subregion
  2. Climatology base (1960-2014) by month
  3. Monthly anomalies (value - climatology)
  4. Annual means and anomalies
  5. Ensemble statistics (p10, p50, p90 across models)
  6. Decadal spatial anomaly maps (full grid, for Objective I maps)

Outputs saved to /media/volume/jay2vol/nexgddp/processed/

Usage:
    source /media/volume/jay2vol/nexgddp/venv/bin/activate
    python 06_preprocessing.py
"""

import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("/media/volume/jay2vol/nexgddp")
RAW_DIR = BASE_DIR / "raw"
PROC_DIR = BASE_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

VARIABLES = ["tas", "hurs"]

SCENARIOS = {
    "historical": (1950, 2014),
    "ssp126": (2015, 2100),
    "ssp245": (2015, 2100),
    "ssp370": (2015, 2100),
    "ssp585": (2015, 2100),
}

BASELINE_START = 1960
BASELINE_END = 2014

SUBREGIONS = {
    "Northern Tropics":    {"lat_min": 0.0,   "lat_max": 13.0},
    "Central Amazonia":    {"lat_min": -15.0,  "lat_max": 0.0},
    "Subtropical Andes":   {"lat_min": -30.0,  "lat_max": -15.0},
    "Southern Cone":       {"lat_min": -56.0,  "lat_max": -30.0},
}

DECADES = {
    "1960s": (1960, 1969),
    "1970s": (1970, 1979),
    "1980s": (1980, 1989),
    "1990s": (1990, 1999),
    "2000s": (2000, 2009),
    "2020s": (2020, 2029),
    "2040s": (2040, 2049),
    "2060s": (2060, 2069),
    "2080s": (2080, 2089),
    "2090s": (2090, 2099),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_available_models():
    models = sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])
    print(f"  Available models: {models}")
    return models


def extract_year_from_filename(filepath):
    parts = Path(filepath).stem.split("_")
    for p in reversed(parts):
        if p.isdigit() and len(p) == 4:
            return int(p)
    return None


def subset_lat(ds, lat_min, lat_max):
    mask = (ds.lat >= lat_min) & (ds.lat <= lat_max)
    return ds.sel(lat=mask)


# =============================================================================
# STEP 1: MONTHLY SUBREGION AVERAGES
# =============================================================================

def compute_monthly_subregion_series(models):
    print("\n" + "=" * 70)
    print("  STEP 1: Monthly subregion averages")
    print("=" * 70)

    records = []
    total_combos = len(models) * len(SCENARIOS) * len(VARIABLES)
    combo_count = 0

    for model in models:
        for scenario, (y_start, y_end) in SCENARIOS.items():
            for variable in VARIABLES:
                combo_count += 1
                var_dir = RAW_DIR / model / scenario / variable
                if not var_dir.exists():
                    continue

                files = sorted(var_dir.glob("*.nc"))
                if not files:
                    continue

                print(f"  [{combo_count}/{total_combos}] "
                      f"{model} / {scenario} / {variable} ({len(files)} files)")

                for f in files:
                    ds = xr.open_dataset(f)

                    for region_name, bounds in SUBREGIONS.items():
                        ds_sub = subset_lat(ds, bounds["lat_min"], bounds["lat_max"])
                        spatial_mean = ds_sub[variable].mean(dim=["lat", "lon"])
                        monthly = spatial_mean.resample(time="1ME").mean()

                        for t, val in zip(monthly["time"].values, monthly.values):
                            if hasattr(t, 'year'):
                                y, m = t.year, t.month
                            else:
                                t_dt = pd.Timestamp(t)
                                y, m = t_dt.year, t_dt.month

                            value = float(val)
                            if variable == "tas":
                                value = value - 273.15

                            records.append({
                                "model": model,
                                "scenario": scenario,
                                "variable": variable,
                                "subregion": region_name,
                                "year": y,
                                "month": m,
                                "value": round(value, 4),
                            })

                    ds.close()

    df = pd.DataFrame(records)
    csv_path = PROC_DIR / "monthly_subregion_means.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    print(f"  Total records: {len(df):,}")
    print(f"  File size: {csv_path.stat().st_size / (1024**2):.1f} MB")

    return df


# =============================================================================
# STEP 2: CLIMATOLOGY AND ANOMALIES
# =============================================================================

def compute_anomalies(df):
    print("\n" + "=" * 70)
    print(f"  STEP 2: Climatology ({BASELINE_START}-{BASELINE_END}) and anomalies")
    print("=" * 70)

    hist = df[(df["scenario"] == "historical") &
              (df["year"] >= BASELINE_START) &
              (df["year"] <= BASELINE_END)]

    climatology = hist.groupby(
        ["model", "variable", "subregion", "month"]
    )["value"].mean().reset_index()
    climatology.rename(columns={"value": "climatology"}, inplace=True)

    clim_path = PROC_DIR / "climatology_1960_2014.csv"
    climatology.to_csv(clim_path, index=False)
    print(f"  Climatology saved: {clim_path}")
    print(f"  Climatology records: {len(climatology):,}")

    df_anom = df.merge(
        climatology,
        on=["model", "variable", "subregion", "month"],
        how="left"
    )
    df_anom["anomaly"] = df_anom["value"] - df_anom["climatology"]

    nan_count = df_anom["anomaly"].isna().sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} records with NaN anomaly (missing climatology)")

    anom_path = PROC_DIR / "monthly_subregion_anomalies.csv"
    df_anom.to_csv(anom_path, index=False)
    print(f"  Anomalies saved: {anom_path}")
    print(f"  Total records: {len(df_anom):,}")
    print(f"  File size: {anom_path.stat().st_size / (1024**2):.1f} MB")

    print(f"\n  Anomaly summary by variable:")
    for var in VARIABLES:
        sub = df_anom[df_anom["variable"] == var]
        print(f"    {var}: mean={sub['anomaly'].mean():.3f}, "
              f"std={sub['anomaly'].std():.3f}, "
              f"min={sub['anomaly'].min():.2f}, "
              f"max={sub['anomaly'].max():.2f}")

    return df_anom


# =============================================================================
# STEP 3: ANNUAL MEANS AND ANOMALIES
# =============================================================================

def compute_annual(df_anom):
    print("\n" + "=" * 70)
    print("  STEP 3: Annual means and anomalies")
    print("=" * 70)

    df_annual = df_anom.groupby(
        ["model", "scenario", "variable", "subregion", "year"]
    ).agg(
        value=("value", "mean"),
        anomaly=("anomaly", "mean"),
    ).reset_index()

    df_annual["value"] = df_annual["value"].round(4)
    df_annual["anomaly"] = df_annual["anomaly"].round(4)

    annual_path = PROC_DIR / "annual_subregion_anomalies.csv"
    df_annual.to_csv(annual_path, index=False)
    print(f"  Saved: {annual_path}")
    print(f"  Total records: {len(df_annual):,}")
    print(f"  File size: {annual_path.stat().st_size / (1024**2):.1f} MB")

    return df_annual


# =============================================================================
# STEP 4: CONTINENTAL AVERAGE
# =============================================================================

def compute_continental_average(df_anom):
    print("\n" + "=" * 70)
    print("  STEP 4: Continental average")
    print("=" * 70)

    df_continental = df_anom.groupby(
        ["model", "scenario", "variable", "year", "month"]
    ).agg(
        value=("value", "mean"),
        anomaly=("anomaly", "mean"),
    ).reset_index()

    df_continental["subregion"] = "South America (continental)"
    df_continental["value"] = df_continental["value"].round(4)
    df_continental["anomaly"] = df_continental["anomaly"].round(4)

    cont_path = PROC_DIR / "monthly_continental_anomalies.csv"
    df_continental.to_csv(cont_path, index=False)
    print(f"  Saved: {cont_path}")
    print(f"  Records: {len(df_continental):,}")

    df_cont_annual = df_continental.groupby(
        ["model", "scenario", "variable", "subregion", "year"]
    ).agg(
        value=("value", "mean"),
        anomaly=("anomaly", "mean"),
    ).reset_index()
    df_cont_annual["value"] = df_cont_annual["value"].round(4)
    df_cont_annual["anomaly"] = df_cont_annual["anomaly"].round(4)

    cont_annual_path = PROC_DIR / "annual_continental_anomalies.csv"
    df_cont_annual.to_csv(cont_annual_path, index=False)
    print(f"  Saved: {cont_annual_path}")

    return df_continental, df_cont_annual


# =============================================================================
# STEP 5: ENSEMBLE STATISTICS
# =============================================================================

def compute_ensemble_stats(df_annual, df_cont_annual):
    print("\n" + "=" * 70)
    print("  STEP 5: Ensemble statistics (across models)")
    print("=" * 70)

    df_all = pd.concat([df_annual, df_cont_annual], ignore_index=True)

    ensemble = df_all.groupby(
        ["scenario", "variable", "subregion", "year"]
    ).agg(
        n_models=("model", "nunique"),
        ens_mean=("anomaly", "mean"),
        ens_std=("anomaly", "std"),
        ens_p10=("anomaly", lambda x: np.percentile(x, 10)),
        ens_p25=("anomaly", lambda x: np.percentile(x, 25)),
        ens_p50=("anomaly", "median"),
        ens_p75=("anomaly", lambda x: np.percentile(x, 75)),
        ens_p90=("anomaly", lambda x: np.percentile(x, 90)),
        ens_min=("anomaly", "min"),
        ens_max=("anomaly", "max"),
        value_mean=("value", "mean"),
    ).reset_index()

    for col in ["ens_mean", "ens_std", "ens_p10", "ens_p25", "ens_p50",
                "ens_p75", "ens_p90", "ens_min", "ens_max", "value_mean"]:
        ensemble[col] = ensemble[col].round(4)

    ens_path = PROC_DIR / "ensemble_annual_stats.csv"
    ensemble.to_csv(ens_path, index=False)
    print(f"  Saved: {ens_path}")
    print(f"  Records: {len(ensemble):,}")
    print(f"  File size: {ens_path.stat().st_size / (1024**2):.1f} MB")

    print(f"\n  Ensemble summary:")
    print(f"    Models per group: {ensemble['n_models'].min()}-{ensemble['n_models'].max()}")
    print(f"    Scenarios: {sorted(ensemble['scenario'].unique())}")
    print(f"    Subregions: {sorted(ensemble['subregion'].unique())}")
    print(f"    Year range: {ensemble['year'].min()}-{ensemble['year'].max()}")

    return ensemble


# =============================================================================
# STEP 6: DECADAL SPATIAL ANOMALY MAPS
# =============================================================================

def compute_decadal_spatial_anomalies(models):
    print("\n" + "=" * 70)
    print("  STEP 6: Decadal spatial anomaly maps")
    print("=" * 70)

    spatial_dir = PROC_DIR / "spatial_decadal"
    spatial_dir.mkdir(parents=True, exist_ok=True)

    for variable in VARIABLES:
        print(f"\n  Processing {variable}...")

        # Baseline spatial climatology
        print(f"    Computing baseline climatology (1960-2014)...")
        baseline_maps = []

        for model in models:
            var_dir = RAW_DIR / model / "historical" / variable
            if not var_dir.exists():
                continue

            model_datasets = []
            for f in sorted(var_dir.glob("*.nc")):
                file_year = extract_year_from_filename(f)
                if file_year is not None and BASELINE_START <= file_year <= BASELINE_END:
                    ds = xr.open_dataset(f)
                    model_datasets.append(ds)

            if model_datasets:
                combined = xr.concat(model_datasets, dim="time")
                mean_map = combined[variable].mean(dim="time")
                baseline_maps.append(mean_map)
                for ds in model_datasets:
                    ds.close()
                print(f"      {model}: {len(model_datasets)} years loaded")

        if not baseline_maps:
            print(f"    No baseline data found for {variable}, skipping.")
            continue

        baseline = xr.concat(baseline_maps, dim="model").mean(dim="model")

        if variable == "tas":
            baseline = baseline - 273.15

        baseline_ds = baseline.to_dataset(name=f"{variable}_baseline")
        baseline_path = spatial_dir / f"baseline_{variable}_1960_2014.nc"
        baseline_ds.to_netcdf(baseline_path)
        print(f"    Baseline saved: {baseline_path}")

        # Each decade
        for decade_name, (d_start, d_end) in DECADES.items():
            if d_end <= 2014:
                scenario = "historical"
            else:
                scenario = "ssp585"

            decade_maps = []
            for model in models:
                var_dir = RAW_DIR / model / scenario / variable
                if not var_dir.exists():
                    continue

                model_datasets = []
                for f in sorted(var_dir.glob("*.nc")):
                    file_year = extract_year_from_filename(f)
                    if file_year is not None and d_start <= file_year <= d_end:
                        ds = xr.open_dataset(f)
                        model_datasets.append(ds)

                if model_datasets:
                    combined = xr.concat(model_datasets, dim="time")
                    mean_map = combined[variable].mean(dim="time")
                    decade_maps.append(mean_map)
                    for ds in model_datasets:
                        ds.close()

            if not decade_maps:
                continue

            decade_mean = xr.concat(decade_maps, dim="model").mean(dim="model")

            if variable == "tas":
                decade_mean = decade_mean - 273.15

            anomaly = decade_mean - baseline

            anomaly_ds = xr.Dataset({
                f"{variable}_anomaly": anomaly,
                f"{variable}_mean": decade_mean,
            })
            nc_path = spatial_dir / f"decade_{variable}_{decade_name}_{scenario}.nc"
            anomaly_ds.to_netcdf(nc_path)
            print(f"    {decade_name} ({scenario}): saved "
                  f"(mean anomaly: {float(anomaly.mean()):.3f})")

    print(f"\n  All spatial files saved in: {spatial_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#" * 70)
    print("  NEX-GDDP-CMIP6 — PREPROCESSING PIPELINE")
    print(f"  Baseline period: {BASELINE_START}-{BASELINE_END}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)

    models = get_available_models()
    t0 = datetime.now()

    df_monthly = compute_monthly_subregion_series(models)
    df_anom = compute_anomalies(df_monthly)
    df_annual = compute_annual(df_anom)
    df_continental, df_cont_annual = compute_continental_average(df_anom)
    df_ensemble = compute_ensemble_stats(df_annual, df_cont_annual)
    compute_decadal_spatial_anomalies(models)

    elapsed = (datetime.now() - t0).total_seconds()

    print("\n" + "=" * 70)
    print(f"  PREPROCESSING COMPLETED in {elapsed / 60:.1f} minutes")
    print(f"\n  Output files in {PROC_DIR}:")
    for f in sorted(PROC_DIR.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 ** 2)
            rel = f.relative_to(PROC_DIR)
            print(f"    {str(rel):55s} ({size_mb:.1f} MB)")

    print(f"\n  Key datasets for Objective I:")
    print(f"    - ensemble_annual_stats.csv      -> anomaly series with p10-p90 bands")
    print(f"    - annual_subregion_anomalies.csv  -> per-model annual anomalies")
    print(f"    - spatial_decadal/*.nc            -> decadal anomaly maps (full grid)")
    print(f"\n  Key datasets for Objective II:")
    print(f"    - monthly_subregion_anomalies.csv -> monthly data for seasonal analysis")
    print(f"    - climatology_1960_2014.csv       -> baseline for correlation analysis")
    print("=" * 70)


if __name__ == "__main__":
    main()
