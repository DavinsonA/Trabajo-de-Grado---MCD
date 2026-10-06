#!/usr/bin/env python3
"""
10_spatial_trends.py (v2 — fix calendar alignment)

Fix critical bug: GCMs use different calendar types (numpy.datetime64,
cftime.DatetimeNoLeap, cftime.Datetime360Day). xr.align(join='inner')
returns 0 timestamps because the types don't match → all outputs NaN.

Solution: normalize each model's time to integer year before stacking.
This eliminates calendar differences while preserving the annual
resolution needed for trend analysis.

Outputs: same 54 NetCDFs as before, but now with valid data.
"""

import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import pymannkendall as mk

warnings.filterwarnings("ignore")

BASE_DIR = Path("/media/volume/jay2vol/nexgddp")
RAW_DIR = BASE_DIR / "raw"
OUT_DIR = BASE_DIR / "outputs" / "10_spatial_trends"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
SSPs = ["ssp126", "ssp245", "ssp370", "ssp585"]
VARIABLES = ["tas", "hurs"]

WINDOWS = {
    "historical_1950_2014": {"hist": (1950, 2014), "ssp": None,         "scenario": None},
    "projected_2015_2100":  {"hist": None,         "ssp": (2015, 2100), "scenario": "per_ssp"},
    "full_1950_2100":       {"hist": (1950, 2014), "ssp": (2015, 2100), "scenario": "per_ssp"},
}


def get_available_models():
    return sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])


def extract_year_from_filename(filepath):
    parts = Path(filepath).stem.split("_")
    for p in reversed(parts):
        if p.isdigit() and len(p) == 4:
            return int(p)
    return None


def time_to_year(time_values):
    """Convert mixed time types (datetime64, cftime) to integer years."""
    years = []
    for t in time_values:
        if hasattr(t, 'year'):
            years.append(int(t.year))
        else:
            years.append(pd.Timestamp(t).year)
    return np.array(years, dtype=int)


def load_model_scenario_annual(model, scenario, variable, year_range):
    y_start, y_end = year_range
    var_dir = RAW_DIR / model / scenario / variable
    if not var_dir.exists():
        return None

    files = []
    for f in sorted(var_dir.glob("*.nc")):
        yr = extract_year_from_filename(f)
        if yr is not None and y_start <= yr <= y_end:
            files.append(f)

    if not files:
        return None

    datasets = []
    for f in files:
        try:
            ds = xr.open_dataset(f)
            if "time" in ds.dims and ds.sizes["time"] > 0 and variable in ds:
                datasets.append(ds)
            else:
                ds.close()
        except Exception as e:
            print(f"      WARN: failed to open {f.name}: {e}")

    if not datasets:
        return None

    try:
        combined = xr.concat(datasets, dim="time")
        annual = combined[variable].resample(time="1YE").mean()
        if variable == "tas":
            annual = annual - 273.15

        # KEY FIX: replace time coord with integer year
        years = time_to_year(annual.time.values)
        annual = annual.assign_coords(time=years).rename({"time": "year"})

        return annual
    except Exception as e:
        print(f"      WARN: concat/resample failed: {e}")
        return None
    finally:
        for ds in datasets:
            ds.close()


def build_combined_series(model, variable, scenario, window_spec):
    parts = []

    if window_spec["hist"] is not None:
        h = load_model_scenario_annual(model, "historical", variable, window_spec["hist"])
        if h is not None:
            parts.append(h)

    if window_spec["ssp"] is not None and scenario is not None:
        s = load_model_scenario_annual(model, scenario, variable, window_spec["ssp"])
        if s is not None:
            parts.append(s)

    if not parts:
        return None

    if len(parts) == 1:
        return parts[0]

    combined = xr.concat(parts, dim="year").sortby("year")
    _, unique_idx = np.unique(combined.year.values, return_index=True)
    combined = combined.isel(year=sorted(unique_idx))
    return combined


def _mk_pixel(series):
    y = series[np.isfinite(series)]
    if len(y) < 10:
        return np.nan, np.nan
    try:
        r = mk.hamed_rao_modification_test(y, alpha=ALPHA)
        return float(r.slope) * 10.0, float(r.p)
    except Exception:
        return np.nan, np.nan


def apply_mk_grid(da_year_latlon):
    slope, pvalue = xr.apply_ufunc(
        _mk_pixel,
        da_year_latlon,
        input_core_dims=[["year"]],
        output_core_dims=[[], []],
        vectorize=True,
        dask="forbidden",
    )
    return slope, pvalue


def compute_pixel_slope(series):
    y = series[np.isfinite(series)]
    if len(y) < 10:
        return np.nan
    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return slope * 10.0


def apply_sign_agreement(model_arrays):
    stack = np.stack(model_arrays, axis=0)
    with np.errstate(invalid="ignore"):
        pos = (stack > 0).sum(axis=0)
        neg = (stack < 0).sum(axis=0)
    n_valid = pos + neg
    n_valid = np.where(n_valid == 0, 1, n_valid)
    return np.maximum(pos, neg) / n_valid * 100.0


def process_combination(variable, window_name, window_spec, scenario, models):
    if scenario is not None:
        label = f"{variable}__{window_name}__{scenario}"
    else:
        label = f"{variable}__{window_name}"

    print(f"\n  >>> {label}")

    model_arrays = []
    loaded_models = []

    for model in models:
        da = build_combined_series(model, variable, scenario, window_spec)
        if da is None:
            print(f"      [SKIP] {model}: no data")
            continue
        model_arrays.append(da)
        loaded_models.append(model)
        valid_first = 100 * np.isfinite(da.isel(year=0).values).sum() / da.isel(year=0).size
        print(f"      {model}: {len(da.year)} years, first-year valid pixels {valid_first:.1f}%")

    if len(model_arrays) < 2:
        print(f"      [SKIP COMBO] not enough models ({len(model_arrays)})")
        return

    all_years = set()
    for da in model_arrays:
        all_years.update(da.year.values.tolist())
    common_years = sorted(all_years)
    print(f"      Common years: {len(common_years)} ({common_years[0]}–{common_years[-1]})")

    aligned = [da.reindex(year=common_years) for da in model_arrays]

    ref = aligned[0]
    for i in range(1, len(aligned)):
        if (len(aligned[i].lat) != len(ref.lat)) or (len(aligned[i].lon) != len(ref.lon)):
            print(f"      WARN: {loaded_models[i]} different grid, reindexing")
            aligned[i] = aligned[i].reindex(lat=ref.lat, lon=ref.lon, method='nearest')

    ensemble = xr.concat(aligned, dim="model").assign_coords(model=loaded_models)
    print(f"      Ensemble: {dict(ensemble.sizes)}")

    print(f"      Computing per-model slopes...")
    per_model_slopes = []
    for mi in range(len(loaded_models)):
        slope_da = xr.apply_ufunc(
            compute_pixel_slope,
            ensemble.isel(model=mi),
            input_core_dims=[["year"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="forbidden",
        )
        per_model_slopes.append(slope_da.values)

    agreement = apply_sign_agreement(per_model_slopes)
    agreement_da = xr.DataArray(
        agreement,
        coords={"lat": ensemble["lat"], "lon": ensemble["lon"]},
        dims=["lat", "lon"],
        name="agreement_pct",
        attrs={
            "long_name": "Inter-model sign agreement",
            "units": "percent",
            "n_models": len(loaded_models),
        },
    )

    print(f"      Computing ensemble median + Mann-Kendall...")
    ensemble_median = ensemble.median(dim="model")
    sen_slope_da, pvalue_da = apply_mk_grid(ensemble_median)

    units_slope = "degC/decade" if variable == "tas" else "percent/decade"
    sen_slope_da = sen_slope_da.rename("sen_slope").assign_attrs({
        "long_name": f"Sen's slope on ensemble-median annual {variable}",
        "units": units_slope,
    })
    pvalue_da = pvalue_da.rename("mk_pvalue").assign_attrs({
        "long_name": f"Hamed-Rao MK p-value on ensemble-median {variable}",
        "alpha_reference": ALPHA,
    })

    encoding = {"zlib": True, "complevel": 4}

    p_slope = OUT_DIR / f"sen_slope_{label}.nc"
    sen_slope_da.to_dataset().to_netcdf(p_slope, encoding={"sen_slope": encoding})

    p_pval = OUT_DIR / f"mk_pvalue_{label}.nc"
    pvalue_da.to_dataset().to_netcdf(p_pval, encoding={"mk_pvalue": encoding})

    p_agree = OUT_DIR / f"agreement_{label}.nc"
    agreement_da.to_dataset().to_netcdf(p_agree, encoding={"agreement_pct": encoding})

    valid = np.isfinite(sen_slope_da.values)
    if valid.any():
        sig = (pvalue_da.values < ALPHA) & valid
        print(f"      Slope: min={np.nanmin(sen_slope_da):.3f}, "
              f"median={np.nanmedian(sen_slope_da):.3f}, "
              f"max={np.nanmax(sen_slope_da):.3f}")
        print(f"      Significant: {sig.sum():,}/{valid.sum():,} ({100*sig.sum()/valid.sum():.1f}%)")
        print(f"      Mean agreement: {np.nanmean(agreement):.1f}%")
    else:
        print(f"      ⚠️ ALL NaN — algo sigue mal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variables", nargs="+", default=VARIABLES, choices=VARIABLES)
    parser.add_argument("--windows", nargs="+", default=list(WINDOWS.keys()), choices=list(WINDOWS.keys()))
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("  SCRIPT 10 v2 — Spatial trends with calendar fix")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)

    t0 = datetime.now()
    models = get_available_models()
    print(f"\n  Models: {models}")

    for variable in args.variables:
        for window_name in args.windows:
            window_spec = WINDOWS[window_name]
            if window_spec["scenario"] == "per_ssp":
                for ssp in SSPs:
                    process_combination(variable, window_name, window_spec, ssp, models)
            else:
                process_combination(variable, window_name, window_spec, None, models)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n  Total: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
