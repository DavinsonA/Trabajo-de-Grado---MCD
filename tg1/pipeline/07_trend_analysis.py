#!/usr/bin/env python3
"""
07_trend_analysis.py
Trend analysis using Mann-Kendall (with three variants) and Sen's slope
on annual anomalies of tas and hurs.

Inputs (from script 06):
    processed/annual_subregion_anomalies.csv
    processed/annual_continental_anomalies.csv
    processed/ensemble_annual_stats.csv

Outputs:
    outputs/07_trends/trend_results_per_model.csv
    outputs/07_trends/trend_results_ensemble.csv

Per (subregion x scenario x variable x model x window) and equivalent for
the ensemble median, reports:
    - MK original tau, p-value
    - MK Hamed-Rao tau, p-value         <- canonical (used in thesis tables)
    - MK Yue-Wang TFPW tau, p-value
    - Sen's slope (units per decade)
    - Sen's slope 95% confidence interval
    - intercept
    - lag-1 autocorrelation
    - n_obs
    - significance flag at alpha=0.05 (based on Hamed-Rao p)

Time windows analyzed:
    - 'historical_1950_2014': only the historical experiment
    - 'projected_2015_2100':  only the SSP experiment
    - 'full_1950_2100':       historical concatenated with the SSP, per model

Usage:
    source /media/volume/jay2vol/nexgddp/venv/bin/activate
    python 07_trend_analysis.py
"""

import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pymannkendall as mk

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("/media/volume/jay2vol/nexgddp")
PROC_DIR = BASE_DIR / "processed"
OUT_DIR = BASE_DIR / "outputs" / "07_trends"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05

SSPs = ["ssp126", "ssp245", "ssp370", "ssp585"]

WINDOWS = {
    "historical_1950_2014": {"hist": (1950, 2014), "ssp": None},
    "projected_2015_2100":  {"hist": None,         "ssp": (2015, 2100)},
    "full_1950_2100":       {"hist": (1950, 2014), "ssp": (2015, 2100)},
}


# =============================================================================
# CORE STATISTICAL HELPERS
# =============================================================================

def lag1_autocorr(y):
    """Pearson autocorrelation at lag 1. Returns NaN if undefined."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 3:
        return np.nan
    a = y[:-1] - y[:-1].mean()
    b = y[1:]  - y[1:].mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom == 0:
        return np.nan
    return float((a * b).sum() / denom)


def run_mk_full(series):
    """
    Run three Mann-Kendall variants + Sen's slope on a 1D series.
    Returns a dict with all metrics. Slope is reported per decade.

    The pymannkendall outputs include slope and intercept already; the
    95% CI for Sen's slope is computed from the empirical distribution
    of pairwise slopes (Sen's original definition).
    """
    y = np.asarray(series, dtype=float)
    mask = np.isfinite(y)
    y = y[mask]
    n = len(y)

    out = {
        "n_obs": n,
        "lag1_autocorr": lag1_autocorr(y),
        "mk_orig_tau": np.nan,        "mk_orig_p": np.nan,
        "mk_hr_tau": np.nan,          "mk_hr_p": np.nan,
        "mk_yw_tau": np.nan,          "mk_yw_p": np.nan,
        "sen_slope_per_year": np.nan,
        "sen_slope_per_decade": np.nan,
        "sen_slope_lo95_per_decade": np.nan,
        "sen_slope_hi95_per_decade": np.nan,
        "intercept": np.nan,
        "significant_005": False,
    }

    if n < 10:
        return out  # not enough data for trend tests

    # Original MK
    try:
        r = mk.original_test(y, alpha=ALPHA)
        out["mk_orig_tau"] = float(r.Tau)
        out["mk_orig_p"]   = float(r.p)
    except Exception:
        pass

    # Hamed-Rao (autocorrelation-corrected) -> canonical
    try:
        r = mk.hamed_rao_modification_test(y, alpha=ALPHA)
        out["mk_hr_tau"] = float(r.Tau)
        out["mk_hr_p"]   = float(r.p)
        out["sen_slope_per_year"] = float(r.slope)
        out["sen_slope_per_decade"] = float(r.slope) * 10.0
        out["intercept"] = float(r.intercept)
        out["significant_005"] = bool(r.p < ALPHA)
    except Exception:
        pass

    # Yue-Wang Trend-Free Pre-Whitening
    try:
        r = mk.yue_wang_modification_test(y, alpha=ALPHA)
        out["mk_yw_tau"] = float(r.Tau)
        out["mk_yw_p"]   = float(r.p)
    except Exception:
        pass

    # Sen's slope 95% CI (non-parametric, from pairwise slopes)
    try:
        idx = np.arange(n)
        diffs_y = y[None, :] - y[:, None]
        diffs_x = (idx[None, :] - idx[:, None]).astype(float)
        upper = np.triu_indices(n, k=1)
        slopes = diffs_y[upper] / diffs_x[upper]
        slopes = slopes[np.isfinite(slopes)]
        if len(slopes) >= 10:
            lo = np.percentile(slopes, 2.5) * 10.0
            hi = np.percentile(slopes, 97.5) * 10.0
            out["sen_slope_lo95_per_decade"] = float(lo)
            out["sen_slope_hi95_per_decade"] = float(hi)
    except Exception:
        pass

    return out


# =============================================================================
# SERIES BUILDERS
# =============================================================================

def build_series_per_model(df_annual, model, scenario, variable, subregion, window_spec):
    """
    Returns (years, values) for one (model, scenario, variable, subregion, window).
    For the 'full' window, concatenates historical + SSP for the same model.
    """
    rows = []

    if window_spec["hist"] is not None:
        y0, y1 = window_spec["hist"]
        df_h = df_annual[
            (df_annual["model"] == model) &
            (df_annual["scenario"] == "historical") &
            (df_annual["variable"] == variable) &
            (df_annual["subregion"] == subregion) &
            (df_annual["year"] >= y0) & (df_annual["year"] <= y1)
        ][["year", "anomaly"]]
        rows.append(df_h)

    if window_spec["ssp"] is not None:
        y0, y1 = window_spec["ssp"]
        df_s = df_annual[
            (df_annual["model"] == model) &
            (df_annual["scenario"] == scenario) &
            (df_annual["variable"] == variable) &
            (df_annual["subregion"] == subregion) &
            (df_annual["year"] >= y0) & (df_annual["year"] <= y1)
        ][["year", "anomaly"]]
        rows.append(df_s)

    if not rows:
        return np.array([]), np.array([])

    s = pd.concat(rows, ignore_index=True).sort_values("year")
    s = s.dropna(subset=["anomaly"]).drop_duplicates(subset=["year"])
    return s["year"].values, s["anomaly"].values


def build_series_ensemble(df_ens, scenario, variable, subregion, window_spec):
    """
    Returns (years, values) for the ensemble median (p50) for one combination.
    """
    rows = []

    if window_spec["hist"] is not None:
        y0, y1 = window_spec["hist"]
        df_h = df_ens[
            (df_ens["scenario"] == "historical") &
            (df_ens["variable"] == variable) &
            (df_ens["subregion"] == subregion) &
            (df_ens["year"] >= y0) & (df_ens["year"] <= y1)
        ][["year", "ens_p50"]]
        rows.append(df_h)

    if window_spec["ssp"] is not None:
        y0, y1 = window_spec["ssp"]
        df_s = df_ens[
            (df_ens["scenario"] == scenario) &
            (df_ens["variable"] == variable) &
            (df_ens["subregion"] == subregion) &
            (df_ens["year"] >= y0) & (df_ens["year"] <= y1)
        ][["year", "ens_p50"]]
        rows.append(df_s)

    if not rows:
        return np.array([]), np.array([])

    s = pd.concat(rows, ignore_index=True).sort_values("year")
    s = s.dropna(subset=["ens_p50"]).drop_duplicates(subset=["year"])
    return s["year"].values, s["ens_p50"].values


# =============================================================================
# MAIN ANALYSES
# =============================================================================

def analyze_per_model(df_annual):
    print("\n" + "=" * 70)
    print("  Per-model trend analysis")
    print("=" * 70)

    models     = sorted(df_annual["model"].unique())
    variables  = sorted(df_annual["variable"].unique())
    subregions = sorted(df_annual["subregion"].unique())

    print(f"  Models: {len(models)} | Variables: {len(variables)} "
          f"| Subregions: {len(subregions)} | Windows: {len(WINDOWS)}")

    results = []
    total = len(models) * len(variables) * len(subregions) * len(WINDOWS) * len(SSPs)
    counter = 0

    for model in models:
        for variable in variables:
            for subregion in subregions:
                for window_name, window_spec in WINDOWS.items():

                    if window_spec["ssp"] is None:
                        # historical-only window: scenario field is irrelevant
                        scenarios_to_loop = ["historical"]
                    else:
                        scenarios_to_loop = SSPs

                    for scenario in scenarios_to_loop:
                        counter += 1
                        years, vals = build_series_per_model(
                            df_annual, model, scenario, variable, subregion, window_spec
                        )

                        if len(vals) < 10:
                            continue

                        stats = run_mk_full(vals)
                        stats.update({
                            "model": model,
                            "variable": variable,
                            "subregion": subregion,
                            "scenario": scenario,
                            "window": window_name,
                            "year_start": int(years.min()),
                            "year_end": int(years.max()),
                        })
                        results.append(stats)

    df_res = pd.DataFrame(results)
    cols_order = [
        "model", "variable", "subregion", "scenario", "window",
        "year_start", "year_end", "n_obs",
        "mk_orig_tau", "mk_orig_p",
        "mk_hr_tau", "mk_hr_p",
        "mk_yw_tau", "mk_yw_p",
        "sen_slope_per_year", "sen_slope_per_decade",
        "sen_slope_lo95_per_decade", "sen_slope_hi95_per_decade",
        "intercept", "lag1_autocorr", "significant_005",
    ]
    df_res = df_res[cols_order]
    for c in df_res.select_dtypes(include=[np.number]).columns:
        df_res[c] = df_res[c].round(6)

    out_path = OUT_DIR / "trend_results_per_model.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(f"  Records: {len(df_res):,}")
    print(f"  Significant (Hamed-Rao, p<{ALPHA}): "
          f"{df_res['significant_005'].sum():,} / {len(df_res):,}")
    return df_res


def analyze_ensemble(df_ens):
    print("\n" + "=" * 70)
    print("  Ensemble-median trend analysis")
    print("=" * 70)

    variables  = sorted(df_ens["variable"].unique())
    subregions = sorted(df_ens["subregion"].unique())

    print(f"  Variables: {len(variables)} | Subregions: {len(subregions)} "
          f"| Windows: {len(WINDOWS)}")

    results = []

    for variable in variables:
        for subregion in subregions:
            for window_name, window_spec in WINDOWS.items():

                if window_spec["ssp"] is None:
                    scenarios_to_loop = ["historical"]
                else:
                    scenarios_to_loop = SSPs

                for scenario in scenarios_to_loop:
                    years, vals = build_series_ensemble(
                        df_ens, scenario, variable, subregion, window_spec
                    )

                    if len(vals) < 10:
                        continue

                    stats = run_mk_full(vals)
                    stats.update({
                        "variable": variable,
                        "subregion": subregion,
                        "scenario": scenario,
                        "window": window_name,
                        "year_start": int(years.min()),
                        "year_end": int(years.max()),
                    })
                    results.append(stats)

    df_res = pd.DataFrame(results)
    cols_order = [
        "variable", "subregion", "scenario", "window",
        "year_start", "year_end", "n_obs",
        "mk_orig_tau", "mk_orig_p",
        "mk_hr_tau", "mk_hr_p",
        "mk_yw_tau", "mk_yw_p",
        "sen_slope_per_year", "sen_slope_per_decade",
        "sen_slope_lo95_per_decade", "sen_slope_hi95_per_decade",
        "intercept", "lag1_autocorr", "significant_005",
    ]
    df_res = df_res[cols_order]
    for c in df_res.select_dtypes(include=[np.number]).columns:
        df_res[c] = df_res[c].round(6)

    out_path = OUT_DIR / "trend_results_ensemble.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(f"  Records: {len(df_res):,}")
    print(f"  Significant (Hamed-Rao, p<{ALPHA}): "
          f"{df_res['significant_005'].sum():,} / {len(df_res):,}")
    return df_res


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#" * 70)
    print("  NEX-GDDP-CMIP6 — SCRIPT 07: TREND ANALYSIS")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output dir: {OUT_DIR}")
    print("#" * 70)

    t0 = datetime.now()

    # Load inputs
    print("\n  Loading inputs...")
    annual_sub_path  = PROC_DIR / "annual_subregion_anomalies.csv"
    annual_cont_path = PROC_DIR / "annual_continental_anomalies.csv"
    ens_path         = PROC_DIR / "ensemble_annual_stats.csv"

    df_sub  = pd.read_csv(annual_sub_path)
    df_cont = pd.read_csv(annual_cont_path)
    df_ens  = pd.read_csv(ens_path)

    df_annual = pd.concat([df_sub, df_cont], ignore_index=True)
    print(f"  Annual per-model records: {len(df_annual):,}")
    print(f"  Ensemble records: {len(df_ens):,}")

    # Run analyses
    df_per_model = analyze_per_model(df_annual)
    df_ensemble  = analyze_ensemble(df_ens)

    elapsed = (datetime.now() - t0).total_seconds()

    print("\n" + "=" * 70)
    print(f"  SCRIPT 07 COMPLETED in {elapsed:.1f} seconds")
    print(f"\n  Outputs:")
    for f in sorted(OUT_DIR.glob("*.csv")):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:45s} ({size_kb:.1f} KB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
