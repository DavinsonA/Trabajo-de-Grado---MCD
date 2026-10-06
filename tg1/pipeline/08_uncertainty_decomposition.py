#!/usr/bin/env python3
"""
08_uncertainty_decomposition.py
Hawkins & Sutton (2009) variance decomposition for tas and hurs projections
in South America under SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5.

Decomposes the total variance of climate projections into three sources:
    I  - internal variability (residual around smoothed trend, ~constant in t)
    M  - model uncertainty   (between-model variance of smoothed trends)
    S  - scenario uncertainty (between-scenario variance of multi-model means)

T(t) = I + M(t) + S(t)
F_X(t) = X(t) / T(t)   with F_I + F_M + F_S = 1

Inputs (from script 06):
    processed/annual_subregion_anomalies.csv
    processed/annual_continental_anomalies.csv

Outputs:
    outputs/08_hawkins_sutton/hs_variance_components.csv
    outputs/08_hawkins_sutton/hs_summary_by_horizon.csv

Method notes:
    - Anomalies are re-referenced internally to 1995-2014 (H&S convention),
      not to the 1960-2014 baseline used elsewhere. This makes the initial
      variance close to zero, which is the canonical presentation.
    - Trends are smoothed per (model, scenario) with a degree-4 polynomial
      fit over the full 1950-2100 window (concatenating historical + SSP).
    - Internal variability I is the average over (model, scenario) of the
      residual variance, assumed constant in time.

Usage:
    source /media/volume/jay2vol/nexgddp/venv/bin/activate
    python 08_uncertainty_decomposition.py
"""

import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("/media/volume/jay2vol/nexgddp")
PROC_DIR = BASE_DIR / "processed"
OUT_DIR = BASE_DIR / "outputs" / "08_hawkins_sutton"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SSPs = ["ssp126", "ssp245", "ssp370", "ssp585"]

# Reference window for re-referencing anomalies (Hawkins & Sutton convention)
HS_REF_START = 1995
HS_REF_END = 2014

# Full analysis window
YEAR_START = 1950
YEAR_END = 2100

# Polynomial degree for trend smoothing
POLY_DEGREE = 4

# Horizons used in the summary table (mid-decade)
HORIZONS = {
    "2020s": (2020, 2029),
    "2050s": (2050, 2059),
    "2090s": (2090, 2099),
}


# =============================================================================
# SERIES BUILDERS
# =============================================================================

def build_full_series(df, model, scenario, variable, subregion):
    """
    Build the full 1950-2100 series for one (model, scenario, variable,
    subregion): historical for 1950-2014, SSP for 2015-2100.

    Returns a DataFrame indexed by year with column 'anomaly'.
    """
    h = df[(df["model"] == model) &
           (df["scenario"] == "historical") &
           (df["variable"] == variable) &
           (df["subregion"] == subregion) &
           (df["year"] >= YEAR_START) & (df["year"] <= 2014)]

    s = df[(df["model"] == model) &
           (df["scenario"] == scenario) &
           (df["variable"] == variable) &
           (df["subregion"] == subregion) &
           (df["year"] >= 2015) & (df["year"] <= YEAR_END)]

    series = pd.concat([h[["year", "anomaly"]], s[["year", "anomaly"]]],
                       ignore_index=True)
    series = series.drop_duplicates(subset=["year"]).sort_values("year")
    series = series.set_index("year")
    return series


def rereference_to_hs_baseline(series):
    """Subtract the 1995-2014 mean from the series."""
    ref_mean = series.loc[HS_REF_START:HS_REF_END, "anomaly"].mean()
    series = series.copy()
    series["anomaly"] = series["anomaly"] - ref_mean
    return series


# =============================================================================
# CORE H&S DECOMPOSITION
# =============================================================================

def smooth_trend(years, values, degree=POLY_DEGREE):
    """Fit polynomial of given degree; return smoothed values at the same years."""
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() < degree + 2:
        return np.full_like(values, np.nan)
    coefs = np.polyfit(years[mask] - years[mask].mean(),
                       values[mask], degree)
    smoothed = np.polyval(coefs, years - years[mask].mean())
    return smoothed


def decompose_one_combination(df_annual, variable, subregion, models):
    """
    Perform H&S decomposition for one (variable, subregion).

    Steps:
        1. Build series per (model, scenario), re-reference to 1995-2014.
        2. Smooth each series with degree-4 polynomial -> trends X[model,scenario,t].
        3. Compute residuals -> internal variability I (single number per
           model-scenario, then averaged).
        4. For each year and scenario, compute multi-model mean of trends
           -> mu[scenario, t].
        5. Model uncertainty M(t): average over scenarios of the between-model
           variance of trends.
        6. Scenario uncertainty S(t): between-scenario variance of mu[scenario,t].
        7. Total T(t) = I + M(t) + S(t) and fractions.

    Returns a DataFrame indexed by year with columns: I, M, S, T,
    F_I, F_M, F_S, signal (mean of multi-model means across scenarios).
    """
    years = np.arange(YEAR_START, YEAR_END + 1)
    n_years = len(years)
    n_models = len(models)
    n_scen = len(SSPs)

    # trends[s, m, t] and residual_vars[s, m]
    trends = np.full((n_scen, n_models, n_years), np.nan)
    residual_vars = np.full((n_scen, n_models), np.nan)

    for si, scenario in enumerate(SSPs):
        for mi, model in enumerate(models):
            series = build_full_series(df_annual, model, scenario,
                                       variable, subregion)
            if len(series) < 50:
                continue
            series = rereference_to_hs_baseline(series)
            series = series.reindex(years)

            y = series["anomaly"].values
            sm = smooth_trend(years, y, degree=POLY_DEGREE)
            trends[si, mi, :] = sm

            resid = y - sm
            resid = resid[np.isfinite(resid)]
            if len(resid) > 10:
                residual_vars[si, mi] = np.var(resid, ddof=1)

    # I: average residual variance across all (scenario, model). Constant in t.
    I_const = np.nanmean(residual_vars)

    # mu[s, t]: multi-model mean of trends per scenario per year
    mu_sm = np.nanmean(trends, axis=1)  # shape (n_scen, n_years)

    # M(t): average over scenarios of between-model variance of trends
    M_t = np.nanmean(np.nanvar(trends, axis=1, ddof=1), axis=0)

    # S(t): between-scenario variance of mu_sm
    S_t = np.nanvar(mu_sm, axis=0, ddof=1)

    # signal: mean across scenarios of the multi-model means
    signal = np.nanmean(mu_sm, axis=0)

    T_t = I_const + M_t + S_t

    with np.errstate(divide="ignore", invalid="ignore"):
        F_I = I_const / T_t
        F_M = M_t / T_t
        F_S = S_t / T_t

    out = pd.DataFrame({
        "variable": variable,
        "subregion": subregion,
        "year": years,
        "I": I_const,
        "M": M_t,
        "S": S_t,
        "T": T_t,
        "F_I": F_I,
        "F_M": F_M,
        "F_S": F_S,
        "signal": signal,
    })
    return out


# =============================================================================
# DRIVERS
# =============================================================================

def run_decomposition(df_annual):
    print("\n" + "=" * 70)
    print("  Hawkins & Sutton variance decomposition")
    print("=" * 70)

    models = sorted(df_annual["model"].unique())
    variables = sorted(df_annual["variable"].unique())
    subregions = sorted(df_annual["subregion"].unique())

    print(f"  Models: {len(models)} | Variables: {len(variables)} "
          f"| Subregions: {len(subregions)}")
    print(f"  SSPs: {SSPs}")
    print(f"  Polynomial degree: {POLY_DEGREE}")
    print(f"  HS reference window: {HS_REF_START}-{HS_REF_END}")

    all_results = []
    for variable in variables:
        for subregion in subregions:
            print(f"  [{variable}] {subregion}")
            df_one = decompose_one_combination(df_annual, variable, subregion, models)
            all_results.append(df_one)

    df_components = pd.concat(all_results, ignore_index=True)
    for c in ["I", "M", "S", "T", "F_I", "F_M", "F_S", "signal"]:
        df_components[c] = df_components[c].round(6)

    out_path = OUT_DIR / "hs_variance_components.csv"
    df_components.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(f"  Records: {len(df_components):,}")
    return df_components


def build_horizon_summary(df_components):
    print("\n" + "=" * 70)
    print("  Horizon summary (2020s, 2050s, 2090s)")
    print("=" * 70)

    rows = []
    for (variable, subregion), grp in df_components.groupby(["variable", "subregion"]):
        for horizon_name, (y0, y1) in HORIZONS.items():
            sub = grp[(grp["year"] >= y0) & (grp["year"] <= y1)]
            if len(sub) == 0:
                continue
            rows.append({
                "variable": variable,
                "subregion": subregion,
                "horizon": horizon_name,
                "year_start": y0,
                "year_end": y1,
                "I_mean": sub["I"].mean(),
                "M_mean": sub["M"].mean(),
                "S_mean": sub["S"].mean(),
                "T_mean": sub["T"].mean(),
                "F_I_mean": sub["F_I"].mean(),
                "F_M_mean": sub["F_M"].mean(),
                "F_S_mean": sub["F_S"].mean(),
                "signal_mean": sub["signal"].mean(),
            })

    df_summary = pd.DataFrame(rows)
    for c in df_summary.select_dtypes(include=[np.number]).columns:
        df_summary[c] = df_summary[c].round(6)

    out_path = OUT_DIR / "hs_summary_by_horizon.csv"
    df_summary.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    print(f"  Records: {len(df_summary):,}")

    print("\n  Fraction summary (F_I / F_M / F_S) by horizon:")
    pivot = df_summary.pivot_table(
        index=["variable", "subregion"],
        columns="horizon",
        values=["F_I_mean", "F_M_mean", "F_S_mean"],
    ).round(3)
    print(pivot.to_string())

    return df_summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#" * 70)
    print("  NEX-GDDP-CMIP6 — SCRIPT 08: HAWKINS-SUTTON DECOMPOSITION")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output dir: {OUT_DIR}")
    print("#" * 70)

    t0 = datetime.now()

    print("\n  Loading inputs...")
    df_sub = pd.read_csv(PROC_DIR / "annual_subregion_anomalies.csv")
    df_cont = pd.read_csv(PROC_DIR / "annual_continental_anomalies.csv")
    df_annual = pd.concat([df_sub, df_cont], ignore_index=True)
    print(f"  Annual per-model records: {len(df_annual):,}")

    df_components = run_decomposition(df_annual)
    df_summary = build_horizon_summary(df_components)

    elapsed = (datetime.now() - t0).total_seconds()

    print("\n" + "=" * 70)
    print(f"  SCRIPT 08 COMPLETED in {elapsed:.1f} seconds")
    print(f"\n  Outputs:")
    for f in sorted(OUT_DIR.glob("*.csv")):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:45s} ({size_kb:.1f} KB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
