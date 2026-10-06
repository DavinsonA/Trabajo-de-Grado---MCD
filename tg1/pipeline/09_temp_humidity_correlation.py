#!/usr/bin/env python3
"""
09_temp_humidity_correlation.py
Temperature-humidity correlation analysis for regime-change detection
(Objective II).

Computes Spearman, Pearson, and Kendall correlations between tas and hurs
monthly anomalies for each (subregion x period x scenario), per-model and
on the ensemble median.

A weakening or sign change of T-H correlation across periods is interpreted
as evidence of regional hydrological regime change (decoupling from
Clausius-Clapeyron equilibrium).

Inputs (from script 06):
    processed/monthly_subregion_anomalies.csv

Outputs:
    outputs/09_correlation/corr_th_per_model.csv
    outputs/09_correlation/corr_th_ensemble.csv

Periods analyzed:
    - baseline:    1960-2014, historical only
    - recent:      1995-2014, historical only
    - mid_century: 2040-2069, per SSP
    - end_century: 2070-2099, per SSP

Method notes:
    - Correlations computed on monthly anomalies (not raw values) to remove
      the seasonal cycle, which otherwise dominates T-H correlation.
    - 95% confidence intervals via Fisher z-transform.
    - Ensemble median is computed first (across models, per month), then
      correlations are computed on the resulting median series.

Usage:
    source /media/volume/jay2vol/nexgddp/venv/bin/activate
    python 09_temp_humidity_correlation.py
"""

import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("/media/volume/jay2vol/nexgddp")
PROC_DIR = BASE_DIR / "processed"
OUT_DIR = BASE_DIR / "outputs" / "09_correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SSPs = ["ssp126", "ssp245", "ssp370", "ssp585"]

PERIODS = {
    "baseline":    {"years": (1960, 2014), "scope": "historical"},
    "recent":      {"years": (1995, 2014), "scope": "historical"},
    "mid_century": {"years": (2040, 2069), "scope": "ssp"},
    "end_century": {"years": (2070, 2099), "scope": "ssp"},
}


# =============================================================================
# CORE CORRELATION HELPERS
# =============================================================================

def fisher_ci(r, n, alpha=0.05):
    """95% CI for a correlation coefficient via Fisher z-transform."""
    if n < 4 or not np.isfinite(r) or abs(r) >= 1.0:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return float(lo), float(hi)


def compute_all_correlations(x, y):
    """Returns dict with Spearman, Pearson, Kendall + CIs and n."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)

    out = {
        "n": n,
        "spearman_r": np.nan, "spearman_p": np.nan,
        "spearman_lo95": np.nan, "spearman_hi95": np.nan,
        "pearson_r": np.nan,  "pearson_p": np.nan,
        "pearson_lo95": np.nan, "pearson_hi95": np.nan,
        "kendall_tau": np.nan, "kendall_p": np.nan,
    }

    if n < 10:
        return out

    try:
        r, p = stats.spearmanr(x, y)
        lo, hi = fisher_ci(r, n)
        out["spearman_r"] = float(r); out["spearman_p"] = float(p)
        out["spearman_lo95"] = lo; out["spearman_hi95"] = hi
    except Exception:
        pass

    try:
        r, p = stats.pearsonr(x, y)
        lo, hi = fisher_ci(r, n)
        out["pearson_r"] = float(r); out["pearson_p"] = float(p)
        out["pearson_lo95"] = lo; out["pearson_hi95"] = hi
    except Exception:
        pass

    try:
        tau, p = stats.kendalltau(x, y)
        out["kendall_tau"] = float(tau); out["kendall_p"] = float(p)
    except Exception:
        pass

    return out


# =============================================================================
# DATA BUILDERS
# =============================================================================

def build_paired_series_per_model(df_monthly, model, scenario, subregion, period_spec):
    """
    For one (model, scenario, subregion, period), returns aligned tas and hurs
    anomaly arrays (one entry per month in the period).
    """
    y0, y1 = period_spec["years"]

    df_sub = df_monthly[
        (df_monthly["model"] == model) &
        (df_monthly["scenario"] == scenario) &
        (df_monthly["subregion"] == subregion) &
        (df_monthly["year"] >= y0) & (df_monthly["year"] <= y1)
    ]

    tas = df_sub[df_sub["variable"] == "tas"][["year", "month", "anomaly"]]
    tas = tas.rename(columns={"anomaly": "tas_anom"})

    hurs = df_sub[df_sub["variable"] == "hurs"][["year", "month", "anomaly"]]
    hurs = hurs.rename(columns={"anomaly": "hurs_anom"})

    merged = tas.merge(hurs, on=["year", "month"], how="inner")
    merged = merged.dropna(subset=["tas_anom", "hurs_anom"])
    return merged["tas_anom"].values, merged["hurs_anom"].values


def build_paired_series_ensemble(df_monthly, scenario, subregion, period_spec):
    """
    Builds the ensemble median series per month (across models), then returns
    aligned tas and hurs arrays for the period.
    """
    y0, y1 = period_spec["years"]

    df_sub = df_monthly[
        (df_monthly["scenario"] == scenario) &
        (df_monthly["subregion"] == subregion) &
        (df_monthly["year"] >= y0) & (df_monthly["year"] <= y1)
    ]

    ens = df_sub.groupby(["variable", "year", "month"])["anomaly"].median().reset_index()
    ens = ens.rename(columns={"anomaly": "ens_anom"})

    tas = ens[ens["variable"] == "tas"][["year", "month", "ens_anom"]]
    tas = tas.rename(columns={"ens_anom": "tas_anom"})

    hurs = ens[ens["variable"] == "hurs"][["year", "month", "ens_anom"]]
    hurs = hurs.rename(columns={"ens_anom": "hurs_anom"})

    merged = tas.merge(hurs, on=["year", "month"], how="inner")
    merged = merged.dropna(subset=["tas_anom", "hurs_anom"])
    return merged["tas_anom"].values, merged["hurs_anom"].values


# =============================================================================
# MAIN ANALYSES
# =============================================================================

def analyze_per_model(df_monthly):
    print("\n" + "=" * 70)
    print("  Per-model T-H correlation analysis")
    print("=" * 70)

    models     = sorted(df_monthly["model"].unique())
    subregions = sorted(df_monthly["subregion"].unique())

    print(f"  Models: {len(models)} | Subregions: {len(subregions)} "
          f"| Periods: {len(PERIODS)}")

    results = []

    for model in models:
        for subregion in subregions:
            for period_name, period_spec in PERIODS.items():
                if period_spec["scope"] == "historical":
                    scenarios_to_loop = ["historical"]
                else:
                    scenarios_to_loop = SSPs

                for scenario in scenarios_to_loop:
                    tas, hurs = build_paired_series_per_model(
                        df_monthly, model, scenario, subregion, period_spec
                    )
                    if len(tas) < 10:
                        continue

                    stats_dict = compute_all_correlations(tas, hurs)
                    stats_dict.update({
                        "model": model,
                        "subregion": subregion,
                        "period": period_name,
                        "scenario": scenario,
                        "year_start": period_spec["years"][0],
                        "year_end": period_spec["years"][1],
                    })
                    results.append(stats_dict)

    df_res = pd.DataFrame(results)
    cols_order = [
        "model", "subregion", "period", "scenario",
        "year_start", "year_end", "n",
        "spearman_r", "spearman_p", "spearman_lo95", "spearman_hi95",
        "pearson_r", "pearson_p", "pearson_lo95", "pearson_hi95",
        "kendall_tau", "kendall_p",
    ]
    df_res = df_res[cols_order]
    for c in df_res.select_dtypes(include=[np.number]).columns:
        df_res[c] = df_res[c].round(6)

    out_path = OUT_DIR / "corr_th_per_model.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(f"  Records: {len(df_res):,}")
    return df_res


def analyze_ensemble(df_monthly):
    print("\n" + "=" * 70)
    print("  Ensemble-median T-H correlation analysis")
    print("=" * 70)

    subregions = sorted(df_monthly["subregion"].unique())

    print(f"  Subregions: {len(subregions)} | Periods: {len(PERIODS)}")

    results = []

    for subregion in subregions:
        for period_name, period_spec in PERIODS.items():
            if period_spec["scope"] == "historical":
                scenarios_to_loop = ["historical"]
            else:
                scenarios_to_loop = SSPs

            for scenario in scenarios_to_loop:
                tas, hurs = build_paired_series_ensemble(
                    df_monthly, scenario, subregion, period_spec
                )
                if len(tas) < 10:
                    continue

                stats_dict = compute_all_correlations(tas, hurs)
                stats_dict.update({
                    "subregion": subregion,
                    "period": period_name,
                    "scenario": scenario,
                    "year_start": period_spec["years"][0],
                    "year_end": period_spec["years"][1],
                })
                results.append(stats_dict)

    df_res = pd.DataFrame(results)
    cols_order = [
        "subregion", "period", "scenario",
        "year_start", "year_end", "n",
        "spearman_r", "spearman_p", "spearman_lo95", "spearman_hi95",
        "pearson_r", "pearson_p", "pearson_lo95", "pearson_hi95",
        "kendall_tau", "kendall_p",
    ]
    df_res = df_res[cols_order]
    for c in df_res.select_dtypes(include=[np.number]).columns:
        df_res[c] = df_res[c].round(6)

    out_path = OUT_DIR / "corr_th_ensemble.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(f"  Records: {len(df_res):,}")
    return df_res


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#" * 70)
    print("  NEX-GDDP-CMIP6 — SCRIPT 09: T-H CORRELATION ANALYSIS")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output dir: {OUT_DIR}")
    print("#" * 70)

    t0 = datetime.now()

    print("\n  Loading inputs...")
    df_monthly = pd.read_csv(PROC_DIR / "monthly_subregion_anomalies.csv")
    print(f"  Monthly records: {len(df_monthly):,}")

    df_per_model = analyze_per_model(df_monthly)
    df_ensemble  = analyze_ensemble(df_monthly)

    elapsed = (datetime.now() - t0).total_seconds()

    print("\n" + "=" * 70)
    print(f"  SCRIPT 09 COMPLETED in {elapsed:.1f} seconds")
    print(f"\n  Outputs:")
    for f in sorted(OUT_DIR.glob("*.csv")):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:45s} ({size_kb:.1f} KB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
