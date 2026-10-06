"""
analysis_lib.py
Reusable analysis functions for the NEX-GDDP-CMIP6 thesis notebook.

This module contains only pure functions: they take in-memory data
(DataFrames, ndarrays, xarray objects) and return in-memory results.
No file I/O, no prints, no hard-coded paths.

The notebook that imports this module is responsible for:
    - reading CSVs and NetCDFs from disk (once, at startup)
    - caching them in memory
    - invoking these functions whenever widgets change
    - rendering the returned data with hvPlot / Panel / Cartopy

Sections:
    1. Constants & lightweight helpers
    2. Series builders (annual & monthly, per-model & ensemble)
    3. Trend tests (Mann-Kendall variants + Sen's slope)
    4. Correlation analysis (Spearman / Pearson / Kendall + Fisher CI)
    5. Hawkins & Sutton variance decomposition
    6. Spatial field loaders (for Cartopy maps)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

try:
    import pymannkendall as mk
    _HAS_PYMK = True
except ImportError:  # pragma: no cover
    _HAS_PYMK = False

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONSTANTS & LIGHTWEIGHT HELPERS
# =============================================================================

SSPs = ("ssp126", "ssp245", "ssp370", "ssp585")
VARIABLES = ("tas", "hurs")

SUBREGIONS = {
    "Northern Tropics":    {"lat_min": 0.0,   "lat_max": 13.0},
    "Central Amazonia":    {"lat_min": -15.0,  "lat_max": 0.0},
    "Subtropical Andes":   {"lat_min": -30.0,  "lat_max": -15.0},
    "Southern Cone":       {"lat_min": -56.0,  "lat_max": -30.0},
}

VAR_UNITS = {
    "tas":  {"value": "°C",       "slope": "°C/decade"},
    "hurs": {"value": "%",        "slope": "%/decade"},
}

DEFAULT_ALPHA = 0.05


def subset_lat(ds, lat_min: float, lat_max: float):
    """Boolean-mask subset on latitude. Use instead of slice() for
    xarray versions that don't accept floats in slice()."""
    mask = (ds.lat >= lat_min) & (ds.lat <= lat_max)
    return ds.sel(lat=mask)


def lag1_autocorr(y) -> float:
    """Pearson autocorrelation at lag 1. NaN if undefined."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 3:
        return float("nan")
    a = y[:-1] - y[:-1].mean()
    b = y[1:]  - y[1:].mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((a * b).sum() / denom)


def fisher_ci(r: float, n: int, alpha: float = DEFAULT_ALPHA):
    """95% (default) confidence interval for a correlation coefficient
    via Fisher z-transform. Returns (lo, hi) or (nan, nan) if undefined."""
    if n < 4 or not np.isfinite(r) or abs(r) >= 1.0:
        return float("nan"), float("nan")
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo = float(np.tanh(z - z_crit * se))
    hi = float(np.tanh(z + z_crit * se))
    return lo, hi


# =============================================================================
# 2. SERIES BUILDERS
# =============================================================================

def build_annual_series_per_model(
    df_annual: pd.DataFrame,
    model: str,
    scenario: str,
    variable: str,
    subregion: str,
    year_range: tuple[int, int] | None = None,
    use_historical_before_2015: bool = True,
):
    """
    Build the annual anomaly series for one (model, scenario, variable, subregion).

    If year_range crosses 2015 and use_historical_before_2015 is True,
    pre-2015 years come from the historical experiment and post-2015 from
    the chosen SSP scenario (the standard convention).

    Returns
    -------
    years  : 1D ndarray of int
    values : 1D ndarray of float (anomaly)
    """
    y0, y1 = year_range if year_range is not None else (1950, 2100)
    parts = []

    if use_historical_before_2015 and y0 <= 2014 and scenario != "historical":
        h = df_annual[
            (df_annual["model"] == model) &
            (df_annual["scenario"] == "historical") &
            (df_annual["variable"] == variable) &
            (df_annual["subregion"] == subregion) &
            (df_annual["year"] >= y0) & (df_annual["year"] <= min(2014, y1))
        ][["year", "anomaly"]]
        parts.append(h)

        if y1 >= 2015:
            s = df_annual[
                (df_annual["model"] == model) &
                (df_annual["scenario"] == scenario) &
                (df_annual["variable"] == variable) &
                (df_annual["subregion"] == subregion) &
                (df_annual["year"] >= 2015) & (df_annual["year"] <= y1)
            ][["year", "anomaly"]]
            parts.append(s)
    else:
        s = df_annual[
            (df_annual["model"] == model) &
            (df_annual["scenario"] == scenario) &
            (df_annual["variable"] == variable) &
            (df_annual["subregion"] == subregion) &
            (df_annual["year"] >= y0) & (df_annual["year"] <= y1)
        ][["year", "anomaly"]]
        parts.append(s)

    if not parts:
        return np.array([]), np.array([])

    s = pd.concat(parts, ignore_index=True).sort_values("year")
    s = s.dropna(subset=["anomaly"]).drop_duplicates(subset=["year"])
    return s["year"].values.astype(int), s["anomaly"].values.astype(float)


def build_annual_series_ensemble(
    df_annual: pd.DataFrame,
    scenario: str,
    variable: str,
    subregion: str,
    year_range: tuple[int, int] | None = None,
    selected_models: Sequence[str] | None = None,
    use_historical_before_2015: bool = True,
) -> pd.DataFrame:
    """
    Build ensemble statistics across the selected models, on the fly.

    The notebook calls this whenever the user changes the model multi-select
    widget. Percentiles are recomputed from the subset of selected models
    (not read from the pre-computed ensemble_annual_stats.csv).

    Returns
    -------
    DataFrame with columns: year, n_models, ens_mean, ens_p10, ens_p25,
                             ens_p50, ens_p75, ens_p90, ens_min, ens_max
    """
    all_models = sorted(df_annual["model"].unique())
    if selected_models is None:
        selected_models = all_models
    selected_models = list(selected_models)

    rows = []
    for model in selected_models:
        years, values = build_annual_series_per_model(
            df_annual, model, scenario, variable, subregion,
            year_range=year_range,
            use_historical_before_2015=use_historical_before_2015,
        )
        if len(years) == 0:
            continue
        rows.append(pd.DataFrame({
            "model": model, "year": years, "anomaly": values
        }))

    if not rows:
        return pd.DataFrame(columns=[
            "year", "n_models", "ens_mean", "ens_p10", "ens_p25",
            "ens_p50", "ens_p75", "ens_p90", "ens_min", "ens_max"
        ])

    long = pd.concat(rows, ignore_index=True)

    stats_df = long.groupby("year").agg(
        n_models=("model", "nunique"),
        ens_mean=("anomaly", "mean"),
        ens_p10=("anomaly", lambda x: np.percentile(x, 10)),
        ens_p25=("anomaly", lambda x: np.percentile(x, 25)),
        ens_p50=("anomaly", "median"),
        ens_p75=("anomaly", lambda x: np.percentile(x, 75)),
        ens_p90=("anomaly", lambda x: np.percentile(x, 90)),
        ens_min=("anomaly", "min"),
        ens_max=("anomaly", "max"),
    ).reset_index()

    return stats_df


def build_monthly_paired_th(
    df_monthly: pd.DataFrame,
    scenario: str,
    subregion: str,
    year_range: tuple[int, int],
    selected_models: Sequence[str] | None = None,
    use_ensemble_median: bool = True,
    use_anomalies: bool = True,
):
    """
    Build paired tas/hurs monthly series for T-H correlation analysis.

    If use_ensemble_median is True, computes the median across selected
    models first, then returns paired series. Otherwise returns long-format
    paired series with one row per (model, year, month).

    Parameters
    ----------
    use_anomalies : if True, uses 'anomaly' column; else 'value'

    Returns
    -------
    tas_arr, hurs_arr : 1D ndarrays of equal length
    """
    y0, y1 = year_range
    col = "anomaly" if use_anomalies else "value"

    df = df_monthly[
        (df_monthly["scenario"] == scenario) &
        (df_monthly["subregion"] == subregion) &
        (df_monthly["year"] >= y0) & (df_monthly["year"] <= y1)
    ]

    if selected_models is not None:
        df = df[df["model"].isin(list(selected_models))]

    if use_ensemble_median:
        ens = df.groupby(["variable", "year", "month"])[col].median().reset_index()
        ens = ens.rename(columns={col: "val"})

        tas = ens[ens["variable"] == "tas"][["year", "month", "val"]]
        tas = tas.rename(columns={"val": "tas_val"})
        hurs = ens[ens["variable"] == "hurs"][["year", "month", "val"]]
        hurs = hurs.rename(columns={"val": "hurs_val"})

        merged = tas.merge(hurs, on=["year", "month"], how="inner")
    else:
        tas = df[df["variable"] == "tas"][["model", "year", "month", col]]
        tas = tas.rename(columns={col: "tas_val"})
        hurs = df[df["variable"] == "hurs"][["model", "year", "month", col]]
        hurs = hurs.rename(columns={col: "hurs_val"})
        merged = tas.merge(hurs, on=["model", "year", "month"], how="inner")

    merged = merged.dropna(subset=["tas_val", "hurs_val"])
    return merged["tas_val"].values, merged["hurs_val"].values


# =============================================================================
# 3. TREND TESTS — MANN-KENDALL + SEN'S SLOPE
# =============================================================================

def run_mk_full(series, alpha: float = DEFAULT_ALPHA, min_n: int = 10) -> dict:
    """
    Run three Mann-Kendall variants + Sen's slope on a 1D series.

    Variants:
        - mk_orig  : original MK (assumes independence)
        - mk_hr    : Hamed-Rao modification (canonical for autocorrelated series)
        - mk_yw    : Yue-Wang trend-free pre-whitening

    Sen's slope is reported in two scales: per year and per decade.
    95% CI for Sen's slope is computed from the empirical distribution of
    pairwise slopes (Sen's original definition, robust to autocorrelation).

    Returns
    -------
    dict with keys: n_obs, lag1_autocorr, mk_orig_tau, mk_orig_p,
                    mk_hr_tau, mk_hr_p, mk_yw_tau, mk_yw_p,
                    sen_slope_per_year, sen_slope_per_decade,
                    sen_slope_lo95_per_decade, sen_slope_hi95_per_decade,
                    intercept, significant_005
    """
    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)

    out = {
        "n_obs": n,
        "lag1_autocorr": lag1_autocorr(y),
        "mk_orig_tau": np.nan, "mk_orig_p": np.nan,
        "mk_hr_tau": np.nan,   "mk_hr_p": np.nan,
        "mk_yw_tau": np.nan,   "mk_yw_p": np.nan,
        "sen_slope_per_year": np.nan,
        "sen_slope_per_decade": np.nan,
        "sen_slope_lo95_per_decade": np.nan,
        "sen_slope_hi95_per_decade": np.nan,
        "intercept": np.nan,
        "significant_005": False,
    }

    if n < min_n or not _HAS_PYMK:
        return out

    try:
        r = mk.original_test(y, alpha=alpha)
        out["mk_orig_tau"] = float(r.Tau)
        out["mk_orig_p"]   = float(r.p)
    except Exception:
        pass

    try:
        r = mk.hamed_rao_modification_test(y, alpha=alpha)
        out["mk_hr_tau"] = float(r.Tau)
        out["mk_hr_p"]   = float(r.p)
        out["sen_slope_per_year"]   = float(r.slope)
        out["sen_slope_per_decade"] = float(r.slope) * 10.0
        out["intercept"] = float(r.intercept)
        out["significant_005"] = bool(r.p < alpha)
    except Exception:
        pass

    try:
        r = mk.yue_wang_modification_test(y, alpha=alpha)
        out["mk_yw_tau"] = float(r.Tau)
        out["mk_yw_p"]   = float(r.p)
    except Exception:
        pass

    # Sen's slope 95% CI from pairwise distribution
    try:
        idx = np.arange(n)
        diffs_y = y[None, :] - y[:, None]
        diffs_x = (idx[None, :] - idx[:, None]).astype(float)
        upper = np.triu_indices(n, k=1)
        slopes = diffs_y[upper] / diffs_x[upper]
        slopes = slopes[np.isfinite(slopes)]
        if len(slopes) >= 10:
            out["sen_slope_lo95_per_decade"] = float(np.percentile(slopes, 2.5) * 10.0)
            out["sen_slope_hi95_per_decade"] = float(np.percentile(slopes, 97.5) * 10.0)
    except Exception:
        pass

    return out


def trend_line_from_sen(years, slope_per_year: float, intercept: float):
    """Reconstruct the trend line y = intercept + slope * (year - year[0])
    on the same year axis. Useful for overlaying on hvplot.line."""
    years = np.asarray(years, dtype=float)
    if not np.isfinite(slope_per_year) or not np.isfinite(intercept):
        return np.full_like(years, np.nan, dtype=float)
    # pymannkendall returns slope/intercept in normalized index, not year scale.
    # The line is: y_hat = intercept + slope * i, where i is 0..n-1.
    i = np.arange(len(years), dtype=float)
    return intercept + slope_per_year * i


# =============================================================================
# 4. CORRELATION ANALYSIS
# =============================================================================

def compute_all_correlations(x, y, alpha: float = DEFAULT_ALPHA) -> dict:
    """
    Compute Spearman, Pearson, and Kendall correlations between x and y,
    with 95% confidence intervals for Spearman and Pearson via Fisher z.

    Returns dict with keys:
        n, spearman_r, spearman_p, spearman_lo95, spearman_hi95,
        pearson_r, pearson_p, pearson_lo95, pearson_hi95,
        kendall_tau, kendall_p
    """
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
        lo, hi = fisher_ci(r, n, alpha=alpha)
        out["spearman_r"] = float(r); out["spearman_p"] = float(p)
        out["spearman_lo95"] = lo; out["spearman_hi95"] = hi
    except Exception:
        pass

    try:
        r, p = stats.pearsonr(x, y)
        lo, hi = fisher_ci(r, n, alpha=alpha)
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
# 5. HAWKINS & SUTTON DECOMPOSITION
# =============================================================================

def smooth_trend(years, values, degree: int = 4):
    """Polynomial fit of given degree centered for numerical stability."""
    years  = np.asarray(years,  dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() < degree + 2:
        return np.full_like(values, np.nan)
    coefs = np.polyfit(years[mask] - years[mask].mean(), values[mask], degree)
    return np.polyval(coefs, years - years[mask].mean())


def hawkins_sutton_decompose(
    df_annual: pd.DataFrame,
    variable: str,
    subregion: str,
    selected_models: Sequence[str] | None = None,
    scenarios: Sequence[str] = SSPs,
    year_start: int = 1950,
    year_end: int = 2100,
    hs_ref_start: int = 1995,
    hs_ref_end: int = 2014,
    poly_degree: int = 4,
) -> pd.DataFrame:
    """
    Compute Hawkins & Sutton (2009) variance decomposition for one
    (variable, subregion). All series are re-referenced to the
    [hs_ref_start, hs_ref_end] window before decomposition.

    Parameters
    ----------
    df_annual : annual anomalies DataFrame (already in tas in °C, hurs in %)
    selected_models : if None, uses all available models in df_annual

    Returns
    -------
    DataFrame with one row per year, columns:
        year, I, M, S, T, F_I, F_M, F_S, signal
    """
    all_models = sorted(df_annual["model"].unique())
    if selected_models is None:
        selected_models = all_models
    selected_models = list(selected_models)
    scenarios = list(scenarios)

    years = np.arange(year_start, year_end + 1)
    n_years = len(years)
    n_models = len(selected_models)
    n_scen = len(scenarios)

    if n_models < 2 or n_scen < 2:
        return pd.DataFrame()  # not enough to decompose

    trends = np.full((n_scen, n_models, n_years), np.nan)
    resid_vars = np.full((n_scen, n_models), np.nan)

    for si, scenario in enumerate(scenarios):
        for mi, model in enumerate(selected_models):
            yrs, vals = build_annual_series_per_model(
                df_annual, model, scenario, variable, subregion,
                year_range=(year_start, year_end),
                use_historical_before_2015=True,
            )
            if len(yrs) < 50:
                continue

            # Re-reference to HS baseline
            ref_mask = (yrs >= hs_ref_start) & (yrs <= hs_ref_end)
            if ref_mask.sum() < 5:
                continue
            ref_mean = np.nanmean(vals[ref_mask])
            vals = vals - ref_mean

            # Reindex into the full years axis
            full_vals = np.full(n_years, np.nan)
            idx = np.searchsorted(years, yrs)
            full_vals[idx] = vals

            sm = smooth_trend(years, full_vals, degree=poly_degree)
            trends[si, mi, :] = sm

            resid = full_vals - sm
            resid = resid[np.isfinite(resid)]
            if len(resid) > 10:
                resid_vars[si, mi] = np.var(resid, ddof=1)

    I_const = np.nanmean(resid_vars)
    mu_sm = np.nanmean(trends, axis=1)                       # (n_scen, n_years)
    M_t   = np.nanmean(np.nanvar(trends, axis=1, ddof=1), axis=0)
    S_t   = np.nanvar(mu_sm, axis=0, ddof=1)
    signal = np.nanmean(mu_sm, axis=0)
    T_t = I_const + M_t + S_t

    with np.errstate(divide="ignore", invalid="ignore"):
        F_I = I_const / T_t
        F_M = M_t / T_t
        F_S = S_t / T_t

    return pd.DataFrame({
        "year": years,
        "I": I_const, "M": M_t, "S": S_t, "T": T_t,
        "F_I": F_I, "F_M": F_M, "F_S": F_S,
        "signal": signal,
    })


# =============================================================================
# 6. SPATIAL FIELD LOADERS
# =============================================================================

def build_spatial_label(variable: str, window: str, scenario: str | None = None) -> str:
    """Build the file label used by script 10's outputs."""
    if scenario is None:
        return f"{variable}__{window}"
    return f"{variable}__{window}__{scenario}"


def load_spatial_trend_field(
    spatial_dir: str | Path,
    variable: str,
    window: str,
    scenario: str | None = None,
) -> xr.Dataset:
    """
    Load the three pre-computed 2D fields produced by script 10:
        - sen_slope    (units/decade)
        - mk_pvalue    (dimensionless)
        - agreement    (percent)

    Returns a single xarray.Dataset with all three variables. The notebook
    uses this for Cartopy maps (slope as base layer, p-value < 0.05 as
    significance overlay, agreement as optional hatching).

    Parameters
    ----------
    window : one of 'historical_1950_2014', 'projected_2015_2100',
             'full_1950_2100'
    scenario : required for 'projected_*' and 'full_*' windows; must be None
               for 'historical_1950_2014'
    """
    spatial_dir = Path(spatial_dir)
    label = build_spatial_label(variable, window, scenario)

    slope_path = spatial_dir / f"sen_slope_{label}.nc"
    pvalue_path = spatial_dir / f"mk_pvalue_{label}.nc"
    agree_path  = spatial_dir / f"agreement_{label}.nc"

    for p in (slope_path, pvalue_path, agree_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing spatial field: {p}")

    ds_slope = xr.open_dataset(slope_path)
    ds_pval  = xr.open_dataset(pvalue_path)
    ds_agree = xr.open_dataset(agree_path)

    out = xr.Dataset({
        "sen_slope": ds_slope["sen_slope"],
        "mk_pvalue": ds_pval["mk_pvalue"],
        "agreement_pct": ds_agree["agreement_pct"],
    })
    out.attrs["variable"] = variable
    out.attrs["window"]   = window
    out.attrs["scenario"] = scenario if scenario else "n/a"
    out.attrs["slope_units"] = VAR_UNITS[variable]["slope"]
    return out


def load_decadal_spatial_anomaly(
    spatial_dir: str | Path,
    variable: str,
    decade: str,
    scenario: str,
) -> xr.Dataset:
    """
    Load the decadal spatial anomaly NetCDF produced by script 06 step 6.

    File pattern: decade_<variable>_<decade>_<scenario>.nc
    Example:      decade_tas_2050s_ssp585.nc

    Returns Dataset with variables: <variable>_anomaly, <variable>_mean
    """
    spatial_dir = Path(spatial_dir)
    p = spatial_dir / f"decade_{variable}_{decade}_{scenario}.nc"
    if not p.exists():
        raise FileNotFoundError(f"Missing decadal field: {p}")
    return xr.open_dataset(p)
