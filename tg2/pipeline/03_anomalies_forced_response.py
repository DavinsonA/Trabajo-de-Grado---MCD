#!/usr/bin/env python3
"""Anomalies, reference variability, forced response and rolling T-H coupling (TG II, B3).

Input : processed_tg2/monthly_region_means.parquet, column mean_weighted_qc
        (cos-lat weighted regional means, daily hurs clipped to [0, 100]).
Steps :
  1. Monthly anomalies vs the 1960-2014 climatology of each model, variable, region and calendar month;
     annual anomalies = calendar-year mean of the monthly anomalies.
  2. sigma_ref: SD (ddof=1) of annual anomalies 1960-2014 after a linear detrend.
  3. Forced response: degree-4 polynomial fitted to each model-SSP-variable-region annual series
     1950-2100 (historical + SSP), as in Hawkins & Sutton (2009); residual SD overall, 1960-2014, 2015-2100.
  4. Rolling Spearman tas-hurs, 30-year windows, step 1 year, on monthly anomalies (raw) and on
     monthly anomalies minus the annual forced response (detrended); lag-1 autocorrelation of each series.
Outputs (processed_tg2/): monthly_anomalies.parquet, annual_anomalies.parquet, sigma_ref.csv,
        forced_response.parquet, forced_response_summary.csv, rolling_spearman.parquet
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(os.environ.get("NEXGDDP_ROOT", "/media/volume/jay2vol/nexgddp"))
OUT = ROOT / "processed_tg2"
BASE_COL = "mean_weighted_qc"
REF = (1960, 2014)
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
POLY_DEGREE = 4
WINDOW = 30
SERIES = ["model", "scenario", "variable", "region"]


def monthly_anomalies(df):
    df = df[["model", "scenario", "variable", "region", "year", "month", BASE_COL]].rename(columns={BASE_COL: "value"})
    keys = ["model", "variable", "region", "month"]
    clim = df[(df.scenario == "historical") & df.year.between(*REF)].groupby(keys)["value"].mean().rename("clim")
    df = df.join(clim, on=keys)
    df["anomaly"] = df["value"] - df["clim"]
    return df.drop(columns="clim")


def with_history(df):
    hist = df[df.scenario == "historical"]
    return pd.concat([pd.concat([hist.assign(scenario=s), df[df.scenario == s]]) for s in SSPS], ignore_index=True)


def detrended_sd(g):
    x, y = g["year"].to_numpy(float), g["anomaly"].to_numpy()
    return np.std(y - np.polyval(np.polyfit(x, y, 1), x), ddof=1)


def fit_forced(g):
    x, y = g["year"].to_numpy(float), g["anomaly"].to_numpy()
    xc = x - x.mean()
    return pd.Series(np.polyval(np.polyfit(xc, y, POLY_DEGREE), xc), index=g.index)


def lag1(r):
    r = r[np.isfinite(r)]
    return np.corrcoef(r[:-1], r[1:])[0, 1] if r.size > 3 else np.nan


def rolling_spearman(wide):
    rows = []
    for (model, scenario, region), g in wide.groupby(["model", "scenario", "region"]):
        g = g.sort_values(["year", "month"])
        years = g["year"].to_numpy()
        for start in range(years.min(), years.max() - WINDOW + 2):
            w = g[(years >= start) & (years < start + WINDOW)]
            rows.append((model, scenario, region, start, start + WINDOW - 1,
                         spearmanr(w["tas"], w["hurs"]).statistic,
                         spearmanr(w["tas_resid"], w["hurs_resid"]).statistic))
    return pd.DataFrame(rows, columns=["model", "scenario", "region", "year_start", "year_end", "rho_raw", "rho_detrended"])


def main():
    pd.set_option("display.width", 170)
    mon = monthly_anomalies(pd.read_parquet(OUT / "monthly_region_means.parquet"))
    mon.to_parquet(OUT / "monthly_anomalies.parquet", index=False)
    ann = mon.groupby(SERIES + ["year"], as_index=False)["anomaly"].mean()
    ann.to_parquet(OUT / "annual_anomalies.parquet", index=False)
    print(f"monthly anomalies: {len(mon):,} rows | annual: {len(ann):,} rows")

    hist = ann[(ann.scenario == "historical") & ann.year.between(*REF)]
    sigma = hist.groupby(["model", "variable", "region"]).apply(detrended_sd, include_groups=False).rename("sigma_ref").reset_index()
    sigma.to_csv(OUT / "sigma_ref.csv", index=False)
    print("\n## sigma_ref (SD of detrended annual anomalies 1960-2014): median [min-max] across models")
    print(sigma.groupby(["variable", "region"])["sigma_ref"].agg(lambda s: f"{s.median():.3f} [{s.min():.3f}-{s.max():.3f}]").unstack(0).to_string())

    full = with_history(ann).sort_values(SERIES + ["year"])
    forced = full.reset_index(drop=True)
    forced["forced"] = forced.groupby(SERIES, group_keys=False)[["year", "anomaly"]].apply(fit_forced)
    forced["residual"] = forced["anomaly"] - forced["forced"]
    forced.to_parquet(OUT / "forced_response.parquet", index=False)
    summ = forced.groupby(SERIES).apply(lambda g: pd.Series({
        "resid_sd_all": g.residual.std(ddof=1),
        "resid_sd_1960_2014": g.loc[g.year.between(*REF), "residual"].std(ddof=1),
        "resid_sd_2015_2100": g.loc[g.year >= 2015, "residual"].std(ddof=1),
    }), include_groups=False).reset_index().merge(sigma, on=["model", "variable", "region"])
    summ["ratio_all"] = summ.resid_sd_all / summ.sigma_ref
    summ["ratio_future"] = summ.resid_sd_2015_2100 / summ.sigma_ref
    summ.to_csv(OUT / "forced_response_summary.csv", index=False)
    print("\n## Residual SD around the degree-4 forced response / sigma_ref: median [min-max] across models and SSPs")
    agg = lambda s: f"{s.median():.2f} [{s.min():.2f}-{s.max():.2f}]"
    print(summ.groupby(["variable", "region"]).agg(ratio_all=("ratio_all", agg), ratio_future=("ratio_future", agg)).to_string())

    mon_full = with_history(mon).merge(forced[SERIES + ["year", "forced"]], on=SERIES + ["year"])
    mon_full["resid"] = mon_full["anomaly"] - mon_full["forced"]
    wide = mon_full.pivot_table(index=["model", "scenario", "region", "year", "month"], columns="variable",
                                values=["anomaly", "resid"]).reset_index()
    wide.columns = ["model", "scenario", "region", "year", "month", "hurs", "tas", "hurs_resid", "tas_resid"]
    roll = rolling_spearman(wide)
    roll.to_parquet(OUT / "rolling_spearman.parquet", index=False)
    ac = roll.groupby(["model", "scenario", "region"]).agg(lag1_raw=("rho_raw", lambda r: lag1(r.to_numpy())),
                                                          lag1_detrended=("rho_detrended", lambda r: lag1(r.to_numpy())))
    print(f"\n## Rolling Spearman: {len(roll):,} windows ({roll.groupby(['model', 'scenario', 'region']).size().iloc[0]} per series)")
    print(f"lag-1 autocorrelation of rho series, median [min-max]: raw {agg(ac.lag1_raw)} | detrended {agg(ac.lag1_detrended)}")
    first, last = roll.year_start.min(), roll.year_start.max()
    sel = roll[(roll.scenario == "ssp585") & roll.year_start.isin([first, last])]
    tab = sel.groupby(["region", "year_start"])[["rho_raw", "rho_detrended"]].median().unstack("year_start").round(2)
    print(f"\nEnsemble-median rho, SSP5-8.5, window {first}-{first + WINDOW - 1} vs {last}-{last + WINDOW - 1}:")
    print(tab.to_string())


if __name__ == "__main__":
    main()
