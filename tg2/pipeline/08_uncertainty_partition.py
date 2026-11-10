#!/usr/bin/env python3
"""OE3: partition of projection uncertainty into internal variability, model and scenario (TG II).

Hawkins & Sutton (2009) on the degree-4 forced responses of 03_anomalies_forced_response.py:
  - every series is re-referenced to its 1995-2014 forced mean (uncertainty in the projected change);
  - I    = mean over model-SSP pairs of the residual variance around the fit (constant in time),
           for decadal means (main, 10-year running mean of the residuals) and annual values (sensitivity);
  - M(t) = mean over SSPs of the variance across models of the forced response;
  - S(t) = variance across SSPs of the multi-model mean forced response;
  - T(t) = I + M(t) + S(t); fractions F_x = x / T.
Evaluated 2015-2100 only (before 2015 the scenarios are identical by construction). Two ensembles:
all 8 GCMs and the TCR-constrained subset (5 GCMs). Horizons follow AR6: 2021-2040, 2041-2060, 2081-2100.

Inputs : processed_tg2/forced_response.parquet, tg2/data/gcm_sensitivity.csv
Outputs: processed_tg2/uncertainty_partition.csv, processed_tg2/uncertainty_partition_horizons.csv,
         tg2/figures/uncertainty_partition.png
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(os.environ.get("NEXGDDP_ROOT", "/media/volume/jay2vol/nexgddp"))
REPO = Path(__file__).resolve().parents[2]
OUT = ROOT / "processed_tg2"
REF = (1995, 2014)
HORIZONS = {"near (2021-2040)": (2021, 2040), "mid (2041-2060)": (2041, 2060), "long (2081-2100)": (2081, 2100)}
SUBREGIONS = ["Northern Tropics", "Central Amazonia", "Subtropical Andes", "Southern Cone"]
COLORS = {"F_I": "#f28e2b", "F_M": "#4e79a7", "F_S": "#59a14f"}
NAMES = {"F_I": "Internal variability", "F_M": "Model", "F_S": "Scenario"}
SERIES = ["model", "scenario", "variable", "region"]


def partition(df, timescale):
    df = df.sort_values(SERIES + ["year"])
    ref = df[df.year.between(*REF)].groupby(SERIES)["forced"].mean().rename("ref")
    df = df.join(ref, on=SERIES)
    df["forced_rr"] = df["forced"] - df["ref"]
    resid = df.groupby(SERIES)["residual"]
    if timescale == "decadal":
        resid = resid.transform(lambda r: r.rolling(10, center=True).mean()).groupby([df[c] for c in SERIES])
    internal = resid.var(ddof=1).groupby(["variable", "region"]).mean().rename("I")
    fut = df[df.year >= 2015]
    model = (fut.groupby(["scenario", "variable", "region", "year"])["forced_rr"].var(ddof=1)
             .groupby(["variable", "region", "year"]).mean().rename("M"))
    scenario = (fut.groupby(["scenario", "variable", "region", "year"])["forced_rr"].mean()
                .groupby(["variable", "region", "year"]).var(ddof=1).rename("S"))
    out = pd.concat([model, scenario], axis=1).reset_index().join(internal, on=["variable", "region"])
    out["T"] = out[["I", "M", "S"]].sum(axis=1)
    for c in "IMS":
        out[f"F_{c}"] = out[c] / out["T"]
    return out


def horizons(part):
    rows = []
    for label, (a, b) in HORIZONS.items():
        h = (part[part.year.between(a, b)].groupby(["ensemble", "timescale", "variable", "region"])
             [["F_I", "F_M", "F_S", "I", "M", "S"]].mean())
        rows.append(h.assign(horizon=label).reset_index())
    return pd.concat(rows, ignore_index=True)


def crossover(part):
    rows = []
    for (ens, ts, var, reg), g in part.groupby(["ensemble", "timescale", "variable", "region"]):
        g = g.sort_values("year")
        above = (g["F_S"] > g["F_M"]).to_numpy()
        year = next((int(y) for i, y in enumerate(g["year"]) if above[i:].all()), None)
        rows.append({"ensemble": ens, "timescale": ts, "variable": var, "region": reg, "year": year})
    return pd.DataFrame(rows)


def plot(part, path):
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8})
    fig, axes = plt.subplots(2, 4, figsize=(6.5, 3.6), sharex=True, sharey=True)
    sub = part[(part.ensemble == "all8") & (part.timescale == "decadal")]
    for i, var in enumerate(["tas", "hurs"]):
        for j, reg in enumerate(SUBREGIONS):
            ax = axes[i, j]
            g = sub[(sub.variable == var) & (sub.region == reg)].sort_values("year")
            ax.stackplot(g.year, g.F_I, g.F_M, g.F_S, colors=[COLORS[k] for k in ("F_I", "F_M", "F_S")],
                         labels=[NAMES[k] for k in ("F_I", "F_M", "F_S")], alpha=0.9)
            ax.set_xlim(2015, 2100)
            ax.set_ylim(0, 1)
            ax.set_xticks([2030, 2060, 2090])
            if i == 0:
                ax.set_title(reg, loc="left")
            if j == 0:
                ax.set_ylabel(f"{'Temperature' if var == 'tas' else 'Relative humidity'}\nfraction of variance")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=0.6, w_pad=0.4)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    pd.set_option("display.width", 170)
    forced = pd.read_parquet(OUT / "forced_response.parquet")
    subset = pd.read_csv(REPO / "tg2" / "data" / "gcm_sensitivity.csv").query("in_constrained_subset")["model"]
    ensembles = {"all8": forced, "constrained5": forced[forced.model.isin(subset)]}
    part = pd.concat([partition(df, ts).assign(ensemble=ens, timescale=ts)
                      for ens, df in ensembles.items() for ts in ("decadal", "annual")], ignore_index=True)
    part.to_csv(OUT / "uncertainty_partition.csv", index=False)
    hz = horizons(part)
    hz.to_csv(OUT / "uncertainty_partition_horizons.csv", index=False)

    cell = lambda d: d.apply(lambda r: f"{100 * r.F_I:3.0f}/{100 * r.F_M:3.0f}/{100 * r.F_S:3.0f}", axis=1)
    main_tab = hz[(hz.ensemble == "all8") & (hz.timescale == "decadal")].assign(cell=cell)
    print("## Main result (8 GCMs, decadal means): % of variance internal / model / scenario")
    print(main_tab.pivot_table(index=["variable", "region"], columns="horizon", values="cell", aggfunc="first")[list(HORIZONS)].to_string())
    long = hz[hz.horizon == "long (2081-2100)"].assign(cell=cell, case=lambda d: d.ensemble + " " + d.timescale)
    print("\n## Sensitivity at 2081-2100: % internal / model / scenario")
    print(long.pivot_table(index=["variable", "region"], columns="case", values="cell", aggfunc="first").to_string())
    co = crossover(part[part.timescale == "decadal"])
    tab = co.set_index(["variable", "region", "ensemble"])["year"].unstack("ensemble")
    print("\n## First year from which scenario uncertainty exceeds model uncertainty until 2100 (decadal)")
    print(tab.map(lambda v: "never" if pd.isna(v) else str(int(v))).to_string())
    (REPO / "tg2" / "figures").mkdir(parents=True, exist_ok=True)
    plot(part, REPO / "tg2" / "figures" / "uncertainty_partition.png")
    print(f"\nSaved: {OUT / 'uncertainty_partition.csv'}, {OUT / 'uncertainty_partition_horizons.csv'}, "
          f"{REPO / 'tg2/figures/uncertainty_partition.png'}")


if __name__ == "__main__":
    main()
