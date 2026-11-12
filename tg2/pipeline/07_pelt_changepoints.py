#!/usr/bin/env python3
"""OE2: structural change points with PELT on data, calibrated penalties and multi-model consensus (TG II).

Detections per model, scenario and region on the annual series 1950-2100 (historical + SSP):
  slope    : piecewise-linear regression (cost "linear") of the annual anomaly -> change in trend;
  variance : Gaussian segmentation (cost "normal") of the residual around the degree-4 forced response;
  coupling : mean-shift segmentation (cost "l2") of the annual mean of the product of standardized
             detrended monthly anomalies of tas and hurs (normalized T-H covariance).
Penalties are calibrated per kind, variable and region on AR(1) null series without change, pooled over
models and scenarios: the smallest penalty of the grid whose false-positive rate is <= ALPHA.
Breaks before SPLIT are pooled per model across scenarios (identical data) and reported as "historical".
Consensus: clusters of model breaks within WINDOW years supported by >= MIN_MODELS of the 8 GCMs.
Inputs : processed_tg2/forced_response.parquet, processed_tg2/monthly_anomalies.parquet
Outputs: processed_tg2/pelt_changepoints_all.csv, pelt_changepoints_ensemble.csv, pelt_penalties.csv,
         th_coupling_annual.csv; tg2/figures/pelt_changepoints.png, th_coupling.png
Usage  : python 07_pelt_changepoints.py [--n-null 12] [--alpha 0.05]
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt

ROOT = Path(os.environ.get("NEXGDDP_ROOT", "/media/volume/jay2vol/nexgddp"))
REPO = Path(__file__).resolve().parents[2]
OUT = ROOT / "processed_tg2"
FIGS = REPO / "tg2" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

SUBREGIONS = ["Northern Tropics", "Central Amazonia", "Subtropical Andes", "Southern Cone"]
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
KEYS = ["model", "scenario", "variable", "region"]
REF = (1960, 2014)
SPLIT = 2015
KINDS = {"slope": ("linear", 15), "variance": ("normal", 20), "coupling": ("l2", 15)}
PENS = np.array([1, 2, 3, 5, 8, 12, 20, 30, 50], float)
WINDOW = 5
MIN_MODELS = 5
MIN_RUNS = 2
FOUND_COLS = ["kind", "variable", "region", "model", "scenario", "year"]
CONS_COLS = ["kind", "variable", "region", "scenario", "n_models", "year_median", "year_min", "year_max"]
RNG = np.random.default_rng(42)


def with_history(df):
    hist = df[df.scenario == "historical"]
    return pd.concat([pd.concat([hist.assign(scenario=s), df[df.scenario == s]]) for s in SSPS], ignore_index=True)


def coupling_series(mon, forced):
    mon = with_history(mon[KEYS + ["year", "month", "anomaly"]])
    mon = mon.merge(forced[KEYS + ["year", "forced"]], on=KEYS + ["year"])
    mon["resid"] = mon["anomaly"] - mon["forced"]
    base = mon[(mon.scenario == "ssp245") & mon.year.between(*REF)]
    sd = base.groupby(["model", "variable", "region"])["resid"].std(ddof=1).rename("sd")
    mon = mon.join(sd, on=["model", "variable", "region"])
    mon["z"] = mon["resid"] / mon["sd"]
    wide = mon.pivot_table(index=["model", "scenario", "region", "year", "month"], columns="variable", values="z").reset_index()
    wide["product"] = wide["tas"] * wide["hurs"]
    out = wide.groupby(["model", "scenario", "region", "year"], as_index=False)["product"].mean()
    return out.rename(columns={"product": "coupling"})


def collect(forced, coup):
    groups = {}
    for (m, s, v, r), g in forced.groupby(KEYS):
        g = g.sort_values("year")
        y, res, t = g["anomaly"].to_numpy(float), g["residual"].to_numpy(float), g["year"].to_numpy(float)
        groups.setdefault(("slope", v, r), []).append((m, s, y, res, t))
        groups.setdefault(("variance", v, r), []).append((m, s, res, res, t))
    for (m, s, r), g in coup.groupby(["model", "scenario", "region"]):
        g = g.sort_values("year")
        y, t = g["coupling"].to_numpy(float), g["year"].to_numpy(float)
        groups.setdefault(("coupling", "tas-hurs", r), []).append((m, s, y, y - y.mean(), t))
    return groups


def ar1_like(resid, n):
    resid = resid[np.isfinite(resid)]
    rho = float(np.clip(np.corrcoef(resid[:-1], resid[1:])[0, 1], -0.95, 0.95))
    sd = resid.std(ddof=1)
    x = np.empty(n)
    x[0] = RNG.normal(0.0, sd)
    eps = RNG.normal(0.0, sd * np.sqrt(1.0 - rho ** 2), n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    return x


def null_series(y, res, t, kind):
    if kind == "slope":
        return np.polyval(np.polyfit(t, y, 1), t) + ar1_like(res, len(y))
    if kind == "variance":
        return ar1_like(res, len(y))
    return y.mean() + ar1_like(res, len(y))


def segment(y, kind, pen):
    model, min_size = KINDS[kind]
    if model == "linear":
        t = np.linspace(-1.0, 1.0, len(y))
        sig = np.column_stack([y, t, np.ones_like(t)])
    else:
        sig = y.reshape(-1, 1)
    return rpt.Pelt(model=model, min_size=min_size, jump=1).fit(sig).predict(pen=pen)[:-1]


def calibrate(items, kind, n_null, alpha):
    nulls = [null_series(y, res, t, kind) for (_, _, y, res, t) in items for _ in range(n_null)]
    scale = 1.0 if KINDS[kind][0] == "normal" else float(np.median([np.var(res, ddof=1) for (_, _, _, res, _) in items]))
    for pen in PENS * scale:
        fpr = float(np.mean([len(segment(s, kind, pen)) > 0 for s in nulls]))
        if fpr <= alpha:
            break
    return pen, fpr, scale


def clusters(g):
    out, cur = [], []
    for row in g.sort_values("year").itertuples(index=False):
        if cur and row.year - cur[-1].year > WINDOW:
            out.append(cur)
            cur = []
        cur.append(row)
    return out + [cur] if cur else out


def pool_historical(found):
    fut, hist, rows = found[found.year >= SPLIT], found[found.year < SPLIT], []
    for (kind, var, reg, m), g in hist.groupby(["kind", "variable", "region", "model"]):
        for cl in clusters(g):
            if len({r.scenario for r in cl}) >= MIN_RUNS:
                rows.append({"kind": kind, "variable": var, "region": reg, "model": m,
                             "scenario": "historical", "year": int(np.median([r.year for r in cl]))})
    return pd.concat([fut, pd.DataFrame(rows, columns=FOUND_COLS)], ignore_index=True)


def consensus(found):
    rows = []
    for (kind, var, reg, s), g in found.groupby(["kind", "variable", "region", "scenario"]):
        for cl in clusters(g):
            models = {r.model for r in cl}
            if len(models) >= MIN_MODELS:
                yrs = [r.year for r in cl]
                rows.append({"kind": kind, "variable": var, "region": reg, "scenario": s, "n_models": len(models),
                             "year_median": int(np.median(yrs)), "year_min": min(yrs), "year_max": max(yrs)})
    return pd.DataFrame(rows, columns=CONS_COLS)


def plot_breaks(forced, cons, path):
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    colors, styles = {"tas": "#d62728", "hurs": "#1f77b4"}, {"slope": "--", "variance": ":"}
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.6), sharex=True)
    for ax, reg in zip(axes.flat, SUBREGIONS):
        ax2 = ax.twinx()
        for var, a in (("tas", ax), ("hurs", ax2)):
            sub = forced[(forced.region == reg) & (forced.variable == var) & (forced.scenario == "ssp585")]
            ens = sub.groupby("year")["forced"].quantile([0.1, 0.5, 0.9]).unstack()
            a.fill_between(ens.index, ens[0.1], ens[0.9], color=colors[var], alpha=0.15, lw=0)
            a.plot(ens.index, ens[0.5], color=colors[var], lw=1.3)
            a.set_ylabel(f"{var} anomaly ({'°C' if var == 'tas' else '%'})", color=colors[var])
            a.tick_params(axis="y", colors=colors[var])
        marks = cons[(cons.region == reg) & cons.scenario.isin(["ssp585", "historical"]) & (cons.kind != "coupling")]
        lo, hi = ax.get_ylim()
        for i, row in enumerate(marks.sort_values("year_median").itertuples()):
            ax.axvline(row.year_median, color=colors[row.variable], ls=styles[row.kind], lw=1.1)
            ax.text(row.year_median + 1, lo + (i % 3) * 0.12 * (hi - lo), str(row.year_median),
                    color=colors[row.variable], fontsize=6.5, rotation=90, va="bottom")
        ax.set_title(reg, loc="left")
        ax.set_xlim(1950, 2100)
        ax.grid(alpha=0.3, lw=0.5)
    fig.text(0.5, 0.01, "Dashed: slope change. Dotted: variance change. Consensus of >= 5 of 8 GCMs, SSP5-8.5 and historical.",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_coupling(coup, cons, path):
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    styles = {"ssp126": ("SSP1-2.6", "#1f77b4"), "ssp585": ("SSP5-8.5", "#d62728")}
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.2), sharex=True, sharey=True)
    for ax, reg in zip(axes.flat, SUBREGIONS):
        for s, (label, color) in styles.items():
            sub = coup[(coup.region == reg) & (coup.scenario == s)]
            ens = sub.groupby("year")["coupling"].quantile([0.1, 0.5, 0.9]).unstack()
            ax.fill_between(ens.index, ens[0.1], ens[0.9], color=color, alpha=0.15, lw=0)
            ax.plot(ens.index, ens[0.5], color=color, lw=1.2, label=label)
        ax.axhline(0, color="gray", lw=0.5)
        marks = cons[(cons.region == reg) & (cons.kind == "coupling") & cons.scenario.isin(["ssp585", "historical"])]
        for row in marks.itertuples():
            ax.axvline(row.year_median, color="black", ls="--", lw=1.0)
            ax.text(row.year_median + 1, ax.get_ylim()[1], str(row.year_median), fontsize=6.5, rotation=90, va="top")
        ax.set_title(reg, loc="left")
        ax.set_xlim(1950, 2100)
        ax.grid(alpha=0.3, lw=0.5)
    axes[0, 0].set_ylabel("Normalized T-H covariance")
    axes[1, 0].set_ylabel("Normalized T-H covariance")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-null", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    pd.set_option("display.width", 170)

    forced = pd.read_parquet(OUT / "forced_response.parquet")
    mon = pd.read_parquet(OUT / "monthly_anomalies.parquet")
    coup = coupling_series(mon, forced)
    coup.to_csv(OUT / "th_coupling_annual.csv", index=False)
    print(f"coupling series: {len(coup):,} rows")

    found, pens = [], []
    for (kind, var, reg), items in collect(forced, coup).items():
        pen, fpr, scale = calibrate(items, kind, args.n_null, args.alpha)
        pens.append({"kind": kind, "variable": var, "region": reg, "pen": pen, "scale": scale, "fpr_null": fpr})
        for m, s, y, _, t in items:
            found += [{"kind": kind, "variable": var, "region": reg, "model": m, "scenario": s, "year": int(t[i])}
                      for i in segment(y, kind, pen)]
        print(f"{kind:9s} {var:9s} {reg:18s} pen={pen:9.4f} fpr_null={fpr:.3f}", flush=True)

    pens = pd.DataFrame(pens)
    found = pool_historical(pd.DataFrame(found, columns=FOUND_COLS))
    cons = consensus(found)
    pens.to_csv(OUT / "pelt_penalties.csv", index=False)
    found.to_csv(OUT / "pelt_changepoints_all.csv", index=False)
    cons.to_csv(OUT / "pelt_changepoints_ensemble.csv", index=False)

    print("\n## Breaks per model (mean count per series) by kind and variable")
    n_series = forced.groupby(KEYS).ngroups
    print(found[found.scenario != "historical"].groupby(["kind", "variable"]).size().div(n_series).round(2).to_string())
    for s in ["ssp585", "historical"]:
        sub = cons[cons.scenario == s]
        print(f"\n## Consensus breaks, {s} (median year [min-max], n models)")
        if sub.empty:
            print("none")
            continue
        sub = sub.assign(txt=lambda d: d.year_median.astype(str) + " [" + d.year_min.astype(str) + "-" + d.year_max.astype(str) + "] n=" + d.n_models.astype(str))
        print(sub.pivot_table(index=["region", "kind"], columns="variable", values="txt", aggfunc=lambda x: "; ".join(x)).to_string())

    plot_breaks(forced, cons, FIGS / "pelt_changepoints.png")
    plot_coupling(coup, cons, FIGS / "th_coupling.png")
    print(f"\nSaved to {OUT} and {FIGS}")


if __name__ == "__main__":
    main()
