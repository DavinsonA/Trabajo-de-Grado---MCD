#!/usr/bin/env python3
"""OE4: probabilistic emulators of monthly subregional anomalies (TG II).

Models : linear baseline (Gaussian, constant variance), multi-quantile XGBoost (19 quantiles),
         deep ensemble of 5 heteroscedastic MLPs (Gaussian NLL, mixture predictive).
Data   : processed_tg2/emulator_table.parquet; folds and feature sets from tg2/data/emulator_folds.json
         (4 leave-one-scenario-out + 8 leave-one-model-out; main features integrated_ma20, ablation base).
Metrics: CRPS approximated on a common grid of 19 quantiles (0.05-0.95) for all models,
         RMSE of the central prediction (monthly and annual means), coverage and width of the 90 % interval;
         for the deep ensemble, aleatoric (mean member variance) and epistemic (variance of member means) SD.
SHAP   : XGBoost native TreeSHAP on a median model fitted to all rows (main features), month and region grouped.
Resumable: one chunk per fold x target x feature set in processed_tg2/oe4/chunks/.

Usage: python 09_emulators.py [--feature-sets integrated_ma20 base] [--targets tas_anom hurs_anom] [--skip-shap]
"""
import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

ROOT = Path(os.environ.get("NEXGDDP_ROOT", "/media/volume/jay2vol/nexgddp"))
REPO = Path(__file__).resolve().parents[2]
OUT = ROOT / "processed_tg2" / "oe4"
CHUNKS = OUT / "chunks"
QUANTILES = np.round(np.arange(0.05, 0.951, 0.05), 2)
I05, I50, I95 = 0, 9, 18
N_MEMBERS = 5
THREADS = os.cpu_count()
KEY = ["model", "scenario", "region", "year", "month"]
LABELS = {"ghg_rf_wm2": "GHG forcing", "ghg_rf_ma20": "GHG forcing, 20-yr mean", "ghg_rf_cum": "GHG forcing, cumulative",
          "tcr": "Model TCR", "month": "Month", "region": "Subregion"}


def fit_linear(xtr, ytr, xte):
    lin = LinearRegression().fit(xtr, ytr)
    sd = np.std(ytr - lin.predict(xtr), ddof=xtr.shape[1] + 1)
    mu = lin.predict(xte)
    return mu, mu[:, None] + sd * norm.ppf(QUANTILES)[None, :]


def fit_xgb(xtr, ytr, xte):
    import xgboost as xgb
    model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=QUANTILES, n_estimators=500,
                             learning_rate=0.05, max_depth=6, tree_method="hist", n_jobs=THREADS, random_state=0)
    model.fit(xtr, ytr)
    q = np.sort(model.predict(xte), axis=1)
    return q[:, I50], q


def fit_member(xtr, ytr, xte, seed, epochs=30, batch=1024):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(THREADS)
    xm, xs, ym, ys = xtr.mean(0), xtr.std(0) + 1e-9, ytr.mean(), ytr.std()
    xt = torch.tensor((xtr - xm) / xs, dtype=torch.float32)
    yt = torch.tensor((ytr - ym) / ys, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    nll = torch.nn.GaussianNLLLoss()
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), batch):
            idx = perm[i:i + batch]
            out = net(xt[idx])
            loss = nll(out[:, 0], yt[idx], torch.nn.functional.softplus(out[:, 1]) + 1e-6)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        out = net(torch.tensor((xte - xm) / xs, dtype=torch.float32))
        mu = out[:, 0].numpy() * ys + ym
        sd = np.sqrt(torch.nn.functional.softplus(out[:, 1]).numpy() + 1e-6) * ys
    return mu, sd


def mixture_quantiles(mu, sd, levels, iters=50):
    lo = np.repeat((mu - 8 * sd).min(1, keepdims=True), len(levels), 1)
    hi = np.repeat((mu + 8 * sd).max(1, keepdims=True), len(levels), 1)
    for _ in range(iters):
        mid = (lo + hi) / 2
        cdf = norm.cdf((mid[:, :, None] - mu[:, None, :]) / sd[:, None, :]).mean(2)
        below = cdf < levels[None, :]
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return (lo + hi) / 2


def fit_ensemble(xtr, ytr, xte):
    members = [fit_member(xtr, ytr, xte, seed) for seed in range(N_MEMBERS)]
    mu, sd = np.column_stack([m[0] for m in members]), np.column_stack([m[1] for m in members])
    return mu.mean(1), mixture_quantiles(mu, sd, QUANTILES), float(np.sqrt((sd ** 2).mean())), float(np.sqrt(mu.var(1).mean()))


def scores(y, central, q, ids):
    pinball = [np.mean(np.maximum(a * (y - q[:, i]), (a - 1) * (y - q[:, i]))) for i, a in enumerate(QUANTILES)]
    annual = pd.DataFrame({"y": y, "c": central}).groupby([ids[k] for k in ["model", "scenario", "region", "year"]]).mean()
    return {"crps": 2 * float(np.mean(pinball)), "rmse": float(np.sqrt(np.mean((y - central) ** 2))),
            "rmse_annual": float(np.sqrt(np.mean((annual.y - annual.c) ** 2))),
            "coverage_90": float(np.mean((y >= q[:, I05]) & (y <= q[:, I95]))), "width_90": float(np.mean(q[:, I95] - q[:, I05]))}


def run_chunk(table, spec, fold, target, fset):
    tag = f"{fold['fold']}__{target}__{fset}"
    mfile, pfile = CHUNKS / f"{tag}.json", CHUNKS / f"{tag}.parquet"
    if mfile.exists() and pfile.exists():
        return json.loads(mfile.read_text()), "skipped"
    feats = spec["feature_sets"][fset]
    col = "scenario" if fold["kind"] == "loso" else "model"
    test = (table[col] == fold["held_out"]).to_numpy()
    xtr, ytr = table.loc[~test, feats].to_numpy(float), table.loc[~test, target].to_numpy(float)
    xte, yte = table.loc[test, feats].to_numpy(float), table.loc[test, target].to_numpy(float)
    ids = table.loc[test, KEY].reset_index(drop=True)
    pred = ids.assign(target=yte)
    rows, base = [], {"fold": fold["fold"], "kind": fold["kind"], "held_out": fold["held_out"], "target": target, "feature_set": fset}
    for name, fit in (("linear", fit_linear), ("xgb_quantile", fit_xgb), ("deep_ensemble", fit_ensemble)):
        t0 = time.time()
        out = fit(xtr, ytr, xte)
        central, q = out[0], out[1]
        row = {**base, "emulator": name, "seconds": round(time.time() - t0, 1), **scores(yte, central, q, ids)}
        if name == "deep_ensemble":
            row.update(aleatoric_sd=out[2], epistemic_sd=out[3])
        rows.append(row)
        pred[f"{name}_central"], pred[f"{name}_q05"], pred[f"{name}_q95"] = central, q[:, I05], q[:, I95]
    pred.to_parquet(pfile, index=False)
    mfile.write_text(json.dumps(rows, indent=1))
    return rows, "done"


def shap_importance(table, spec, targets):
    import xgboost as xgb
    feats = spec["feature_sets"]["integrated_ma20"]
    groups = {f: ("region" if f.startswith("region_") else "month" if f.startswith("month_") else f) for f in feats}
    sample = table.sample(n=min(20000, len(table)), random_state=0)
    rows = []
    for target in targets:
        model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, n_estimators=500, learning_rate=0.05,
                                 max_depth=6, tree_method="hist", n_jobs=THREADS, random_state=0)
        model.fit(table[feats].to_numpy(float), table[target].to_numpy(float))
        contrib = model.get_booster().predict(xgb.DMatrix(sample[feats].to_numpy(float)), pred_contribs=True)[:, :-1]
        grouped = pd.DataFrame(contrib, columns=feats).T.groupby(groups).sum().T
        for g, v in grouped.abs().mean().items():
            rows.append({"target": target, "feature": g, "mean_abs_shap": float(v)})
    imp = pd.DataFrame(rows)
    imp.to_csv(OUT / "shap_importance.csv", index=False)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5})
    fig, axes = plt.subplots(1, len(targets), figsize=(6.5, 2.4))
    for ax, target in zip(np.atleast_1d(axes), targets):
        d = imp[imp.target == target].sort_values("mean_abs_shap")
        ax.barh(d.feature.map(LABELS), d.mean_abs_shap, color="#4e79a7")
        ax.set_title("Temperature (°C)" if target == "tas_anom" else "Relative humidity (%)", loc="left")
        ax.set_xlabel("mean |SHAP|")
        ax.grid(axis="x", alpha=0.3, lw=0.5)
    fig.tight_layout(pad=0.4, w_pad=1.5)
    fig.savefig(REPO / "tg2" / "figures" / "oe4_shap_importance.png", dpi=300)
    plt.close(fig)
    return imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-sets", nargs="+", default=["integrated_ma20", "base"])
    ap.add_argument("--targets", nargs="+", default=["tas_anom", "hurs_anom"])
    ap.add_argument("--skip-shap", action="store_true")
    args = ap.parse_args()
    pd.set_option("display.width", 170)
    CHUNKS.mkdir(parents=True, exist_ok=True)
    (REPO / "tg2" / "figures").mkdir(parents=True, exist_ok=True)
    spec = json.loads((REPO / "tg2" / "data" / "emulator_folds.json").read_text())
    table = pd.read_parquet(ROOT / "processed_tg2" / "emulator_table.parquet")
    jobs = [(f, t, s) for s in args.feature_sets for t in args.targets for f in spec["folds"]]
    t0, metrics = time.time(), []
    for i, (fold, target, fset) in enumerate(jobs, 1):
        rows, status = run_chunk(table, spec, fold, target, fset)
        metrics += rows
        print(f"[{i:2d}/{len(jobs)}] {fold['fold']:20s} {target:9s} {fset:15s} {status:7s} | elapsed {(time.time() - t0) / 60:5.1f} min", flush=True)
    m = pd.DataFrame(metrics)
    m.to_csv(OUT / "oe4_metrics.csv", index=False)

    main_fs = m[m.feature_set == "integrated_ma20"]
    if not main_fs.empty:
        summ = main_fs.groupby(["target", "kind", "emulator"])[["crps", "rmse", "rmse_annual", "coverage_90", "width_90"]].mean()
        print("\n## Main feature set (integrated_ma20): mean over folds")
        print(summ.round(3).to_string())
        lin = main_fs[main_fs.emulator == "linear"].set_index(["fold", "target"])["crps"]
        skill = main_fs.assign(crpss=lambda d: 1 - d.crps / lin.reindex(pd.MultiIndex.from_frame(d[["fold", "target"]])).to_numpy())
        print("\n## CRPS skill vs linear baseline (1 - CRPS/CRPS_linear): mean [min-max] over folds")
        print(skill[skill.emulator != "linear"].groupby(["target", "kind", "emulator"])["crpss"]
              .agg(lambda s: f"{s.mean():.2f} [{s.min():.2f}-{s.max():.2f}]").to_string())
        ens = main_fs[main_fs.emulator == "deep_ensemble"]
        print("\n## Deep ensemble uncertainty: mean SD over folds")
        print(ens.groupby(["target", "kind"])[["aleatoric_sd", "epistemic_sd"]].mean().round(3).to_string())
        print("\n## Per-fold CRPS (integrated_ma20)")
        print(main_fs.pivot_table(index=["target", "fold"], columns="emulator", values="crps").round(3).to_string())
    if {"integrated_ma20", "base"} <= set(m.feature_set):
        abl = m.pivot_table(index=["target", "kind", "emulator"], columns="feature_set", values="crps", aggfunc="mean")
        abl["gain_pct"] = 100 * (abl["base"] - abl["integrated_ma20"]) / abl["base"]
        print("\n## Ablation: CRPS with base vs integrated_ma20 features (gain_pct > 0 = memory term helps)")
        print(abl.round(3).to_string())
    if not args.skip_shap:
        imp = shap_importance(table, spec, args.targets)
        print("\n## SHAP mean |contribution| (XGBoost median, all rows, integrated_ma20)")
        print(imp.pivot_table(index="feature", columns="target", values="mean_abs_shap").round(3).to_string())
    print(f"\nSaved: {OUT / 'oe4_metrics.csv'}, chunks in {CHUNKS}, {OUT / 'shap_importance.csv'}, "
          f"{REPO / 'tg2/figures/oe4_shap_importance.png'}")


if __name__ == "__main__":
    main()
