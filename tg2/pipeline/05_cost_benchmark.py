#!/usr/bin/env python3
"""Computational cost of the OE4 emulators on one fold (TG II, B5).

Fits, on a single fold and target: a linear baseline, a multi-quantile XGBoost
(q = 0.05, 0.50, 0.95, pinball loss) and one small heteroscedastic MLP (Gaussian NLL, PyTorch).
Reports wall time, peak extra memory and sanity skill, and projects the cost of
12 folds x 2 targets (x 5 members for the deep ensemble).

Usage: python 05_cost_benchmark.py [--fold loso_ssp245] [--target tas_anom] [--features integrated_ma20]
"""
import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

ROOT = Path(os.environ.get("NEXGDDP_ROOT", "/media/volume/jay2vol/nexgddp"))
REPO = Path(__file__).resolve().parents[2]
QUANTILES = np.array([0.05, 0.50, 0.95])
THREADS = os.cpu_count()
N_FOLDS, N_TARGETS, N_MEMBERS = 12, 2, 5


class PeakMemory:
    def __enter__(self):
        self.proc, self.base, self.peak, self.run = psutil.Process(), psutil.Process().memory_info().rss, 0, True
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()
        self.t0 = time.perf_counter()
        return self

    def _watch(self):
        while self.run:
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(0.05)

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        self.run = False
        self.thread.join()
        self.extra_mib = max(self.peak - self.base, 0) / 2**20


def pinball(y, p, q):
    d = y - p
    return np.mean(np.maximum(q * d, (q - 1) * d))


def fit_xgb(xtr, ytr, xte):
    import xgboost as xgb
    model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=QUANTILES, n_estimators=500,
                             learning_rate=0.05, max_depth=6, tree_method="hist", n_jobs=THREADS, random_state=0)
    model.fit(xtr, ytr)
    return np.sort(model.predict(xte), axis=1)


def fit_mlp(xtr, ytr, xte, epochs=30, batch=1024, seed=0):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="loso_ssp245")
    ap.add_argument("--target", default="tas_anom")
    ap.add_argument("--features", default="integrated_ma20")
    args = ap.parse_args()

    spec = json.loads((REPO / "tg2" / "data" / "emulator_folds.json").read_text())
    feats = spec["feature_sets"][args.features]
    kind, held_out = args.fold.split("_", 1)
    table = pd.read_parquet(ROOT / "processed_tg2" / "emulator_table.parquet")
    test = (table["scenario"] if kind == "loso" else table["model"]) == held_out
    xtr, ytr = table.loc[~test, feats].to_numpy(float), table.loc[~test, args.target].to_numpy(float)
    xte, yte = table.loc[test, feats].to_numpy(float), table.loc[test, args.target].to_numpy(float)
    print(f"fold {args.fold} | target {args.target} | features {args.features} ({len(feats)}) | "
          f"train {len(ytr):,} | test {len(yte):,} | threads {THREADS}")

    res = {}
    with PeakMemory() as m:
        lin = LinearRegression().fit(xtr, ytr).predict(xte)
    res["linear"] = {"seconds": m.seconds, "extra_mib": m.extra_mib, "rmse": float(np.sqrt(np.mean((yte - lin) ** 2)))}

    with PeakMemory() as m:
        q = fit_xgb(xtr, ytr, xte)
    res["xgb_quantile"] = {"seconds": m.seconds, "extra_mib": m.extra_mib,
                           "rmse_median": float(np.sqrt(np.mean((yte - q[:, 1]) ** 2))),
                           "mean_pinball": float(np.mean([pinball(yte, q[:, i], a) for i, a in enumerate(QUANTILES)])),
                           "coverage_90": float(np.mean((yte >= q[:, 0]) & (yte <= q[:, 2]))),
                           "width_90": float(np.mean(q[:, 2] - q[:, 0]))}

    try:
        with PeakMemory() as m:
            mu, sd = fit_mlp(xtr, ytr, xte)
        z = (yte - mu) / sd
        crps = sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
        res["mlp_member"] = {"seconds": m.seconds, "extra_mib": m.extra_mib,
                             "rmse_mean": float(np.sqrt(np.mean((yte - mu) ** 2))), "crps": float(crps.mean()),
                             "coverage_90": float(np.mean(np.abs(z) <= 1.645)), "width_90": float(np.mean(2 * 1.645 * sd))}
    except ImportError:
        print("torch not installed: MLP skipped")

    print("\n## Results on this fold")
    print(pd.DataFrame(res).T.round(4).to_string())
    fits = {"linear": N_FOLDS * N_TARGETS, "xgb_quantile": N_FOLDS * N_TARGETS, "mlp_member": N_FOLDS * N_TARGETS * N_MEMBERS}
    print(f"\n## Projection: {N_FOLDS} folds x {N_TARGETS} targets (x {N_MEMBERS} members for the deep ensemble)")
    for name, n in fits.items():
        if name in res:
            print(f"{name:13s} {n:4d} fits x {res[name]['seconds']:7.1f} s = {n * res[name]['seconds'] / 3600:6.2f} h "
                  f"| peak extra memory per fit {res[name]['extra_mib']:7.0f} MiB")
    (ROOT / "processed_tg2" / "cost_benchmark.json").write_text(json.dumps({"args": vars(args), "results": res}, indent=2))


if __name__ == "__main__":
    main()
