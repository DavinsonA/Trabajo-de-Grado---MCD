#!/usr/bin/env python3
"""Monthly regional means of tas and hurs from the NEX-GDDP-CMIP6 daily files in raw/.

One pass computing, per model, scenario, variable, region, year and month:
  mean_unweighted   simple mean over land cells (TG I method)
  mean_weighted     cos(latitude)-weighted mean over land cells
  mean_weighted_qc  as mean_weighted, with daily hurs clipped to [0, 100] (tas unchanged)
  n_qc_out          land cell-days outside Thrasher et al. (2022) bounds (tas 200-340 K, hurs 0-102 %)
  n_clipped         land cell-days changed by the hurs clip
tas is returned in degC. One parquet per model-scenario-variable is written to
processed_tg2/chunks/; existing chunks are skipped, so the run resumes after interruptions.

Usage:
  python 01_subregional_monthly_means.py                                   # full run
  python 01_subregional_monthly_means.py --only ACCESS-CM2 historical tas  # single chunk
"""
import argparse
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(os.environ.get("NEXGDDP_ROOT", "/media/volume/jay2vol/nexgddp"))
RAW = ROOT / "raw"
OUT = ROOT / "processed_tg2"
CHUNKS = OUT / "chunks"
WORKERS = int(os.environ.get("WORKERS", 7))

MODELS = ["ACCESS-CM2", "CanESM5", "CESM2", "EC-Earth3", "GFDL-ESM4", "IPSL-CM6A-LR", "MIROC6", "MPI-ESM1-2-HR"]
SCENARIOS = {"historical": range(1950, 2015), **{s: range(2015, 2101) for s in ("ssp126", "ssp245", "ssp370", "ssp585")}}
VARIABLES = ("tas", "hurs")
REGIONS = {
    "Northern Tropics": (0, 13),
    "Central Amazonia": (-15, 0),
    "Subtropical Andes": (-30, -15),
    "Southern Cone": (-56, -30),
    "All land": (-56, 13),
}
QC_BOUNDS = {"tas": (200.0, 340.0), "hurs": (0.0, 102.0)}
REFERENCE = RAW / "ACCESS-CM2" / "historical" / "tas" / "tas_ACCESS-CM2_historical_1950_SA.nc"
COLUMNS = ["model", "scenario", "variable", "region", "year", "month", "n_days", "n_cells",
           "mean_unweighted", "mean_weighted", "mean_weighted_qc", "n_qc_out", "n_clipped"]


def build_grid():
    with xr.open_dataset(REFERENCE) as ds:
        lat = ds.lat.values
        land = np.isfinite(ds["tas"].isel(time=0).values)
    cell_lat = lat[np.nonzero(land)[0]]
    weights = np.cos(np.deg2rad(cell_lat))
    regions = {name: np.flatnonzero((cell_lat >= lo) & (cell_lat <= hi)) for name, (lo, hi) in REGIONS.items()}
    return land, weights, regions


def process_file(path, var, land, weights, regions):
    with xr.open_dataset(path) as ds:
        months = ds.time.dt.month.values
        data = ds[var].values
    ocean_values = int(np.isfinite(data[:, ~land]).sum())
    x = data[:, land].astype(np.float64)
    land_nan = int((~np.isfinite(x)).sum())
    lo, hi = QC_BOUNDS[var]
    qc_out = (x < lo) | (x > hi)
    if var == "hurs":
        clipped = (x < 0.0) | (x > 100.0)
        xq = np.clip(x, 0.0, 100.0)
    else:
        clipped = np.zeros_like(qc_out)
        xq = x
    offset = 273.15 if var == "tas" else 0.0
    rows = []
    for month in range(1, 13):
        t = months == month
        mean_raw = np.nanmean(x[t], axis=0)
        mean_qc = np.nanmean(xq[t], axis=0)
        n_qc = qc_out[t].sum(axis=0)
        n_cl = clipped[t].sum(axis=0)
        for name, idx in regions.items():
            v, vq, w = mean_raw[idx], mean_qc[idx], weights[idx]
            ok = np.isfinite(v)
            rows.append((name, month, int(t.sum()), int(ok.sum()),
                         v[ok].mean() - offset,
                         np.average(v[ok], weights=w[ok]) - offset,
                         np.average(vq[ok], weights=w[ok]) - offset,
                         int(n_qc[idx].sum()), int(n_cl[idx].sum())))
    return rows, ocean_values, land_nan


def run_chunk(model, scenario, var):
    target = CHUNKS / f"{model}__{scenario}__{var}.parquet"
    if target.exists():
        return model, scenario, var, "skipped", 0.0, []
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    land, weights, regions = build_grid()
    records, issues, fatal = [], [], False
    for year in SCENARIOS[scenario]:
        path = RAW / model / scenario / var / f"{var}_{model}_{scenario}_{year}_SA.nc"
        if not path.exists():
            issues.append(f"missing {path.name}")
            fatal = True
            continue
        try:
            rows, ocean_values, land_nan = process_file(path, var, land, weights, regions)
        except Exception as exc:
            issues.append(f"error {path.name}: {type(exc).__name__}: {exc}")
            fatal = True
            continue
        if ocean_values or land_nan:
            issues.append(f"mask {path.name}: ocean_values={ocean_values} land_nan={land_nan}")
        records += [(model, scenario, var, region, year, *rest) for region, *rest in rows]
    if fatal:
        return model, scenario, var, "FAILED", time.time() - t0, issues
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(records, columns=COLUMNS).to_parquet(tmp, index=False)
    tmp.rename(target)
    return model, scenario, var, "done", time.time() - t0, issues


def report(i, n, result, t0, fh):
    model, scenario, var, status, secs, issues = result
    line = f"[{i:2d}/{n}] {model} {scenario} {var}: {status} ({secs:.0f} s, {len(issues)} issues) | elapsed {(time.time() - t0) / 60:.1f} min"
    print(line, flush=True)
    fh.write(line + "\n" + "".join(f"    {issue}\n" for issue in issues))
    fh.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs=3, metavar=("MODEL", "SCENARIO", "VARIABLE"))
    args = parser.parse_args()
    CHUNKS.mkdir(parents=True, exist_ok=True)
    jobs = [tuple(args.only)] if args.only else [(m, s, v) for m in MODELS for s in SCENARIOS for v in VARIABLES]
    t0 = time.time()
    with open(OUT / "run_log.txt", "a") as fh:
        if args.only:
            report(1, 1, run_chunk(*jobs[0]), t0, fh)
            return
        with ProcessPoolExecutor(WORKERS) as pool:
            futures = [pool.submit(run_chunk, *job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                report(i, len(jobs), future.result(), t0, fh)
    parts = sorted(CHUNKS.glob("*.parquet"))
    if len(parts) < len(jobs):
        print(f"Incomplete: {len(parts)}/{len(jobs)} chunks. Re-run the same command to resume.")
        return
    df = pd.concat(map(pd.read_parquet, parts), ignore_index=True)
    df.to_parquet(OUT / "monthly_region_means.parquet", index=False)
    print(f"Done: {len(df):,} rows -> {OUT / 'monthly_region_means.parquet'}")


if __name__ == "__main__":
    main()
