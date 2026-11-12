#!/usr/bin/env python3
"""Annual means per land cell of tas and hurs, input of the time-of-emergence maps (TG II, OE1).

One pass over raw/: for every model, scenario, variable and year, the mean of the 12 monthly means
of each of the 25,829 land cells (daily hurs clipped to [0, 100], tas in degC), consistent with the
regional tables. One .npz per model-scenario-variable in processed_tg2/cells/ (years x cells, float32);
existing files are skipped, so the run resumes after interruptions. grid.npz stores lat, lon and land mask.

Usage: python 11_cell_annual_means.py [--scenarios historical ssp245 ssp585]
"""
import argparse
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(os.environ.get("NEXGDDP_ROOT", "/media/volume/jay2vol/nexgddp"))
RAW = ROOT / "raw"
CELLS = ROOT / "processed_tg2" / "cells"
WORKERS = int(os.environ.get("WORKERS", 7))
MODELS = ["ACCESS-CM2", "CanESM5", "CESM2", "EC-Earth3", "GFDL-ESM4", "IPSL-CM6A-LR", "MIROC6", "MPI-ESM1-2-HR"]
YEARS = {"historical": range(1950, 2015), **{s: range(2015, 2101) for s in ("ssp126", "ssp245", "ssp370", "ssp585")}}
VARIABLES = ("tas", "hurs")
REFERENCE = RAW / "ACCESS-CM2" / "historical" / "tas" / "tas_ACCESS-CM2_historical_1950_SA.nc"


def land_grid():
    with xr.open_dataset(REFERENCE) as ds:
        return ds.lat.values, ds.lon.values, np.isfinite(ds["tas"].isel(time=0).values)


def annual_cells(path, var, land):
    with xr.open_dataset(path) as ds:
        months = ds.time.dt.month.values
        x = ds[var].values[:, land].astype(np.float64)
    if var == "hurs":
        x = np.clip(x, 0.0, 100.0)
    annual = np.stack([np.nanmean(x[months == m], axis=0) for m in range(1, 13)]).mean(axis=0)
    return annual - 273.15 if var == "tas" else annual


def run_chunk(model, scenario, var):
    target = CELLS / f"{model}__{scenario}__{var}.npz"
    if target.exists():
        return model, scenario, var, "skipped", 0.0, []
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    land = land_grid()[2]
    years, rows, issues = list(YEARS[scenario]), [], []
    for year in years:
        path = RAW / model / scenario / var / f"{var}_{model}_{scenario}_{year}_SA.nc"
        if not path.exists():
            issues.append(f"missing {path.name}")
            continue
        try:
            rows.append(annual_cells(path, var, land))
        except Exception as exc:
            issues.append(f"error {path.name}: {type(exc).__name__}: {exc}")
    if issues:
        return model, scenario, var, "FAILED", time.time() - t0, issues
    tmp = CELLS / f"{model}__{scenario}__{var}.tmp.npz"
    np.savez(tmp, years=np.array(years), data=np.vstack(rows).astype(np.float32))
    tmp.rename(target)
    return model, scenario, var, "done", time.time() - t0, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["historical", "ssp245", "ssp585"])
    args = ap.parse_args()
    CELLS.mkdir(parents=True, exist_ok=True)
    lat, lon, land = land_grid()
    np.savez(CELLS / "grid.npz", lat=lat, lon=lon, land=land)
    jobs = [(m, s, v) for m in MODELS for s in args.scenarios for v in VARIABLES]
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as pool, open(CELLS / "run_log.txt", "a") as fh:
        futures = [pool.submit(run_chunk, *job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            model, scenario, var, status, secs, issues = future.result()
            line = f"[{i:2d}/{len(jobs)}] {model} {scenario} {var}: {status} ({secs:.0f} s) | elapsed {(time.time() - t0) / 60:.1f} min"
            print(line, flush=True)
            fh.write(line + "\n" + "".join(f"    {x}\n" for x in issues))
    done = len(list(CELLS.glob("*__*__*.npz")))
    print(f"Cell files: {done} | land cells: {int(land.sum()):,} | output: {CELLS}")


if __name__ == "__main__":
    main()
