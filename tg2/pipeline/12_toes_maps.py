#!/usr/bin/env python3
"""OE1 maps: time of emergence (ToE) per land cell (TG II).

Per model and cell, the method of 06_toe_divergence.py: annual anomalies vs 1960-2014; sigma_ref = SD
(ddof=1) of the linearly detrended anomalies 1960-2014; forced response = degree-4 polynomial over
1950-2100 (historical + SSP); signal-to-noise oriented by the sign of the 2100 signal; ToE = first year
from which SNR >= 2 holds until 2100. Ensemble: median over the 8 GCMs with non-emergence counted as
later than 2100, so a cell emerges in the ensemble only if at least 5 of 8 models emerge.

Inputs : processed_tg2/cells/*.npz (11_cell_annual_means.py)
Outputs: tg2/figures/toe_maps_ssp585.png, processed_tg2/toe_cells_<scenario>.npz,
         processed_tg2/toe_cells_summary.csv (area-weighted share of land emerged by 2040/2060/2100)
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("NEXGDDP_ROOT", "/media/volume/jay2vol/nexgddp"))
REPO = Path(__file__).resolve().parents[2]
OUT = ROOT / "processed_tg2"
CELLS = OUT / "cells"
MODELS = ["ACCESS-CM2", "CanESM5", "CESM2", "EC-Earth3", "GFDL-ESM4", "IPSL-CM6A-LR", "MIROC6", "MPI-ESM1-2-HR"]
SCENARIOS = ["ssp585", "ssp245"]
MAP_SCENARIO = "ssp585"
THRESHOLD = 2.0
REF = (1960, 2014)
BANDS = {"Northern Tropics": (0, 13), "Central Amazonia": (-15, 0), "Subtropical Andes": (-30, -15),
         "Southern Cone": (-56, -30), "All land": (-56, 13)}
BOUNDS = [1990, 2000, 2020, 2040, 2060, 2080, 2100]
NAMES = {"tas": "Temperature", "hurs": "Relative humidity"}
SSP_LABELS = {"ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}


def load(model, scenario, var):
    h, f = np.load(CELLS / f"{model}__historical__{var}.npz"), np.load(CELLS / f"{model}__{scenario}__{var}.npz")
    return np.concatenate([h["years"], f["years"]]), np.vstack([h["data"], f["data"]]).astype(np.float64)


def toe_cells(years, data):
    ref = (years >= REF[0]) & (years <= REF[1])
    anom = data - data[ref].mean(axis=0)
    yr = years[ref].astype(float)
    slope, intercept = np.polyfit(yr, anom[ref], 1)
    sigma = (anom[ref] - (np.outer(yr, slope) + intercept)).std(axis=0, ddof=1)
    xc = years - years.mean()
    forced = np.vander(xc, 5) @ np.polyfit(xc, anom, 4)
    snr = forced / sigma
    ok = snr * np.sign(snr[-1]) >= THRESHOLD
    sustained = np.flip(np.logical_and.accumulate(np.flip(ok, axis=0), axis=0), axis=0)
    return np.where(sustained.any(axis=0), years[np.argmax(sustained, axis=0)], np.inf)


def summarize(toe, cell_lat, var, scenario):
    weights = np.cos(np.deg2rad(cell_lat))
    rows = []
    for name, (lo, hi) in BANDS.items():
        sel = (cell_lat >= lo) & (cell_lat <= hi)
        w = weights[sel] / weights[sel].sum()
        row = {"scenario": scenario, "variable": var, "region": name}
        row.update({f"share_by_{y}": float((w * (toe[sel] <= y)).sum()) for y in (2040, 2060, 2100)})
        rows.append(row)
    return rows


def plot(maps, lat, lon, land, path):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7.5})
    cmap = ListedColormap(plt.get_cmap("magma")(np.linspace(0.15, 0.95, len(BOUNDS) - 1)))
    norm = BoundaryNorm(BOUNDS, cmap.N)
    lon_w = np.where(lon > 180, lon - 360, lon)
    features = [cfeature.NaturalEarthFeature("physical", "coastline", "50m", facecolor="none", edgecolor="black", linewidth=0.5),
                cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", "50m", facecolor="none",
                                             edgecolor="gray", linewidth=0.3)]
    try:
        for f in features:
            next(iter(f.geometries()))
    except Exception as exc:
        features = []
        print(f"coastlines skipped (Natural Earth unavailable): {type(exc).__name__}")
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.6), subplot_kw={"projection": ccrs.PlateCarree()})
    for ax, (var, toe) in zip(axes, maps.items()):
        grid = np.full(land.shape, np.nan)
        grid[land] = toe
        never = np.where(np.isinf(grid), 1.0, np.nan)
        ax.set_extent([-82, -34, -56, 13], crs=ccrs.PlateCarree())
        ax.pcolormesh(lon_w, lat, never, cmap=ListedColormap(["#c8c8c8"]), shading="nearest", transform=ccrs.PlateCarree())
        mesh = ax.pcolormesh(lon_w, lat, np.where(np.isfinite(grid), np.clip(grid, BOUNDS[0], BOUNDS[-1]), np.nan),
                             cmap=cmap, norm=norm, shading="nearest", transform=ccrs.PlateCarree())
        for f in features:
            ax.add_feature(f)
        for b in (0, -15, -30):
            ax.plot([-82, -34], [b, b], color="black", lw=0.5, ls="--", transform=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, color="gray", alpha=0.5)
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 7}
        ax.set_title(f"{NAMES[var]}, {SSP_LABELS[MAP_SCENARIO]}", loc="left")
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", fraction=0.05, pad=0.08, aspect=40, extend="min")
    cbar.set_label("Year of emergence (ensemble median, SNR ≥ 2)")
    fig.legend(handles=[Patch(color="#c8c8c8", label="No emergence by 2100")], loc="lower right", frameon=False,
               bbox_to_anchor=(0.98, 0.0))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    pd.set_option("display.width", 170)
    g = np.load(CELLS / "grid.npz")
    lat, lon, land = g["lat"], g["lon"], g["land"]
    cell_lat = np.repeat(lat[:, None], lon.size, axis=1)[land]
    rows, maps = [], {}
    for scenario in SCENARIOS:
        for var in ("tas", "hurs"):
            per_model = []
            for model in MODELS:
                years, data = load(model, scenario, var)
                per_model.append(toe_cells(years, data))
            per_model = np.vstack(per_model)
            ensemble = np.median(per_model, axis=0)
            np.savez(OUT / f"toe_cells_{scenario}_{var}.npz", toe_ensemble=ensemble,
                     n_models_emerged=np.isfinite(per_model).sum(axis=0))
            rows += summarize(ensemble, cell_lat, var, scenario)
            if scenario == MAP_SCENARIO:
                maps[var] = ensemble
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "toe_cells_summary.csv", index=False)
    print("## Area-weighted share of land where the ensemble-median ToE (SNR >= 2) has occurred")
    print(summary.set_index(["scenario", "variable", "region"]).round(2).to_string())
    (REPO / "tg2" / "figures").mkdir(parents=True, exist_ok=True)
    plot(maps, lat, lon, land, REPO / "tg2" / "figures" / f"toe_maps_{MAP_SCENARIO}.png")
    print(f"\nSaved: {REPO / 'tg2/figures' / f'toe_maps_{MAP_SCENARIO}.png'}, {OUT / 'toe_cells_summary.csv'}")


if __name__ == "__main__":
    main()
