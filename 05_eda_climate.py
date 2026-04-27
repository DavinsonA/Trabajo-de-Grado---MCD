#!/usr/bin/env python3
"""
05_eda_climate.py
Análisis Exploratorio de Datos (EDA) para tas y hurs.

Genera figuras y estadísticas para los entregables:
  - Distribuciones por subregión y periodo
  - Series temporales anuales por modelo
  - Ciclo estacional por subregión
  - Comparación inter-modelo
  - Correlación tas-hurs
  - Mapas espaciales de promedios y variabilidad

Uso:
    source /media/volume/jay2vol/nexgddp/venv/bin/activate
    python 05_eda_climate.py
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

BASE_DIR = Path("/media/volume/jay2vol/nexgddp")
RAW_DIR = BASE_DIR / "raw"
OUT_DIR = BASE_DIR / "outputs" / "05_eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBREGIONS = {
    "Trópico Norte":       {"lat_min": 0.0,   "lat_max": 13.0},
    "Amazonía Central":    {"lat_min": -15.0,  "lat_max": 0.0},
    "Andes Subtropicales": {"lat_min": -30.0,  "lat_max": -15.0},
    "Cono Sur":            {"lat_min": -56.0,  "lat_max": -30.0},
}

SUBREGION_COLORS = {
    "Trópico Norte": "#E74C3C",
    "Amazonía Central": "#27AE60",
    "Andes Subtropicales": "#F39C12",
    "Cono Sur": "#3498DB",
}

MODEL_COLORS = {
    "ACCESS-CM2": "#1f77b4",
    "CanESM5": "#ff7f0e",
    "CESM2": "#2ca02c",
    "MIROC6": "#d62728",
    "MPI-ESM1-2-HR": "#9467bd",
}

SCENARIO_COLORS = {
    "historical": "#333333",
    "ssp126": "#2196F3",
    "ssp245": "#FFC107",
    "ssp370": "#FF9800",
    "ssp585": "#F44336",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.grid": True,
    "axes.gridcolor": "#E0E0E0",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def get_available_models():
    """Retorna lista de modelos disponibles."""
    models = sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])
    print(f"  Modelos disponibles: {models}")
    return models


def load_annual_series(model, scenario, variable, subregion_bounds=None):
    """
    Carga todos los archivos de una combinación modelo/escenario/variable,
    calcula promedio anual (y promedio espacial si se dan bounds de subregión).
    Retorna un pandas Series indexado por año.
    """
    var_dir = RAW_DIR / model / scenario / variable
    if not var_dir.exists():
        return None

    files = sorted(var_dir.glob("*.nc"))
    if not files:
        return None

    annual_values = []

    for f in files:
        ds = xr.open_dataset(f)

        if subregion_bounds:
            ds = ds.sel(lat=slice(subregion_bounds["lat_min"],
                                  subregion_bounds["lat_max"]))

        # Promedio espacial y temporal (anual)
        val = float(ds[variable].mean().values)
        year = int(ds["time"].dt.year[0].values)
        annual_values.append({"year": year, "value": val})

        ds.close()

    df = pd.DataFrame(annual_values).set_index("year").sort_index()
    return df["value"]


def load_monthly_series(model, scenario, variable, subregion_bounds=None,
                        year_start=None, year_end=None):
    """
    Carga archivos y calcula promedios mensuales con promedio espacial.
    Retorna DataFrame con columnas [year, month, value].
    """
    var_dir = RAW_DIR / model / scenario / variable
    if not var_dir.exists():
        return None

    files = sorted(var_dir.glob("*.nc"))
    if not files:
        return None

    monthly_records = []

    for f in files:
        # Extraer año del nombre del archivo
        parts = f.stem.split("_")
        file_year = int(parts[-2])  # penúltimo es el año (antes de "SA")

        if year_start and file_year < year_start:
            continue
        if year_end and file_year > year_end:
            continue

        ds = xr.open_dataset(f)

        if subregion_bounds:
            ds = ds.sel(lat=slice(subregion_bounds["lat_min"],
                                  subregion_bounds["lat_max"]))

        # Promedio espacial por día, luego agrupar por mes
        spatial_mean = ds[variable].mean(dim=["lat", "lon"])
        monthly = spatial_mean.resample(time="1ME").mean()

        for t, val in zip(monthly["time"].values, monthly.values):
            t_dt = pd.Timestamp(t)
            monthly_records.append({
                "year": t_dt.year,
                "month": t_dt.month,
                "value": float(val),
            })

        ds.close()

    if not monthly_records:
        return None

    return pd.DataFrame(monthly_records)


def load_spatial_mean(model, scenario, variable, year_start, year_end):
    """Carga y promedia espacialmente un rango de años. Retorna DataArray 2D (lat × lon)."""
    var_dir = RAW_DIR / model / scenario / variable
    if not var_dir.exists():
        return None

    files = sorted(var_dir.glob("*.nc"))
    selected = []
    for f in files:
        parts = f.stem.split("_")
        file_year = int(parts[-2])
        if year_start <= file_year <= year_end:
            selected.append(f)

    if not selected:
        return None

    datasets = [xr.open_dataset(f) for f in selected]
    combined = xr.concat(datasets, dim="time")
    mean_map = combined[variable].mean(dim="time")

    for ds in datasets:
        ds.close()

    return mean_map


# =============================================================================
# EDA 1: SERIES TEMPORALES ANUALES POR MODELO
# =============================================================================

def eda_annual_timeseries(models):
    """Series temporales anuales de tas por modelo, todos los escenarios."""
    print("\n" + "=" * 70)
    print("  EDA 1: Series temporales anuales — tas")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()

    for idx, (region_name, bounds) in enumerate(SUBREGIONS.items()):
        ax = axes[idx]
        ax.set_title(region_name, fontweight="bold")

        for model in models:
            for scenario in ["historical", "ssp245", "ssp585"]:
                series = load_annual_series(model, scenario, "tas", bounds)
                if series is None:
                    continue

                # Convertir K a °C
                series = series - 273.15

                color = SCENARIO_COLORS[scenario]
                alpha = 0.5 if scenario == "historical" else 0.35
                lw = 1.2 if scenario == "historical" else 0.8

                label = f"{model} ({scenario})" if model == models[0] else None
                ax.plot(series.index, series.values, color=color,
                        alpha=alpha, linewidth=lw)

        ax.set_ylabel("Temperatura (°C)")
        ax.set_xlabel("Año")

    # Leyenda manual por escenario
    legend_patches = [
        plt.Line2D([0], [0], color=SCENARIO_COLORS["historical"], lw=2, label="historical"),
        plt.Line2D([0], [0], color=SCENARIO_COLORS["ssp245"], lw=2, label="SSP2-4.5"),
        plt.Line2D([0], [0], color=SCENARIO_COLORS["ssp585"], lw=2, label="SSP5-8.5"),
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, 1.02))

    plt.suptitle("Temperatura media anual por subregión (todos los modelos)",
                 fontweight="bold", y=1.05)
    plt.tight_layout()
    fig_path = OUT_DIR / "fig01_annual_timeseries_tas.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Guardada: {fig_path}")


def eda_annual_timeseries_hurs(models):
    """Series temporales anuales de hurs por modelo."""
    print("\n" + "=" * 70)
    print("  EDA 1b: Series temporales anuales — hurs")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()

    for idx, (region_name, bounds) in enumerate(SUBREGIONS.items()):
        ax = axes[idx]
        ax.set_title(region_name, fontweight="bold")

        for model in models:
            for scenario in ["historical", "ssp245", "ssp585"]:
                series = load_annual_series(model, scenario, "hurs", bounds)
                if series is None:
                    continue

                color = SCENARIO_COLORS[scenario]
                alpha = 0.5 if scenario == "historical" else 0.35
                ax.plot(series.index, series.values, color=color,
                        alpha=alpha, linewidth=0.8)

        ax.set_ylabel("Humedad relativa (%)")
        ax.set_xlabel("Año")

    legend_patches = [
        plt.Line2D([0], [0], color=SCENARIO_COLORS["historical"], lw=2, label="historical"),
        plt.Line2D([0], [0], color=SCENARIO_COLORS["ssp245"], lw=2, label="SSP2-4.5"),
        plt.Line2D([0], [0], color=SCENARIO_COLORS["ssp585"], lw=2, label="SSP5-8.5"),
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, 1.02))

    plt.suptitle("Humedad relativa media anual por subregión (todos los modelos)",
                 fontweight="bold", y=1.05)
    plt.tight_layout()
    fig_path = OUT_DIR / "fig02_annual_timeseries_hurs.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Guardada: {fig_path}")


# =============================================================================
# EDA 2: DISTRIBUCIONES POR PERIODO
# =============================================================================

def eda_distributions(models):
    """Distribuciones de tas y hurs comparando periodo histórico vs futuro."""
    print("\n" + "=" * 70)
    print("  EDA 2: Distribuciones por periodo")
    print("=" * 70)

    periods = {
        "Histórico\n(1960-1990)": ("historical", 1960, 1990),
        "Presente\n(1991-2014)": ("historical", 1991, 2014),
        "Futuro cercano\n(2030-2060, SSP2-4.5)": ("ssp245", 2030, 2060),
        "Futuro lejano\n(2070-2100, SSP5-8.5)": ("ssp585", 2070, 2100),
    }

    for variable, units, title in [("tas", "°C", "Temperatura"), ("hurs", "%", "Humedad relativa")]:
        fig, axes = plt.subplots(len(SUBREGIONS), len(periods), figsize=(16, 12),
                                 sharex="row", sharey="row")

        for row, (region_name, bounds) in enumerate(SUBREGIONS.items()):
            for col, (period_name, (scenario, y_start, y_end)) in enumerate(periods.items()):
                ax = axes[row, col]

                all_values = []
                for model in models[:3]:  # Muestrear 3 modelos para velocidad
                    series = load_monthly_series(model, scenario, variable, bounds,
                                                 y_start, y_end)
                    if series is not None:
                        vals = series["value"].values
                        if variable == "tas":
                            vals = vals - 273.15
                        all_values.extend(vals)

                if all_values:
                    ax.hist(all_values, bins=40, alpha=0.7,
                            color=SUBREGION_COLORS[region_name], edgecolor="white",
                            linewidth=0.3)
                    ax.axvline(np.mean(all_values), color="black", linestyle="--",
                               linewidth=1, alpha=0.7)
                    ax.text(0.95, 0.95, f"μ={np.mean(all_values):.1f}",
                            transform=ax.transAxes, ha="right", va="top", fontsize=8)

                if row == 0:
                    ax.set_title(period_name, fontsize=9, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(region_name, fontsize=9, fontweight="bold")
                if row == len(SUBREGIONS) - 1:
                    ax.set_xlabel(f"{variable} ({units})")

        plt.suptitle(f"Distribución de {title} mensual por subregión y periodo",
                     fontweight="bold", fontsize=13)
        plt.tight_layout()
        fig_path = OUT_DIR / f"fig03_distributions_{variable}.png"
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        print(f"  Guardada: {fig_path}")


# =============================================================================
# EDA 3: CICLO ESTACIONAL
# =============================================================================

def eda_seasonal_cycle(models):
    """Ciclo estacional promedio por subregión (climatología 1961-1990)."""
    print("\n" + "=" * 70)
    print("  EDA 3: Ciclo estacional")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    month_names = ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    for var_idx, (variable, units, ylabel) in enumerate([
        ("tas", "°C", "Temperatura (°C)"),
        ("hurs", "%", "Humedad relativa (%)")
    ]):
        ax = axes[var_idx]

        for region_name, bounds in SUBREGIONS.items():
            monthly_all = []

            for model in models[:3]:
                series = load_monthly_series(model, "historical", variable, bounds,
                                             1961, 1990)
                if series is not None:
                    monthly_all.append(series)

            if not monthly_all:
                continue

            combined = pd.concat(monthly_all)
            if variable == "tas":
                combined["value"] = combined["value"] - 273.15

            climatology = combined.groupby("month")["value"].agg(["mean", "std"])

            ax.plot(range(1, 13), climatology["mean"],
                    color=SUBREGION_COLORS[region_name],
                    linewidth=2, marker="o", markersize=4,
                    label=region_name)
            ax.fill_between(range(1, 13),
                            climatology["mean"] - climatology["std"],
                            climatology["mean"] + climatology["std"],
                            color=SUBREGION_COLORS[region_name], alpha=0.15)

        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_names)
        ax.set_xlabel("Mes")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Ciclo estacional — {variable} (1961-1990)", fontweight="bold")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = OUT_DIR / "fig04_seasonal_cycle.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Guardada: {fig_path}")


# =============================================================================
# EDA 4: COMPARACIÓN INTER-MODELO
# =============================================================================

def eda_model_comparison(models):
    """Comparación del spread inter-modelo por subregión."""
    print("\n" + "=" * 70)
    print("  EDA 4: Comparación inter-modelo")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (region_name, bounds) in enumerate(SUBREGIONS.items()):
        ax = axes[idx]
        ax.set_title(region_name, fontweight="bold")

        model_means = {}

        for model in models:
            series = load_annual_series(model, "historical", "tas", bounds)
            if series is not None:
                series = series - 273.15
                model_means[model] = series

        if not model_means:
            continue

        # Plot cada modelo
        for model, series in model_means.items():
            ax.plot(series.index, series.values,
                    color=MODEL_COLORS.get(model, "#888"),
                    linewidth=1.2, alpha=0.7, label=model)

        # Ensemble mean
        df_all = pd.DataFrame(model_means)
        ens_mean = df_all.mean(axis=1)
        ens_std = df_all.std(axis=1)

        ax.plot(ens_mean.index, ens_mean.values, color="black",
                linewidth=2, label="Ensemble mean", zorder=10)
        ax.fill_between(ens_mean.index,
                        ens_mean - 2*ens_std,
                        ens_mean + 2*ens_std,
                        color="gray", alpha=0.2, label="±2σ")

        ax.set_ylabel("Temperatura (°C)")
        ax.set_xlabel("Año")
        ax.legend(fontsize=7, loc="upper left")

        # Anotar spread
        spread = df_all.max(axis=1) - df_all.min(axis=1)
        ax.text(0.98, 0.05, f"Spread medio: {spread.mean():.2f}°C",
                transform=ax.transAxes, ha="right", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.suptitle("Comparación inter-modelo — tas histórico (1950-2014)",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig_path = OUT_DIR / "fig05_model_comparison_tas.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Guardada: {fig_path}")


# =============================================================================
# EDA 5: CORRELACIÓN TAS vs HURS
# =============================================================================

def eda_correlation_tas_hurs(models):
    """Correlación entre temperatura y humedad por subregión y década."""
    print("\n" + "=" * 70)
    print("  EDA 5: Correlación tas vs hurs")
    print("=" * 70)

    decades = {
        "1960-1980": ("historical", 1960, 1980),
        "1990-2014": ("historical", 1990, 2014),
        "2040-2060\n(SSP2-4.5)": ("ssp245", 2040, 2060),
        "2080-2100\n(SSP5-8.5)": ("ssp585", 2080, 2100),
    }

    fig, axes = plt.subplots(len(SUBREGIONS), len(decades), figsize=(16, 14))

    corr_records = []

    for row, (region_name, bounds) in enumerate(SUBREGIONS.items()):
        for col, (decade_name, (scenario, y_start, y_end)) in enumerate(decades.items()):
            ax = axes[row, col]

            tas_vals = []
            hurs_vals = []

            for model in models[:3]:  # 3 modelos para velocidad
                tas_series = load_monthly_series(model, scenario, "tas", bounds,
                                                  y_start, y_end)
                hurs_series = load_monthly_series(model, scenario, "hurs", bounds,
                                                   y_start, y_end)

                if tas_series is not None and hurs_series is not None:
                    # Alinear por año-mes
                    tas_series["key"] = tas_series["year"]*100 + tas_series["month"]
                    hurs_series["key"] = hurs_series["year"]*100 + hurs_series["month"]

                    merged = pd.merge(tas_series, hurs_series, on="key",
                                      suffixes=("_tas", "_hurs"))

                    tas_vals.extend((merged["value_tas"] - 273.15).values)
                    hurs_vals.extend(merged["value_hurs"].values)

            if tas_vals and hurs_vals:
                tas_arr = np.array(tas_vals)
                hurs_arr = np.array(hurs_vals)

                # Scatter
                ax.scatter(tas_arr, hurs_arr, s=3, alpha=0.3,
                          color=SUBREGION_COLORS[region_name])

                # Línea de tendencia
                valid = np.isfinite(tas_arr) & np.isfinite(hurs_arr)
                if valid.sum() > 10:
                    z = np.polyfit(tas_arr[valid], hurs_arr[valid], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(tas_arr[valid].min(), tas_arr[valid].max(), 50)
                    ax.plot(x_line, p(x_line), "k--", linewidth=1, alpha=0.7)

                    # Correlación
                    r = np.corrcoef(tas_arr[valid], hurs_arr[valid])[0, 1]
                    ax.text(0.05, 0.95, f"r = {r:.3f}\nn = {valid.sum()}",
                            transform=ax.transAxes, fontsize=8, va="top",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

                    corr_records.append({
                        "region": region_name,
                        "period": decade_name.replace("\n", " "),
                        "scenario": scenario,
                        "r": round(r, 4),
                        "slope": round(z[0], 4),
                        "n": int(valid.sum()),
                    })

            if row == 0:
                ax.set_title(decade_name, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{region_name}\nhurs (%)", fontsize=8)
            if row == len(SUBREGIONS) - 1:
                ax.set_xlabel("tas (°C)", fontsize=8)

    plt.suptitle("Correlación temperatura–humedad por subregión y periodo",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig_path = OUT_DIR / "fig06_correlation_tas_hurs.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Guardada: {fig_path}")

    # Guardar tabla de correlaciones
    if corr_records:
        df_corr = pd.DataFrame(corr_records)
        csv_path = OUT_DIR / "correlation_tas_hurs.csv"
        df_corr.to_csv(csv_path, index=False)
        print(f"  Tabla de correlaciones: {csv_path}")
        print(f"\n  Resumen de correlaciones tas-hurs:")
        print(df_corr.to_string(index=False))


# =============================================================================
# EDA 6: MAPAS ESPACIALES
# =============================================================================

def eda_spatial_maps(models):
    """Mapas de temperatura y humedad media por periodo."""
    print("\n" + "=" * 70)
    print("  EDA 6: Mapas espaciales")
    print("=" * 70)

    periods = {
        "Histórico (1961-1990)": ("historical", 1961, 1990),
        "Futuro (2071-2100, SSP5-8.5)": ("ssp585", 2071, 2100),
    }

    for variable, cmap, units, label in [
        ("tas", "RdYlBu_r", "°C", "Temperatura media"),
        ("hurs", "BrBG", "%", "Humedad relativa media"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for col, (period_name, (scenario, y_start, y_end)) in enumerate(periods.items()):
            ax = axes[col]

            # Promediar sobre modelos disponibles
            maps = []
            for model in models[:3]:
                m = load_spatial_mean(model, scenario, variable, y_start, y_end)
                if m is not None:
                    maps.append(m)

            if not maps:
                ax.set_title(f"{period_name}\n(sin datos)")
                continue

            # Ensemble mean
            ensemble = xr.concat(maps, dim="model").mean(dim="model")

            lon_vals = ensemble.lon.values
            lat_vals = ensemble.lat.values
            lon_180 = np.where(lon_vals > 180, lon_vals - 360, lon_vals)

            data = ensemble.values
            if variable == "tas":
                data = data - 273.15

            im = ax.pcolormesh(lon_180, lat_vals, data, cmap=cmap, shading="auto")
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label(f"{label} ({units})")

            # Subregiones como líneas
            for reg_name, bounds in SUBREGIONS.items():
                ax.axhline(bounds["lat_min"], color="gray", linewidth=0.5, linestyle=":")
                ax.axhline(bounds["lat_max"], color="gray", linewidth=0.5, linestyle=":")

            ax.set_xlabel("Longitud")
            ax.set_ylabel("Latitud")
            ax.set_title(period_name, fontweight="bold")
            ax.set_aspect("equal")

        plt.suptitle(f"Mapa espacial — {label}", fontweight="bold", fontsize=13)
        plt.tight_layout()
        fig_path = OUT_DIR / f"fig07_spatial_map_{variable}.png"
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        print(f"  Guardada: {fig_path}")

    # Mapa de DIFERENCIA (cambio proyectado)
    print("\n  Generando mapa de cambio proyectado...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for var_idx, (variable, cmap, units, label) in enumerate([
        ("tas", "RdBu_r", "°C", "Cambio en temperatura"),
        ("hurs", "BrBG", "%", "Cambio en humedad relativa"),
    ]):
        ax = axes[var_idx]

        hist_maps = []
        future_maps = []

        for model in models[:3]:
            h = load_spatial_mean(model, "historical", variable, 1961, 1990)
            f = load_spatial_mean(model, "ssp585", variable, 2071, 2100)
            if h is not None and f is not None:
                hist_maps.append(h)
                future_maps.append(f)

        if hist_maps and future_maps:
            hist_ens = xr.concat(hist_maps, dim="model").mean(dim="model")
            fut_ens = xr.concat(future_maps, dim="model").mean(dim="model")
            diff = fut_ens - hist_ens

            lon_vals = diff.lon.values
            lat_vals = diff.lat.values
            lon_180 = np.where(lon_vals > 180, lon_vals - 360, lon_vals)

            data = diff.values

            # Symmetric colorbar
            vmax = np.nanpercentile(np.abs(data), 95)
            im = ax.pcolormesh(lon_180, lat_vals, data, cmap=cmap,
                               shading="auto", vmin=-vmax, vmax=vmax)
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label(f"Δ{variable} ({units})")

            ax.set_xlabel("Longitud")
            ax.set_ylabel("Latitud")
            ax.set_title(f"{label}\n(SSP5-8.5 2071-2100 vs Hist 1961-1990)", fontweight="bold",
                        fontsize=10)
            ax.set_aspect("equal")

    plt.tight_layout()
    fig_path = OUT_DIR / "fig08_spatial_change_map.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Guardada: {fig_path}")


# =============================================================================
# EDA 7: RESUMEN ESTADÍSTICO PARA FEATURES
# =============================================================================

def eda_feature_summary(models):
    """
    Genera tabla resumen de estadísticas por variable, subregión y periodo.
    Útil para justificar la selección/descarte de features.
    """
    print("\n" + "=" * 70)
    print("  EDA 7: Resumen estadístico de features")
    print("=" * 70)

    records = []

    for variable in ["tas", "hurs"]:
        for region_name, bounds in SUBREGIONS.items():
            for period_name, (scenario, y_start, y_end) in [
                ("Histórico 1961-1990", ("historical", 1961, 1990)),
                ("Histórico 1991-2014", ("historical", 1991, 2014)),
                ("SSP2-4.5 2041-2070", ("ssp245", 2041, 2070)),
                ("SSP5-8.5 2071-2100", ("ssp585", 2071, 2100)),
            ]:
                all_vals = []
                for model in models[:3]:
                    series = load_monthly_series(model, scenario, variable, bounds,
                                                 y_start, y_end)
                    if series is not None:
                        vals = series["value"].values
                        if variable == "tas":
                            vals = vals - 273.15
                        all_vals.extend(vals)

                if all_vals:
                    arr = np.array(all_vals)
                    records.append({
                        "variable": variable,
                        "region": region_name,
                        "period": period_name,
                        "mean": round(np.nanmean(arr), 2),
                        "std": round(np.nanstd(arr), 2),
                        "min": round(np.nanmin(arr), 2),
                        "p25": round(np.nanpercentile(arr, 25), 2),
                        "median": round(np.nanmedian(arr), 2),
                        "p75": round(np.nanpercentile(arr, 75), 2),
                        "max": round(np.nanmax(arr), 2),
                        "iqr": round(np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25), 2),
                        "cv": round(np.nanstd(arr) / np.nanmean(arr) * 100, 2) if np.nanmean(arr) != 0 else None,
                        "n_samples": len(arr),
                    })

    df_features = pd.DataFrame(records)
    csv_path = OUT_DIR / "feature_summary_stats.csv"
    df_features.to_csv(csv_path, index=False)
    print(f"  Guardado: {csv_path}")
    print(f"\n  Resumen (primeras 10 filas):")
    print(df_features.head(10).to_string(index=False))

    return df_features


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#" * 70)
    print("  NEX-GDDP-CMIP6 — EDA CLIMÁTICO")
    print(f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)

    models = get_available_models()

    # Ejecutar todos los análisis
    t0 = datetime.now()

    eda_annual_timeseries(models)          # fig01
    eda_annual_timeseries_hurs(models)     # fig02
    eda_distributions(models)              # fig03
    eda_seasonal_cycle(models)             # fig04
    eda_model_comparison(models)           # fig05
    eda_correlation_tas_hurs(models)       # fig06
    eda_spatial_maps(models)               # fig07, fig08
    eda_feature_summary(models)            # CSV

    elapsed = (datetime.now() - t0).total_seconds()

    print(f"\n" + "=" * 70)
    print(f"  EDA completado en {elapsed/60:.1f} minutos")
    print(f"  Todos los outputs en: {OUT_DIR}")
    print(f"\n  Archivos generados:")
    for f in sorted(OUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:45s} ({size_kb:.0f} KB)")
    print(f"=" * 70)


if __name__ == "__main__":
    main()
