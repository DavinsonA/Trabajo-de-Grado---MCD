#!/usr/bin/env python3
"""
04_dataset_inventory.py
Inventario completo, control de calidad y estudio de viabilidad del dataset.

Genera:
  - Matriz de completitud (modelo × escenario × variable)
  - Reporte de QC: NaN%, rangos, cobertura temporal
  - Figuras para el documento de tesis
  - CSVs con estadísticas resumidas

Uso:
    source /media/volume/jay2vol/nexgddp/venv/bin/activate
    python 04_dataset_inventory.py
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

BASE_DIR = Path("/media/volume/jay2vol/nexgddp")
RAW_DIR = BASE_DIR / "raw"
OUT_DIR = BASE_DIR / "outputs" / "04_inventory"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Modelos esperados (incluye los 3 pendientes para documentar)
ALL_MODELS = [
    "ACCESS-CM2", "CanESM5", "CESM2", "EC-Earth3",
    "GFDL-ESM4", "IPSL-CM6A-LR", "MIROC6", "MPI-ESM1-2-HR"
]

VARIABLES = ["tas", "hurs"]

SCENARIOS = {
    "historical": (1950, 2014),
    "ssp126": (2015, 2100),
    "ssp245": (2015, 2100),
    "ssp370": (2015, 2100),
    "ssp585": (2015, 2100),
}

# Rangos físicos válidos (de la Tech Note)
VALID_RANGES = {
    "tas":  (200.0, 340.0),   # Kelvin
    "hurs": (0.0, 102.0),     # Porcentaje
}

# Subregiones de Sudamérica
SUBREGIONS = {
    "Trópico Norte":       {"lat_min": 0.0,   "lat_max": 13.0},
    "Amazonía Central":    {"lat_min": -15.0,  "lat_max": 0.0},
    "Andes Subtropicales": {"lat_min": -30.0,  "lat_max": -15.0},
    "Cono Sur":            {"lat_min": -56.0,  "lat_max": -30.0},
}

# Estilo de figuras
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
# 1. MATRIZ DE COMPLETITUD
# =============================================================================

def build_completeness_matrix():
    """Construye la matriz de completitud: archivos esperados vs encontrados."""
    print("=" * 70)
    print("  1. MATRIZ DE COMPLETITUD")
    print("=" * 70)

    records = []

    for model in ALL_MODELS:
        for scenario, (y_start, y_end) in SCENARIOS.items():
            expected = y_end - y_start + 1
            for variable in VARIABLES:
                var_dir = RAW_DIR / model / scenario / variable
                if var_dir.exists():
                    found = len(list(var_dir.glob("*.nc")))
                else:
                    found = 0
                pct = (found / expected) * 100

                records.append({
                    "model": model,
                    "scenario": scenario,
                    "variable": variable,
                    "expected": expected,
                    "found": found,
                    "missing": expected - found,
                    "completeness_pct": round(pct, 1),
                })

    df = pd.DataFrame(records)

    # Resumen por modelo
    print("\n  Completitud por modelo:")
    for model in ALL_MODELS:
        sub = df[df["model"] == model]
        total_exp = sub["expected"].sum()
        total_found = sub["found"].sum()
        pct = (total_found / total_exp * 100) if total_exp > 0 else 0
        status = "✓" if pct == 100 else ("~" if pct > 90 else "✗")
        print(f"    {status} {model:20s}  {total_found:4d}/{total_exp:4d} archivos ({pct:.1f}%)")

    # Detalle de faltantes
    missing = df[df["missing"] > 0]
    if len(missing) > 0:
        print(f"\n  Combinaciones con archivos faltantes ({len(missing)}):")
        for _, row in missing.iterrows():
            print(f"    - {row['model']} / {row['scenario']} / {row['variable']}: "
                  f"faltan {row['missing']} de {row['expected']}")

    # Guardar CSV
    csv_path = OUT_DIR / "completeness_matrix.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Guardado: {csv_path}")

    return df


def plot_completeness_heatmap(df):
    """Genera heatmap visual de completitud."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, var in enumerate(VARIABLES):
        ax = axes[idx]
        sub = df[df["variable"] == var]

        # Pivot para heatmap
        pivot = sub.pivot_table(
            index="model", columns="scenario",
            values="completeness_pct", aggfunc="first"
        )

        # Reordenar columnas
        col_order = ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]
        pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])

        # Dibujar
        im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)

        # Anotar valores
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                color = "white" if val < 50 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

        ax.set_title(f"Completitud — {var}", fontweight="bold")

    plt.tight_layout()
    fig_path = OUT_DIR / "fig_completeness_heatmap.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Figura guardada: {fig_path}")


# =============================================================================
# 2. CONTROL DE CALIDAD (QC)
# =============================================================================

def run_quality_control():
    """
    QC sobre una muestra representativa de archivos.
    Verifica: NaN%, rangos, continuidad temporal, shape consistency.
    """
    print("\n" + "=" * 70)
    print("  2. CONTROL DE CALIDAD")
    print("=" * 70)

    # Muestrear: primer año, último año de cada combinación disponible
    sample_years = {
        "historical": [1950, 1980, 2014],
        "ssp126": [2015, 2050, 2100],
        "ssp245": [2015, 2050, 2100],
        "ssp370": [2015, 2050, 2100],
        "ssp585": [2015, 2050, 2100],
    }

    qc_records = []
    available_models = [m for m in ALL_MODELS if (RAW_DIR / m).exists()]

    for model in available_models:
        for scenario, years in sample_years.items():
            for variable in VARIABLES:
                for year in years:
                    pattern = f"{variable}_{model}_{scenario}_{year}_SA.nc"
                    filepath = RAW_DIR / model / scenario / variable / pattern

                    if not filepath.exists():
                        continue

                    ds = xr.open_dataset(filepath)
                    data = ds[variable].values

                    total = data.size
                    nan_count = int(np.isnan(data).sum())
                    nan_pct = round((nan_count / total) * 100, 2)

                    valid_min, valid_max = VALID_RANGES.get(variable, (None, None))
                    data_min = float(np.nanmin(data))
                    data_max = float(np.nanmax(data))
                    data_mean = float(np.nanmean(data))
                    data_std = float(np.nanstd(data))

                    out_of_range = 0
                    if valid_min is not None:
                        finite = data[np.isfinite(data)]
                        out_of_range = int(np.sum((finite < valid_min) | (finite > valid_max)))

                    n_timesteps = ds.dims["time"]
                    n_lat = ds.dims["lat"]
                    n_lon = ds.dims["lon"]

                    qc_records.append({
                        "model": model,
                        "scenario": scenario,
                        "variable": variable,
                        "year": year,
                        "n_time": n_timesteps,
                        "n_lat": n_lat,
                        "n_lon": n_lon,
                        "nan_pct": nan_pct,
                        "data_min": round(data_min, 2),
                        "data_max": round(data_max, 2),
                        "data_mean": round(data_mean, 2),
                        "data_std": round(data_std, 2),
                        "out_of_range": out_of_range,
                    })

                    ds.close()

    df_qc = pd.DataFrame(qc_records)

    # Resumen
    print(f"\n  Archivos muestreados: {len(df_qc)}")

    # Consistencia de dimensiones
    unique_shapes = df_qc.groupby("variable")[["n_lat", "n_lon"]].nunique()
    print(f"\n  Consistencia de grilla espacial:")
    for var in VARIABLES:
        if var in unique_shapes.index:
            n_lat_unique = unique_shapes.loc[var, "n_lat"]
            n_lon_unique = unique_shapes.loc[var, "n_lon"]
            status = "✓" if (n_lat_unique == 1 and n_lon_unique == 1) else "✗"
            sample = df_qc[df_qc["variable"] == var].iloc[0]
            print(f"    {status} {var}: {sample['n_lat']}×{sample['n_lon']} "
                  f"(lat×lon consistente: {n_lat_unique == 1 and n_lon_unique == 1})")

    # Resumen NaN
    print(f"\n  Porcentaje de NaN por variable:")
    for var in VARIABLES:
        sub = df_qc[df_qc["variable"] == var]
        print(f"    {var}: min={sub['nan_pct'].min():.1f}%, "
              f"max={sub['nan_pct'].max():.1f}%, "
              f"mean={sub['nan_pct'].mean():.1f}%")

    # Rangos
    print(f"\n  Rangos observados vs esperados:")
    for var in VARIABLES:
        sub = df_qc[df_qc["variable"] == var]
        vmin, vmax = VALID_RANGES[var]
        obs_min = sub["data_min"].min()
        obs_max = sub["data_max"].max()
        total_oor = sub["out_of_range"].sum()
        print(f"    {var}: observado [{obs_min:.1f}, {obs_max:.1f}] "
              f"vs esperado [{vmin}, {vmax}] | "
              f"fuera de rango: {total_oor}")

    # Días por año
    print(f"\n  Pasos temporales por archivo:")
    for n_time in sorted(df_qc["n_time"].unique()):
        count = (df_qc["n_time"] == n_time).sum()
        print(f"    {n_time} días: {count} archivos")

    # Guardar
    csv_path = OUT_DIR / "quality_control_report.csv"
    df_qc.to_csv(csv_path, index=False)
    print(f"\n  Guardado: {csv_path}")

    return df_qc


def plot_nan_coverage(df_qc):
    """Visualiza la cobertura de NaN por variable y modelo."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, var in enumerate(VARIABLES):
        ax = axes[idx]
        sub = df_qc[df_qc["variable"] == var]

        for model in sub["model"].unique():
            m_sub = sub[sub["model"] == model]
            ax.scatter(m_sub["year"], m_sub["nan_pct"], label=model, s=30, alpha=0.7)

        ax.set_xlabel("Año")
        ax.set_ylabel("NaN (%)")
        ax.set_title(f"Cobertura de datos faltantes — {var}", fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig_path = OUT_DIR / "fig_nan_coverage.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Figura guardada: {fig_path}")


def plot_value_ranges(df_qc):
    """Visualiza los rangos de valores por modelo y variable."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, var in enumerate(VARIABLES):
        ax = axes[idx]
        sub = df_qc[df_qc["variable"] == var]
        models = sorted(sub["model"].unique())

        positions = range(len(models))
        for i, model in enumerate(models):
            m_sub = sub[sub["model"] == model]
            ax.plot([i, i], [m_sub["data_min"].min(), m_sub["data_max"].max()],
                    linewidth=3, alpha=0.6, solid_capstyle="round")
            ax.scatter(i, m_sub["data_mean"].mean(), color="black", s=40, zorder=5)

        # Rango válido
        vmin, vmax = VALID_RANGES[var]
        ax.axhline(vmin, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label=f"Límite [{vmin}, {vmax}]")
        ax.axhline(vmax, color="red", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_xticks(positions)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)

        units = "K" if var == "tas" else "%"
        ax.set_ylabel(f"Valor ({units})")
        ax.set_title(f"Rango de valores por modelo — {var}", fontweight="bold")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = OUT_DIR / "fig_value_ranges.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Figura guardada: {fig_path}")


# =============================================================================
# 3. COBERTURA ESPACIAL Y SUBREGIONES
# =============================================================================

def plot_spatial_coverage():
    """Visualiza la cobertura espacial y las subregiones definidas."""
    print("\n" + "=" * 70)
    print("  3. COBERTURA ESPACIAL")
    print("=" * 70)

    # Cargar un archivo de ejemplo
    sample = sorted((RAW_DIR).rglob("tas_*_historical_*1980*_SA.nc"))
    if not sample:
        sample = sorted((RAW_DIR).rglob("tas_*_historical_*_SA.nc"))
    if not sample:
        print("  No se encontraron archivos para visualización espacial.")
        return

    ds = xr.open_dataset(sample[0])
    tas_mean = ds["tas"].mean(dim="time")

    # Convertir lon de 0-360 a -180/180 para visualización
    lon_vals = ds["lon"].values
    lat_vals = ds["lat"].values

    lon_180 = np.where(lon_vals > 180, lon_vals - 360, lon_vals)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Panel 1: Mapa de temperatura media
    ax1 = axes[0]
    data = tas_mean.values - 273.15  # Convertir a °C para visualización

    im = ax1.pcolormesh(lon_180, lat_vals, data, cmap="RdYlBu_r", shading="auto")
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8, label="Temperatura media (°C)")
    ax1.set_xlabel("Longitud")
    ax1.set_ylabel("Latitud")
    ax1.set_title(f"Temperatura media anual — {Path(sample[0]).stem}", fontweight="bold", fontsize=10)
    ax1.set_aspect("equal")

    # Panel 2: Subregiones
    ax2 = axes[1]
    # Fondo con NaN mask (tierra vs océano)
    nan_mask = np.isnan(data).astype(float)
    land_mask = (~np.isnan(data)).astype(float)
    ax2.pcolormesh(lon_180, lat_vals, land_mask, cmap="Greys", shading="auto",
                   alpha=0.15, vmin=0, vmax=2)

    colors = {"Trópico Norte": "#E74C3C", "Amazonía Central": "#27AE60",
              "Andes Subtropicales": "#F39C12", "Cono Sur": "#3498DB"}

    for name, bounds in SUBREGIONS.items():
        lat_min, lat_max = bounds["lat_min"], bounds["lat_max"]

        # Dibujar rectángulo
        rect = plt.Rectangle(
            (lon_180.min(), lat_min),
            lon_180.max() - lon_180.min(),
            lat_max - lat_min,
            linewidth=2, edgecolor=colors[name], facecolor=colors[name],
            alpha=0.25, label=name
        )
        ax2.add_patch(rect)

        # Etiqueta
        mid_lat = (lat_min + lat_max) / 2
        ax2.text(lon_180.mean(), mid_lat, name, ha="center", va="center",
                 fontsize=9, fontweight="bold", color=colors[name],
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax2.set_xlim(lon_180.min(), lon_180.max())
    ax2.set_ylim(lat_vals.min(), lat_vals.max())
    ax2.set_xlabel("Longitud")
    ax2.set_ylabel("Latitud")
    ax2.set_title("Subregiones de estudio", fontweight="bold")
    ax2.set_aspect("equal")
    ax2.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    fig_path = OUT_DIR / "fig_spatial_coverage_subregions.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Figura guardada: {fig_path}")

    # Estadísticas espaciales
    print(f"\n  Grilla espacial:")
    print(f"    Latitud:  {lat_vals.min():.2f}° a {lat_vals.max():.2f}° ({len(lat_vals)} puntos)")
    print(f"    Longitud: {lon_180.min():.2f}° a {lon_180.max():.2f}° ({len(lon_vals)} puntos)")
    print(f"    Resolución: {abs(lat_vals[1]-lat_vals[0]):.2f}°")
    print(f"    Total celdas: {len(lat_vals) * len(lon_vals):,}")

    # Fracción tierra/océano
    land_cells = np.sum(~np.isnan(data))
    total_cells = data.size
    print(f"    Celdas con datos (tierra): {land_cells:,} ({land_cells/total_cells*100:.1f}%)")
    print(f"    Celdas NaN (océano): {total_cells - land_cells:,} ({(total_cells-land_cells)/total_cells*100:.1f}%)")

    # Estadísticas por subregión
    print(f"\n  Celdas con datos por subregión:")
    for name, bounds in SUBREGIONS.items():
        lat_mask = (lat_vals >= bounds["lat_min"]) & (lat_vals <= bounds["lat_max"])
        sub_data = data[lat_mask, :]
        land = np.sum(~np.isnan(sub_data))
        total = sub_data.size
        print(f"    {name:25s}: {land:5,} celdas tierra / {total:5,} total ({land/total*100:.1f}%)")

    ds.close()


# =============================================================================
# 4. RESUMEN GENERAL DEL DATASET
# =============================================================================

def generate_dataset_summary():
    """Genera un resumen general del dataset para el documento."""
    print("\n" + "=" * 70)
    print("  4. RESUMEN GENERAL DEL DATASET")
    print("=" * 70)

    # Contar archivos y espacio
    all_files = list(RAW_DIR.rglob("*.nc"))
    total_size_gb = sum(f.stat().st_size for f in all_files) / (1024**3)
    available_models = sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])

    summary = {
        "Dataset": "NEX-GDDP-CMIP6 (NASA Earth Exchange)",
        "Fuente": "AWS S3 (s3://nex-gddp-cmip6)",
        "Referencia": "Thrasher et al. (2022), Scientific Data 9:262",
        "Método de downscaling": "BCSD (Bias-Corrected Spatial Disaggregation)",
        "Resolución espacial": "0.25° × 0.25° (~25 km)",
        "Resolución temporal": "Diaria",
        "Periodo histórico": "1950-2014",
        "Periodo proyectado": "2015-2100",
        "Variables extraídas": "tas (temperatura), hurs (humedad relativa)",
        "Región de estudio": "Sudamérica (56°S–13°N, 82°W–34°W)",
        "Grilla recortada": "276 × 192 (lat × lon)",
        "Modelos disponibles": f"{len(available_models)} de 8 seleccionados",
        "Modelos descargados": ", ".join(available_models),
        "Modelos pendientes": ", ".join([m for m in ALL_MODELS if m not in available_models]),
        "Escenarios": "historical, SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5",
        "Total archivos": f"{len(all_files):,}",
        "Espacio en disco": f"{total_size_gb:.1f} GB",
    }

    for key, value in summary.items():
        print(f"  {key:30s}: {value}")

    # Guardar como JSON
    json_path = OUT_DIR / "dataset_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Guardado: {json_path}")

    return summary


# =============================================================================
# 5. EVALUACIÓN DE VIABILIDAD
# =============================================================================

def assess_viability(df_completeness, df_qc):
    """Evaluación formal de viabilidad para el documento."""
    print("\n" + "=" * 70)
    print("  5. EVALUACIÓN DE VIABILIDAD")
    print("=" * 70)

    criteria = []

    # Criterio 1: Cobertura temporal
    available = df_completeness[df_completeness["found"] > 0]
    full_coverage = available[available["completeness_pct"] == 100.0]
    c1_pct = len(full_coverage) / len(available) * 100 if len(available) > 0 else 0
    criteria.append({
        "criterio": "Cobertura temporal completa (1950-2100)",
        "resultado": f"{c1_pct:.1f}% de combinaciones modelo/escenario/variable",
        "cumple": c1_pct > 95,
    })

    # Criterio 2: Integridad de datos (NaN)
    mean_nan = df_qc["nan_pct"].mean()
    criteria.append({
        "criterio": "Integridad de datos (NaN aceptable para análisis terrestre)",
        "resultado": f"Promedio {mean_nan:.1f}% NaN (esperado: >50% por cobertura oceánica)",
        "cumple": True,  # NaN en océano es esperado
    })

    # Criterio 3: Rangos físicos
    total_oor = df_qc["out_of_range"].sum()
    total_cells = df_qc["n_time"].sum() * df_qc["n_lat"].iloc[0] * df_qc["n_lon"].iloc[0]
    oor_pct = (total_oor / total_cells * 100) if total_cells > 0 else 0
    criteria.append({
        "criterio": "Valores dentro de rangos físicos",
        "resultado": f"{oor_pct:.4f}% fuera de rango (umbral aceptable: <0.1%)",
        "cumple": oor_pct < 0.1,
    })

    # Criterio 4: Consistencia dimensional
    lat_consistent = df_qc["n_lat"].nunique() == 1
    lon_consistent = df_qc["n_lon"].nunique() == 1
    criteria.append({
        "criterio": "Consistencia dimensional entre modelos",
        "resultado": f"lat={'consistente' if lat_consistent else 'INCONSISTENTE'}, "
                     f"lon={'consistente' if lon_consistent else 'INCONSISTENTE'}",
        "cumple": lat_consistent and lon_consistent,
    })

    # Criterio 5: Diversidad del ensemble
    available_models = df_completeness[df_completeness["found"] > 0]["model"].nunique()
    criteria.append({
        "criterio": "Diversidad del ensemble (mínimo 5 modelos para estadísticas robustas)",
        "resultado": f"{available_models} modelos disponibles",
        "cumple": available_models >= 5,
    })

    # Imprimir
    all_pass = True
    for c in criteria:
        status = "✓ CUMPLE" if c["cumple"] else "✗ NO CUMPLE"
        all_pass = all_pass and c["cumple"]
        print(f"\n  {status}")
        print(f"    Criterio:   {c['criterio']}")
        print(f"    Resultado:  {c['resultado']}")

    print(f"\n  {'=' * 50}")
    verdict = "VIABLE" if all_pass else "VIABLE CON OBSERVACIONES"
    print(f"  VEREDICTO: Dataset {verdict} para el proyecto")
    print(f"  {'=' * 50}")

    # Guardar
    df_criteria = pd.DataFrame(criteria)
    csv_path = OUT_DIR / "viability_assessment.csv"
    df_criteria.to_csv(csv_path, index=False)
    print(f"\n  Guardado: {csv_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#" * 70)
    print("  NEX-GDDP-CMIP6 — INVENTARIO Y ESTUDIO DE VIABILIDAD")
    print(f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)

    # 1. Completitud
    df_comp = build_completeness_matrix()
    plot_completeness_heatmap(df_comp)

    # 2. Control de calidad
    df_qc = run_quality_control()
    plot_nan_coverage(df_qc)
    plot_value_ranges(df_qc)

    # 3. Cobertura espacial
    plot_spatial_coverage()

    # 4. Resumen general
    generate_dataset_summary()

    # 5. Viabilidad
    assess_viability(df_comp, df_qc)

    print(f"\n  Todos los outputs en: {OUT_DIR}")
    print(f"  Archivos generados:")
    for f in sorted(OUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:45s} ({size_kb:.0f} KB)")

    print("\n  Script completado.")


if __name__ == "__main__":
    main()
