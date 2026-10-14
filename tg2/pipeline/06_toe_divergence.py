#!/usr/bin/env python3
"""Time of Emergence (ToE) and Exhaustive Scenario Divergence Window (TG II, B1).

Input : processed_tg2/forced_response.parquet
        processed_tg2/sigma_ref.csv
Outputs: 
  - processed_tg2/toe_summary.csv
  - processed_tg2/divergence_all_pairs.csv (Detalle completo por modelo y par)
  - processed_tg2/divergence_ensemble_summary.csv (Resumen de todos los pares)
  - tg2/figures/fan_charts_toe.png (Gráfico comparativo SSP1-2.6 vs SSP5-8.5)
"""
import os
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("NEXGDDP_ROOT", "/media/volume/jay2vol/nexgddp"))
REPO = Path(__file__).resolve().parents[2]
OUT = ROOT / "processed_tg2"
FIGS = REPO / "tg2" / "figures"

SUBREGIONS = ["Northern Tropics", "Central Amazonia", "Subtropical Andes", "Southern Cone"]
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
VARIABLES = ["tas", "hurs"]

# Genera exhaustivamente las 6 combinaciones posibles de pares entre los 4 escenarios
ALL_PAIRS = list(combinations(SSPS, 2))  


def compute_toe(forced, sigma):
    """Calcula el año de emergencia inicial (SNR>=1) y robusta (SNR>=2)."""
    df = forced.merge(sigma, on=["model", "variable", "region"])
    df["snr"] = df["forced"] / df["sigma_ref"]
    
    rows = []
    for (model, scenario, variable, region), g in df.groupby(["model", "scenario", "variable", "region"]):
        g = g.sort_values("year")
        years = g["year"].to_numpy()
        snr = g["snr"].to_numpy()
        
        # Encontrar primer año donde SNR >= threshold de forma sostenida hasta 2100
        toe_1 = next((int(y) for i, y in enumerate(years) if np.all(snr[i:] >= 1.0)), None)
        toe_2 = next((int(y) for i, y in enumerate(years) if np.all(snr[i:] >= 2.0)), None)
        
        rows.append({
            "model": model, "scenario": scenario, "variable": variable,
            "region": region, "toe_snr1": toe_1, "toe_snr2": toe_2
        })
    return pd.DataFrame(rows)


def compute_all_divergences(forced, sigma):
    """Calcula el año en que dos escenarios divergen significativamente para todos los pares."""
    df = forced.merge(sigma, on=["model", "variable", "region"])
    rows = []
    
    for (model, variable, region), g in df.groupby(["model", "variable", "region"]):
        sigma_val = g["sigma_ref"].iloc[0]
        threshold = np.sqrt(2) * sigma_val
        piv = g.pivot(index="year", columns="scenario", values="forced")
        
        for s1, s2 in ALL_PAIRS:
            if s1 in piv.columns and s2 in piv.columns:
                diff = (piv[s1] - piv[s2]).abs()
                years = diff.index.to_numpy()
                diff_vals = diff.to_numpy()
                
                # Primer año sostenido hasta 2100
                div_year = next((int(y) for i, y in enumerate(years) if np.all(diff_vals[i:] >= threshold)), None)
                
                rows.append({
                    "model": model,
                    "variable": variable,
                    "region": region,
                    "scenario_1": s1,
                    "scenario_2": s2,
                    "pair": f"{s1}_vs_{s2}",
                    "divergence_year": div_year
                })
    return pd.DataFrame(rows)


def summarize_divergence(div_df):
    """Genera un resumen agrupado con mediana, min, max y formato listo de publicación."""
    def format_stats(s):
        valid = s.dropna()
        if len(valid) == 0:
            return "Sin divergencia"
        return f"{valid.median():.0f} [{valid.min():.0f}-{valid.max():.0f}]"

    summary = div_df.groupby(["pair", "variable", "region"])["divergence_year"].agg(
        ensemble_median="median",
        ensemble_min="min",
        ensemble_max="max",
        n_models_diverged=lambda s: int(s.notna().sum()),
        formatted=format_stats
    ).reset_index()
    
    return summary


def plot_fan_charts_ssp1_vs_ssp5(forced, sigma, path):
    """Genera Fan Charts comparando SSP1-2.6 vs SSP5-8.5 con escala unificada por variable."""
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})
    
    # sharex=True unifica el eje X (Años 1950-2100)
    # sharey="col" unifica el eje Y de la col 1 (tas) y el eje Y de la col 2 (hurs)
    fig, axes = plt.subplots(len(SUBREGIONS), len(VARIABLES), figsize=(8, 10), sharex=True, sharey="col")
    
    selected_ssps = ["ssp126", "ssp585"]
    colors = {"ssp126": "#1f77b4", "ssp585": "#d62728"}
    
    # Calcular límites Y globales por variable para asegurar simetría y limpieza
    tas_data = forced[forced.variable == "tas"]["forced"]
    hurs_data = forced[forced.variable == "hurs"]["forced"]
    
    y_limits = {
        "tas": (min(0, tas_data.min() - 0.5), tas_data.max() + 0.5),      # Ej: -0.5 °C a +6.0 °C
        "hurs": (hurs_data.min() - 1.0, max(0, hurs_data.max() + 1.0))   # Ej: -12.0 % a +4.0 %
    }
    
    for i, reg in enumerate(SUBREGIONS):
        for j, var in enumerate(VARIABLES):
            ax = axes[i, j]
            for ssp in selected_ssps:
                sub = forced[(forced.region == reg) & (forced.variable == var) & (forced.scenario == ssp)]
                ens = sub.groupby("year")["forced"].agg(
                    med="median",
                    p10=lambda x: np.percentile(x, 10),
                    p90=lambda x: np.percentile(x, 90)
                )
                ax.plot(ens.index, ens["med"], color=colors[ssp], label=ssp.upper() if (i == 0 and j == 0) else "")
                ax.fill_between(ens.index, ens["p10"], ens["p90"], color=colors[ssp], alpha=0.15)
            
            # Línea de referencia 2*sigma_ref (mediana entre modelos)
            sig_med = sigma[(sigma.region == reg) & (sigma.variable == var)]["sigma_ref"].median()
            ax.axhline(2 * sig_med, color="black", ls="--", lw=0.8, alpha=0.7, label="$2\\sigma_{ref}$" if (i == 0 and j == 0) else "")
            if var == "hurs":
                ax.axhline(-2 * sig_med, color="black", ls="--", lw=0.8, alpha=0.7)
                
            # Fijar explícitamente el rango uniforme del eje Y
            ax.set_ylim(y_limits[var])
            ax.set_title(f"{reg} - {var.upper()}")
            ax.grid(alpha=0.3)
            
            # Etiquetas limpias solo en los bordes
            if j == 0:
                ax.set_ylabel("Anomalía (°C)")
            elif j == 1:
                ax.set_ylabel("Anomalía (%)")
    
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    pd.set_option("display.width", 160)
    forced = pd.read_parquet(OUT / "forced_response.parquet")
    sigma = pd.read_csv(OUT / "sigma_ref.csv")
    
    # 1. Calcular ToE
    toe = compute_toe(forced, sigma)
    toe.to_csv(OUT / "toe_summary.csv", index=False)
    
    # 2. Calcular Divergencia para TODOS los pares y guardar archivos
    div_all = compute_all_divergences(forced, sigma)
    div_all.to_csv(OUT / "divergence_all_pairs.csv", index=False)
    
    div_summary = summarize_divergence(div_all)
    div_summary.to_csv(OUT / "divergence_ensemble_summary.csv", index=False)
    
    # 3. Reporte en consola (se mantiene idéntico)
    print("## Time of Emergence (ToE SNR>=2) Ensemble Median [Min-Max] Year (SSP5-8.5)")
    toe_s5 = toe[toe.scenario == "ssp585"]
    print(toe_s5.groupby(["variable", "region"])["toe_snr2"].agg(
        lambda s: f"{s.median():.0f} [{s.min():.0f}-{s.max():.0f}]" if s.notna().any() else "No emerge"
    ).unstack(0).to_string())
    
    print("\n## Scenario Divergence Year (SSP1-2.6 vs SSP5-8.5) Ensemble Median [Min-Max]")
    p1 = div_summary[div_summary.pair == "ssp126_vs_ssp585"]
    print(p1.pivot(index="region", columns="variable", values="formatted").to_string())
    
    # 4. Generar Figura
    plot_fan_charts_ssp1_vs_ssp5(forced, sigma, FIGS / "fan_charts_toe.png")
    
    print(f"\nSaved: {OUT / 'toe_summary.csv'}, {OUT / 'divergence_all_pairs.csv'}, {OUT / 'divergence_ensemble_summary.csv'} and {FIGS / 'fan_charts_toe.png'}")


if __name__ == "__main__":
    main()