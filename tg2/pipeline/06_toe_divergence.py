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
        snr = g["snr"].to_numpy() * np.sign(g["snr"].to_numpy()[-1])
        
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
                keep = years >= 2015
                years, diff_vals = years[keep], diff_vals[keep]
                
                # Primer año sostenido hasta 2100
                div_year = next((int(y) for i, y in enumerate(years) if np.all(diff_vals[i:] >= threshold)), None)
                div_year_strict = next((int(y) for i, y in enumerate(years) if np.all(diff_vals[i:] >= 2 * threshold)), None)
                
                rows.append({
                    "model": model,
                    "variable": variable,
                    "region": region,
                    "scenario_1": s1,
                    "scenario_2": s2,
                    "pair": f"{s1}_vs_{s2}",
                    "divergence_year": div_year,
                    "divergence_year_strict": div_year_strict
                })
    return pd.DataFrame(rows)


def summarize_divergence(div_df, col="divergence_year"):
    """Genera un resumen agrupado con mediana, min, max y formato listo de publicación."""
    def format_stats(s):
        valid = s.dropna()
        if len(valid) == 0:
            return "Sin divergencia"
        return f"{valid.median():.0f} [{valid.min():.0f}-{valid.max():.0f}]"

    summary = div_df.groupby(["pair", "variable", "region"])[col].agg(
        ensemble_median="median",
        ensemble_min="min",
        ensemble_max="max",
        n_models_diverged=lambda s: int(s.notna().sum()),
        formatted=format_stats
    ).reset_index()
    
    return summary


def plot_fan_charts_ssp1_vs_ssp5(forced, sigma, path):
    """Fan charts SSP1-2.6 vs SSP5-8.5: ensemble median and p10-p90 of the forced response, 2 x 4 panels."""
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8})
    styles = {"ssp126": ("SSP1-2.6", "#1f77b4"), "ssp585": ("SSP5-8.5", "#d62728")}
    rows = (("tas", "Temperature\nanomaly (°C)"), ("hurs", "Relative humidity\nanomaly (%)"))
    fig, axes = plt.subplots(2, len(SUBREGIONS), figsize=(6.5, 3.9), sharex=True, sharey="row")
    for i, (var, ylabel) in enumerate(rows):
        for j, reg in enumerate(SUBREGIONS):
            ax = axes[i, j]
            for ssp, (label, color) in styles.items():
                sub = forced[(forced.region == reg) & (forced.variable == var) & (forced.scenario == ssp)]
                ens = sub.groupby("year")["forced"].quantile([0.1, 0.5, 0.9]).unstack()
                ax.fill_between(ens.index, ens[0.1], ens[0.9], color=color, alpha=0.18, lw=0)
                ax.plot(ens.index, ens[0.5], color=color, lw=1.3, label=label)
            thr = 2 * sigma[(sigma.region == reg) & (sigma.variable == var)]["sigma_ref"].median()
            ax.axhline(thr if var == "tas" else -thr, color="black", ls="--", lw=0.7, label="2 $\\sigma_{ref}$")
            ax.axhline(0, color="gray", lw=0.4)
            ax.set_xlim(1950, 2100)
            ax.set_xticks([1960, 2020, 2080])
            ax.grid(alpha=0.3, lw=0.5)
            if i == 0:
                ax.set_title(reg, loc="left")
            if j == 0:
                ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.8, w_pad=0.5)
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
    div_strict = summarize_divergence(div_all, "divergence_year_strict")
    div_strict.to_csv(OUT / "divergence_ensemble_summary_strict.csv", index=False)
    
    # 3. Reporte en consola (se mantiene idéntico)
    print("## Time of Emergence (ToE SNR>=2) Ensemble Median [Min-Max] Year (SSP5-8.5)")
    toe_s5 = toe[toe.scenario == "ssp585"]
    print(toe_s5.groupby(["variable", "region"])["toe_snr2"].agg(
        lambda s: f"{s.median():.0f} [{s.min():.0f}-{s.max():.0f}] n={s.notna().sum()}" if s.notna().any() else "No emerge"
    ).unstack(0).to_string())
    
    print("\n## Scenario Divergence Year (SSP1-2.6 vs SSP5-8.5) Ensemble Median [Min-Max]")
    p1 = div_summary[div_summary.pair == "ssp126_vs_ssp585"]
    print(p1.pivot(index="region", columns="variable", values="formatted").to_string())
    print("\n## Same pair, strict threshold 2*sqrt(2)*sigma_ref")
    p1s = div_strict[div_strict.pair == "ssp126_vs_ssp585"]
    print(p1s.pivot(index="region", columns="variable", values="formatted").to_string())
    
    # 4. Generar Figura
    plot_fan_charts_ssp1_vs_ssp5(forced, sigma, FIGS / "fan_charts_toe.png")
    
    print(f"\nSaved: {OUT / 'toe_summary.csv'}, {OUT / 'divergence_all_pairs.csv'}, {OUT / 'divergence_ensemble_summary.csv'} and {FIGS / 'fan_charts_toe.png'}")


if __name__ == "__main__":
    main()