#!/usr/bin/env python3
"""Detección de Puntos de Cambio Estructural (PELT) y Acoplamiento T-H (TG II, B2).

Inputs : processed_tg2/forced_response.parquet
Outputs: 
  - processed_tg2/pelt_changepoints_all.csv
  - processed_tg2/pelt_changepoints_ensemble.csv
  - processed_tg2/th_coupling_summary.csv
  - tg2/figures/pelt_changepoints.png
  - tg2/figures/th_coupling_phase.png
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
import warnings

# Ocultar warnings de divisiones por cero en correlaciones constantes
warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("NEXGDDP_ROOT", "/media/volume/jay2vol/nexgddp"))
REPO = Path(__file__).resolve().parents[2] if "__file__" in locals() else Path(".")
OUT = ROOT / "processed_tg2"
FIGS = REPO / "tg2" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

SUBREGIONS = ["Northern Tropics", "Central Amazonia", "Subtropical Andes", "Southern Cone"]
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
VARIABLES = ["tas", "hurs"]
MODELS_COUNT = 8

def generate_ar1_surrogate(series):
    """Genera una serie sustituta AR(1) conservando la varianza y autocorrelación original."""
    n = len(series)
    if n < 2: return series
    # Autocorrelación lag-1
    rho = np.corrcoef(series[1:], series[:-1])[0, 1]
    std = np.std(series)
    
    surrogate = np.zeros(n)
    surrogate[0] = series[0]
    # Ruido blanco escalado por la varianza residual
    noise = np.random.normal(0, std * np.sqrt(max(0, 1 - rho**2)), n)
    
    for t in range(1, n):
        surrogate[t] = rho * surrogate[t-1] + noise[t]
    return surrogate

def detect_slope_changepoints(series, years, pen=5.0):
    """Aplica PELT sobre la primera derivada estandarizada para detectar cambios de pendiente."""
    diff_series = np.diff(series)
    std_diff = np.std(diff_series)
    
    if std_diff == 0:
        return []
        
    # Estandarizar para que la penalización sea invariante a la escala (tas vs hurs)
    diff_norm = (diff_series - np.mean(diff_series)) / std_diff
    algo = rpt.Pelt(model="l2", min_size=10).fit(diff_norm)
    
    try:
        result = algo.predict(pen=pen)
        bps_indices = [idx for idx in result if idx < len(diff_series)]
        return [years[idx] for idx in bps_indices]
    except Exception:
        return []

def extract_all_changepoints(df):
    """Extrae puntos de cambio para todos los modelos usando PELT y estima FPR."""
    rows = []
    fpr_records = []
    
    for (model, scenario, variable, region), g in df.groupby(["model", "scenario", "variable", "region"]):
        g = g.sort_values("year")
        years = g["year"].to_numpy()
        forced = g["forced"].to_numpy()
        
        # 1. Detección en la serie real
        bps = detect_slope_changepoints(forced, years)
        for bp in bps:
            rows.append({
                "model": model, "scenario": scenario, 
                "variable": variable, "region": region, 
                "changepoint_year": bp, "type": "real"
            })
            
        # 2. Control de Falsos Positivos (Surrogates)
        surrogate_bps = 0
        n_surrogates = 5
        for _ in range(n_surrogates):
            surr_series = generate_ar1_surrogate(forced)
            surr_bps = detect_slope_changepoints(surr_series, years)
            surrogate_bps += len(surr_bps)
            
        fpr_records.append({
            "scenario": scenario, "variable": variable, "region": region,
            "false_positives": surrogate_bps / n_surrogates
        })
        
    # Prevenir KeyError forzando la estructura de columnas incluso si la lista está vacía
    cols = ["model", "scenario", "variable", "region", "changepoint_year", "type"]
    return pd.DataFrame(rows, columns=cols), pd.DataFrame(fpr_records)

def apply_consensus_filter(bps_df, window=4, min_models=5):
    """Filtra los puntos de cambio requiriendo consenso multi-modelo (>= 5 de 8)."""
    consensus_rows = []
    
    for (scenario, variable, region), g in bps_df.groupby(["scenario", "variable", "region"]):
        years = g["changepoint_year"].tolist()
        
        # Evaluar densidad de quiebres por año
        year_range = np.arange(1950, 2101)
        counts = [sum(1 for y in years if abs(y - yr) <= window) for yr in year_range]
        
        # Encontrar picos locales que superen el umbral
        valid_bps = []
        for i in range(1, len(counts)-1):
            if counts[i] >= min_models:
                # Verificar si es un máximo local en la ventana
                window_slice = counts[max(0, i-window):min(len(counts), i+window+1)]
                if counts[i] == max(window_slice):
                    valid_bps.append(year_range[i])
        
        # Remover duplicados muy cercanos
        valid_bps = sorted(list(set(valid_bps)))
        final_bps = []
        for bp in valid_bps:
            if not final_bps or (bp - final_bps[-1]) > window:
                final_bps.append(bp)
                
        for bp in final_bps:
            consensus_rows.append({
                "scenario": scenario, "variable": variable, 
                "region": region, "consensus_year": bp
            })
            
    return pd.DataFrame(consensus_rows)

def compute_thermodynamic_coupling(df, window=20):
    """Calcula la correlación móvil (Acoplamiento T-H) a lo largo del tiempo."""
    # Pivotar para tener tas y hurs como columnas
    piv = df.pivot_table(index=["model", "scenario", "region", "year"], 
                         columns="variable", values="forced").reset_index()
    
    rows = []
    for (model, scenario, region), g in piv.groupby(["model", "scenario", "region"]):
        g = g.sort_values("year").set_index("year")
        # Correlación de Pearson móvil centrada
        rolling_corr = g["tas"].rolling(window=window, center=True).corr(g["hurs"])
        
        for year, corr in rolling_corr.dropna().items():
            rows.append({
                "model": model, "scenario": scenario, "region": region,
                "year": year, "th_correlation": corr, "tas": g.loc[year, "tas"], "hurs": g.loc[year, "hurs"]
            })
            
    return pd.DataFrame(rows)

def plot_structural_breaks(forced, consensus_df, path):
    """Generates Figure A: Forced response with PELT structural breaks in 2x2 grid."""
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    ssp = "ssp585"
    colors = {"tas": "#d62728", "hurs": "#1f77b4"}
    
    # 1. Calcular límites globales
    tas_mins, tas_maxs, hurs_mins, hurs_maxs = [], [], [], []
    for reg in SUBREGIONS:
        sub_tas = forced[(forced.region == reg) & (forced.variable == "tas") & (forced.scenario == ssp)]
        ens_tas = sub_tas.groupby("year")["forced"].agg(p10=lambda x: np.percentile(x, 10), p90=lambda x: np.percentile(x, 90))
        tas_mins.append(ens_tas["p10"].min())
        tas_maxs.append(ens_tas["p90"].max())
        
        sub_hurs = forced[(forced.region == reg) & (forced.variable == "hurs") & (forced.scenario == ssp)]
        ens_hurs = sub_hurs.groupby("year")["forced"].agg(p10=lambda x: np.percentile(x, 10), p90=lambda x: np.percentile(x, 90))
        hurs_mins.append(ens_hurs["p10"].min())
        hurs_maxs.append(ens_hurs["p90"].max())
        
    tas_min, tas_max = min(tas_mins), max(tas_maxs)
    hurs_min, hurs_max = min(hurs_mins), max(hurs_maxs)
    
    for i, reg in enumerate(SUBREGIONS):
        ax1 = axes[i]
        ax2 = ax1.twinx()
        
        for var, ax, color in zip(["tas", "hurs"], [ax1, ax2], [colors["tas"], colors["hurs"]]):
            sub = forced[(forced.region == reg) & (forced.variable == var) & (forced.scenario == ssp)]
            ens = sub.groupby("year")["forced"].agg(med="median", p10=lambda x: np.percentile(x, 10), p90=lambda x: np.percentile(x, 90))
            
            ax.plot(ens.index, ens["med"], color=color, label=var.upper(), lw=2)
            ax.fill_between(ens.index, ens["p10"], ens["p90"], color=color, alpha=0.15)
            
            if var == "tas":
                ax.set_ylabel("TAS Anomaly (°C)", color=color, fontweight="bold")
                ax.tick_params(axis='y', colors=color)
                ax.set_ylim(tas_min - 0.2, tas_max + 0.5)
            else:
                ax.set_ylabel("HURS Anomaly (%)", color=color, fontweight="bold")
                ax.tick_params(axis='y', colors=color)
                ax.set_ylim(hurs_min - 0.5, hurs_max + 0.5)
        
        # Marcar Puntos de Cambio con el COLOR de su variable
        bps = consensus_df[(consensus_df.region == reg) & (consensus_df.scenario == ssp)]
        bps = bps.sort_values(by="consensus_year")
        
        for j, row in bps.reset_index().iterrows():
            bp = row["consensus_year"]
            var = row["variable"]
            line_color = colors[var]
            
            # Línea vertical del color correspondiente (Rojo TAS o Azul HURS)
            ax1.axvline(bp, color=line_color, linestyle="--", lw=1.8, zorder=10)
            
            y_offset = (j % 3) * ((tas_max - tas_min) * 0.15)
            y_pos = ax1.get_ylim()[0] + 0.5 + y_offset
            
            # Etiqueta del año con el mismo color que la línea para fácil asociación
            ax1.text(bp + 2, y_pos, str(bp), fontsize=12, fontweight='bold', rotation=90, color=line_color,
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
        ax1.set_title(reg, fontweight="bold", fontsize=16)
        ax1.set_xlabel("Year", fontweight="bold")
        ax1.grid(alpha=0.3)
        ax1.set_xlim(1950, 2100)

    fig.tight_layout(h_pad=3.0, w_pad=2.0)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_phase_diagrams(coupling_df, consensus_df, path):
    """Generates Figure B: Phase Diagram TAS vs HURS in 2x2 grid."""
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13})
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()
    ssp = "ssp585"
    colors = {"tas": "#d62728", "hurs": "#1f77b4"}
    
    # === DICCIONARIO DE POSICIONES MANUALES ===
    # Formato -> ("Región", Año): (Desplazamiento en X, Desplazamiento en Y)
    # X positivo: Mueve a la derecha | X negativo: Mueve a la izquierda
    # Y positivo: Mueve hacia arriba | Y negativo: Mueve hacia abajo
    label_offsets = {
        # Northern Tropics
        ("Northern Tropics", 1981): (0, 15),
        ("Northern Tropics", 2016): (-30, -5),
        ("Northern Tropics", 2021): (35, -5),
        
        # Central Amazonia
        ("Central Amazonia", 2021): (20, 15),
        ("Central Amazonia", 2081): (20, 15),
        
        # Subtropical Andes
        ("Subtropical Andes", 1956): (-20, 15),
        ("Subtropical Andes", 1961): (30, 5),
        ("Subtropical Andes", 1971): (40, -10),
        ("Subtropical Andes", 2026): (0, -20),
        
        # Southern Cone
        ("Southern Cone", 1961): (-10, 15),
        ("Southern Cone", 1969): (5, -25),
        ("Southern Cone", 2031): (15, 15),
        ("Southern Cone", 2081): (-10, -20),
        ("Southern Cone", 2086): (20, 10)
    }
    
    # 1. Calcular límites globales basados estrictamente en la MEDIANA
    tas_mins, tas_maxs, hurs_mins, hurs_maxs = [], [], [], []
    for reg in SUBREGIONS:
        sub = coupling_df[(coupling_df.region == reg) & (coupling_df.scenario == ssp)]
        ens = sub.groupby("year").agg({"tas": "median", "hurs": "median"})
        tas_mins.append(ens["tas"].min())
        tas_maxs.append(ens["tas"].max())
        hurs_mins.append(ens["hurs"].min())
        hurs_maxs.append(ens["hurs"].max())
        
    tas_min, tas_max = min(tas_mins), max(tas_maxs)
    hurs_min, hurs_max = min(hurs_mins), max(hurs_maxs)
    
    for i, reg in enumerate(SUBREGIONS):
        ax = axes[i]
        sub = coupling_df[(coupling_df.region == reg) & (coupling_df.scenario == ssp)]
        ens = sub.groupby("year").agg({"tas": "median", "hurs": "median", "th_correlation": "median"})
        
        ax.set_xlim(tas_min - 0.5, tas_max + 0.5)
        ax.set_ylim(hurs_min - 0.5, hurs_max + 0.5)
        
        # Trayectoria de fase
        sc = ax.scatter(ens["tas"], ens["hurs"], c=ens.index, cmap="viridis", s=25, alpha=0.9)
        ax.plot(ens["tas"], ens["hurs"], color="gray", lw=1.2, alpha=0.6)
        
        # Marcar todos los puntos de cambio (TAS y HURS)
        bps_reg = consensus_df[(consensus_df.region == reg) & (consensus_df.scenario == ssp)]
        bps_reg = bps_reg.sort_values(by="consensus_year")
        
        for j, row in bps_reg.reset_index().iterrows():
            bp = row["consensus_year"]
            var = row["variable"]
            
            if bp in ens.index:
                # Estrellas rojas para Temp, Cuadrados azules para Humedad
                if var == "tas":
                    marker_style = {"color": colors["tas"], "marker": "*", "s": 180, "edgecolor": "black"}
                else:
                    marker_style = {"color": colors["hurs"], "marker": "s", "s": 70, "edgecolor": "black"}
                
                ax.scatter(ens.loc[bp, "tas"], ens.loc[bp, "hurs"], zorder=10, **marker_style)
                
                # Buscar las coordenadas en el diccionario. Si no existe, usar (15, 15) por defecto.
                x_offset, y_offset = label_offsets.get((reg, bp), (15, 15))
                
                # Etiqueta limpia: SOLO el año
                ax.annotate(str(bp), (ens.loc[bp, "tas"], ens.loc[bp, "hurs"]), 
                            xytext=(x_offset, y_offset), textcoords="offset points", 
                            fontsize=12, fontweight="bold", ha="center",
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        ax.set_title(reg, fontweight="bold", fontsize=16)
        ax.set_xlabel("Temperature Anomaly (°C)", fontweight="bold")
        ax.set_ylabel("Humidity Anomaly (%)", fontweight="bold")
        ax.grid(alpha=0.3)
        
    cbar = fig.colorbar(sc, ax=axes.tolist(), pad=0.02, aspect=40)
    cbar.set_label("Year", fontweight="bold", fontsize=14)
    
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    pd.set_option("display.width", 160)
    print("Cargando respuesta forzada...")
    forced = pd.read_parquet(OUT / "forced_response.parquet")
    
    # 1. Detección PELT y Falsos Positivos
    print("Aplicando algoritmo PELT (Pruned Exact Linear Time) y generando AR(1) surrogates...")
    bps_all, fpr_df = extract_all_changepoints(forced)
    bps_all.to_csv(OUT / "pelt_changepoints_all.csv", index=False)
    
    # 2. Filtro de Consenso (>=5 modelos)
    print("Aplicando filtro de consenso inter-modelo...")
    bps_consensus = apply_consensus_filter(bps_all)
    bps_consensus.to_csv(OUT / "pelt_changepoints_ensemble.csv", index=False)
    
    # 3. Acoplamiento Termodinámico T-H
    print("Calculando acoplamiento termodinámico y correlaciones móviles...")
    th_coupling = compute_thermodynamic_coupling(forced)
    th_coupling.to_csv(OUT / "th_coupling_summary.csv", index=False)
    
    # 4. Resumen en consola
    print("\n## Tasa de Falsos Positivos Estimada (Surrogates AR-1) por subregión")
    print(fpr_df.groupby(["region"])["false_positives"].mean().reset_index().to_string(index=False))
    
    print("\n## Años de Quiebre Estructural por Consenso (SSP5-8.5)")
    c_s5 = bps_consensus[bps_consensus.scenario == "ssp585"]
    print(c_s5.pivot_table(index="region", columns="variable", values="consensus_year", aggfunc=list).to_string())
    
    # 5. Gráficos
    print("\nGenerando Figuras de Fase y Quiebres Estructurales...")
    plot_structural_breaks(forced, bps_consensus, FIGS / "pelt_changepoints.png")
    plot_phase_diagrams(th_coupling, bps_consensus, FIGS / "th_coupling_phase.png")
    
    print(f"\n¡Completado! Archivos guardados en: {OUT.name}/ y {FIGS.name}/")

if __name__ == "__main__":
    main()