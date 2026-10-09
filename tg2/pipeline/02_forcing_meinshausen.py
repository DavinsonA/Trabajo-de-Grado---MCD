#!/usr/bin/env python3
"""Annual greenhouse-gas forcing covariates per scenario from Meinshausen et al. (2020).

Source: SUPPLEMENT_DataTables_Meinshausen_6May2020.xlsx (GMD 13, 3571-3605, supplement),
global-mean ("World") mole fractions; T1 = history, T4/T5/T6/T11 = SSP1-2.6/2-4.5/3-7.0/5-8.5.
Radiative forcing relative to 1750 with the simplified expressions of Myhre et al. (1998)
for CO2, CH4 and N2O, plus the two non-overlapping equivalence species
(CFC12-eq: ozone-depleting substances; HFC-134a-eq: remaining minor gases) times the AR5
radiative efficiencies of CFC-12 (0.32) and HFC-134a (0.16 W m-2 ppb-1).
CO2-eq = CO2(1750) * exp(F / 5.35). Greenhouse gases only: aerosols are not included.

Outputs (repo): tg2/data/forcing_ssp.parquet, tg2/figures/forcing_trajectories.png
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
XLSX = (Path(os.environ.get("NEXGDDP_ROOT", "/media/volume/jay2vol/nexgddp"))
        / "external" / "meinshausen2020" / "SUPPLEMENT_DataTables_Meinshausen_6May2020.xlsx")
SHEETS = {"historical": "T1", "ssp126": "T4", "ssp245": "T5", "ssp370": "T6", "ssp585": "T11"}
GASES = {"co2_ppm": "CO2", "ch4_ppb": "CH4", "n2o_ppb": "N2O", "cfc12eq_ppt": "CFC12-eq", "hfc134aeq_ppt": "HFC-134a-eq"}
RE_CFC12, RE_HFC134A = 0.32, 0.16
COLORS = {"historical": "#333333", "ssp126": "#1f77b4", "ssp245": "#daa520", "ssp370": "#ff7f0e", "ssp585": "#d62728"}
LABELS = {"historical": "Historical", "ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}


def read_sheet(df):
    gas, region = df.iloc[8].astype(str).str.strip(), df.iloc[10].astype(str).str.strip()
    cols = {k: next(c for c in df.columns if gas[c] == g and region[c] == "World") for k, g in GASES.items()}
    body = df.iloc[12:]
    out = pd.DataFrame({k: pd.to_numeric(body[c], errors="coerce") for k, c in cols.items()})
    out.insert(0, "year", pd.to_numeric(body[0], errors="coerce"))
    return out.dropna(subset=["year"]).astype({"year": int}).set_index("year")


def overlap(m, n):
    return 0.47 * np.log(1 + 2.01e-5 * (m * n) ** 0.75 + 5.31e-15 * m * (m * n) ** 1.52)


def ghg_forcing(df, ref):
    c0, m0, n0 = ref["co2_ppm"], ref["ch4_ppb"], ref["n2o_ppb"]
    m, n = df["ch4_ppb"], df["n2o_ppb"]
    f_co2 = 5.35 * np.log(df["co2_ppm"] / c0)
    f_ch4 = 0.036 * (np.sqrt(m) - np.sqrt(m0)) - (overlap(m, n0) - overlap(m0, n0))
    f_n2o = 0.12 * (np.sqrt(n) - np.sqrt(n0)) - (overlap(m0, n) - overlap(m0, n0))
    f_minor = 1e-3 * (RE_CFC12 * (df["cfc12eq_ppt"] - ref["cfc12eq_ppt"])
                      + RE_HFC134A * (df["hfc134aeq_ppt"] - ref["hfc134aeq_ppt"]))
    return f_co2 + f_ch4 + f_n2o + f_minor


def main():
    names = pd.ExcelFile(XLSX).sheet_names
    sheet_of = {s: next(n for n in names if n.startswith(code + " ")) for s, code in SHEETS.items()}
    raw = pd.read_excel(XLSX, sheet_name=list(sheet_of.values()), header=None)
    data = {s: read_sheet(raw[sheet_of[s]]) for s in SHEETS}
    hist = data["historical"]
    ref = hist.loc[1750]

    frames = []
    for scenario in SHEETS:
        series = hist.loc[1850:2014] if scenario == "historical" else pd.concat([hist.loc[1850:2014], data[scenario].loc[2015:2100]])
        f = ghg_forcing(series, ref)
        out = series[["co2_ppm"]].assign(
            co2eq_ppm=ref["co2_ppm"] * np.exp(f / 5.35),
            ghg_rf_wm2=f,
            ghg_rf_ma20=f.rolling(20).mean(),
            ghg_rf_cum=f.cumsum(),
        )
        keep = (1950, 2014) if scenario == "historical" else (2015, 2100)
        frames.append(out.loc[keep[0]:keep[1]].assign(scenario=scenario).reset_index())
    forcing = pd.concat(frames, ignore_index=True)[["year", "scenario", "co2eq_ppm", "co2_ppm", "ghg_rf_wm2", "ghg_rf_ma20", "ghg_rf_cum"]]

    (REPO / "tg2" / "data").mkdir(parents=True, exist_ok=True)
    (REPO / "tg2" / "figures").mkdir(parents=True, exist_ok=True)
    forcing.to_parquet(REPO / "tg2" / "data" / "forcing_ssp.parquet", index=False)

    pd.set_option("display.width", 160)
    print(f"Reference 1750: CO2 {ref['co2_ppm']:.2f} ppm | CH4 {ref['ch4_ppb']:.1f} ppb | N2O {ref['n2o_ppb']:.1f} ppb")
    print(f"Rows: {len(forcing)} | per scenario: {forcing.groupby('scenario').size().to_dict()}")
    h = forcing[forcing.scenario == "historical"].set_index("year")
    print("\n## Continuity 2013-2016 (historical -> each SSP)")
    for scenario in list(SHEETS)[1:]:
        s = forcing[forcing.scenario == scenario].set_index("year")
        joined = pd.concat([h.loc[2013:2014], s.loc[2015:2016]])[["co2_ppm", "co2eq_ppm", "ghg_rf_wm2"]]
        step = joined.diff().loc[2015]
        typical = pd.concat([h.loc[2005:2014], s.loc[2015:2025]])[["co2_ppm", "co2eq_ppm", "ghg_rf_wm2"]].diff().abs().median()
        flag = "OK" if (step.abs() <= 3 * typical).all() else "CHECK"
        print(f"{scenario}: 2014->2015 step co2 {step.co2_ppm:+.2f} ppm, co2eq {step.co2eq_ppm:+.2f} ppm, rf {step.ghg_rf_wm2:+.3f} W/m2 | typical |step| co2eq {typical.co2eq_ppm:.2f} -> {flag}")
    print("\n## Values in 2014 and 2100")
    print(forcing[forcing.year.isin([2014, 2100])].round(3).to_string(index=False))
    s126 = forcing[forcing.scenario == "ssp126"].set_index("year")
    print(f"\nSSP1-2.6 peak: CO2-eq {s126.co2eq_ppm.max():.1f} ppm in {s126.co2eq_ppm.idxmax()} | "
          f"CO2 {s126.co2_ppm.max():.1f} ppm in {s126.co2_ppm.idxmax()} | GHG RF {s126.ghg_rf_wm2.max():.2f} W/m2 in {s126.ghg_rf_wm2.idxmax()}")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    hist_end = forcing[(forcing.scenario == "historical") & (forcing.year == 2014)]
    for scenario in SHEETS:
        s = forcing[forcing.scenario == scenario]
        if scenario != "historical":
            s = pd.concat([hist_end, s])
        ax.plot(s.year, s.ghg_rf_wm2, color=COLORS[scenario], lw=2, label=LABELS[scenario])
    peak_year, peak = int(s126.ghg_rf_wm2.idxmax()), s126.ghg_rf_wm2.max()
    ax.plot(peak_year, peak, "o", ms=7, mfc="white", mec=COLORS["ssp126"], mew=2, zorder=5)
    ax.annotate(f"{peak_year}, {peak:.2f} W m$^{{-2}}$", (peak_year, peak), xytext=(0, -12), textcoords="offset points",
                ha="center", va="top", fontsize=9, color=COLORS["ssp126"])
    ax.axvline(2014.5, color="gray", ls="--", lw=0.8)
    ax.set_xlim(1950, 2100)
    ax.set_ylim(0, None)
    ax.set_xlabel("Year")
    ax.set_ylabel("GHG radiative forcing (W m$^{-2}$)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(REPO / "tg2" / "figures" / "forcing_trajectories.png", dpi=150)
    print(f"\nSaved: {REPO / 'tg2/data/forcing_ssp.parquet'} and {REPO / 'tg2/figures/forcing_trajectories.png'}")


if __name__ == "__main__":
    main()
