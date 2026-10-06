#!/usr/bin/env python3
"""Analysis-ready emulator table and structural hold-out folds (TG II, B4).

One row per model, scenario, subregion, year and month; historical rows appear once per model
(scenario = "historical") and are shared by all SSPs. Targets: tas_anom (degC) and hurs_anom (%),
monthly anomalies vs 1960-2014 (cos-lat weighted, hurs QC). Features: GHG forcing (current,
20-year mean, cumulative since 1850), model TCR (ECS optional), month as sin/cos, subregion one-hot.
Year is kept as an identifier only.

Folds (12): leave-one-scenario-out (test = 2015-2100 of the held-out SSP; train = everything else,
including the shared historical rows) and leave-one-model-out (test = every row of the held-out model).

Inputs : processed_tg2/monthly_anomalies.parquet, tg2/data/forcing_ssp.parquet, tg2/data/gcm_sensitivity.csv
Outputs: processed_tg2/emulator_table.parquet, tg2/data/emulator_folds.json
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("NEXGDDP_ROOT", "/media/volume/jay2vol/nexgddp"))
REPO = Path(__file__).resolve().parents[2]
OUT = ROOT / "processed_tg2"
SUBREGIONS = ["Northern Tropics", "Central Amazonia", "Subtropical Andes", "Southern Cone"]
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
KEY = ["model", "scenario", "region", "year", "month"]
ONE_HOT = {r: "region_" + r.lower().replace(" ", "_") for r in SUBREGIONS}
FEATURES = {
    "base": ["ghg_rf_wm2", "tcr", "month_sin", "month_cos", *ONE_HOT.values()],
    "integrated_ma20": ["ghg_rf_wm2", "ghg_rf_ma20", "tcr", "month_sin", "month_cos", *ONE_HOT.values()],
    "integrated_cum": ["ghg_rf_wm2", "ghg_rf_cum", "tcr", "month_sin", "month_cos", *ONE_HOT.values()],
}
TARGETS = ["tas_anom", "hurs_anom"]


def build_table():
    mon = pd.read_parquet(OUT / "monthly_anomalies.parquet")
    mon = mon[mon.region.isin(SUBREGIONS)]
    table = (mon.pivot_table(index=KEY, columns="variable", values="anomaly").reset_index()
             .rename(columns={"tas": "tas_anom", "hurs": "hurs_anom"}))
    table.columns.name = None
    forcing = pd.read_parquet(REPO / "tg2" / "data" / "forcing_ssp.parquet")
    sens = pd.read_csv(REPO / "tg2" / "data" / "gcm_sensitivity.csv")[["model", "tcr", "ecs", "in_constrained_subset"]]
    table = table.merge(forcing, on=["year", "scenario"], how="left", validate="many_to_one")
    table = table.merge(sens, on="model", how="left", validate="many_to_one")
    angle = 2 * np.pi * (table["month"] - 1) / 12
    table["month_sin"], table["month_cos"] = np.sin(angle), np.cos(angle)
    for region, col in ONE_HOT.items():
        table[col] = (table["region"] == region).astype("int8")
    return table.sort_values(KEY).reset_index(drop=True)


def make_folds(table):
    folds = [("loso", s, table["scenario"] == s) for s in SSPS]
    folds += [("lomo", m, table["model"] == m) for m in sorted(table["model"].unique())]
    return folds


def check_fold(table, kind, held_out, test):
    train = ~test
    col = "scenario" if kind == "loso" else "model"
    assert set(table.loc[test, col]) == {held_out}, f"{kind}/{held_out}: test contains other {col}s"
    assert held_out not in set(table.loc[train, col]), f"{kind}/{held_out}: held-out {col} leaks into train"
    if kind == "loso":
        assert table.loc[test, "year"].min() >= 2015, f"loso/{held_out}: test includes historical years"
    overlap = table.loc[train, KEY].merge(table.loc[test, KEY], on=KEY)
    assert overlap.empty, f"{kind}/{held_out}: {len(overlap)} identical keys in train and test"


def main():
    pd.set_option("display.width", 170)
    table = build_table()
    assert not table.duplicated(KEY).any(), "duplicated keys"
    needed = TARGETS + sorted({f for cols in FEATURES.values() for f in cols})
    missing = table[needed].isna().sum()
    assert missing.sum() == 0, f"missing values:\n{missing[missing > 0]}"
    table.to_parquet(OUT / "emulator_table.parquet", index=False)

    n_models, n_regions = table.model.nunique(), table.region.nunique()
    n_hist = table[table.scenario == "historical"].groupby(["model", "region"]).size().iloc[0]
    n_ssp = table[table.scenario == "ssp585"].groupby(["model", "region"]).size().iloc[0]
    print(f"Emulator table: {len(table):,} rows per target ({n_models} models x {n_regions} subregions x "
          f"({n_hist} historical + {len(SSPS)} x {n_ssp} SSP months)), {table.shape[1]} columns")
    print(f"Historical rows appear once per model ({(table.scenario == 'historical').sum():,}) and are shared by all SSPs")
    print(f"Constrained subset (TCR 1.4-2.2): {sorted(table.loc[table.in_constrained_subset, 'model'].unique())}")

    summary = []
    for kind, held_out, test in make_folds(table):
        check_fold(table, kind, held_out, test)
        summary.append({"fold": f"{kind}_{held_out}", "kind": kind, "held_out": held_out,
                        "n_train": int((~test).sum()), "n_test": int(test.sum()),
                        "test_years": f"{table.loc[test, 'year'].min()}-{table.loc[test, 'year'].max()}"})
    folds = pd.DataFrame(summary)
    print(f"\n## {len(folds)} folds, all leakage checks passed")
    print(folds.to_string(index=False))

    spec = {"targets": TARGETS, "feature_sets": FEATURES, "identifiers": KEY,
            "fold_rules": {"loso": "test: scenario == held_out (2015-2100); train: scenario != held_out",
                           "lomo": "test: model == held_out; train: model != held_out"},
            "folds": summary}
    (REPO / "tg2" / "data" / "emulator_folds.json").write_text(json.dumps(spec, indent=2))
    print(f"\nSaved: {OUT / 'emulator_table.parquet'} and {REPO / 'tg2/data/emulator_folds.json'}")
    print("\n## Feature ranges")
    print(table[["ghg_rf_wm2", "ghg_rf_ma20", "ghg_rf_cum", "co2eq_ppm", "tcr", "ecs", *TARGETS]].describe().loc[["min", "mean", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
