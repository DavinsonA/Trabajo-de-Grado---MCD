#!/usr/bin/env python3
"""OE4 figure: CRPS skill vs the linear baseline and 90 % interval coverage, per fold (TG II).

Input : processed_tg2/oe4/oe4_metrics.csv (from 09_emulators.py), tg2/data/gcm_sensitivity.csv
Output: tg2/figures/oe4_skill_coverage.png (main feature set integrated_ma20)
Folds are ordered: held-out scenarios by forcing, then held-out models by TCR (shown in brackets).
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(os.environ.get("NEXGDDP_ROOT", "/media/volume/jay2vol/nexgddp"))
REPO = Path(__file__).resolve().parents[2]
STYLE = {"deep_ensemble": ("Deep ensemble", "o", "#4e79a7"), "xgb_quantile": ("Quantile XGBoost", "s", "#f28e2b")}
SSP_LABELS = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}


def main():
    m = pd.read_csv(ROOT / "processed_tg2" / "oe4" / "oe4_metrics.csv").query("feature_set == 'integrated_ma20'")
    tcr = pd.read_csv(REPO / "tg2" / "data" / "gcm_sensitivity.csv").set_index("model")["tcr"]
    lin = m[m.emulator == "linear"].set_index(["fold", "target"])["crps"]
    m = m.assign(crpss=1 - m.crps / lin.reindex(pd.MultiIndex.from_frame(m[["fold", "target"]])).to_numpy())
    loso = [f"loso_{s}" for s in SSP_LABELS]
    lomo = [f"lomo_{x}" for x in tcr.sort_values(kind="stable").index]
    order = loso + lomo
    labels = [SSP_LABELS[f[5:]] for f in loso] + [f"{f[5:]} ({tcr[f[5:]]:.1f})" for f in lomo]
    ypos = {f: i for i, f in enumerate(order)}

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8})
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.2), sharey=True)
    for i, (target, name) in enumerate((("tas_anom", "Temperature"), ("hurs_anom", "Relative humidity"))):
        for j, (col, xlabel, ref) in enumerate((("crpss", "CRPS skill vs linear baseline", 0.0),
                                               ("coverage_90", "Coverage of the 90 % interval", 0.90))):
            ax = axes[i, j]
            for emu, (lab, marker, color) in STYLE.items():
                d = m[(m.target == target) & (m.emulator == emu)]
                ax.scatter(d[col], d.fold.map(ypos), marker=marker, s=22, color=color, label=lab, zorder=3)
            ax.axvline(ref, color="gray", lw=0.8, ls="--")
            ax.axhline(len(loso) - 0.5, color="black", lw=0.6)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(labels)
            ax.set_ylim(len(order) - 0.5, -0.5)
            ax.grid(axis="x", alpha=0.3, lw=0.5)
            ax.set_title(f"{name}: {'skill' if j == 0 else 'coverage'}", loc="left")
            if i == 1:
                ax.set_xlabel(xlabel)
    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.8, w_pad=0.8)
    out = REPO / "tg2" / "figures" / "oe4_skill_coverage.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
