#!/usr/bin/env python
"""Figures for the write-up."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from octant_cyp import io, normalize  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 140, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
CALL_COLORS = {"substrate": "#c0392b", "non-substrate": "#2c7fb8",
               "inconclusive": "#b0b0b0"}


def fig_offset():
    """The systematic control-vs-treatment offset, and why CYP3A4 can't use it."""
    s = io.load_summary()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    grid = np.linspace(-8, 3, 1200)
    for ax, enz in zip(axes, ["CYP2J2", "CYP3A4"]):
        x = s.loc[s.enzyme == enz, "log2fc"].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(x, bins=80, color="#d9d9d9", density=True, label="compounds")
        ax.plot(grid, st.gaussian_kde(x, bw_method=0.2)(grid), color="#333", lw=1.2)
        mode = normalize.inactive_mode(x.values)
        sep = normalize.mode_separation(x.values)
        ax.axvline(0, color="#2c7fb8", ls="--", lw=1.2, label="no change")
        ax.axvline(mode, color="#c0392b", lw=1.4,
                   label=f"inactive mode {mode:+.2f} ({100*2**mode:.0f}%)")
        ok = "identifiable" if sep > normalize.MIN_MODE_SEPARATION else "NOT identifiable"
        ax.set_title(f"{enz}  —  separation {sep:,.0f} ({ok})")
        ax.set_xlabel("log2 fold-change (treatment / control)")
        ax.set_xlim(-8, 3)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("density")
    fig.suptitle("Unreactive compounds should sit at 0. In CYP2J2 they sit at +0.19.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_systematic_offset.png", bbox_inches="tight")
    plt.close(fig)


def fig_volcano():
    calls = pd.read_csv(RES / "reactivity_calls_all.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, enz in zip(axes, ["CYP3A4", "CYP2J2"]):
        g = calls[calls.enzyme == enz]
        for call, sub in g.groupby("call"):
            ax.scatter(sub["delta_adj"], -np.log10(sub["q_value"].clip(lower=1e-12)),
                       s=7, alpha=0.55, c=CALL_COLORS.get(call, "#999"),
                       label=f"{call} ({len(sub)})", edgecolors="none")
        bounded = g[g["effect_bounded"]]
        ax.scatter(bounded["delta_adj"], -np.log10(bounded["q_value"].clip(lower=1e-12)),
                   s=16, facecolors="none", edgecolors="k", lw=0.4,
                   label=f"fully censored ({len(bounded)})")
        ax.axvline(np.log10(0.8), color="k", ls=":", lw=1)
        ax.set_title(enz)
        ax.set_xlabel("log10 fold-change (adjusted)")
        ax.legend(fontsize=7, frameon=False, loc="lower left")
    axes[0].set_ylabel("-log10 q")
    fig.suptitle("Substrate calls. Ringed points are fully censored — "
                 "their effect is a bound, and a log pipeline would drop them.",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_volcano.png", bbox_inches="tight")
    plt.close(fig)


def fig_qc():
    w = io.load_wells()
    from octant_cyp import qc
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    cv = qc.control_cv(w)
    axes[0].hist(cv["cv"].dropna(), bins=60, color="#2c7fb8")
    axes[0].axvline(cv["cv"].median(), color="#c0392b", lw=1.4,
                    label=f"median {cv['cv'].median():.3f}")
    axes[0].set_xlabel("control CV per compound"); axes[0].set_ylabel("count")
    axes[0].set_title("Assay noise floor"); axes[0].legend(fontsize=7, frameon=False)

    pat = qc.censoring_pattern(w)
    counts = pat["pattern"].value_counts()
    axes[1].bar(counts.index, counts.values,
                color=["#b0b0b0", "#c0392b", "#e67e22"][:len(counts)])
    for i, v in enumerate(counts.values):
        axes[1].text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    axes[1].set_title("Non-detects per treatment group")
    axes[1].set_ylabel("compound x enzyme groups")

    lay = qc.add_plate_coordinates(w, qc.plate_geometry(w))
    for cond, c in [("control", "#2c7fb8"), ("treatment", "#c0392b")]:
        sub = lay[lay.condition == cond]
        axes[2].scatter(sub["col_idx"], sub["row_idx"], s=0.4, c=c, label=cond,
                        alpha=0.5, edgecolors="none")
    axes[2].invert_yaxis()
    axes[2].set_xlabel("plate column"); axes[2].set_ylabel("plate row")
    axes[2].set_title("Controls occupy rows A–D only")
    axes[2].legend(fontsize=7, frameon=False, markerscale=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_qc.png", bbox_inches="tight")
    plt.close(fig)


def fig_model():
    cal = pd.read_csv(RES / "part2_calibration.csv")
    cmp_ = pd.read_csv(RES / "part2_split_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    axes[0].plot(cal["mean_predicted"], cal["observed_frequency"], "o-",
                 color="#c0392b", ms=4)
    axes[0].set_xlabel("mean predicted probability")
    axes[0].set_ylabel("observed substrate frequency")
    axes[0].set_title("Calibration (scaffold-grouped CV)")
    axes[0].legend(fontsize=7, frameon=False)

    m = cmp_[cmp_.model != "baseline_prevalence"]
    x = np.arange(len(m)); wdt = 0.35
    axes[1].bar(x - wdt/2, m["roc_auc_random"], wdt, label="random split",
                color="#b0b0b0")
    axes[1].bar(x + wdt/2, m["roc_auc_scaffold"], wdt, label="scaffold split",
                color="#2c7fb8")
    axes[1].set_xticks(x); axes[1].set_xticklabels(m["model"], fontsize=7)
    axes[1].set_ylabel("ROC-AUC"); axes[1].set_ylim(0.5, 0.95)
    axes[1].set_title("Optimism from random splitting")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_model.png", bbox_inches="tight")
    plt.close(fig)


def fig_chemistry():
    enr = pd.read_csv(RES / "part2_substructure_enrichment.csv")
    assoc = pd.read_csv(RES / "part2_descriptor_association.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    sig = enr[enr.q_value < 0.05].sort_values("log2_odds")
    colors = ["#2c7fb8" if v < 0 else "#c0392b" for v in sig["log2_odds"]]
    axes[0].barh(sig["feature"], sig["log2_odds"], color=colors)
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel("log2 odds ratio (substrate)")
    axes[0].set_title("Substructures (q < 0.05)")
    axes[0].tick_params(labelsize=7)

    a = assoc.sort_values("rho").head(10)
    axes[1].barh(a["descriptor"], a["rho"], color="#2c7fb8")
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_xlabel("Spearman rho vs log10 fold-change")
    axes[1].set_title("More negative = more depletion")
    axes[1].tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_chemistry.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for fn in (fig_offset, fig_volcano, fig_qc, fig_model, fig_chemistry):
        fn()
        print("wrote", fn.__name__)
