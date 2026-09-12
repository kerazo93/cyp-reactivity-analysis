#!/usr/bin/env python
"""Part 1 -- identify CYP3A4 and CYP2J2 substrates with statistical rationale."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from octant_cyp import io, normalize, qc, substrates  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def main() -> None:
    wells = io.load_wells()
    lod = io.estimate_lod(wells)
    print(f"[qc] wells={len(wells)}  LOD={lod}")

    # ---- QC -------------------------------------------------------------
    qc.mass_accuracy_summary(wells).to_csv(RESULTS / "qc_mass_accuracy.csv", index=False)
    cv = qc.control_cv(wells)
    cv.to_csv(RESULTS / "qc_control_cv.csv", index=False)
    print(f"[qc] control CV median={cv['cv'].median():.4f} IQR="
          f"{cv['cv'].quantile(.25):.4f}-{cv['cv'].quantile(.75):.4f}")
    pattern = qc.censoring_pattern(wells)
    pattern.to_csv(RESULTS / "qc_censoring_pattern.csv", index=False)
    print("[qc] censoring:", pattern["pattern"].value_counts().to_dict())
    layout = qc.condition_layout(wells)
    layout.to_csv(RESULTS / "qc_condition_layout.csv", index=False)
    print("[qc] plate geometry:", qc.plate_geometry(wells),
          "| control rows", layout.loc[layout.condition == "control", "row_min"].item(),
          "-", layout.loc[layout.condition == "control", "row_max"].item(),
          "| treatment rows", layout.loc[layout.condition == "treatment", "row_min"].item(),
          "-", layout.loc[layout.condition == "treatment", "row_max"].item(),
          "(NOT position-matched)")
    pd.concat([qc.edge_vs_interior(wells, c).assign(condition=c)
               for c in ("control", "treatment")]).to_csv(
        RESULTS / "qc_edge_effects.csv", index=False)

    # ---- systematic offset ---------------------------------------------
    summary = io.load_summary()
    plate_map = wells[["ocnt_batch", "enzyme", "plate"]].drop_duplicates()
    offsets = normalize.estimate_plate_offsets(summary, plate_map)
    offsets.to_csv(RESULTS / "qc_plate_offsets.csv", index=False)
    print("\n[offset] inactive-mode offset per plate:")
    print(offsets[["enzyme", "plate", "offset_pct_of_control",
                   "mode_separation", "identifiable"]].round(2).to_string(index=False))

    usable = offsets[offsets["identifiable"]]
    offset_map = dict(zip(usable["plate"], usable["offset_log2"]))
    print(f"[offset] applying correction on plates: {sorted(offset_map)}")

    # ---- contrasts ------------------------------------------------------
    contrasts = substrates.fit_all_contrasts(wells, lod=lod)
    print(f"\n[fit] contrasts={len(contrasts)}  "
          f"non-converged={(~contrasts['converged']).sum()}")

    called = substrates.call_substrates(contrasts, offsets=offset_map)
    smiles = io.compound_table(wells)
    called = called.merge(smiles, on="ocnt_batch", how="left")

    # ---- validation -----------------------------------------------------
    print("\n[eBayes] prior parameters:")
    for enzyme, g in called.groupby("enzyme"):
        print(f"   {enzyme}: s2_prior={g['s2_prior'].iloc[0]:.5f} "
              f"df_prior={g['df_prior'].iloc[0]:.2f} "
              f"(residual df per compound = {g['df'].median():.0f})")
        mde = substrates.minimum_detectable_effect(
            g["s2_prior"].iloc[0], g["v"].median(), g["df_total"].median())
        print(f"      minimum detectable effect @95% power: "
              f"{100*mde:.1f}% remaining ({100*(1-mde):.1f}% depletion)")

    print("\n[validation] blog anchor -- uncorrected <50% remaining:")
    anchor = (summary.assign(hit=summary["pct_remaining"] < 50)
                     .groupby("enzyme")["hit"].mean() * 100)
    print(anchor.round(1).to_string(), "  (blog reports ~61% / ~13%)")

    null = substrates.permutation_null(wells, lod=lod, n_rep=2, seed=0)
    null.to_csv(RESULTS / "validation_permutation_null.csv", index=False)
    from scipy import stats as st
    ks = st.kstest(null["p_value"].dropna(), "uniform")
    print(f"\n[validation] permutation null: n={len(null)} "
          f"KS vs uniform p={ks.pvalue:.3f}, "
          f"frac p<0.05 = {(null['p_value'] < 0.05).mean():.4f} (expect ~0.05), "
          f"discoveries at q<0.05 = {(null['q_value'] < 0.05).sum()}")

    # ---- output ---------------------------------------------------------
    cols = ["ocnt_batch", "standardized_smiles", "enzyme", "call",
            "pct_remaining_est", "pct_remaining_ci_low", "pct_remaining_ci_high",
            "delta_adj", "se", "t_mod", "p_value", "q_value",
            "p_substrate", "q_substrate", "p_nonsubstrate", "q_nonsubstrate",
            "censoring", "effect_bounded", "n_control", "n_treatment",
            "n_censored", "offset_log2", "plate", "converged"]
    print("\n[calls]")
    for enzyme, g in called.groupby("enzyme"):
        vc = g["call"].value_counts()
        print(f"   {enzyme}: " + ", ".join(f"{k}={v} ({100*v/len(g):.1f}%)"
                                           for k, v in vc.items()))
        path = RESULTS / f"substrates_{enzyme.lower()}.csv"
        (g[cols].sort_values("delta_adj").to_csv(path, index=False))
        print(f"      -> {path.relative_to(ROOT)}")

    called.to_csv(RESULTS / "reactivity_calls_all.csv", index=False)

    sens = substrates.threshold_sensitivity(contrasts, offset_map)
    sens.to_csv(RESULTS / "validation_threshold_sensitivity.csv", index=False)
    print("\n[sensitivity] effect threshold x normalisation:")
    print(sens.to_string(index=False))


if __name__ == "__main__":
    main()
