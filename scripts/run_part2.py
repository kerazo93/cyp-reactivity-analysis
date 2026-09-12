#!/usr/bin/env python
"""Part 2 -- chemical analysis of the CYP3A4 reactivity data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from octant_cyp import cliffs, enrichment, features, io, models  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
ENZYME = "CYP3A4"


def main() -> None:
    calls = pd.read_csv(RESULTS / "reactivity_calls_all.csv")
    df = calls[calls["enzyme"] == ENZYME].reset_index(drop=True)
    print(f"[part2] {ENZYME}: {len(df)} compounds "
          f"({df['call'].value_counts().to_dict()})")

    feats = features.build_feature_table(df["standardized_smiles"])
    df = pd.concat([df, feats], axis=1)
    mols = [features.parse_smiles(s) for s in df["standardized_smiles"]]
    fps = features.bit_fingerprints(mols)

    labelled = df[df["call"] != "inconclusive"].copy()
    labelled["is_substrate"] = labelled["call"] == "substrate"
    print(f"[part2] labelled for enrichment/modelling: {len(labelled)} "
          f"({labelled['is_substrate'].mean():.1%} substrate)")

    # ---- 1. soft-spot substructure enrichment ---------------------------
    spot_cols = [c for c in features.SOFT_SPOT_SMARTS if c in labelled.columns]
    spot = enrichment.binary_enrichment(labelled[spot_cols],
                                        labelled["is_substrate"], min_count=15)
    spot.to_csv(RESULTS / "part2_substructure_enrichment.csv", index=False)
    print("\n[enrichment] substructures with q<0.05:")
    sig = spot[spot["q_value"] < 0.05]
    if sig.empty:
        print("   (none)")
    else:
        print(sig[["feature", "n_with_feature", "rate_with", "rate_without",
                   "odds_ratio", "q_value", "direction"]].round(3).to_string(index=False))

    # ---- 2. scaffold enrichment -----------------------------------------
    scaf = enrichment.scaffold_enrichment(labelled["murcko_scaffold"],
                                          labelled["is_substrate"], min_members=5)
    scaf.to_csv(RESULTS / "part2_scaffold_enrichment.csv", index=False)
    n_testable = 0 if scaf.empty else len(scaf)
    print(f"\n[enrichment] scaffolds with >=5 members: {n_testable}; "
          f"q<0.05: {0 if scaf.empty else int((scaf['q_value'] < 0.05).sum())}")
    if not scaf.empty:
        print(scaf.head(5)[["scaffold", "n_members", "rate", "background_rate",
                            "odds_ratio", "q_value"]].round(3).to_string(index=False))

    # ---- 3. continuous descriptor trends --------------------------------
    desc_cols = list(features.DESCRIPTORS)
    assoc = enrichment.continuous_association(df[desc_cols], df["delta_adj"])
    assoc.to_csv(RESULTS / "part2_descriptor_association.csv", index=False)
    print("\n[descriptors] rank correlation with log10 fold-change "
          "(negative rho => higher values mean MORE depletion):")
    print(assoc.head(8).round(4).to_string(index=False))

    # ---- 4. activity cliffs ---------------------------------------------
    cl = cliffs.find_cliffs(df, fps, sim_threshold=0.70)
    cl.to_csv(RESULTS / "part2_activity_cliffs.csv", index=False)
    if cl.empty:
        print("\n[cliffs] no similar pairs above threshold")
    else:
        strong = cl[(cl["significant"]) & (cl["abs_delta"] > 0.5)]
        print(f"\n[cliffs] similar pairs (Tanimoto>=0.70): {len(cl)}; "
              f"statistically significant differences: {int(cl['significant'].sum())}; "
              f"significant AND >3-fold apart: {len(strong)}")
        print(strong.head(8)[["id_i", "id_j", "tanimoto", "activity_i",
                              "activity_j", "abs_delta", "sali", "q_value"]]
              .round(3).to_string(index=False))

    # ---- 5. matched molecular pairs -------------------------------------
    mmp = cliffs.matched_pairs(df["standardized_smiles"], df["ocnt_batch"])
    if not mmp.empty:
        mmp = cliffs.annotate_pairs(mmp, df)
        mmp.to_csv(RESULTS / "part2_matched_pairs.csv", index=False)
        trans = cliffs.transformation_summary(mmp, min_count=3)
        trans.to_csv(RESULTS / "part2_mmp_transformations.csv", index=False)
        print(f"\n[MMP] pairs={len(mmp)}  significant={int(mmp['significant'].sum())}  "
              f"recurring transformations (>=3 pairs)={len(trans)}")
        if not trans.empty:
            print(trans.head(5).round(3).to_string(index=False))

    # ---- 6. QSAR with honest validation ---------------------------------
    X_fp = features.fingerprint_matrix(
        [features.parse_smiles(s) for s in labelled["standardized_smiles"]])
    X_desc = labelled[desc_cols].fillna(labelled[desc_cols].median()).to_numpy(float)
    X = np.hstack([X_fp, X_desc])
    y = labelled["is_substrate"].to_numpy(int)

    cmp_ = models.compare_splits(X, y, labelled["murcko_scaffold"])
    cmp_.to_csv(RESULTS / "part2_split_comparison.csv", index=False)
    print("\n[model] random vs scaffold-grouped CV (optimism = random - scaffold):")
    print(cmp_.round(3).to_string(index=False))

    folds = models.scaffold_folds(labelled["murcko_scaffold"])
    metrics, oof = models.cross_validate(X, y, groups=folds)
    metrics.to_csv(RESULTS / "part2_model_metrics.csv", index=False)
    best = metrics.sort_values("pr_auc", ascending=False)["model"].iloc[0]
    print(f"\n[model] scaffold-CV metrics (best by PR-AUC: {best}):")
    print(metrics.round(3).to_string(index=False))

    cal = models.calibration_table(y, oof[best])
    cal.to_csv(RESULTS / "part2_calibration.csv", index=False)
    print("\n[model] calibration of", best)
    print(cal.round(3).to_string(index=False))

    labelled["oof_pred"] = oof[best]
    labelled[["ocnt_batch", "standardized_smiles", "call", "is_substrate",
              "oof_pred", "murcko_scaffold"]].to_csv(
        RESULTS / "part2_oof_predictions.csv", index=False)

    # ---- 7. substrate vs inhibitor --------------------------------------
    inh = io.load_inhibition()
    merged = df.merge(inh[["ocnt_batch", "CYP3A4_pIC50", "activity_status",
                           "qc_flag_primary"]], on="ocnt_batch", how="inner")
    merged.to_csv(RESULTS / "part2_substrate_vs_inhibitor.csv", index=False)
    print(f"\n[substrate x inhibitor] overlap: {len(merged)} compounds")
    ok = merged[merged["call"] != "inconclusive"]
    tab = pd.crosstab(ok["call"], ok["activity_status"])
    print(tab.to_string())
    both = ok.dropna(subset=["CYP3A4_pIC50"])
    if len(both) > 30:
        from scipy import stats as st
        r, p = st.spearmanr(both["CYP3A4_pIC50"], both["delta_adj"])
        print(f"   Spearman(pIC50, log10 FC) = {r:.3f} (p={p:.2g}, n={len(both)})")
        print("   potent inhibitors (pIC50>6) that are NOT substrates: "
              f"{int(((both['CYP3A4_pIC50'] > 6) & (both['call'] == 'non-substrate')).sum())}")

    # ---- 8. CYP3A4 vs CYP2J2 selectivity --------------------------------
    wide = calls.pivot_table(index="ocnt_batch", columns="enzyme",
                            values="delta_adj")
    wide = wide.dropna()
    sel = calls.pivot_table(index="ocnt_batch", columns="enzyme",
                           values="call", aggfunc="first").dropna()
    print(f"\n[selectivity] compounds with both enzymes: {len(wide)}")
    from scipy import stats as st
    r, p = st.spearmanr(wide["CYP3A4"], wide["CYP2J2"])
    print(f"   Spearman(CYP3A4, CYP2J2 log10 FC) = {r:.3f} (p={p:.2g})")
    print(pd.crosstab(sel["CYP3A4"], sel["CYP2J2"]).to_string())
    wide.join(sel, rsuffix="_call").to_csv(RESULTS / "part2_selectivity.csv")

    # ---- 9. ionisation artefact check -----------------------------------
    fly = io.load_will_it_fly()
    f2 = df.merge(fly, on="standardized_smiles", how="inner")
    print(f"\n[artefact] compounds with ionisation data: {len(f2)}")
    if len(f2) > 20:
        r, p = st.spearmanr(np.log10(f2["ammonium_fluoride_area"].clip(lower=1)),
                            f2["delta_adj"])
        print(f"   Spearman(log ionisation area, log10 FC) = {r:.3f} (p={p:.2g})")
    r2, p2 = st.spearmanr(df["mu_control"], df["delta_adj"])
    print(f"   Spearman(control signal, log10 FC) = {r2:.3f} (p={p2:.2g}) "
          "-- tests whether apparent depletion tracks raw signal strength")


if __name__ == "__main__":
    main()
