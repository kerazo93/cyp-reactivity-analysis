#!/usr/bin/env python
"""Part 3 -- select 1,000 purchasable compounds for follow-up CYP3A4 testing."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from octant_cyp import features, io, models, selection  # noqa: E402

RESULTS = ROOT / "results"
PROC = ROOT / "data" / "processed"
BUDGET = 1000
CHUNK = 50_000

# Bucket sizes. Rationale in selection.py's module docstring.
BUCKETS = {
    "model_validation": 200,
    "cliff_resolution": 250,
    "substructure_test": 250,
    "uncertainty_sampling": 200,
    "space_expansion": 100,
}


def featurise(smiles: pd.Series, desc_cols: list[str]) -> np.ndarray:
    mols = [features.parse_smiles(s) for s in smiles]
    fp = features.fingerprint_matrix(mols)
    d = features.descriptor_frame(mols)
    d = d[desc_cols].fillna(d[desc_cols].median())
    return np.hstack([fp, d.to_numpy(float)]), mols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-size", type=int, default=300_000,
                    help="candidates sampled from the in-stock pool (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidates", type=Path, default=PROC / "zinc_candidates.parquet")
    ap.add_argument("--out", type=Path, default=RESULTS / "followup_1000.csv")
    ap.add_argument("--budget", type=int, default=BUDGET)
    args = ap.parse_args()

    scale = args.budget / BUDGET
    buckets = {k: max(1, int(round(v * scale))) for k, v in BUCKETS.items()}

    desc_cols = list(features.DESCRIPTORS)


    # ---- training data ---------------------------------------------------
    calls = pd.read_csv(RESULTS / "reactivity_calls_all.csv")
    train = calls[calls["enzyme"] == "CYP3A4"].reset_index(drop=True)
    X_train, train_mols = featurise(train["standardized_smiles"], desc_cols)
    y_train = train["delta_adj"].to_numpy(float)
    train_fps = features.bit_fingerprints(train_mols)
    train_feats = features.build_feature_table(train["standardized_smiles"])
    train = pd.concat([train, train_feats], axis=1)

    folds = models.scaffold_folds(train["murcko_scaffold"])
    reg_metrics, _ = models.cross_validate_regression(X_train, y_train, folds)
    reg_metrics.to_csv(RESULTS / "part3_regressor_cv.csv", index=False)
    print("[model] scaffold-grouped CV for the ranking model:")
    print(reg_metrics.round(3).to_string(index=False))

    reg = models.fit_final_regressor(X_train, y_train)

    # ---- MS-detectability model -----------------------------------------
    fly = io.load_will_it_fly()
    X_fly, _ = featurise(fly["standardized_smiles"], desc_cols)
    y_fly = selection.detectability_labels(fly)
    det_metrics, _ = models.cross_validate(X_fly, y_fly, groups=None, n_splits=5)
    det_metrics.to_csv(RESULTS / "part3_detectability_cv.csv", index=False)
    print(f"\n[model] MS-detectability (n={len(fly):,}, "
          f"{100*y_fly.mean():.1f}% detectable at area>={selection.DETECTABILITY_FLOOR:.0f}):")
    print(det_metrics.round(3).to_string(index=False))
    det = selection.train_detectability_model(X_fly, y_fly)

    # ---- cliff references -----------------------------------------------
    cl = pd.read_csv(RESULTS / "part2_activity_cliffs.csv")
    sig = cl[cl["significant"] & (cl["abs_delta"] > 0.5)]
    cliff_ids = pd.unique(pd.concat([sig["id_i"], sig["id_j"]]))
    id_to_pos = {b: i for i, b in enumerate(train["ocnt_batch"])}
    cliff_fps = [train_fps[id_to_pos[b]] for b in cliff_ids if b in id_to_pos]
    print(f"\n[cliffs] {len(sig)} significant cliff pairs -> "
          f"{len(cliff_fps)} reference compounds")

    # implicated substructures from Part 2
    enr = pd.read_csv(RESULTS / "part2_substructure_enrichment.csv")
    implicated = enr[enr["q_value"] < 0.05]["feature"].tolist()
    smarts = {k: features.SOFT_SPOT_SMARTS[k] for k in implicated
              if k in features.SOFT_SPOT_SMARTS}
    print(f"[chem] implicated substructures to test: {implicated}")

    # ---- candidate pool --------------------------------------------------
    cand = pd.read_parquet(args.candidates)
    print(f"\n[pool] in-stock candidates after filtering: {len(cand):,}")
    if args.pool_size and len(cand) > args.pool_size:
        cand = cand.sample(args.pool_size, random_state=args.seed).reset_index(drop=True)
        print(f"[pool] sampled {len(cand):,} for scoring (seed={args.seed}); "
              "re-run with --pool-size 0 to score the full pool")

    # ---- chunked scoring -------------------------------------------------
    n = len(cand)
    pred = np.zeros(n); spread = np.zeros(n); detect = np.zeros(n)
    ad_sim = np.zeros(n); cliff_sim = np.zeros(n)
    sub_flags = np.zeros((n, len(smarts)), dtype=bool)
    keep_fps: list = [None] * n

    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        sl = slice(start, stop)
        Xc, mols = featurise(cand["smiles"].iloc[sl], desc_cols)
        pred[sl] = reg.predict(Xc)
        spread[sl] = models.forest_spread(reg, Xc)
        detect[sl] = det.predict_proba(Xc)[:, 1]
        fps = features.bit_fingerprints(mols)
        keep_fps[sl] = fps
        ad_sim[sl] = models.applicability_domain(fps, train_fps)
        if cliff_fps:
            cs = np.zeros(stop - start)
            for i, f in enumerate(fps):
                if f is not None:
                    cs[i] = max(DataStructs.BulkTanimotoSimilarity(f, cliff_fps))
            cliff_sim[sl] = cs
        if smarts:
            sub_flags[sl] = selection.substructure_pool(mols, smarts).to_numpy()
        print(f"   scored {stop:,}/{n:,}", flush=True)

    cand["pred_log10fc"] = pred
    cand["pred_uncertainty"] = spread
    cand["p_detectable"] = detect
    cand["max_sim_to_training"] = ad_sim
    cand["max_sim_to_cliff"] = cliff_sim
    for j, name in enumerate(smarts):
        cand[f"has_{name}"] = sub_flags[:, j]

    # ---- hard constraints ------------------------------------------------
    eligible = cand.index[
        (cand["p_detectable"] >= 0.5) &
        (cand["max_sim_to_training"] < 0.95)   # not already effectively tested
    ].to_numpy()
    print(f"\n[filter] eligible after detectability + novelty: {len(eligible):,} "
          f"of {len(cand):,}")

    picks: dict[str, list[int]] = {}
    taken: set[int] = set()

    def avail(mask: np.ndarray | None = None) -> np.ndarray:
        pool = np.array([i for i in eligible if i not in taken], dtype=int)
        if mask is None or pool.size == 0:
            return pool
        return pool[mask[pool]]

    # 1. calibration across the predicted range
    p = cand["pred_log10fc"].to_numpy()
    prob_like = (p - p.min()) / max(p.max() - p.min(), 1e-9)
    idx = selection.stratified_by_probability(prob_like, avail(), buckets["model_validation"],
                                              keep_fps)
    picks["model_validation"] = idx; taken.update(idx)

    # 2. cliff neighbourhoods
    near = cand["max_sim_to_cliff"].to_numpy() >= 0.55
    pool = avail(near)
    idx = selection.greedy_diverse(pool, keep_fps, buckets["cliff_resolution"],
                                   max_sim=0.75,
                                   priority=cand["max_sim_to_cliff"].to_numpy()[pool])
    picks["cliff_resolution"] = idx; taken.update(idx)

    # 3. substructure hypothesis tests -- split evenly over implicated groups
    per_group = max(1, buckets["substructure_test"] // max(len(smarts), 1))
    sub_idx: list[int] = []
    for name in smarts:
        col = cand[f"has_{name}"].to_numpy()
        pool = avail(col)
        if len(pool) == 0:
            continue
        # span the predicted range within each group so the test is not confounded
        pri = -np.abs(prob_like[pool] - 0.5)
        got = selection.greedy_diverse(pool, keep_fps, per_group, 0.75, pri)
        sub_idx.extend(got); taken.update(got)
    picks["substructure_test"] = sub_idx

    # 4. uncertainty sampling
    pool = avail()
    idx = selection.greedy_diverse(pool, keep_fps, buckets["uncertainty_sampling"],
                                   max_sim=0.75,
                                   priority=cand["pred_uncertainty"].to_numpy()[pool])
    picks["uncertainty_sampling"] = idx; taken.update(idx)

    # 5. chemical-space expansion -- deliberately outside the training domain
    far = cand["max_sim_to_training"].to_numpy() < 0.35
    pool = avail(far)
    idx = selection.greedy_diverse(pool, keep_fps, buckets["space_expansion"],
                                   max_sim=0.70,
                                   priority=-cand["max_sim_to_training"].to_numpy()[pool])
    picks["space_expansion"] = idx; taken.update(idx)

    # ---- assemble --------------------------------------------------------
    frames = []
    for bucket, idx in picks.items():
        if not idx:
            continue
        sub = cand.loc[idx].copy()
        sub["bucket"] = bucket
        frames.append(sub)
    out = pd.concat(frames).drop_duplicates(subset="zinc_id")

    if len(out) < args.budget:
        short = args.budget - len(out)
        pool = avail()
        extra = selection.greedy_diverse(pool, keep_fps, short, 0.75,
                                         cand["pred_uncertainty"].to_numpy()[pool])
        if extra:
            ex = cand.loc[extra].copy(); ex["bucket"] = "uncertainty_sampling"
            out = pd.concat([out, ex])
    out = out.head(args.budget)

    out["pct_remaining_pred"] = 100 * 10 ** out["pred_log10fc"]
    out["in_applicability_domain"] = out["max_sim_to_training"] >= 0.35
    out["vendor_lookup"] = "https://zinc20.docking.org/substances/" + out["zinc_id"].astype(str)

    cols = ["zinc_id", "smiles", "bucket", "pred_log10fc", "pct_remaining_pred",
            "pred_uncertainty", "p_detectable", "max_sim_to_training",
            "in_applicability_domain", "max_sim_to_cliff", "mw", "clogp",
            "tranche", "purchasability", "vendor_lookup"] + \
           [f"has_{k}" for k in smarts]
    out[cols].to_csv(args.out, index=False)

    print(f"\n[select] {len(out)} compounds -> {args.out}")
    print(out["bucket"].value_counts().to_string())
    print("\n[select] predicted % remaining by bucket:")
    print(out.groupby("bucket")["pct_remaining_pred"].describe()[
        ["count", "min", "50%", "max"]].round(1).to_string())
    print(f"\n[select] median nearest-neighbour similarity to training set: "
          f"{out['max_sim_to_training'].median():.3f}")
    print(f"[select] outside applicability domain (by design, expansion bucket): "
          f"{int((~out['in_applicability_domain']).sum())}")
    cand.drop(columns=["smiles"]).to_parquet(PROC / "zinc_scored.parquet", index=False)


if __name__ == "__main__":
    main()
