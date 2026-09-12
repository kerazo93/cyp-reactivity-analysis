"""Part 3: choosing 1,000 purchasable compounds for follow-up.

A budget of 1,000 assays spent on the 1,000 highest-scoring predicted substrates
would be close to wasted.  The model already says they are substrates, the screen
already shows that most of this chemical space is turned over by CYP3A4
(68.6% called substrate), and a confirmatory result teaches nothing that changes
a decision.  The budget is therefore allocated across objectives that each
resolve a *stated uncertainty*:

===========================  =====  ======================================
Model validation / calibration  200  does the probability mean what it says?
Activity-cliff resolution       250  which side of a cliff does new chemistry fall on?
Substructure hypothesis tests   250  are the enriched/depleted groups causal?
Uncertainty sampling            200  where is the model least able to decide?
Chemical-space expansion        100  does the model transfer off its training domain?
===========================  =====  ======================================

Two hard constraints apply to every bucket.  Compounds must be predicted
**MS-detectable** -- a compound that does not ionise cannot yield a depletion
measurement at all, and the blog is explicit that this pre-filter blinds the
platform to part of chemical space -- and every selection is **diversity-capped**
so a single series cannot consume a bucket.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from sklearn.ensemble import RandomForestClassifier

#: A compound is treated as assay-ready if its ionisation peak area reaches the
#: weakest control well that actually supported a reactivity measurement in this
#: screen (2,824).  Derived from the data rather than assumed.
DETECTABILITY_FLOOR = 2824.0


def detectability_labels(fly: pd.DataFrame,
                         floor: float = DETECTABILITY_FLOOR) -> np.ndarray:
    """Binary assay-readiness label from the ionisation screen."""
    return (fly["ammonium_fluoride_area"].to_numpy(float) >= floor).astype(int)


def train_detectability_model(X: np.ndarray, y: np.ndarray,
                              seed: int = 0) -> RandomForestClassifier:
    """Predict whether a compound will give usable MS signal."""
    mdl = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, n_jobs=-1,
        random_state=seed, class_weight="balanced_subsample")
    mdl.fit(X, y)
    return mdl


def forest_uncertainty(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """Disagreement between trees -- an epistemic uncertainty proxy.

    Prediction entropy alone peaks at p=0.5 even when every tree agrees; tree
    variance instead marks regions where the training data does not determine
    the answer, which is what active learning should target.
    """
    per_tree = np.stack([t.predict_proba(X)[:, 1] for t in model.estimators_])
    return per_tree.std(axis=0)


def greedy_diverse(candidate_idx: np.ndarray, fps: list, n: int,
                   max_sim: float = 0.80,
                   priority: np.ndarray | None = None) -> list[int]:
    """Pick ``n`` candidates, rejecting any within ``max_sim`` of one already picked.

    Candidates are visited in ``priority`` order (descending), so the selection
    stays close to the bucket's objective while spreading across chemotypes.
    If the similarity constraint cannot be satisfied the shortfall is returned
    rather than back-filled with near-duplicates.
    """
    candidate_idx = np.asarray(candidate_idx, dtype=int)
    if candidate_idx.size == 0:
        return []
    order = (candidate_idx[np.argsort(-np.asarray(priority, float))]
             if priority is not None else candidate_idx)
    picked: list[int] = []
    picked_fps: list = []
    for i in order:
        f = fps[i]
        if f is None:
            continue
        if picked_fps and max(DataStructs.BulkTanimotoSimilarity(f, picked_fps)) > max_sim:
            continue
        picked.append(int(i))
        picked_fps.append(f)
        if len(picked) >= n:
            break
    return picked


def stratified_by_probability(prob: np.ndarray, pool: np.ndarray, n: int,
                              fps: list, n_bins: int = 10,
                              max_sim: float = 0.80) -> list[int]:
    """Spread picks evenly across predicted-probability bins.

    Calibration can only be checked where predictions actually exist, so this
    deliberately buys low- and mid-probability compounds too -- the opposite of
    a top-N selection.
    """
    pool = np.asarray(pool, dtype=int)
    if pool.size == 0:
        return []
    edges = np.linspace(0, 1, n_bins + 1)
    per_bin = max(1, n // n_bins)
    out: list[int] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = pool[(prob[pool] >= lo) & (prob[pool] < hi)]
        if len(in_bin) == 0:
            continue
        # Prefer the middle of each bin so picks represent the bin, not its edge.
        centre = (lo + hi) / 2
        pri = -np.abs(prob[in_bin] - centre)
        out.extend(greedy_diverse(in_bin, fps, per_bin, max_sim, pri))
    return out[:n]


def neighbours_of(reference_fps: list, pool: np.ndarray, fps: list,
                  min_sim: float = 0.55, top_k: int = 50) -> pd.DataFrame:
    """Vendor compounds structurally close to a set of reference molecules."""
    rows = []
    pool_fps = [fps[i] for i in pool]
    valid = [(i, f) for i, f in zip(pool, pool_fps) if f is not None]
    if not valid:
        return pd.DataFrame(columns=["idx", "ref", "similarity"])
    idxs, vecs = zip(*valid)
    for r, rf in enumerate(reference_fps):
        if rf is None:
            continue
        sims = np.array(DataStructs.BulkTanimotoSimilarity(rf, list(vecs)))
        keep = np.where(sims >= min_sim)[0]
        if len(keep) == 0:
            continue
        keep = keep[np.argsort(-sims[keep])][:top_k]
        for k in keep:
            rows.append({"idx": int(idxs[k]), "ref": r, "similarity": float(sims[k])})
    return pd.DataFrame(rows)


def substructure_pool(mols: list, smarts: dict[str, str]) -> pd.DataFrame:
    """Boolean matrix of which candidates carry each implicated substructure."""
    compiled = {k: Chem.MolFromSmarts(v) for k, v in smarts.items()}
    data = {}
    for name, patt in compiled.items():
        if patt is None:
            continue
        data[name] = np.array([m is not None and m.HasSubstructMatch(patt)
                               for m in mols])
    return pd.DataFrame(data)
