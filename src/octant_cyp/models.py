"""QSAR modelling of CYP3A4 reactivity, with honest validation.

Two choices here are load-bearing.

**Scaffold splitting.**  A random split puts analogues of the same series on both
sides of the fold boundary.  The model then scores well by recognising series it
has already seen, which is not the task -- Part 3 applies it to vendor compounds
that are by construction *not* in the training set.  Scaffold-grouped CV measures
the thing we actually need.  :func:`compare_splits` quantifies the gap.

**Labels.**  Only compounds Part 1 could classify are used.  The ``inconclusive``
class is dropped rather than merged into the negatives: those compounds were not
measured well enough to say, and teaching a model that they are unreactive would
poison the negatives with exactly the borderline cases the model most needs to
get right.

An applicability-domain measure (nearest-neighbour Tanimoto to the training set)
accompanies every prediction, because Part 3 scores compounds that may sit far
outside the chemistry seen here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import DataStructs
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def scaffold_folds(scaffolds: pd.Series, n_splits: int = 5,
                   seed: int = 0) -> np.ndarray:
    """Assign fold ids so that a scaffold never spans two folds.

    Scaffolds are distributed largest-first into the currently smallest fold,
    which keeps folds balanced despite a very skewed class-size distribution
    (this library has 871 scaffolds over 1,223 compounds, 693 of them singletons).
    """
    counts = scaffolds.value_counts()
    sizes = np.zeros(n_splits, dtype=int)
    assign: dict[str, int] = {}
    for scaf, n in counts.items():
        k = int(np.argmin(sizes))
        assign[scaf] = k
        sizes[k] += n
    return scaffolds.map(assign).to_numpy()


def _models(seed: int = 0) -> dict:
    return {
        "baseline_prevalence": DummyClassifier(strategy="prior"),
        "logistic_descriptors": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, C=1.0)),
        "random_forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, n_jobs=-1,
            random_state=seed, class_weight="balanced_subsample"),
    }


def cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None,
                   n_splits: int = 5, seed: int = 0,
                   models: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Grouped (or plain) CV returning per-model metrics and out-of-fold predictions."""
    models = models or _models(seed)
    if groups is not None:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = list(splitter.split(X, y, groups))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = list(splitter.split(X, y))

    oof = {name: np.full(len(y), np.nan) for name in models}
    for train_idx, test_idx in split_iter:
        for name, proto in models.items():
            from sklearn.base import clone
            mdl = clone(proto)
            mdl.fit(X[train_idx], y[train_idx])
            oof[name][test_idx] = mdl.predict_proba(X[test_idx])[:, 1]

    rows = []
    for name, pred in oof.items():
        ok = np.isfinite(pred)
        rows.append({
            "model": name,
            "roc_auc": roc_auc_score(y[ok], pred[ok]),
            "pr_auc": average_precision_score(y[ok], pred[ok]),
            "brier": brier_score_loss(y[ok], pred[ok]),
            "prevalence": float(y.mean()),
            "n": int(ok.sum()),
        })
    return pd.DataFrame(rows), oof


def compare_splits(X: np.ndarray, y: np.ndarray, scaffolds: pd.Series,
                   n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    """Same models under random vs scaffold-grouped CV.

    The difference is the size of the optimism a random split would have bought.
    """
    rand, _ = cross_validate(X, y, groups=None, n_splits=n_splits, seed=seed)
    scaf, _ = cross_validate(X, y, groups=scaffold_folds(scaffolds, n_splits, seed),
                             n_splits=n_splits, seed=seed)
    rand["split"] = "random"
    scaf["split"] = "scaffold"
    out = pd.concat([rand, scaf], ignore_index=True)
    wide = out.pivot(index="model", columns="split",
                     values=["roc_auc", "pr_auc"]).reset_index()
    wide.columns = ["model", "roc_auc_random", "roc_auc_scaffold",
                    "pr_auc_random", "pr_auc_scaffold"]
    wide["roc_auc_optimism"] = wide["roc_auc_random"] - wide["roc_auc_scaffold"]
    wide["pr_auc_optimism"] = wide["pr_auc_random"] - wide["pr_auc_scaffold"]
    return wide


def calibration_table(y: np.ndarray, pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Reliability curve: predicted probability versus observed frequency."""
    ok = np.isfinite(pred)
    frac_pos, mean_pred = calibration_curve(y[ok], pred[ok], n_bins=n_bins,
                                            strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "observed_frequency": frac_pos})


def applicability_domain(query_fps: list, train_fps: list) -> np.ndarray:
    """Maximum Tanimoto similarity of each query to the training set.

    A prediction for a compound with low nearest-neighbour similarity is an
    extrapolation, and Part 3 treats it as such.
    """
    train = [f for f in train_fps if f is not None]
    out = np.zeros(len(query_fps))
    for i, f in enumerate(query_fps):
        if f is None or not train:
            out[i] = np.nan
            continue
        out[i] = max(DataStructs.BulkTanimotoSimilarity(f, train))
    return out
