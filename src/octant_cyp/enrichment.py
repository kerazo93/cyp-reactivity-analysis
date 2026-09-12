"""Scaffold and substructure enrichment against CYP3A4 reactivity.

Every test here is two-sided Fisher's exact on a 2x2 of (feature present) x
(substrate), FDR-controlled across the family of tests.  Depletion is reported
as well as enrichment: a scaffold that is reliably *unreactive* is as useful for
design as one that is reliably turned over.

Compounds called ``inconclusive`` in Part 1 are excluded rather than folded into
the negatives.  Treating unmeasurable compounds as non-substrates would dilute
exactly the effects these tests are trying to find.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

from .stats import benjamini_hochberg


def _fisher(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Odds ratio (Haldane-corrected) and two-sided Fisher p for a 2x2."""
    table = [[a, b], [c, d]]
    _, p = st.fisher_exact(table)
    odds = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    return float(odds), float(p)


def binary_enrichment(features: pd.DataFrame, is_substrate: pd.Series,
                      min_count: int = 10) -> pd.DataFrame:
    """Test each boolean feature column for association with substrate status."""
    y = np.asarray(is_substrate, dtype=bool)
    rows = []
    for col in features.columns:
        x = np.asarray(features[col], dtype=bool)
        n_pos = int(x.sum())
        if n_pos < min_count or n_pos > len(x) - min_count:
            continue
        a = int((x & y).sum())        # feature present, substrate
        b = int((x & ~y).sum())       # feature present, not substrate
        c = int((~x & y).sum())
        d = int((~x & ~y).sum())
        odds, p = _fisher(a, b, c, d)
        rows.append({
            "feature": col, "n_with_feature": n_pos,
            "n_substrate_with": a, "n_substrate_without": c,
            "rate_with": a / max(a + b, 1), "rate_without": c / max(c + d, 1),
            "odds_ratio": odds, "log2_odds": np.log2(odds), "p_value": p,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value"] = benjamini_hochberg(out["p_value"].values)
    out["direction"] = np.where(out["odds_ratio"] > 1, "enriched", "depleted")
    return out.sort_values("p_value").reset_index(drop=True)


def scaffold_enrichment(scaffolds: pd.Series, is_substrate: pd.Series,
                        min_members: int = 5) -> pd.DataFrame:
    """Per-scaffold substrate rate versus the rest of the library."""
    df = pd.DataFrame({"scaffold": scaffolds.values,
                       "y": np.asarray(is_substrate, dtype=bool)})
    df = df[df["scaffold"].astype(str).str.len() > 0]
    rows = []
    total_pos = int(df["y"].sum())
    total = len(df)
    for scaf, g in df.groupby("scaffold"):
        n = len(g)
        if n < min_members:
            continue
        a = int(g["y"].sum()); b = n - a
        c = total_pos - a; d = (total - n) - c
        odds, p = _fisher(a, b, c, d)
        rows.append({
            "scaffold": scaf, "n_members": n, "n_substrate": a,
            "rate": a / n, "background_rate": c / max(c + d, 1),
            "odds_ratio": odds, "p_value": p,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value"] = benjamini_hochberg(out["p_value"].values)
    out["direction"] = np.where(out["odds_ratio"] > 1, "enriched", "depleted")
    return out.sort_values("p_value").reset_index(drop=True)


def continuous_association(descriptors: pd.DataFrame, activity: pd.Series,
                           method: str = "spearman") -> pd.DataFrame:
    """Rank correlation of each descriptor with the continuous log fold-change.

    Using the continuous effect rather than the binary call keeps the weak-but-real
    end of the range, which a threshold would discard.
    """
    y = np.asarray(activity, dtype=float)
    rows = []
    for col in descriptors.columns:
        x = np.asarray(descriptors[col], dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 30 or np.nanstd(x[ok]) == 0:
            continue
        if method == "spearman":
            r, p = st.spearmanr(x[ok], y[ok])
        else:
            r, p = st.pearsonr(x[ok], y[ok])
        rows.append({"descriptor": col, "n": int(ok.sum()), "rho": float(r),
                     "p_value": float(p)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value"] = benjamini_hochberg(out["p_value"].values)
    return out.sort_values("p_value").reset_index(drop=True)
