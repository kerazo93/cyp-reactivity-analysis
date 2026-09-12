"""Activity cliffs and matched molecular pairs for CYP3A4 reactivity.

An activity cliff is only interesting if the activity difference is real.  In a
screen with n=4 wells per compound, ranking pairs by raw |delta activity| mostly
surfaces pairs where one member happens to be noisy.  Every pair here therefore
carries a test of the difference itself, using the per-compound standard errors
from Part 1, FDR-controlled across all pairs considered.

SALI = |activity_i - activity_j| / (1 - Tanimoto_ij)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMMPA
from scipy import stats as st

from .stats import benjamini_hochberg


def similarity_pairs(fps: list, threshold: float = 0.70) -> list[tuple[int, int, float]]:
    """All index pairs with Tanimoto >= ``threshold``.

    Uses RDKit's bulk similarity so the O(n^2) sweep stays tractable for a
    library of this size (~1,200 compounds => ~750k comparisons).
    """
    pairs = []
    for i in range(len(fps) - 1):
        if fps[i] is None:
            continue
        rest = fps[i + 1:]
        valid = [(j + i + 1, f) for j, f in enumerate(rest) if f is not None]
        if not valid:
            continue
        idx, vec = zip(*valid)
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], list(vec))
        for j, s in zip(idx, sims):
            if s >= threshold:
                pairs.append((i, j, float(s)))
    return pairs


def find_cliffs(
    data: pd.DataFrame,
    fps: list,
    activity_col: str = "delta_adj",
    se_col: str = "se",
    smiles_col: str = "standardized_smiles",
    id_col: str = "ocnt_batch",
    sim_threshold: float = 0.70,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Similar pairs ranked by SALI, with a significance test on the difference.

    ``significant`` marks pairs whose activity difference survives FDR control --
    these are cliffs the data can actually support, as opposed to pairs separated
    by measurement noise.
    """
    act = data[activity_col].to_numpy(float)
    se = data[se_col].to_numpy(float)
    ids = data[id_col].to_numpy()
    smi = data[smiles_col].to_numpy()

    rows = []
    for i, j, sim in similarity_pairs(fps, sim_threshold):
        d = act[i] - act[j]
        sed = np.hypot(se[i], se[j])
        rows.append({
            "id_i": ids[i], "id_j": ids[j],
            "smiles_i": smi[i], "smiles_j": smi[j],
            "activity_i": act[i], "activity_j": act[j],
            "delta_activity": d, "abs_delta": abs(d),
            "se_diff": sed, "tanimoto": sim,
            # SALI is undefined when the fingerprints are identical; those pairs
            # are reported separately rather than given a meaningless huge score.
            "sali": (abs(d) / (1.0 - sim)) if sim < 0.999 else np.nan,
            "fingerprint_identical": sim >= 0.999,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    z = out["delta_activity"] / out["se_diff"]
    out["p_value"] = 2 * st.norm.sf(np.abs(z))
    out["q_value"] = benjamini_hochberg(out["p_value"].values)
    out["significant"] = out["q_value"] < alpha
    # Rank by SALI where defined; fingerprint-degenerate pairs sort by raw gap.
    return out.sort_values(["sali", "abs_delta"], ascending=False,
                           na_position="last").reset_index(drop=True)


def matched_pairs(smiles: pd.Series, ids: pd.Series,
                  min_core_atoms: int = 8,
                  max_sub_atoms: int = 13) -> pd.DataFrame:
    """Matched molecular pairs sharing a common core, via single-bond cuts.

    ``rdMMPA.FragmentMol`` with ``maxCuts=1`` returns ``('', 'fragA.fragB')`` and
    does *not* say which fragment is the core -- so the larger fragment is taken
    as the core and the smaller as the substituent.  Without that step, tiny
    fragments such as ``C[*:1]`` become "cores" shared by hundreds of unrelated
    molecules and the pair count explodes combinatorially.

    ``min_core_atoms`` and ``max_sub_atoms`` keep pairs to genuine single-point
    changes on a substantial shared scaffold, which is what makes an MMP
    interpretable.

    MMPs make an enrichment result actionable: "molecules containing X are
    depleted" becomes "replacing X with Y changed turnover in these pairs".
    """
    def _heavy(frag: str) -> int:
        m = Chem.MolFromSmiles(frag)
        return m.GetNumHeavyAtoms() if m is not None else 0

    frags: dict[str, dict[str, str]] = {}
    for cid, smi in zip(ids, smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        try:
            cuts = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False)
        except Exception:
            continue
        for _core, chains in cuts:
            parts = chains.split(".")
            if len(parts) != 2:
                continue
            a, b = parts
            na, nb = _heavy(a), _heavy(b)
            core, sub = (a, b) if na >= nb else (b, a)
            n_core, n_sub = max(na, nb), min(na, nb)
            if n_core < min_core_atoms or n_sub > max_sub_atoms:
                continue
            frags.setdefault(core, {}).setdefault(cid, sub)

    rows = []
    for core, members in frags.items():
        if len(members) < 2:
            continue
        items = list(members.items())
        for a in range(len(items) - 1):
            for b in range(a + 1, len(items)):
                (id_i, sub_i), (id_j, sub_j) = items[a], items[b]
                if sub_i == sub_j:
                    continue
                rows.append({"core": core, "id_i": id_i, "id_j": id_j,
                             "sub_i": sub_i, "sub_j": sub_j})
    return pd.DataFrame(rows).drop_duplicates(subset=["id_i", "id_j", "core"])


def annotate_pairs(pairs: pd.DataFrame, data: pd.DataFrame,
                   activity_col: str = "delta_adj", se_col: str = "se",
                   id_col: str = "ocnt_batch", alpha: float = 0.05) -> pd.DataFrame:
    """Attach activities and a difference test to a table of id pairs."""
    if pairs.empty:
        return pairs
    look = data.set_index(id_col)
    out = pairs.copy()
    for side in ("i", "j"):
        out[f"activity_{side}"] = out[f"id_{side}"].map(look[activity_col])
        out[f"se_{side}"] = out[f"id_{side}"].map(look[se_col])
    out = out.dropna(subset=["activity_i", "activity_j"])
    if out.empty:
        return out
    out["delta_activity"] = out["activity_i"] - out["activity_j"]
    out["abs_delta"] = out["delta_activity"].abs()
    out["se_diff"] = np.hypot(out["se_i"], out["se_j"])
    z = out["delta_activity"] / out["se_diff"]
    out["p_value"] = 2 * st.norm.sf(np.abs(z))
    out["q_value"] = benjamini_hochberg(out["p_value"].values)
    out["significant"] = out["q_value"] < alpha
    return out.sort_values("abs_delta", ascending=False).reset_index(drop=True)


def transformation_summary(mmp: pd.DataFrame, min_count: int = 3) -> pd.DataFrame:
    """Aggregate MMP transformations, to find substituent swaps that move turnover."""
    if mmp.empty:
        return mmp
    df = mmp.copy()
    # Orient each transformation consistently so A>>B and B>>A aggregate together.
    swap = df["sub_i"] > df["sub_j"]
    df.loc[swap, ["sub_i", "sub_j"]] = df.loc[swap, ["sub_j", "sub_i"]].values
    df.loc[swap, "delta_activity"] *= -1
    df["transformation"] = df["sub_i"] + ">>" + df["sub_j"]
    g = df.groupby("transformation").agg(
        n_pairs=("delta_activity", "size"),
        mean_delta=("delta_activity", "mean"),
        median_delta=("delta_activity", "median"),
        n_significant=("significant", "sum"),
    ).reset_index()
    g = g[g["n_pairs"] >= min_count]
    return g.sort_values("mean_delta").reset_index(drop=True)
