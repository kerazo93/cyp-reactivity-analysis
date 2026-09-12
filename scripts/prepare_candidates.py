#!/usr/bin/env python
"""Filter the ZINC in-stock pool down to an assay-relevant candidate set.

Tranche files are processed in parallel (one worker per file) so the full
2.7M-compound pool never sits in memory at once and the PAINS/Brenk sweep --
which dominates the runtime -- spreads across cores.

Filters, cheapest first so the expensive one sees the fewest molecules:

1. valid, single-component structure
2. property window matched to the *assayed* library, not generic drug-likeness
3. PAINS removal (assay-interference only)

Deliberately NOT applied: the Brenk "unwanted functionality" catalogue. Brenk is
a lead-discovery filter and rejects anilines, nitroaromatics, halides and
Michael acceptors -- which is precisely the chemistry Part 2 found to be
associated with CYP3A4 turnover (anilines are the strongest depleted
substructure). Screening it out would remove the compounds most able to test our
own conclusions; on this pool it also rejected ~99% of candidates. PAINS is kept
because it targets compounds that confound *assays*, which is a different and
legitimate concern here.

Output: data/processed/zinc_candidates.parquet
"""
from __future__ import annotations

import glob
import multiprocessing as mp
import os
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "external" / "zinc"
DEST = ROOT / "data" / "processed"

# Window bracketing the assayed library (MW IQR 365-465, cLogP IQR 2.8-4.5),
# widened enough to permit extrapolation without leaving the assay's comfort zone.
MW_RANGE = (300.0, 550.0)
LOGP_RANGE = (1.0, 6.0)

_CATALOG: FilterCatalog | None = None


def _init_worker() -> None:
    global _CATALOG
    RDLogger.DisableLog("rdApp.*")
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    _CATALOG = FilterCatalog(params)


def _process_file(path: str) -> tuple[pd.DataFrame, dict]:
    tranche = Path(path).stem
    rows = []
    stats = {"read": 0, "bad": 0, "window": 0, "pains": 0}
    with open(path) as fh:
        fh.readline()  # header
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            smi, zid = parts[0], parts[1]
            stats["read"] += 1
            if "." in smi:
                stats["bad"] += 1
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                stats["bad"] += 1
                continue
            mw = Descriptors.MolWt(mol)
            if not (MW_RANGE[0] <= mw <= MW_RANGE[1]):
                stats["window"] += 1
                continue
            logp = Crippen.MolLogP(mol)
            if not (LOGP_RANGE[0] <= logp <= LOGP_RANGE[1]):
                stats["window"] += 1
                continue
            if _CATALOG.HasMatch(mol):
                stats["pains"] += 1
                continue
            rows.append((zid, Chem.MolToSmiles(mol), tranche, mw, logp))
    df = pd.DataFrame(rows, columns=["zinc_id", "smiles", "tranche", "mw", "clogp"])
    return df, stats


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(SRC / "*.smi")))
    if not files:
        sys.exit("no tranche files found -- run scripts/fetch_zinc.sh first")
    n_proc = max(1, min(len(files), (os.cpu_count() or 4)))
    print(f"[prep] {len(files)} tranche files across {n_proc} workers", flush=True)

    frames, totals = [], {"read": 0, "bad": 0, "window": 0, "pains": 0}
    with mp.Pool(n_proc, initializer=_init_worker) as pool:
        for i, (df, st) in enumerate(pool.imap_unordered(_process_file, files), 1):
            frames.append(df)
            for k, v in st.items():
                totals[k] += v
            if i % 25 == 0 or i == len(files):
                kept = sum(len(f) for f in frames)
                print(f"   [{i}/{len(files)}] read={totals['read']:,} "
                      f"kept={kept:,}", flush=True)

    allc = pd.concat(frames, ignore_index=True)
    n_pre = len(allc)
    allc = allc.drop_duplicates(subset="smiles").reset_index(drop=True)
    allc["purchasability"] = allc["tranche"].str[3]
    allc["reactivity_class"] = allc["tranche"].str[2]

    out = DEST / "zinc_candidates.parquet"
    allc.to_parquet(out, index=False)

    print(f"\n[prep] read {totals['read']:,}")
    print(f"[prep] rejected -- unparseable/multi-component {totals['bad']:,}, "
          f"outside property window {totals['window']:,}, "
          f"PAINS {totals['pains']:,}, duplicates {n_pre - len(allc):,}")
    print(f"[prep] kept {len(allc):,} unique candidates -> {out}")
    print(allc[["mw", "clogp"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
