#!/usr/bin/env python
"""Filter the ZINC in-stock pool down to an assay-relevant candidate set.

The tranche files hold ~2.7M in-stock compounds. Only 1,000 are ultimately
bought, so the pool is randomly subsampled *before* the expensive structural
work rather than after: PAINS matching dominates the runtime, and sampling first
makes the cost proportional to the pool we actually intend to score. The pool is
homogeneous by construction -- every tranche is already restricted to the same
MW/logP region -- so a uniform sample is representative, and the sampling seed
is recorded so the selection is reproducible. Pass --sample 0 to use all of it.

Filters, cheapest first:

1. valid, single-component structure
2. property window matched to the *assayed* library, not generic drug-likeness
3. PAINS removal (assay interference)

Deliberately NOT applied: the Brenk "unwanted functionality" catalogue. Brenk
rejects anilines, nitroaromatics, halides and Michael acceptors -- precisely the
chemistry Part 2 found associated with CYP3A4 turnover (anilines are the
strongest depleted substructure). Screening it out would remove the compounds
most able to test our own conclusions; on this pool it also rejected ~99% of
candidates. PAINS is kept because it targets compounds that confound *assays*,
which is a separate and legitimate concern here.

Output: data/processed/zinc_candidates.parquet
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "external" / "zinc"
DEST = ROOT / "data" / "processed"

MW_RANGE = (300.0, 550.0)
LOGP_RANGE = (1.0, 6.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=400_000,
                    help="compounds to sample before filtering (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(str(SRC / "*.smi")))
    if not files:
        raise SystemExit("no tranche files -- run scripts/fetch_zinc.sh first")

    # ---- cheap text pass -------------------------------------------------
    records: list[tuple[str, str, str]] = []
    for path in files:
        tranche = Path(path).stem
        with open(path) as fh:
            fh.readline()
            for line in fh:
                p = line.split()
                if len(p) >= 2 and "." not in p[0]:
                    records.append((p[1], p[0], tranche))
    print(f"[prep] {len(records):,} in-stock compounds across {len(files)} tranches",
          flush=True)

    if args.sample and len(records) > args.sample:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(records), args.sample, replace=False)
        records = [records[i] for i in idx]
        print(f"[prep] sampled {len(records):,} (seed={args.seed})", flush=True)

    # ---- structural pass -------------------------------------------------
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)

    rows = []
    n_bad = n_window = n_pains = 0
    for i, (zid, smi, tranche) in enumerate(records, 1):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_bad += 1
            continue
        mw = Descriptors.MolWt(mol)
        if not (MW_RANGE[0] <= mw <= MW_RANGE[1]):
            n_window += 1
            continue
        logp = Crippen.MolLogP(mol)
        if not (LOGP_RANGE[0] <= logp <= LOGP_RANGE[1]):
            n_window += 1
            continue
        if catalog.HasMatch(mol):
            n_pains += 1
            continue
        rows.append((zid, Chem.MolToSmiles(mol), tranche, mw, logp))
        if i % 50_000 == 0:
            print(f"   [{i:,}/{len(records):,}] kept={len(rows):,}", flush=True)

    df = pd.DataFrame(rows, columns=["zinc_id", "smiles", "tranche", "mw", "clogp"])
    n_pre = len(df)
    df = df.drop_duplicates(subset="smiles").reset_index(drop=True)
    df["purchasability"] = df["tranche"].str[3]
    df["reactivity_class"] = df["tranche"].str[2]

    DEST.mkdir(parents=True, exist_ok=True)
    out = DEST / "zinc_candidates.parquet"
    df.to_parquet(out, index=False)

    print(f"\n[prep] rejected -- unparseable {n_bad:,}, outside window {n_window:,}, "
          f"PAINS {n_pains:,}, duplicates {n_pre - len(df):,}")
    print(f"[prep] kept {len(df):,} unique candidates -> {out}")
    print(df[["mw", "clogp"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
