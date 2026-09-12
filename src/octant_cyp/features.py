"""Molecular featurisation for the CYP3A4 chemical analysis.

Three feature families, each earning its place:

* **ECFP4 count fingerprints** -- neighbourhood similarity, for cliffs and as
  model input.
* **Interpretable physicochemical descriptors** -- CYP3A4 has a large, lipophilic
  active site and a well-documented preference for bigger, greasier substrates,
  so these are hypotheses to test, not filler.
* **Metabolic soft-spot SMARTS** -- a curated panel of the transformations
  CYP3A4 is actually known to perform.  Enrichment against a generic
  substructure sweep would mostly rediscover correlated fragments; asking
  directly about N-dealkylation, O-demethylation, benzylic oxidation and the
  rest gives results a chemist can act on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

#: Known CYP3A4 metabolic soft spots and other groups worth testing.
#: Written to be reasonably specific -- e.g. the tertiary-amine pattern excludes
#: amides and anilines, which are not N-dealkylation substrates in the same way.
SOFT_SPOT_SMARTS: dict[str, str] = {
    # --- classic CYP3A4 transformations ---------------------------------
    "tertiary_aliphatic_amine": "[NX3;H0;!$(N[#6]=[O,S,N]);!$(N[a]);!$(N[#16]=O);!$(N[#15])]([CX4])([CX4])[CX4]",
    "secondary_aliphatic_amine": "[NX3;H1;!$(N[#6]=[O,S,N]);!$(N[a]);!$(N[#16]=O)]([CX4])[CX4]",
    "N_methyl_amine": "[NX3;!$(N[#6]=[O,S,N]);!$(N[#16]=O)][CH3]",
    "aryl_methyl_ether": "[OX2]([CH3])[c]",
    "alkyl_methyl_ether": "[OX2]([CH3])[CX4]",
    "benzylic_CH": "[CX4;H1,H2;!$(C[O,N,S])][c]",
    "unsubstituted_phenyl": "[cH]1[cH][cH][cH][cH]c1",
    "thioether": "[#16X2H0]([#6])[#6]",
    "piperazine": "[NX3]1[CX4][CX4][NX3][CX4][CX4]1",
    "piperidine": "[NX3]1[CX4][CX4][CX4][CX4][CX4]1",
    "morpholine": "[OX2]1[CX4][CX4][NX3][CX4][CX4]1",
    "pyrrolidine": "[NX3]1[CX4][CX4][CX4][CX4]1",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "terminal_alkyne": "[CX2;H1]#[CX2]",
    # --- groups expected to *resist* or block oxidation -------------------
    "sulfonamide": "[#16X4](=[OX1])(=[OX1])[NX3]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "nitro": "[NX3+](=O)[O-]",
    "trifluoromethyl": "[CX4](F)(F)F",
    "aryl_halide": "[c][F,Cl,Br,I]",
    "tert_butyl": "[CX4]([CH3])([CH3])[CH3]",
    # --- hydrolysis / bioactivation liabilities ---------------------------
    "ester": "[CX3](=O)[OX2H0][#6]",
    "amide": "[CX3](=[OX1])[NX3]",
    "aniline": "[NX3;H1,H2][c]",
    "phenol": "[OX2H][c]",
    "aliphatic_hydroxyl": "[OX2H][CX4]",
}


def parse_smiles(smiles: str) -> Chem.Mol | None:
    """Parse a SMILES string, returning ``None`` on failure."""
    return Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None


def validate_smarts(patterns: dict[str, str] | None = None) -> dict[str, str]:
    """Raise if any SMARTS in the panel fails to compile."""
    patterns = patterns or SOFT_SPOT_SMARTS
    bad = [k for k, v in patterns.items() if Chem.MolFromSmarts(v) is None]
    if bad:
        raise ValueError(f"invalid SMARTS: {bad}")
    return patterns


def morgan_generator(radius: int = 2, n_bits: int = 2048):
    """ECFP4-equivalent fingerprint generator (radius 2 => diameter 4)."""
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def fingerprint_matrix(mols, radius: int = 2, n_bits: int = 2048,
                       counts: bool = True) -> np.ndarray:
    """Stack Morgan fingerprints into a dense array for modelling."""
    gen = morgan_generator(radius, n_bits)
    rows = []
    for m in mols:
        if m is None:
            rows.append(np.zeros(n_bits, dtype=np.int16))
            continue
        fp = gen.GetCountFingerprintAsNumPy(m) if counts else gen.GetFingerprintAsNumPy(m)
        rows.append(fp.astype(np.int16))
    return np.vstack(rows)


def bit_fingerprints(mols, radius: int = 2, n_bits: int = 2048):
    """RDKit ExplicitBitVect fingerprints, for fast Tanimoto similarity."""
    gen = morgan_generator(radius, n_bits)
    return [gen.GetFingerprint(m) if m is not None else None for m in mols]


#: Interpretable descriptors.  Each is a testable hypothesis about CYP3A4.
DESCRIPTORS = {
    "mw": Descriptors.MolWt,
    "clogp": Crippen.MolLogP,
    "tpsa": Descriptors.TPSA,
    "hbd": Descriptors.NumHDonors,
    "hba": Descriptors.NumHAcceptors,
    "rotb": Descriptors.NumRotatableBonds,
    "aromatic_rings": Descriptors.NumAromaticRings,
    "rings": Descriptors.RingCount,
    "fsp3": Descriptors.FractionCSP3,
    "heavy_atoms": Descriptors.HeavyAtomCount,
    "formal_charge": lambda m: Chem.GetFormalCharge(m),
    "n_hetero": Descriptors.NumHeteroatoms,
    "molar_refractivity": Crippen.MolMR,
}


def descriptor_frame(mols) -> pd.DataFrame:
    """Physicochemical descriptor table, one row per molecule."""
    rows = []
    for m in mols:
        if m is None:
            rows.append({k: np.nan for k in DESCRIPTORS})
            continue
        rec = {}
        for name, fn in DESCRIPTORS.items():
            try:
                rec[name] = float(fn(m))
            except Exception:
                rec[name] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def soft_spot_frame(mols, patterns: dict[str, str] | None = None) -> pd.DataFrame:
    """Boolean substructure matrix over the soft-spot panel."""
    patterns = validate_smarts(patterns)
    compiled = {k: Chem.MolFromSmarts(v) for k, v in patterns.items()}
    rows = []
    for m in mols:
        if m is None:
            rows.append({k: False for k in compiled})
            continue
        rows.append({k: m.HasSubstructMatch(p) for k, p in compiled.items()})
    return pd.DataFrame(rows)


def murcko_scaffold(mol: Chem.Mol, generic: bool = False) -> str:
    """Bemis-Murcko scaffold as SMILES; ``generic`` erases atom/bond types."""
    if mol is None:
        return ""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaf = MurckoScaffold.MakeScaffoldGeneric(scaf)
        return Chem.MolToSmiles(scaf)
    except Exception:
        return ""


def build_feature_table(smiles: pd.Series) -> pd.DataFrame:
    """Descriptors + soft spots + scaffolds, indexed like ``smiles``."""
    mols = [parse_smiles(s) for s in smiles]
    desc = descriptor_frame(mols)
    spots = soft_spot_frame(mols)
    out = pd.concat([desc, spots], axis=1)
    out["murcko_scaffold"] = [murcko_scaffold(m) for m in mols]
    out["murcko_generic"] = [murcko_scaffold(m, generic=True) for m in mols]
    out["n_scaffold_atoms"] = [
        Chem.MolFromSmiles(s).GetNumHeavyAtoms() if s and Chem.MolFromSmiles(s) else 0
        for s in out["murcko_scaffold"]
    ]
    out.index = smiles.index
    return out
