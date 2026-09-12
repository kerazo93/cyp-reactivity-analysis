"""Loading and provenance for the OpenADMET CYP release.

Data: https://huggingface.co/datasets/openadmet/Octant_CYP_inhibition_reactivity_blog_release
Licence: CC-BY-4.0 (OpenADMET consortium / Octant Bio).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

#: Non-detected wells are recorded as ``area == 0``.  The detection limit is
#: taken as the smallest peak area the vendor pipeline ever reported (21.0),
#: i.e. an empirical floor rather than an assumed one.  See :func:`estimate_lod`.
DEFAULT_LOD = 21.0


def load_wells(path: Path | None = None) -> pd.DataFrame:
    """Well-level Echo-MS reactivity data (19,344 rows)."""
    return pd.read_csv(path or RAW / "reactivity_wells.tsv", sep="\t")


def load_summary(path: Path | None = None) -> pd.DataFrame:
    """Vendor-computed compound x enzyme summary (2,446 rows, no uncertainty)."""
    return pd.read_csv(path or RAW / "reactivity.tsv", sep="\t")


def load_inhibition(path: Path | None = None) -> pd.DataFrame:
    """CYP3A4 dose-response summary (1,340 rows)."""
    return pd.read_csv(path or RAW / "inhibition.tsv", sep="\t")


def load_will_it_fly(path: Path | None = None) -> pd.DataFrame:
    """MS ionisation screen: 11,353 compounds, fluoride vs formate peak areas."""
    return pd.read_csv(path or RAW / "will_it_fly_in_mass_spec.tsv", sep="\t")


def estimate_lod(wells: pd.DataFrame) -> float:
    """Empirical limit of detection: the smallest peak area actually reported.

    Anything below this was returned as ``0`` (no peak found), so the smallest
    *observed* value is the tightest defensible censoring threshold.  Using a
    larger LOD would discard real measurements; a smaller one is unsupported by
    the data.
    """
    detected = wells.loc[wells["area"] > 0, "area"]
    return float(detected.min())


def compound_table(wells: pd.DataFrame) -> pd.DataFrame:
    """One row per compound: batch id and standardised SMILES.

    The summary table ships without structures, so SMILES must come from the
    well-level file.  The mapping is 1:1 (1,223 batches, 1,223 unique SMILES).
    """
    out = (wells[["ocnt_batch", "standardized_smiles"]]
           .drop_duplicates()
           .reset_index(drop=True))
    if out["ocnt_batch"].duplicated().any():
        raise ValueError("a batch maps to more than one SMILES")
    return out
