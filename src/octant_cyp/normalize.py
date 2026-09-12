"""Systematic control-vs-treatment offset in the Echo-MS reactivity plates.

Incubating a compound with a CYP cannot *create* parent compound, so unreactive
compounds must sit at 100% remaining.  They do not.  In the CYP2J2 arm -- where
~82% of the library is unreactive and the inactive population is therefore
cleanly separated -- the inactive mode sits at ``+0.19`` log2, i.e. ~114% of
control, consistently across all four of its plates.  62.5% of CYP2J2 compounds
appear "enriched" by incubation, which is not a biological possibility.

Control and treatment wells are also spatially segregated on the plate: controls
occupy only rows A-D of the 32x48 grid, treatments rows E-AF.  The comparison is
therefore not position-matched, and any plate-level gradient or difference in
matrix between the two zones lands directly on every fold-change.

This module measures that offset so it can be corrected.

A note on bandwidth, because it changes the answer.  ``gaussian_kde``'s
``bw_method`` is a multiple of the data's standard deviation.  The CYP3A4 fold-
change distribution has a much larger spread than CYP2J2's (sd 5.25 vs 2.01)
because most of its library is reactive, so a shared *relative* bandwidth
smooths CYP3A4 roughly 2.6x harder and erases its inactive peak entirely -- which
initially looked like "CYP3A4 has no identifiable offset".  Using a fixed
*absolute* bandwidth, both enzymes show the same offset (~+0.2 log2, ~115% of
control) on all eight plates, which is what one expects from an artefact of the
control/treatment well layout rather than of the enzyme.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

#: Search window (log2 fold-change) in which a genuine inactive mode can live.
#: Wide enough to admit a real offset, narrow enough to exclude the active lobe.
INACTIVE_WINDOW = (-0.6, 1.0)

#: KDE bandwidth in **absolute log2 units**, so the two enzymes are smoothed
#: identically despite very different spreads.  0.08 is well below the offset
#: being measured (~0.2) and well above the bin noise; the estimate moves by
#: <0.05 log2 across 0.05-0.15 (see ``bootstrap_mode``).
KDE_BANDWIDTH = 0.08

#: Minimum peak-to-trough contrast for an inactive mode to be trusted as a null
#: anchor.  Both enzymes clear this comfortably at matched bandwidth (CYP2J2
#: ~1e4, CYP3A4 ~24); the check exists to catch the degenerate case where no
#: separable inactive population exists at all.
MIN_MODE_SEPARATION = 5.0


def inactive_mode(log2fc: np.ndarray,
                  window: tuple[float, float] = INACTIVE_WINDOW,
                  bw: float = KDE_BANDWIDTH) -> float:
    """Locate the inactive population's centre as a KDE mode in ``window``.

    A Gaussian-mixture mean is the obvious alternative but is badly biased here:
    when actives dominate (CYP3A4) the "upper" component absorbs weak actives and
    drifts low.  A restricted mode is robust to that contamination.
    """
    x = np.asarray(log2fc, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 30:
        return np.nan
    grid = np.linspace(window[0], window[1], 1001)
    dens = st.gaussian_kde(x, bw_method=bw / x.std(ddof=1))(grid)
    return float(grid[np.argmax(dens)])


def mode_separation(log2fc: np.ndarray,
                    window: tuple[float, float] = INACTIVE_WINDOW,
                    bw: float = KDE_BANDWIDTH) -> float:
    """Peak-to-trough contrast of the inactive mode, as an identifiability check.

    Returns the ratio of the density at the in-window mode to the smallest
    density between that mode and the active lobe.  A clean, well-separated
    inactive population gives a large value; a contaminated shoulder gives ~1.
    ``nan`` if the mode sits at the window edge (no interior peak at all).
    """
    x = np.asarray(log2fc, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 30:
        return np.nan
    grid = np.linspace(-8.0, window[1], 3001)
    dens = st.gaussian_kde(x, bw_method=bw / x.std(ddof=1))(grid)
    in_win = (grid >= window[0]) & (grid <= window[1])
    peak_i = np.argmax(np.where(in_win, dens, -np.inf))
    if peak_i in (0, len(grid) - 1):
        return np.nan
    left = dens[:peak_i]
    if left.size == 0:
        return np.nan
    return float(dens[peak_i] / max(left.min(), 1e-12))


def estimate_plate_offsets(summary: pd.DataFrame,
                           plate_map: pd.DataFrame | None = None,
                           value: str = "log2fc") -> pd.DataFrame:
    """Per enzyme x plate inactive-mode offset, with an identifiability flag.

    Parameters
    ----------
    summary:
        Compound x enzyme table carrying ``value`` (log2 fold-change).
    plate_map:
        Optional ``ocnt_batch, enzyme, plate`` mapping; when supplied the offset
        is estimated per plate rather than pooled across the enzyme.
    """
    df = summary.copy()
    df[value] = df[value].replace([np.inf, -np.inf], np.nan)
    if plate_map is not None:
        df = df.merge(plate_map.drop_duplicates(["ocnt_batch", "enzyme"]),
                      on=["ocnt_batch", "enzyme"], how="left")
        keys = ["enzyme", "plate"]
    else:
        keys = ["enzyme"]

    rows = []
    for key, g in df.groupby(keys):
        x = g[value].dropna().values
        mode = inactive_mode(x)
        sep = mode_separation(x)
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        rec.update({
            "n": len(x),
            "offset_log2": mode,
            "offset_pct_of_control": 100 * 2.0 ** mode if np.isfinite(mode) else np.nan,
            "mode_separation": sep,
            # A separable inactive lobe is the precondition for trusting the offset.
            "identifiable": bool(np.isfinite(sep) and sep > MIN_MODE_SEPARATION),
            "frac_above_100pct": float((x > 0).mean()),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def apply_offset(delta_log10: np.ndarray, offset_log2: float) -> np.ndarray:
    """Subtract a log2-scaled offset from a log10 fold-change."""
    return np.asarray(delta_log10, float) - float(offset_log2) * np.log10(2.0)


def bootstrap_mode(log2fc: np.ndarray, n_boot: int = 500,
                   bw: float = KDE_BANDWIDTH, seed: int = 0) -> dict:
    """Bootstrap confidence interval for the inactive-mode offset."""
    rng = np.random.default_rng(seed)
    x = np.asarray(log2fc, float)
    x = x[np.isfinite(x)]
    draws = np.array([inactive_mode(rng.choice(x, len(x), replace=True), bw=bw)
                      for _ in range(n_boot)])
    draws = draws[np.isfinite(draws)]
    return {
        "mode": inactive_mode(x, bw=bw),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "sd": float(draws.std()),
        "n_boot": int(len(draws)),
    }
