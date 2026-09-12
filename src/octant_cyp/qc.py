"""Quality-control diagnostics for the Echo-MS reactivity plates.

The OpenADMET blog post flags edge effects, dispensing patterns and outliers as
things "modellers can apply their own criteria" to.  These helpers test for each
so the choices made downstream are evidenced rather than assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_plate_coordinates(wells: pd.DataFrame,
                          geometry: tuple[int, int] | None = None) -> pd.DataFrame:
    """Split the ``well`` label (e.g. ``AC12``) into row index and column index.

    ``geometry`` is ``(n_rows, n_cols)`` for the physical plate.  It must describe
    the *whole* plate, not whichever subset is being analysed: control wells
    occupy only rows A-D, so deriving the geometry from them would place every
    control on the plate edge by construction.  Defaults to the full extent of
    the supplied frame, so pass the complete well table when subsetting later.
    """
    out = wells.copy()
    lab = out["well"].astype(str).str.extract(r"^([A-Za-z]+)(\d+)$")
    letters, cols = lab[0], lab[1]

    def _row_index(s: str) -> int:
        n = 0
        for ch in str(s).upper():
            n = n * 26 + (ord(ch) - 64)
        return n

    out["row_label"] = letters
    out["row_idx"] = letters.map(_row_index)
    out["col_idx"] = cols.astype(int)
    n_rows, n_cols = geometry or (int(out["row_idx"].max()), int(out["col_idx"].max()))
    out["n_rows"], out["n_cols"] = n_rows, n_cols
    # Ring distance from the plate edge: 0 = outermost ring.
    out["edge_ring"] = np.minimum.reduce([
        out["row_idx"] - 1, n_rows - out["row_idx"],
        out["col_idx"] - 1, n_cols - out["col_idx"],
    ])
    return out


def plate_geometry(wells: pd.DataFrame) -> tuple[int, int]:
    """Physical plate dimensions implied by the full set of well labels."""
    c = add_plate_coordinates(wells)
    return int(c["row_idx"].max()), int(c["col_idx"].max())


def condition_layout(wells: pd.DataFrame) -> pd.DataFrame:
    """Which plate rows each condition occupies.

    Control and treatment are *not* interleaved: controls sit in rows A-D and
    treatments in rows E-AF, so the control/treatment contrast is not
    position-matched and any plate-level gradient maps onto every fold-change.
    """
    c = add_plate_coordinates(wells, plate_geometry(wells))
    return (c.groupby("condition")
             .agg(row_min=("row_idx", "min"), row_max=("row_idx", "max"),
                  n_rows_used=("row_idx", "nunique"),
                  col_min=("col_idx", "min"), col_max=("col_idx", "max"),
                  n_wells=("area", "size"))
             .reset_index())


def mass_accuracy_summary(wells: pd.DataFrame) -> pd.DataFrame:
    """Distribution of ``mass_error_ppm``, which is NaN exactly when no peak was found."""
    det = wells[wells["area"] > 0]
    return pd.DataFrame({
        "n_wells": [len(wells)],
        "n_detected": [len(det)],
        "n_nondetect": [int((wells["area"] == 0).sum())],
        "mass_error_nan_equals_zero_area":
            [bool(((wells["area"] == 0) == wells["mass_error_ppm"].isna()).all())],
        "ppm_p01": [det["mass_error_ppm"].quantile(0.01)],
        "ppm_median": [det["mass_error_ppm"].median()],
        "ppm_p99": [det["mass_error_ppm"].quantile(0.99)],
        "ppm_abs_max": [det["mass_error_ppm"].abs().max()],
    })


def control_cv(wells: pd.DataFrame) -> pd.DataFrame:
    """Per-compound coefficient of variation across control replicates.

    This is the assay's noise floor and sets the smallest depletion that can be
    resolved at all.
    """
    ctl = wells[wells["condition"] == "control"]
    g = ctl.groupby(["ocnt_batch", "enzyme"])["area"].agg(["mean", "std", "size"])
    g["cv"] = g["std"] / g["mean"]
    return g.reset_index()


def plate_position_effects(wells: pd.DataFrame) -> pd.DataFrame:
    """Median log10 control signal by column, per plate.

    Controls span all 48 columns but only rows A-D, so column position is the
    axis along which a control-side spatial gradient could actually show up.
    """
    geom = plate_geometry(wells)
    ctl = add_plate_coordinates(
        wells[(wells["condition"] == "control") & (wells["area"] > 0)], geom)
    ctl = ctl.assign(log_area=np.log10(ctl["area"]))
    return (ctl.groupby(["plate", "col_idx"])["log_area"]
               .agg(median="median", n="size").reset_index())


def row_profile(wells: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Median log10 signal by plate row for one condition, pooled over plates."""
    geom = plate_geometry(wells)
    sub = add_plate_coordinates(
        wells[(wells["condition"] == condition) & (wells["area"] > 0)], geom)
    sub = sub.assign(log_area=np.log10(sub["area"]))
    return (sub.groupby(["row_idx", "row_label"])["log_area"]
               .agg(median="median", n="size").reset_index())


def edge_vs_interior(wells: pd.DataFrame, condition: str = "control") -> pd.DataFrame:
    """Edge-versus-interior contrast of signal, per plate, for one condition.

    Returns NaN deltas where a condition does not populate both zones -- which is
    the case for controls, confined to rows A-D.  Reported rather than hidden.
    """
    geom = plate_geometry(wells)
    sub = add_plate_coordinates(
        wells[(wells["condition"] == condition) & (wells["area"] > 0)], geom)
    sub = sub.assign(log_area=np.log10(sub["area"]),
                     zone=np.where(sub["edge_ring"] <= 1, "edge", "interior"))
    piv = (sub.groupby(["plate", "zone"])["log_area"].median().unstack())
    for z in ("edge", "interior"):
        if z not in piv.columns:
            piv[z] = np.nan
    piv["delta_log10"] = piv["edge"] - piv["interior"]
    piv["pct_shift"] = (10 ** piv["delta_log10"] - 1) * 100
    return piv.reset_index()


def censoring_summary(wells: pd.DataFrame) -> pd.DataFrame:
    """How non-detects are distributed -- the key preprocessing decision."""
    w = wells.assign(is_zero=wells["area"] == 0)
    by_cond = (w.groupby(["enzyme", "condition"])["is_zero"]
                 .agg(n_zero="sum", n="size").reset_index())
    by_cond["pct_zero"] = 100 * by_cond["n_zero"] / by_cond["n"]
    return by_cond


def censoring_pattern(wells: pd.DataFrame) -> pd.DataFrame:
    """Per treatment group: how many replicates were non-detects."""
    trt = wells[wells["condition"] == "treatment"]
    g = trt.groupby(["ocnt_batch", "enzyme"])["area"].agg(
        n="size", n_zero=lambda s: int((s == 0).sum()))
    g["pattern"] = np.select(
        [g["n_zero"] == 0, g["n_zero"] == g["n"]],
        ["none", "full"], default="partial")
    return g.reset_index()
