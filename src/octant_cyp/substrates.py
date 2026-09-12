"""Part 1: calling CYP substrates with uncertainty.

Pipeline, per enzyme:

    well areas -> censored log-ratio contrast   (stats.fit_censored_contrast)
               -> optional plate offset         (normalize)
               -> empirical-Bayes moderation    (stats.ebayes_moderated_t)
               -> BH q-values + TOST            (stats.benjamini_hochberg / tost_equivalence)
               -> three-way call

The three-way call is deliberate.  A screen that only emits hits leaves the rest
of the library in an undifferentiated "not a hit" bucket that mixes *confidently
unreactive* compounds with *never adequately measured* ones.  Part 3 needs the
former as negatives for model training and must exclude the latter, so they are
separated here rather than downstream.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

from .stats import (
    benjamini_hochberg,
    ebayes_moderated_t,
    fit_censored_contrast,

)

#: Depletion large enough to matter biologically, as a fold-change.  0.80 = 20%
#: depletion.  Chosen to sit clearly above the assay's resolving power: with a
#: median control CV of ~8% and n=4, the 95% CI on a fold-change is about +/-13%.
DEFAULT_EFFECT = 0.80

#: Threshold used by the OpenADMET blog post, retained for comparison.
BLOG_EFFECT = 0.50


def fit_all_contrasts(wells: pd.DataFrame, lod: float) -> pd.DataFrame:
    """Fit the censored control-vs-treatment contrast for every compound x enzyme."""
    rows = []
    key = ["ocnt_batch", "enzyme"]
    for (batch, enzyme), g in wells.groupby(key, sort=False):
        ctl = g.loc[g["condition"] == "control", "area"].values
        trt = g.loc[g["condition"] == "treatment", "area"].values
        if len(ctl) < 2 or len(trt) < 1:
            continue
        fit = fit_censored_contrast(ctl, trt, lod=lod)
        rows.append({
            "ocnt_batch": batch, "enzyme": enzyme,
            "delta_log10": fit.delta, "s2": fit.s2, "df": fit.df, "v": fit.v,
            "mu_control": fit.mu_control, "mu_treatment": fit.mu_treatment,
            "n_control": fit.n_control, "n_treatment": fit.n_treatment,
            "n_censored": fit.n_censored, "censoring": fit.censoring,
            "converged": fit.converged,
            "plate": g["plate"].iloc[0],
        })
    return pd.DataFrame(rows)


def call_substrates(
    contrasts: pd.DataFrame,
    offsets: dict[str, float] | None = None,
    effect: float = DEFAULT_EFFECT,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Moderate, test, and classify.  ``offsets`` maps plate -> log2 offset.

    Classification partitions compounds by where the confidence interval for the
    log fold-change sits relative to the single decision boundary ``-margin``
    (``margin = |log10(effect)|``):

    ==================  ===========================================
    CI entirely below   ``substrate``      -- confidently >= 20% depleted
    CI entirely above   ``non-substrate``  -- confidently <  20% depleted
    CI straddles        ``inconclusive``   -- not resolved at n=4
    ==================  ===========================================

    The two claims are separate one-sided tests, each FDR-controlled, and they
    are mutually exclusive by construction.  This is stricter than the usual
    "significant *and* big enough" rule -- which certifies neither direction --
    and it is what lets Part 3 draw a trustworthy set of negatives rather than
    treating "not a hit" as evidence of no reactivity.

    A conventional two-sided moderated-t p/q is also reported for comparability
    with standard screening practice.
    """
    df = contrasts.copy()

    off_log2 = df["plate"].map(offsets).fillna(0.0) if offsets else 0.0
    df["offset_log2"] = off_log2
    df["delta_adj"] = df["delta_log10"] - np.asarray(off_log2) * np.log10(2.0)

    margin = abs(np.log10(effect))
    out = []
    for enzyme, g in df.groupby("enzyme", sort=False):
        g = g.copy()
        eb = ebayes_moderated_t(g["delta_adj"].values, g["s2"].values,
                                g["df"].values, g["v"].values)
        g["se"] = eb["se"]
        g["t_mod"] = eb["t"]
        g["p_value"] = eb["p"]          # two-sided, H0: no change
        g["q_value"] = benjamini_hochberg(eb["p"])
        g["df_total"] = eb["df_total"]
        g["s2_prior"] = eb["s2_prior"]
        g["df_prior"] = eb["df_prior"]

        # Two one-sided tests against the effect boundary -margin.
        z = (g["delta_adj"].values + margin) / eb["se"]
        g["p_substrate"] = st.t.cdf(z, eb["df_total"])      # H1: delta < -margin
        g["p_nonsubstrate"] = st.t.sf(z, eb["df_total"])    # H1: delta > -margin
        g["q_substrate"] = benjamini_hochberg(g["p_substrate"].values)
        g["q_nonsubstrate"] = benjamini_hochberg(g["p_nonsubstrate"].values)

        crit = np.abs(st.t.ppf(1 - alpha / 2, g["df_total"]))
        g["ci_low"] = g["delta_adj"] - crit * g["se"]
        g["ci_high"] = g["delta_adj"] + crit * g["se"]
        g["pct_remaining_est"] = 100 * 10 ** g["delta_adj"]
        g["pct_remaining_ci_low"] = 100 * 10 ** g["ci_low"]
        g["pct_remaining_ci_high"] = 100 * 10 ** g["ci_high"]

        is_sub = g["q_substrate"].values < alpha
        is_non = g["q_nonsubstrate"].values < alpha
        g["call"] = np.select([is_sub, is_non], ["substrate", "non-substrate"],
                              default="inconclusive")
        g["effect_bounded"] = g["censoring"] == "full"
        out.append(g)

    return pd.concat(out, ignore_index=True)


def threshold_sensitivity(contrasts: pd.DataFrame,
                          offsets: dict[str, float] | None,
                          effects=(BLOG_EFFECT, DEFAULT_EFFECT),
                          alpha: float = 0.05) -> pd.DataFrame:
    """Hit counts across effect thresholds, with and without offset correction.

    Separates the two reasons our substrate counts differ from the published
    ones: the effect threshold, and the plate-offset correction.
    """
    rows = []
    for label, off in (("uncorrected", None), ("offset-corrected", offsets)):
        for eff in effects:
            called = call_substrates(contrasts, offsets=off, effect=eff, alpha=alpha)
            for enzyme, g in called.groupby("enzyme"):
                vc = g["call"].value_counts()
                rows.append({
                    "enzyme": enzyme, "normalisation": label,
                    "effect_threshold_pct_remaining": 100 * eff,
                    "n": len(g),
                    "substrate": int(vc.get("substrate", 0)),
                    "non_substrate": int(vc.get("non-substrate", 0)),
                    "inconclusive": int(vc.get("inconclusive", 0)),
                    "pct_substrate": round(100 * vc.get("substrate", 0) / len(g), 1),
                })
    return pd.DataFrame(rows)


def permutation_null(wells: pd.DataFrame, lod: float, n_rep: int = 1,
                     seed: int = 0) -> pd.DataFrame:
    """Calibration check: split each compound's *control* wells into pseudo-arms.

    Control wells contain no enzyme, so any apparent depletion is noise.  A
    correctly calibrated test returns uniform p-values here.  This validates the
    inference machinery; it deliberately says nothing about the control-versus-
    treatment positional offset, which is a property of the plate design rather
    than of the estimator (see :mod:`octant_cyp.normalize`).
    """
    rng = np.random.default_rng(seed)
    rows = []
    ctl = wells[wells["condition"] == "control"]
    for rep in range(n_rep):
        for (batch, enzyme), g in ctl.groupby(["ocnt_batch", "enzyme"], sort=False):
            a = g["area"].values
            if len(a) < 4:
                continue
            idx = rng.permutation(len(a))
            half = len(a) // 2
            fit = fit_censored_contrast(a[idx[:half]], a[idx[half:]], lod=lod)
            rows.append({"ocnt_batch": batch, "enzyme": enzyme, "rep": rep,
                         "delta_log10": fit.delta, "s2": fit.s2,
                         "df": fit.df, "v": fit.v})
    null = pd.DataFrame(rows)
    out = []
    for enzyme, g in null.groupby("enzyme", sort=False):
        g = g.copy()
        eb = ebayes_moderated_t(g["delta_log10"].values, g["s2"].values,
                                g["df"].values, g["v"].values)
        g["p_value"] = eb["p"]
        g["q_value"] = benjamini_hochberg(eb["p"])
        out.append(g)
    return pd.concat(out, ignore_index=True)


def minimum_detectable_effect(s2_post: float, v: float, df_total: float,
                              alpha: float = 0.05, power: float = 0.95) -> float:
    """Smallest depletion (as fraction remaining) detectable at the given power."""
    se = np.sqrt(s2_post * v)
    z = st.t.ppf(1 - alpha / 2, df_total) + st.t.ppf(power, df_total)
    return float(10 ** (-z * se))
