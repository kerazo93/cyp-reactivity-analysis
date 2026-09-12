"""Statistical machinery for calling CYP substrates from Echo-MS depletion data.

The released summary table (``reactivity.tsv``) reduces each compound to a ratio of
means -- ``pct_remaining = 100 * mean(treatment) / mean(control)`` -- with no
uncertainty attached, so the published ">50% depletion" rule is an effect-size
rule with no inferential content.  This module supplies the missing inference:

1. :func:`fit_censored_contrast` -- a Tobit (left-censored normal) MLE on
   ``log10(peak area)``.  Non-detected treatment wells are recorded as
   ``area == 0``; they are *complete depletion*, not missing data, and dropping
   them (or taking ``log(0)``) discards the strongest substrates in the screen.
2. :func:`fit_fdist` / :func:`ebayes_moderated_t` -- Smyth's (2004) empirical-Bayes
   variance moderation.  With n=4 wells the per-compound variance estimate is
   nearly worthless; moderation borrows strength across the ~1,200 compounds
   assayed on the same plates.
3. :func:`benjamini_hochberg` -- FDR control across compounds within an enzyme.
4. :func:`tost_equivalence` -- two one-sided tests, so that "not a substrate" can
   be asserted positively rather than inferred from a failure to reject.

Reference: Smyth, G.K. (2004) "Linear models and empirical Bayes methods for
assessing differential expression in microarray experiments", SAGMB 3(1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats

__all__ = [
    "ContrastFit",
    "fit_censored_contrast",
    "trigamma_inverse",
    "fit_fdist",
    "ebayes_moderated_t",
    "benjamini_hochberg",
    "tost_equivalence",
]


# --------------------------------------------------------------------------
# 1. Per-compound censored two-group contrast
# --------------------------------------------------------------------------
@dataclass
class ContrastFit:
    """Result of a single compound x enzyme control-vs-treatment contrast.

    Attributes
    ----------
    delta:
        log10 fold-change, ``mu_treatment - mu_control``.  Negative = depleted.
        When ``censoring == "full"`` this is an *upper bound* (the least extreme
        value consistent with every treatment well falling below the LOD).
    s2:
        Residual variance on the log10 scale, to be moderated across compounds.
    df:
        Residual degrees of freedom backing ``s2`` (uncensored observations only).
    v:
        Unscaled variance factor of the contrast, so ``Var(delta) = v * sigma^2``.
    censoring:
        ``"none"``, ``"partial"`` or ``"full"``.
    """

    delta: float
    s2: float
    df: int
    v: float
    mu_control: float
    mu_treatment: float
    n_control: int
    n_treatment: int
    n_censored: int
    censoring: str
    converged: bool


def fit_censored_contrast(
    control: np.ndarray,
    treatment: np.ndarray,
    lod: float,
) -> ContrastFit:
    """Fit ``log10(area) ~ condition`` with left-censoring on the treatment arm.

    Parameters
    ----------
    control, treatment:
        Raw peak areas (linear scale).  Zeros denote a non-detect and are treated
        as left-censored at ``lod``.  Controls are expected to be fully detected.
    lod:
        Limit of detection on the linear peak-area scale.

    Notes
    -----
    Both arms share one variance.  Echo-MS peak-area noise is multiplicative
    (the observed control CV is ~8% and roughly constant across the intensity
    range), which is precisely the regime where a constant variance on the log
    scale is the right model.

    When *every* treatment well is censored the treatment mean is not identified
    -- the likelihood increases monotonically as ``mu_treatment -> -inf``.  Rather
    than report a fabricated point estimate we pin ``mu_treatment`` at
    ``log10(lod)``, the largest value consistent with the observations, making
    ``delta`` a conservative upper bound: the true depletion is at least this
    large.  For these compounds control means exceed 2,800 against an LOD of ~21,
    so the resulting bound already implies >99% depletion and the substrate call
    is not sensitive to this choice.
    """
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    log_lod = np.log10(lod)

    if np.any(control <= 0):
        raise ValueError("control wells must all be detected (area > 0)")

    y_c = np.log10(control)
    obs_mask = treatment > 0
    y_t = np.log10(treatment[obs_mask])
    n_cens = int((~obs_mask).sum())
    n_c, n_t = len(y_c), len(treatment)

    if n_cens == 0:
        censoring = "none"
    elif n_cens < n_t:
        censoring = "partial"
    else:
        censoring = "full"

    # ---- fully censored: closed form, with mu_t pinned at the LOD ----------
    if censoring == "full":
        mu_c = float(y_c.mean())
        mu_t = log_lod
        # Variance is identified by the control arm alone.
        df = max(n_c - 1, 1)
        s2 = float(((y_c - mu_c) ** 2).sum() / df) if n_c > 1 else np.nan
        return ContrastFit(
            delta=mu_t - mu_c, s2=s2, df=df, v=1.0 / n_c + 1.0 / n_t,
            mu_control=mu_c, mu_treatment=mu_t, n_control=n_c,
            n_treatment=n_t, n_censored=n_cens, censoring=censoring,
            converged=True,
        )

    # ---- uncensored: ordinary pooled two-sample contrast -------------------
    if censoring == "none":
        mu_c, mu_t = float(y_c.mean()), float(y_t.mean())
        df = n_c + n_t - 2
        rss = ((y_c - mu_c) ** 2).sum() + ((y_t - mu_t) ** 2).sum()
        s2 = float(rss / df) if df > 0 else np.nan
        return ContrastFit(
            delta=mu_t - mu_c, s2=s2, df=df, v=1.0 / n_c + 1.0 / n_t,
            mu_control=mu_c, mu_treatment=mu_t, n_control=n_c,
            n_treatment=n_t, n_censored=0, censoring=censoring, converged=True,
        )

    # ---- partial censoring: Tobit MLE --------------------------------------
    def neg_ll(theta: np.ndarray) -> float:
        mu_c, mu_t, log_sigma = theta
        sigma = np.exp(log_sigma)
        ll = stats.norm.logpdf(y_c, mu_c, sigma).sum()
        ll += stats.norm.logpdf(y_t, mu_t, sigma).sum()
        ll += n_cens * stats.norm.logcdf((log_lod - mu_t) / sigma)
        return -ll

    sigma0 = max(np.std(np.concatenate([y_c, y_t]), ddof=1), 1e-3)
    theta0 = np.array([y_c.mean(), min(y_t.mean(), log_lod), np.log(sigma0)])
    res = optimize.minimize(neg_ll, theta0, method="Nelder-Mead",
                            options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
    mu_c, mu_t, log_sigma = res.x
    sigma = float(np.exp(log_sigma))

    # Var(delta) from the observed information; express it as v * sigma^2 so it
    # plugs into the same moderated-t machinery as the uncensored compounds.
    hess = _numeric_hessian(neg_ll, res.x)
    try:
        cov = np.linalg.inv(hess)
        var_delta = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
        v = max(var_delta / sigma**2, 1e-12)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        v = 1.0 / n_c + 1.0 / n_t

    # Degrees of freedom are carried by the *uncensored* observations only.
    df = max(n_c + len(y_t) - 2, 1)
    s2 = float(sigma**2 * (n_c + len(y_t)) / df)  # bias-correct the MLE variance

    return ContrastFit(
        delta=float(mu_t - mu_c), s2=s2, df=df, v=v,
        mu_control=float(mu_c), mu_treatment=float(mu_t), n_control=n_c,
        n_treatment=n_t, n_censored=n_cens, censoring=censoring,
        converged=bool(res.success),
    )


def _numeric_hessian(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central-difference Hessian of ``f`` at ``x``."""
    n = len(x)
    h = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            xi = np.array(x, dtype=float)
            xi[i] += eps; xi[j] += eps; fpp = f(xi)
            xi = np.array(x, dtype=float)
            xi[i] += eps; xi[j] -= eps; fpm = f(xi)
            xi = np.array(x, dtype=float)
            xi[i] -= eps; xi[j] += eps; fmp = f(xi)
            xi = np.array(x, dtype=float)
            xi[i] -= eps; xi[j] -= eps; fmm = f(xi)
            h[i, j] = h[j, i] = (fpp - fpm - fmp + fmm) / (4 * eps**2)
    return h


# --------------------------------------------------------------------------
# 2. Empirical-Bayes variance moderation (Smyth 2004)
# --------------------------------------------------------------------------
def trigamma_inverse(x: np.ndarray | float) -> np.ndarray:
    """Solve ``trigamma(y) = x`` for y, by Newton iteration on ``1/trigamma``.

    Mirrors limma's ``trigammaInverse``; monotone and quadratically convergent.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.where(x > 1e7, 1.0 / np.sqrt(np.maximum(x, 1e-300)), np.nan)
    y = np.where(x < 1e-6, 1.0 / x, y)
    mid = np.isnan(y)
    y[mid] = 0.5 + 1.0 / x[mid]
    for _ in range(60):
        tri = special.polygamma(1, y)
        tetra = special.polygamma(2, y)
        dif = tri * (1.0 - tri / x) / tetra
        y = y + dif
        if np.max(np.abs(dif / np.maximum(y, 1e-12))) < 1e-10:
            break
    return y


def fit_fdist(s2: np.ndarray, df: np.ndarray) -> tuple[float, float]:
    """Estimate the scaled-inverse-chi-square prior ``(s0^2, d0)`` for variances.

    Returns ``(s2_prior, df_prior)``.  ``df_prior`` is ``inf`` when the observed
    variances are less dispersed than sampling noise alone would predict, in
    which case every compound is shrunk to the common variance.
    """
    s2 = np.asarray(s2, dtype=float)
    df = np.asarray(df, dtype=float)
    ok = np.isfinite(s2) & (s2 > 0) & np.isfinite(df) & (df > 0)
    if ok.sum() < 2:
        raise ValueError("need at least two usable variances")
    s2, df = s2[ok], df[ok]

    z = np.log(s2)
    e = z - special.digamma(df / 2.0) + np.log(df / 2.0)
    e_mean = e.mean()
    e_var = np.sum((e - e_mean) ** 2) / (len(e) - 1)
    e_var -= np.mean(special.polygamma(1, df / 2.0))

    if e_var > 0:
        df_prior = float(2.0 * trigamma_inverse(e_var)[0])
        s2_prior = float(np.exp(e_mean + special.digamma(df_prior / 2.0)
                                - np.log(df_prior / 2.0)))
    else:
        df_prior = np.inf
        s2_prior = float(np.exp(e_mean))
    return s2_prior, df_prior


def ebayes_moderated_t(
    delta: np.ndarray,
    s2: np.ndarray,
    df: np.ndarray,
    v: np.ndarray,
    s2_prior: float | None = None,
    df_prior: float | None = None,
) -> dict:
    """Moderated t-statistics and two-sided p-values.

    The posterior variance ``s2_post = (d0*s0^2 + d*s^2) / (d0 + d)`` replaces the
    noisy per-compound estimate, and the null distribution gains the prior's
    degrees of freedom.  With n=4 wells this is the difference between a test
    with 6 df and one with ~6+d0 df.
    """
    delta = np.asarray(delta, float)
    s2 = np.asarray(s2, float)
    df = np.asarray(df, float)
    v = np.asarray(v, float)

    if s2_prior is None or df_prior is None:
        s2_prior, df_prior = fit_fdist(s2, df)

    if np.isinf(df_prior):
        s2_post = np.full_like(s2, s2_prior)
        df_total = np.full_like(df, np.inf)
    else:
        s2_post = (df_prior * s2_prior + df * s2) / (df_prior + df)
        df_total = df + df_prior

    se = np.sqrt(s2_post * v)
    t = delta / se
    p = 2.0 * stats.t.sf(np.abs(t), df_total)
    return {
        "t": t, "p": p, "se": se, "s2_post": s2_post, "df_total": df_total,
        "s2_prior": s2_prior, "df_prior": df_prior,
    }


# --------------------------------------------------------------------------
# 3. Multiplicity and equivalence
# --------------------------------------------------------------------------
def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (step-up, monotonised). NaNs propagate."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    n = pv.size
    if n == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    q[ok] = out
    return q


def tost_equivalence(
    delta: np.ndarray, se: np.ndarray, df: np.ndarray, margin: float
) -> np.ndarray:
    """Two one-sided tests for ``|delta| < margin``.

    Returns the TOST p-value (the larger of the two one-sided p-values).  A small
    value licenses the positive claim "this compound is *not* a substrate", as
    opposed to merely failing to show that it is.  ``margin`` is on the same
    log10 scale as ``delta`` -- e.g. ``abs(log10(0.8))`` for a 20% effect.
    """
    delta = np.asarray(delta, float)
    se = np.asarray(se, float)
    df = np.asarray(df, float)
    p_lower = stats.t.sf((delta + margin) / se, df)   # H0: delta <= -margin
    p_upper = stats.t.cdf((delta - margin) / se, df)  # H0: delta >= +margin
    return np.maximum(p_lower, p_upper)
