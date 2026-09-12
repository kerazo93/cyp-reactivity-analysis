"""Validation of the inference layer against closed forms and simulation."""
import numpy as np
import pytest
from scipy import special, stats

from octant_cyp.stats import (
    benjamini_hochberg,
    ebayes_moderated_t,
    fit_censored_contrast,
    fit_fdist,
    tost_equivalence,
    trigamma_inverse,
)


def test_trigamma_inverse_roundtrip():
    y = np.array([0.05, 0.3, 1.0, 3.0, 25.0, 400.0])
    assert np.allclose(trigamma_inverse(special.polygamma(1, y)), y, rtol=1e-6)


def test_bh_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(0, 1, 200), rng.uniform(0, 1e-4, 20)])
    assert np.allclose(benjamini_hochberg(p), sm.multipletests(p, method="fdr_bh")[1])


def test_bh_is_monotone_and_bounded():
    q = benjamini_hochberg(np.array([0.001, 0.2, 0.5, 0.9]))
    assert np.all(np.diff(q) >= 0) and q.max() <= 1.0


def test_fit_fdist_recovers_prior():
    """Variances drawn from a known scaled-inv-chi2 prior are recovered."""
    rng = np.random.default_rng(7)
    n, d, d0, s20 = 8000, 6.0, 10.0, 0.004
    sigma2 = d0 * s20 / rng.chisquare(d0, n)           # prior draw
    s2 = sigma2 * rng.chisquare(d, n) / d              # sampling draw
    s20_hat, d0_hat = fit_fdist(s2, np.full(n, d))
    assert s20_hat == pytest.approx(s20, rel=0.10)
    assert d0_hat == pytest.approx(d0, rel=0.25)


def test_moderated_t_reduces_to_ordinary_t_with_vague_prior():
    """df_prior -> 0 removes the moderation, recovering Student's t exactly."""
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 6), rng.normal(0.8, 1, 6)
    df = len(a) + len(b) - 2
    s2 = (((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()) / df
    v = 1 / len(a) + 1 / len(b)
    got = ebayes_moderated_t(np.array([b.mean() - a.mean()]), np.array([s2]),
                             np.array([df]), np.array([v]),
                             s2_prior=1.0, df_prior=0.0)
    ref = stats.ttest_ind(b, a, equal_var=True)
    assert got["t"][0] == pytest.approx(ref.statistic, rel=1e-10)
    assert got["p"][0] == pytest.approx(ref.pvalue, rel=1e-10)


def test_moderated_t_shrinks_extreme_variances():
    """A compound with a freakishly small variance is not allowed to look huge."""
    rng = np.random.default_rng(3)
    n = 500
    s2 = rng.chisquare(6, n) / 6 * 0.01
    s2[0] = 1e-6                       # pathological under-dispersion
    delta = np.full(n, -0.05)
    out = ebayes_moderated_t(delta, s2, np.full(n, 6.0), np.full(n, 0.5))
    unmoderated = abs(delta[0]) / np.sqrt(s2[0] * 0.5)
    assert abs(out["t"][0]) < unmoderated / 10


def test_censored_contrast_matches_two_sample_t_when_uncensored():
    ctrl = np.array([10000.0, 11000, 9000, 10500])
    trt = np.array([5000.0, 5200, 4800, 5100])
    fit = fit_censored_contrast(ctrl, trt, lod=21.0)
    yc, yt = np.log10(ctrl), np.log10(trt)
    assert fit.censoring == "none"
    assert fit.delta == pytest.approx(yt.mean() - yc.mean())
    assert fit.df == 6
    assert fit.v == pytest.approx(0.5)


def test_censored_contrast_full_censoring_is_a_conservative_bound():
    ctrl = np.array([30000.0, 31000, 29000, 30500])
    fit = fit_censored_contrast(ctrl, np.zeros(4), lod=21.0)
    assert fit.censoring == "full"
    assert fit.mu_treatment == pytest.approx(np.log10(21.0))
    # bound implies >99% depletion, and is the *least* extreme reading
    assert 10 ** fit.delta < 0.01
    assert fit.delta > np.log10(21.0) - np.log10(30500)


def test_tobit_recovers_truth_under_partial_censoring():
    """Tobit beats naive deletion of non-detects on simulated ground truth."""
    rng = np.random.default_rng(11)
    # Put the treatment mean just above the LOD so replicates straddle it --
    # this is the regime the 107 partially-censored groups actually occupy.
    lod, mu_c, true_delta, sigma = 21.0, 4.5, -3.1, 0.30
    naive_err, tobit_err = [], []
    for _ in range(600):
        ctrl = 10 ** rng.normal(mu_c, sigma, 4)
        yt = rng.normal(mu_c + true_delta, sigma, 4)
        trt = np.where(10**yt < lod, 0.0, 10**yt)
        if (trt == 0).sum() in (0, 4):
            continue
        fit = fit_censored_contrast(ctrl, trt, lod=lod)
        tobit_err.append(fit.delta - true_delta)
        obs = trt[trt > 0]
        naive_err.append(np.log10(obs).mean() - np.log10(ctrl).mean() - true_delta)
    assert len(tobit_err) > 20
    # Discarding non-detects biases the estimate upward (too little depletion).
    assert np.mean(naive_err) > 0.01
    assert abs(np.mean(tobit_err)) < abs(np.mean(naive_err))


def test_tost_flags_equivalence_but_not_real_effects():
    margin = abs(np.log10(0.8))
    flat = tost_equivalence(np.array([0.001]), np.array([0.01]), np.array([50.0]), margin)
    real = tost_equivalence(np.array([-0.30]), np.array([0.01]), np.array([50.0]), margin)
    assert flat[0] < 0.05     # confidently within +/-20%
    assert real[0] > 0.05     # a genuine 50% depletion is not "equivalent"


def test_tost_is_not_fooled_by_noise():
    """A wide confidence interval must not be declared equivalent."""
    noisy = tost_equivalence(np.array([0.0]), np.array([0.5]), np.array([5.0]), abs(np.log10(0.8)))
    assert noisy[0] > 0.05
