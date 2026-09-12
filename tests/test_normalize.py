"""Tests for the control-vs-treatment offset estimator."""
import numpy as np
import pytest

from octant_cyp import normalize


def _mixture(rng, n_inactive, n_active, offset, seed_sd=0.15):
    """Inactive lobe at `offset`, plus a broad active lobe well below it."""
    inactive = rng.normal(offset, seed_sd, n_inactive)
    active = rng.normal(-5.0, 2.5, n_active)
    return np.concatenate([inactive, active])


def test_recovers_known_offset_when_inactives_dominate():
    rng = np.random.default_rng(0)
    x = _mixture(rng, 1000, 200, offset=0.25)
    assert normalize.inactive_mode(x) == pytest.approx(0.25, abs=0.05)


def test_recovers_known_offset_when_actives_dominate():
    """The CYP3A4 regime: the inactive lobe is a minority but still present."""
    rng = np.random.default_rng(1)
    x = _mixture(rng, 300, 900, offset=0.25)
    assert normalize.inactive_mode(x) == pytest.approx(0.25, abs=0.07)


def test_bandwidth_is_absolute_not_relative():
    """Regression: a shared *relative* bandwidth erases the minority-inactive peak.

    scipy's bw_method multiplies the sample standard deviation, so a
    widely-dispersed distribution gets smoothed far harder.  Two datasets with
    the same offset but different spreads must still return the same mode --
    this is what originally made CYP3A4 look as though it had no inactive mode.
    """
    rng = np.random.default_rng(2)
    # Chosen to reproduce the real spread ratio: CYP2J2 sd 2.01, CYP3A4 sd 5.25.
    narrow = _mixture(rng, 1150, 70, offset=0.25)    # mostly inactive
    wide = _mixture(rng, 300, 920, offset=0.25)      # mostly active
    assert np.std(wide) > 2.0 * np.std(narrow)
    m_narrow = normalize.inactive_mode(narrow)
    m_wide = normalize.inactive_mode(wide)
    assert abs(m_narrow - m_wide) < 0.08


def test_mode_estimate_is_stable_across_bandwidths():
    rng = np.random.default_rng(3)
    x = _mixture(rng, 400, 800, offset=0.25)
    modes = [normalize.inactive_mode(x, bw=b) for b in (0.05, 0.08, 0.12, 0.15)]
    assert max(modes) - min(modes) < 0.08


def test_separation_flags_absence_of_an_inactive_population():
    """A single broad active lobe has no separable null to anchor on."""
    rng = np.random.default_rng(4)
    x = rng.normal(-5.0, 2.5, 1200)
    sep = normalize.mode_separation(x)
    assert not (np.isfinite(sep) and sep > normalize.MIN_MODE_SEPARATION)


def test_apply_offset_shifts_by_the_right_amount():
    delta = np.array([0.0, -1.0])
    out = normalize.apply_offset(delta, offset_log2=1.0)  # 1 log2 = log10(2)
    assert out[0] == pytest.approx(-np.log10(2))
    assert out[1] == pytest.approx(-1.0 - np.log10(2))


def test_bootstrap_brackets_the_point_estimate():
    rng = np.random.default_rng(5)
    x = _mixture(rng, 900, 300, offset=0.25)
    b = normalize.bootstrap_mode(x, n_boot=120, seed=1)
    assert b["ci_low"] <= b["mode"] <= b["ci_high"]
    assert b["ci_low"] < 0.25 < b["ci_high"]
