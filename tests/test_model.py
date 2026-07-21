"""
Unit tests for pipeline/model.py -- synthetic data only (Phase 2, plan section 9).

These are NOT a re-run of the Phase 0 validation gate (that requires live data and is
closed, see research/regime-dashboard-plan.md section 7b). They exist to catch
regressions in the mechanics the gate depends on -- label continuity, NaN handling,
smoothing behavior -- using controlled synthetic inputs with known expected properties.
Two tests (test_drift_model_survives_leading_nan_regressor,
test_commit_regime_raises_on_all_nan_row_without_guard) are direct regressions for the
two real bugs found in the first live Phase 1 run (2026-07-21) -- see model.py's and
run_nightly.py's inline comments for the incident detail.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.model import (
    fit_hmm, forward_filter, walk_forward_direction, vol_layer,
    curve_conditioned_drift_posterior, commit_regime,
)


def make_regime_switching_returns(n=900, seed=7):
    """3-state regime-switching returns with well-separated means -- for tests that need
    the HMM to actually recover distinguishable states, not just run without crashing.
    """
    rng = np.random.RandomState(seed)
    mu = {0: -0.004, 1: 0.0002, 2: 0.004}   # deliberately wide apart, unlike real markets
    sg = {0: 0.02, 1: 0.01, 2: 0.01}
    state = 1
    rets = []
    for _ in range(n):
        if rng.rand() < 0.01:
            state = rng.choice([s for s in (0, 1, 2) if s != state])
        rets.append(rng.normal(mu[state], sg[state]))
    dates = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(rets, index=dates)


class TestFitHmmContinuity:
    def test_first_fit_orders_by_mean_no_prev(self):
        r = make_regime_switching_returns(700).values
        m, order, raw_means, separated, z_min = fit_hmm(r, prev_means_ordered=None)
        ordered_means = raw_means[order]
        assert list(ordered_means) == sorted(ordered_means), \
            "first fit (no continuity anchor) must order states bear->neutral->bull by mean"

    def test_continuity_matching_preserves_label_identity_across_refits(self):
        """The v1 bug this guards against: independently re-sorting each refit's raw
        means by value causes label swaps when two state means are close, splicing a
        sign-inverted effect into what should be a stable series. Continuity matching
        (nearest-previous-mean assignment) must keep each state's identity stable even
        when a later window's raw HMM component ordering differs from the first fit's.
        """
        r = make_regime_switching_returns(1100).values
        m1, order1, raw_means1, _, _ = fit_hmm(r[:700], prev_means_ordered=None)
        prev_means = raw_means1[order1]
        m2, order2, raw_means2, _, _ = fit_hmm(r[:900], prev_means_ordered=prev_means)
        new_means_ordered = raw_means2[order2]
        # The meaningful property isn't "means barely moved" (200 additional real data
        # points legitimately shift parameter estimates, especially with genuine regime
        # transitions in between) -- it's that each state matched to its OWN previous
        # mean more closely than to any OTHER state's previous mean. A label swap would
        # show up as a state being closer to a different prior state's mean than to its
        # own, which this checks directly instead of via an arbitrary magnitude bound.
        for i in range(3):
            dist_own = abs(new_means_ordered[i] - prev_means[i])
            dist_others = [abs(new_means_ordered[i] - prev_means[j]) for j in range(3) if j != i]
            assert dist_own <= min(dist_others), (
                f"state {i} ended up closer to a DIFFERENT state's previous mean than "
                f"its own -- label-swap regression"
            )

    def test_separated_flag_is_diagnostic_not_gating(self):
        """v2 regression guard: fit_hmm must always return a usable model/order
        regardless of the `separated` diagnostic value -- it must never raise or return
        None when separation is poor. walk_forward_direction is what must ALWAYS accept
        the fit (v3 fix); this test checks the primitive fit_hmm doesn't itself refuse.
        """
        rng = np.random.RandomState(3)
        r = rng.normal(0, 0.01, 700)  # no real regime structure -> poor separation
        m, order, raw_means, separated, z_min = fit_hmm(r)
        assert m is not None and order is not None
        assert isinstance(separated, bool)  # diagnostic value present, but didn't block


class TestWalkForwardDirection:
    def test_output_shape_and_no_lookahead_columns(self):
        r = make_regime_switching_returns(1100)
        dirpost, transition_mats, diagnostics = walk_forward_direction(
            r, min_train=500, refit_every=300,
        )
        assert list(dirpost.columns) == ["p_bear", "p_neut", "p_bull"]
        assert dirpost.index.is_monotonic_increasing
        # posteriors must sum to ~1 at every row (they're a probability distribution)
        row_sums = dirpost.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_never_freezes_on_poor_separation(self):
        """v2 regression guard, at the orchestration level: even with a return series
        that gives poor state separation throughout, walk_forward_direction must keep
        refitting on schedule (n_refit tracks every window), not freeze on the first fit.
        """
        rng = np.random.RandomState(9)
        n = 1100
        r = pd.Series(rng.normal(0, 0.01, n),
                       index=pd.bdate_range("2010-01-01", periods=n))
        dirpost, transition_mats, diagnostics = walk_forward_direction(
            r, min_train=500, refit_every=300,
        )
        expected_refits = (n - 500) // 300 + (1 if (n - 500) % 300 else 0)
        assert diagnostics["n_refit"] == expected_refits, (
            "n_refit should count every scheduled window regardless of separation "
            "quality -- a lower count would indicate the v2 freeze bug is back"
        )


class TestVolLayer:
    def test_output_in_unit_interval(self):
        n = 500
        idx = pd.bdate_range("2020-01-01", periods=n)
        rng = np.random.RandomState(1)
        vix_pct = pd.Series(rng.uniform(0, 1, n), index=idx)
        backward = pd.Series(rng.randint(0, 2, n).astype(float), index=idx)
        rv_pct = pd.Series(rng.uniform(0, 1, n), index=idx)
        weights = dict(vix_pct=1.2, backwardation=1.0, rv_pct=1.0, bias=-1.6)
        p = vol_layer(vix_pct, backward, rv_pct, weights)
        assert (p >= 0).all() and (p <= 1).all()

    def test_monotonic_in_each_input(self):
        """Higher vix_pct/backwardation/rv_pct should never DECREASE P(high vol) --
        the logistic combination's weights are all configured positive in config.yaml,
        so this should hold structurally, not just empirically."""
        idx = pd.bdate_range("2020-01-01", periods=2)
        weights = dict(vix_pct=1.2, backwardation=1.0, rv_pct=1.0, bias=-1.6)
        low = vol_layer(pd.Series([0.1, 0.1], index=idx), pd.Series([0.0, 0.0], index=idx),
                         pd.Series([0.1, 0.1], index=idx), weights)
        high = vol_layer(pd.Series([0.9, 0.9], index=idx), pd.Series([1.0, 1.0], index=idx),
                          pd.Series([0.9, 0.9], index=idx), weights)
        assert (high > low).all()


class TestCurveConditionedDrift:
    def test_drift_model_survives_leading_nan_regressor(self):
        """REGRESSION TEST for the real bug found in the first live Phase 1 run
        (2026-07-21): the NaN mask must exclude rows where X (slope_z) is NaN, not just
        where y (forward return) is NaN. A leading-NaN slope_z window (VX curve history
        not reaching as far back as price history) must NOT contaminate bn/Vn for the
        entire subsequent series -- before the fix, this zeroed out p_up/beta_slope for
        every single date, not just the leading window.
        """
        n = 1000
        idx = pd.bdate_range("2015-01-01", periods=n)
        rng = np.random.RandomState(2)
        slope_z = pd.Series(rng.normal(0, 1, n), index=idx)
        slope_z.iloc[:300] = np.nan  # simulates VX curve history starting later
        fwd = pd.Series(rng.normal(0.0005, 0.01, n), index=idx)
        fwd.iloc[-5:] = np.nan  # normal trailing NaN from the forward-return shift

        p_up, beta = curve_conditioned_drift_posterior(slope_z, fwd, min_train=252)
        assert p_up.notna().sum() > 0, (
            "drift model produced an entirely-NaN series -- the leading-NaN-regressor "
            "contamination bug has regressed"
        )
        # values after the contamination window clears should be valid probabilities
        valid = p_up.dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_predict_skips_when_todays_regressor_is_nan(self):
        """If the CURRENT date's slope_z is itself NaN (curve data lagging price data),
        the model must skip prediction for that date rather than propagate NaN silently
        through a matrix multiply that would otherwise still "succeed" with a NaN result.
        """
        n = 800
        idx = pd.bdate_range("2015-01-01", periods=n)
        rng = np.random.RandomState(4)
        slope_z = pd.Series(rng.normal(0, 1, n), index=idx)
        slope_z.iloc[-1] = np.nan  # today's regressor missing
        fwd = pd.Series(rng.normal(0.0005, 0.01, n), index=idx)
        fwd.iloc[-5:] = np.nan
        p_up, beta = curve_conditioned_drift_posterior(slope_z, fwd, min_train=252)
        assert pd.isna(p_up.iloc[-1]), "should skip prediction when today's regressor is NaN"


class TestCommitRegime:
    def test_all_nan_row_does_not_silently_produce_a_confident_wrong_label(self):
        """commit_regime assumes it's only ever handed complete rows (see model.py's
        module docstring) -- the real guard against this lives in run_nightly.py's
        cp.dropna(how="any") before commit_regime is ever called, not here. Pandas'
        own idxmax(axis=1) behavior on an all-NaN row is version-dependent: newer
        versions raise ValueError (what actually broke the first live Phase 1 run,
        2026-07-21); older versions return NaN with a deprecation warning instead of
        raising. This test accepts either behavior but requires that if it DOESN'T
        raise, the all-NaN row must not silently produce a different (wrong, falsely
        confident) committed regime than the prior day.
        """
        idx = pd.bdate_range("2020-01-01", periods=5)
        cells = ["bear_hi", "bear_lo", "neut_hi", "neut_lo", "bull_hi", "bull_lo"]
        cp = pd.DataFrame(0.1, index=idx, columns=cells)
        cp.iloc[2] = np.nan  # one fully-NaN row, mid-series
        try:
            committed = commit_regime(cp)
        except ValueError:
            return  # current-pandas behavior -- exactly the documented assumption boundary
        assert committed.iloc[2] == committed.iloc[1], (
            "an all-NaN row degraded to NaN (not a raise) but still produced a "
            "DIFFERENT committed regime than the prior day -- should carry forward "
            "unchanged, not manufacture a confident label from missing data"
        )

    def test_commitment_requires_consecutive_days_above_threshold(self):
        """Anti-flip-flop smoothing (plan section 3.3): a new regime must show
        posterior > commit_p for commit_days consecutive days AND clear min_dwell since
        the last switch before the committed regime actually changes.
        """
        idx = pd.bdate_range("2020-01-01", periods=10)
        cells = ["bear_hi", "bear_lo", "neut_hi", "neut_lo", "bull_hi", "bull_lo"]
        cp = pd.DataFrame(0.05, index=idx, columns=cells)
        cp["neut_lo"] = 0.75  # start firmly in neut_lo
        # day 5 (index 5): a SINGLE day spike in bull_hi above commit_p -- should NOT
        # be enough to switch (commit_days=2 requires 2 consecutive days)
        cp.loc[idx[5], :] = 0.05
        cp.loc[idx[5], "bull_hi"] = 0.85
        committed = commit_regime(cp, commit_p=0.70, commit_days=2, min_dwell=3)
        assert committed.iloc[5] == "neut_lo", (
            "a single day above commit_p should not be enough to switch regimes "
            "(commit_days=2 requires 2 consecutive days) -- anti-flip-flop regression"
        )

    def test_commitment_switches_after_sustained_signal_and_dwell(self):
        idx = pd.bdate_range("2020-01-01", periods=15)
        cells = ["bear_hi", "bear_lo", "neut_hi", "neut_lo", "bull_hi", "bull_lo"]
        cp = pd.DataFrame(0.05, index=idx, columns=cells)
        cp["neut_lo"] = 0.75
        # from day 5 onward, bull_hi is sustained and dominant -- should eventually commit
        cp.loc[idx[5]:, :] = 0.05
        cp.loc[idx[5]:, "bull_hi"] = 0.85
        committed = commit_regime(cp, commit_p=0.70, commit_days=2, min_dwell=3)
        assert committed.iloc[-1] == "bull_hi", (
            "a sustained, dominant signal should eventually commit after "
            "commit_days + min_dwell -- smoothing logic regression"
        )
