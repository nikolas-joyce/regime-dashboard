"""
Unit tests for pipeline/tilt.py (per-name tilt layer, plan section 3.5).

The gating logic (gate_direction_by_market / compute_name_cell) is the highest-stakes
piece here -- it's what stops a name from getting recommended a directionally opposite
structure from the market regime. Manually verified against two live Actions runs
(2026-07-21) already; these tests lock that behavior down so it can't silently regress.
"""
import numpy as np
import pandas as pd

from pipeline.tilt import (
    relative_strength_zscore, direction_tilt, name_vol_state,
    gate_direction_by_market, compute_name_cell,
)


class TestRelativeStrengthZscore:
    """relative_strength_zscore measures deviation from its OWN trailing baseline, not
    absolute over/underperformance level -- a name with a constant excess drift for the
    whole sample gets absorbed into its own rolling mean and should hover near z=0 once
    past warm-up (that's by design: the tilt layer is meant to catch relative-strength
    SHIFTS, not steady-state tilt). These tests exercise that shift-detection property
    directly, using a name that's in-line with the market for a while and then breaks
    into sustained out/underperformance partway through.
    """

    def test_new_outperformance_shows_positive_z_after_the_shift(self):
        n = 600
        idx = pd.bdate_range("2020-01-01", periods=n)
        rng = np.random.RandomState(1)
        market_ret = pd.Series(rng.normal(0.0003, 0.01, n), index=idx)
        excess = np.concatenate([np.zeros(400), np.full(200, 0.0015)])  # shift at day 400
        name_ret = market_ret + pd.Series(rng.normal(0, 0.002, n), index=idx) + excess
        z = relative_strength_zscore(name_ret, market_ret, window=21, z_window=252)
        # Relative comparison (post-shift vs. a pre-shift baseline), not an absolute
        # magnitude threshold -- a genuine within-window regime shift also inflates the
        # rolling std used as z's denominator, self-normalizing the magnitude down even
        # when the direction/timing is exactly right.
        z_pre = z.iloc[350:390].mean()
        z_post = z.iloc[420:460].mean()
        assert z_post > z_pre + 0.2, (
            f"post-shift z ({z_post:.3f}) should be clearly higher than the pre-shift "
            f"baseline ({z_pre:.3f}) for a name that just broke into outperformance"
        )

    def test_new_underperformance_shows_negative_z_after_the_shift(self):
        n = 600
        idx = pd.bdate_range("2020-01-01", periods=n)
        rng = np.random.RandomState(2)
        market_ret = pd.Series(rng.normal(0.0003, 0.01, n), index=idx)
        excess = np.concatenate([np.zeros(400), np.full(200, -0.0015)])
        name_ret = market_ret + pd.Series(rng.normal(0, 0.002, n), index=idx) + excess
        z = relative_strength_zscore(name_ret, market_ret, window=21, z_window=252)
        z_pre = z.iloc[350:390].mean()
        z_post = z.iloc[420:460].mean()
        assert z_post < z_pre - 0.2, (
            f"post-shift z ({z_post:.3f}) should be clearly lower than the pre-shift "
            f"baseline ({z_pre:.3f}) for a name that just broke into underperformance"
        )


class TestDirectionTilt:
    def test_thresholds_are_respected(self):
        idx = pd.bdate_range("2020-01-01", periods=5)
        rs_short = pd.Series([0.8, -0.8, 0.1, np.nan, 0.6], index=idx)
        rs_long = pd.Series([0.8, -0.8, -0.1, 0.5, 0.4], index=idx)
        tilt = direction_tilt(rs_short, rs_long, bull_thresh=0.5, bear_thresh=-0.5)
        assert tilt.iloc[0] == "bull"   # avg 0.8 > 0.5
        assert tilt.iloc[1] == "bear"   # avg -0.8 < -0.5
        assert tilt.iloc[2] == "neut"   # avg 0.0, within band
        assert tilt.iloc[3] is None     # one input NaN -> undefined, not silently "neut"
        assert tilt.iloc[4] == "neut"   # avg 0.5, at the boundary (not > 0.5)


class TestNameVolState:
    def test_high_low_split_at_threshold(self):
        n = 400
        idx = pd.bdate_range("2020-01-01", periods=n)
        # constructed so the back half is clearly higher vol than the front half
        rv = pd.Series(np.concatenate([np.full(200, 0.10), np.full(200, 0.40)]), index=idx)
        state = name_vol_state(rv, pct_window=252, pct_thresh=0.60)
        assert state.iloc[-1] == "hi", "sustained higher realized vol should land in the hi bucket"


class TestGateDirectionByMarket:
    def test_bearish_market_caps_bull_tilt_at_neut(self):
        idx = pd.bdate_range("2020-01-01", periods=4)
        market_dir = pd.Series("bear", index=idx)
        name_tilt = pd.Series(["bull", "neut", "bear", "bull"], index=idx)
        gated = gate_direction_by_market(market_dir, name_tilt)
        assert list(gated) == ["neut", "neut", "bear", "neut"], (
            "every bull tilt under a bearish market must be capped to neut -- "
            "the exact gating rule verified against live data on 2026-07-21"
        )

    def test_bullish_market_caps_bear_tilt_at_neut(self):
        idx = pd.bdate_range("2020-01-01", periods=4)
        market_dir = pd.Series("bull", index=idx)
        name_tilt = pd.Series(["bull", "neut", "bear", "bear"], index=idx)
        gated = gate_direction_by_market(market_dir, name_tilt)
        assert list(gated) == ["bull", "neut", "neut", "neut"]

    def test_neutral_market_imposes_no_cap(self):
        idx = pd.bdate_range("2020-01-01", periods=3)
        market_dir = pd.Series("neut", index=idx)
        name_tilt = pd.Series(["bull", "neut", "bear"], index=idx)
        gated = gate_direction_by_market(market_dir, name_tilt)
        assert list(gated) == ["bull", "neut", "bear"], \
            "a neutral market should not cap the name's own tilt in either direction"

    def test_no_bull_leak_through_across_a_large_random_sample(self):
        """Statistical smoke test mirroring the manual verification done against live
        Actions output: across many random (market_direction, name_tilt) pairs, a
        bearish market must NEVER let a bull-tilted name through as bull, and a bullish
        market must NEVER let a bear-tilted name through as bear.
        """
        rng = np.random.RandomState(5)
        n = 2000
        idx = pd.bdate_range("2020-01-01", periods=n)
        market_dir = pd.Series(rng.choice(["bear", "neut", "bull"], n), index=idx)
        name_tilt = pd.Series(rng.choice(["bear", "neut", "bull"], n), index=idx)
        gated = gate_direction_by_market(market_dir, name_tilt)
        bad_bull_leak = ((market_dir == "bear") & (gated == "bull")).sum()
        bad_bear_leak = ((market_dir == "bull") & (gated == "bear")).sum()
        assert bad_bull_leak == 0
        assert bad_bear_leak == 0


class TestComputeNameCell:
    def test_end_to_end_cell_construction_and_gating(self):
        idx = pd.bdate_range("2020-01-01", periods=3)
        market_committed = pd.Series(["bear_hi", "bull_lo", "neut_hi"], index=idx)
        name_tilt = pd.Series(["bull", "bear", "bull"], index=idx)
        vol_state = pd.Series(["lo", "hi", "lo"], index=idx)
        cell = compute_name_cell(market_committed, name_tilt, vol_state)
        # row 0: market bear -> bull tilt capped to neut; vol lo -> "neut_lo"
        # row 1: market bull -> bear tilt capped to neut; vol hi -> "neut_hi"
        # row 2: market neut -> no cap; bull tilt passes; vol lo -> "bull_lo"
        assert list(cell) == ["neut_lo", "neut_hi", "bull_lo"]

    def test_null_propagates_when_either_input_is_missing(self):
        idx = pd.bdate_range("2020-01-01", periods=2)
        market_committed = pd.Series(["bull_lo", "bull_lo"], index=idx)
        name_tilt = pd.Series([None, "bull"], index=idx)
        vol_state = pd.Series(["lo", None], index=idx)
        cell = compute_name_cell(market_committed, name_tilt, vol_state)
        assert cell.iloc[0] is None or pd.isna(cell.iloc[0])
        assert cell.iloc[1] is None or pd.isna(cell.iloc[1])
