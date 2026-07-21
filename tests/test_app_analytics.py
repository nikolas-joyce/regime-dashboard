"""
Unit tests for app/analytics.py (Phase 3 dashboard logic).

Kept in the same tests/ directory and pytest run as the pipeline tests, even though this
module lives under app/ -- it's still pure, testable logic split out from Streamlit
presentation code for exactly this reason.
"""
import numpy as np
import pandas as pd
import pytest

from app.analytics import (
    empirical_transition_matrix, next_day_probs, exit_probability, regime_runs,
    days_in_current_regime, forward_returns, conditional_vs_unconditional_density,
    structure_terms, call_history_log, CELLS,
)


def _committed_series(sequence: list[str], start="2020-01-01") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(sequence))
    return pd.Series(sequence, index=idx)


class TestEmpiricalTransitionMatrix:
    def test_sticky_regime_has_high_self_transition(self):
        # 3 runs of 20 days each in bull_lo, separated by single-day bear_hi blips
        seq = ["bull_lo"] * 20 + ["bear_hi"] + ["bull_lo"] * 20 + ["bear_hi"] + ["bull_lo"] * 20
        committed = _committed_series(seq)
        m = empirical_transition_matrix(committed)
        assert m.loc["bull_lo", "bull_lo"] > 0.9

    def test_rows_sum_to_one_or_zero(self):
        seq = ["bull_lo"] * 10 + ["bear_hi"] * 10
        committed = _committed_series(seq)
        m = empirical_transition_matrix(committed)
        row_sums = m.sum(axis=1)
        for cell in CELLS:
            assert row_sums[cell] == pytest.approx(1.0) or row_sums[cell] == 0.0

    def test_never_observed_cell_gives_zero_row_not_crash(self):
        seq = ["bull_lo"] * 5
        committed = _committed_series(seq)
        m = empirical_transition_matrix(committed)
        assert (m.loc["bear_hi"] == 0.0).all()


class TestNextDayProbsAndExit:
    def test_exit_probability_low_for_sticky_regime(self):
        seq = ["bull_lo"] * 50 + ["bear_hi"]
        committed = _committed_series(seq)
        m = empirical_transition_matrix(committed)
        assert exit_probability("bull_lo", m) < 0.1

    def test_unknown_current_cell_returns_all_zero(self):
        committed = _committed_series(["bull_lo"] * 10)
        m = empirical_transition_matrix(committed)
        probs = next_day_probs("not_a_cell", m)
        assert (probs == 0.0).all()


class TestRegimeRuns:
    def test_collapses_contiguous_runs_correctly(self):
        seq = ["bull_lo"] * 3 + ["bear_hi"] * 5 + ["bull_lo"] * 2
        committed = _committed_series(seq)
        runs = regime_runs(committed)
        assert list(runs["regime"]) == ["bull_lo", "bear_hi", "bull_lo"]
        assert list(runs["duration_days"]) == [3, 5, 2]

    def test_empty_series_returns_empty_frame(self):
        runs = regime_runs(pd.Series(dtype=object))
        assert runs.empty

    def test_days_in_current_regime_counts_the_last_run(self):
        seq = ["bear_hi"] * 4 + ["bull_lo"] * 7
        committed = _committed_series(seq)
        assert days_in_current_regime(committed) == 7


class TestForwardReturns:
    def test_horizon_shift_matches_manual_calc(self):
        idx = pd.bdate_range("2020-01-01", periods=10)
        price = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], index=idx)
        fwd = forward_returns(price, horizon=2)
        expected_first = np.log(102 / 100)
        assert fwd.iloc[0] == pytest.approx(expected_first)
        # last 2 obs can't have a matured 2-day-forward return
        assert len(fwd) == 8


class TestConditionalVsUnconditionalDensity:
    def test_insufficient_data_flagged_below_20_obs(self):
        idx = pd.bdate_range("2020-01-01", periods=30)
        price = pd.Series(100 * np.exp(np.cumsum(np.full(30, 0.001))), index=idx)
        cells = pd.Series(["bull_lo"] * 5 + ["bear_hi"] * 25, index=idx)
        result = conditional_vs_unconditional_density(price, cells, "bull_lo", horizon=2)
        assert result["insufficient_data"] is True

    def test_detects_a_real_shift_in_conditional_mean(self):
        # Sticky multi-day runs (each >> horizon), mirroring real committed_regime
        # behavior under min_dwell smoothing -- an IID-per-day cell label would make a
        # 3-day-forward window mostly NOT overlap the origin day's own cell, drowning
        # the conditioning signal in noise regardless of how real the underlying effect
        # is. This is how conditional_vs_unconditional_density is actually used (against
        # name_cell_history, itself derived from the sticky market/tilt gating).
        rng = np.random.RandomState(11)
        run_labels = (["bull_lo", "bear_hi", "neut_lo"] * 20)[:60]
        rng.shuffle(run_labels)
        run_length = 15
        cells_list, ret_list = [], []
        for label in run_labels:
            drift = 0.004 if label == "bull_lo" else 0.0
            ret_list.append(rng.normal(0, 0.004, run_length) + drift)
            cells_list.append([label] * run_length)
        idx = pd.bdate_range("2020-01-01", periods=run_length * len(run_labels))
        cells = pd.Series(np.concatenate(cells_list), index=idx)
        price = pd.Series(100 * np.exp(np.cumsum(np.concatenate(ret_list))), index=idx)

        result = conditional_vs_unconditional_density(price, cells, "bull_lo", horizon=3)
        assert result["insufficient_data"] is False
        assert result["conditional_mean"] > result["unconditional_mean"]
        assert result["effect_size_sd"] > 0.2


class TestStructureTerms:
    def test_no_trade_is_all_zero(self):
        t = structure_terms([], S0=100, sigma=0.25, r=0.03, T0=30 / 365)
        assert t["max_gain"] == 0.0 and t["max_loss"] == 0.0 and t["breakevens"] == []

    def test_debit_spread_max_loss_equals_debit_paid(self):
        legs = [{"cp": "call", "delta": 0.55, "pos": 1}, {"cp": "call", "delta": 0.30, "pos": -1}]
        t = structure_terms(legs, S0=100, sigma=0.25, r=0.03, T0=35 / 365)
        assert t["net_credit_debit"] < 0  # debit
        assert t["max_loss"] == pytest.approx(t["net_credit_debit"], abs=0.05)

    def test_short_put_max_gain_equals_credit_received(self):
        legs = [{"cp": "put", "delta": -0.30, "pos": -1}]
        t = structure_terms(legs, S0=100, sigma=0.25, r=0.03, T0=35 / 365)
        assert t["net_credit_debit"] > 0  # credit
        assert t["max_gain"] == pytest.approx(t["net_credit_debit"], abs=0.05)

    def test_debit_spread_has_one_breakeven_between_strikes(self):
        legs = [{"cp": "call", "delta": 0.55, "pos": 1}, {"cp": "call", "delta": 0.30, "pos": -1}]
        t = structure_terms(legs, S0=100, sigma=0.25, r=0.03, T0=35 / 365)
        assert len(t["breakevens"]) == 1
        k_low = min(l["strike"] for l in t["legs"])
        k_high = max(l["strike"] for l in t["legs"])
        assert k_low < t["breakevens"][0] < k_high


class TestCallHistoryLog:
    def test_recent_unmatured_rows_have_nan_forward_return(self):
        idx = pd.bdate_range("2020-01-01", periods=20)
        price = pd.Series(100 + np.arange(20) * 0.5, index=idx)
        cells = pd.Series(["bull_lo"] * 20, index=idx)
        log = call_history_log(price, cells, horizon=5)
        most_recent_date = idx[-1]
        assert pd.isna(log.loc[most_recent_date, "fwd_return"])

    def test_matured_rows_have_a_real_forward_return(self):
        idx = pd.bdate_range("2020-01-01", periods=20)
        price = pd.Series(100 + np.arange(20) * 0.5, index=idx)
        cells = pd.Series(["bull_lo"] * 20, index=idx)
        log = call_history_log(price, cells, horizon=5)
        earliest_date = idx[0]
        assert not pd.isna(log.loc[earliest_date, "fwd_return"])
