"""
Unit tests for pipeline/forecast.py (per-name forecast-density logic, now precomputed
nightly for all 50 names -- see run_nightly.py's forecast-density block).

forward_returns/conditional_vs_unconditional_density tests moved here from
tests/test_app_analytics.py when the logic itself moved from app/analytics.py into
pipeline/forecast.py (2026-07-22) -- testing the source of truth directly rather than
through app.analytics's re-export.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.forecast import (
    forward_returns, conditional_vs_unconditional_density, binned_density,
    compute_name_forecast_density,
)


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
        # is. This is how the function is actually used (against name_cell_history,
        # itself derived from the sticky market/tilt gating).
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


class TestBinnedDensity:
    def test_counts_sum_to_input_length(self):
        values = pd.Series(np.random.RandomState(1).normal(0, 1, 200))
        edges = np.linspace(values.min(), values.max(), 11)
        counts = binned_density(values, edges)
        assert len(counts) == 10
        assert sum(counts) == 200

    def test_values_outside_edges_are_excluded_not_erroring(self):
        values = pd.Series([-100, 0, 0, 0, 100])
        edges = np.linspace(-1, 1, 5)
        counts = binned_density(values, edges)
        assert sum(counts) == 3  # the three zeros; -100/100 fall outside the edges


class TestComputeNameForecastDensity:
    def _sticky_series(self, seed=11, run_length=15):
        rng = np.random.RandomState(seed)
        run_labels = (["bull_lo", "bear_hi", "neut_lo"] * 20)[:60]
        rng.shuffle(run_labels)
        cells_list, ret_list = [], []
        for label in run_labels:
            drift = 0.004 if label == "bull_lo" else 0.0
            ret_list.append(rng.normal(0, 0.004, run_length) + drift)
            cells_list.append([label] * run_length)
        idx = pd.bdate_range("2020-01-01", periods=run_length * len(run_labels))
        cells = pd.Series(np.concatenate(cells_list), index=idx)
        price = pd.Series(100 * np.exp(np.cumsum(np.concatenate(ret_list))), index=idx)
        return price, cells

    def test_full_payload_shape_for_sufficient_data(self):
        price, cells = self._sticky_series()
        result = compute_name_forecast_density(price, cells, "bull_lo", horizon=3, n_bins=10)
        assert result["insufficient_data"] is False
        assert len(result["bin_edges"]) == 11
        assert len(result["conditional_hist"]) == 10
        assert len(result["unconditional_hist"]) == 10
        # both histograms built on the SAME bin edges -- directly comparable shapes
        assert sum(result["conditional_hist"]) == result["n_conditional"]
        assert sum(result["unconditional_hist"]) == result["n_unconditional"]

    def test_insufficient_data_returns_null_histograms_not_a_crash(self):
        idx = pd.bdate_range("2020-01-01", periods=30)
        price = pd.Series(100 * np.exp(np.cumsum(np.full(30, 0.001))), index=idx)
        cells = pd.Series(["bull_lo"] * 5 + ["bear_hi"] * 25, index=idx)
        result = compute_name_forecast_density(price, cells, "bull_lo", horizon=2)
        assert result["insufficient_data"] is True
        assert result["bin_edges"] is None
        assert result["conditional_hist"] is None
        assert result["n_conditional"] == 5  # diagnostic count still present

    def test_degenerate_zero_variance_series_does_not_crash(self):
        idx = pd.bdate_range("2020-01-01", periods=100)
        price = pd.Series([100.0] * 100, index=idx)  # flat price -- all forward returns are 0
        cells = pd.Series(["bull_lo"] * 100, index=idx)
        result = compute_name_forecast_density(price, cells, "bull_lo", horizon=3, n_bins=10)
        assert result["insufficient_data"] is False
        assert result["bin_edges"][0] < result["bin_edges"][-1]  # padded, not zero-width
