"""
Unit tests for pipeline/iv_calc.py (plan section 5 step 3).

Formalizes the ad-hoc checks run manually before the IV snapshot's first live
confirmation (2026-07-21) into a real, repeatable test file.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from pipeline.iv_calc import bs_delta, dte, select_expiries, atm_iv_from_chain, skew_25d_from_chain


class TestDteAndSelectExpiries:
    def test_dte_counts_calendar_days(self):
        today = date(2026, 7, 21)
        assert dte("2026-08-07", today) == 17

    def test_picks_near_and_far_expiries_correctly(self):
        today = date(2026, 7, 21)
        available = [str(today + timedelta(days=d)) for d in [7, 17, 32, 38, 45, 60, 90]]
        near, far = select_expiries(available, as_of=today, near_dte_target=17, far_dte_range=(30, 45))
        assert dte(near, today) == 17
        assert 30 <= dte(far, today) <= 45

    def test_falls_back_to_closest_when_nothing_in_target_range(self):
        """If no expiry falls in the 30-45 DTE window (thin/monthly-only chain), fall
        back to the closest available (by distance to the range midpoint, across ALL
        available expiries -- not excluding whichever one 'near' picked) rather than
        returning None. The caller can still get a usable (if imperfect) snapshot
        instead of skipping the name."""
        today = date(2026, 7, 21)
        available = [str(today + timedelta(days=d)) for d in [10, 60, 90]]  # nothing in 30-45
        near, far = select_expiries(available, as_of=today, near_dte_target=17, far_dte_range=(30, 45))
        assert far is not None
        assert dte(far, today) == 60  # closest to the 37.5 midpoint among what's available

    def test_returns_none_none_for_empty_list(self):
        near, far = select_expiries([], as_of=date(2026, 7, 21))
        assert near is None and far is None

    def test_ignores_already_expired_dates(self):
        today = date(2026, 7, 21)
        available = [str(today - timedelta(days=5)), str(today + timedelta(days=38))]
        near, far = select_expiries(available, as_of=today, near_dte_target=17, far_dte_range=(30, 45))
        assert dte(far, today) == 38


class TestBsDelta:
    def test_atm_call_delta_near_half(self):
        d = bs_delta(100, 100, 30 / 365, 0.25, 0.03, is_call=True)
        assert 0.45 < d < 0.60

    def test_deep_itm_call_approaches_one(self):
        d = bs_delta(100, 60, 30 / 365, 0.25, 0.03, is_call=True)
        assert d > 0.95

    def test_deep_otm_call_approaches_zero(self):
        d = bs_delta(100, 160, 30 / 365, 0.25, 0.03, is_call=True)
        assert d < 0.05

    def test_atm_put_delta_near_negative_half(self):
        d = bs_delta(100, 100, 30 / 365, 0.25, 0.03, is_call=False)
        assert -0.55 < d < -0.40

    def test_zero_or_negative_time_returns_nan(self):
        assert np.isnan(bs_delta(100, 100, 0, 0.25, 0.03, is_call=True))


def _synthetic_smile(strikes, S=100, skew_toward_puts=True):
    def iv(K):
        m = np.log(K / S)
        base = 0.22 + 0.9 * m ** 2
        return base - 0.35 * m if skew_toward_puts else base + 0.35 * m
    return pd.DataFrame({"strike": strikes, "impliedVolatility": [iv(k) for k in strikes]})


class TestAtmIvFromChain:
    def test_returns_value_near_the_true_atm_iv(self):
        strikes = np.arange(60, 145, 2.5)
        calls = _synthetic_smile(strikes)
        puts = _synthetic_smile(strikes)
        atm = atm_iv_from_chain(calls, puts, spot=100)
        assert abs(atm - 0.22) < 0.02

    def test_none_on_empty_chain(self):
        assert atm_iv_from_chain(pd.DataFrame(), pd.DataFrame(), spot=100) is None
        assert atm_iv_from_chain(None, None, spot=100) is None


class TestSkew25dFromChain:
    def test_positive_skew_when_smile_favors_puts(self):
        strikes = np.arange(50, 160, 2.5)
        calls = _synthetic_smile(strikes, skew_toward_puts=True)
        puts = _synthetic_smile(strikes, skew_toward_puts=True)
        skew = skew_25d_from_chain(calls, puts, spot=100, T=37 / 365, r=0.03)
        assert skew > 0, "OTM puts richer than OTM calls should give a positive skew reading"

    def test_negative_skew_when_smile_favors_calls(self):
        strikes = np.arange(50, 160, 2.5)
        calls = _synthetic_smile(strikes, skew_toward_puts=False)
        puts = _synthetic_smile(strikes, skew_toward_puts=False)
        skew = skew_25d_from_chain(calls, puts, spot=100, T=37 / 365, r=0.03)
        assert skew < 0

    def test_none_when_ivs_are_all_zero(self):
        strikes = np.arange(50, 160, 2.5)
        calls = pd.DataFrame({"strike": strikes, "impliedVolatility": np.zeros(len(strikes))})
        puts = pd.DataFrame({"strike": strikes, "impliedVolatility": np.zeros(len(strikes))})
        assert skew_25d_from_chain(calls, puts, spot=100, T=37 / 365, r=0.03) is None
