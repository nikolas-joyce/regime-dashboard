"""
Unit tests for pipeline/matrix.py (regime -> structure lookup, Black-Scholes pricing).
"""
import numpy as np
import pytest

from pipeline.matrix import bs_price, strike_from_delta, confidence_tier, recommend_structure, price_structure


class TestBsPrice:
    def test_put_call_parity(self):
        S, K, T, sig, r = 100, 100, 30 / 365, 0.25, 0.03
        call = bs_price(S, K, T, sig, r, 1)
        put = bs_price(S, K, T, sig, r, -1)
        # C - P = S - K*e^(-rT)
        lhs = call - put
        rhs = S - K * np.exp(-r * T)
        assert abs(lhs - rhs) < 1e-8

    def test_expired_option_is_intrinsic_value(self):
        assert bs_price(110, 100, 0, 0.25, 0.03, 1) == 10  # ITM call, no time value
        assert bs_price(90, 100, 0, 0.25, 0.03, 1) == 0    # OTM call, no time value
        assert bs_price(90, 100, 0, 0.25, 0.03, -1) == 10  # ITM put


class TestStrikeFromDelta:
    def test_call_delta_050_is_close_to_spot(self):
        S, T, sig, r = 100, 30 / 365, 0.25, 0.03
        K = strike_from_delta(S, T, sig, r, 0.50, is_call=True)
        assert abs(K - S) < 5, "0.50-delta call strike should be close to spot"

    def test_higher_call_delta_means_lower_strike(self):
        S, T, sig, r = 100, 30 / 365, 0.25, 0.03
        k_high_delta = strike_from_delta(S, T, sig, r, 0.70, is_call=True)  # more ITM
        k_low_delta = strike_from_delta(S, T, sig, r, 0.30, is_call=True)   # more OTM
        assert k_high_delta < k_low_delta


class TestConfidenceTier:
    def test_bands(self):
        bands = {"high": 0.75, "moderate": 0.55}
        assert confidence_tier(0.90, bands) == "high_confidence"
        assert confidence_tier(0.75, bands) == "high_confidence"  # boundary inclusive
        assert confidence_tier(0.60, bands) == "moderate_confidence"
        assert confidence_tier(0.55, bands) == "moderate_confidence"  # boundary inclusive
        assert confidence_tier(0.40, bands) == "low_confidence"


class TestRecommendStructure:
    def test_looks_up_correct_tier(self):
        matrix = {
            "bull_hi": {
                "high_confidence": {"structure": "short_put", "legs": []},
                "moderate_confidence": {"structure": "bull_put_credit_spread", "legs": []},
                "low_confidence": {"structure": "no_trade", "legs": []},
            }
        }
        bands = {"high": 0.75, "moderate": 0.55}
        assert recommend_structure("bull_hi", 0.90, matrix, bands)["structure"] == "short_put"
        assert recommend_structure("bull_hi", 0.60, matrix, bands)["structure"] == "bull_put_credit_spread"
        assert recommend_structure("bull_hi", 0.30, matrix, bands)["structure"] == "no_trade"

    def test_unknown_cell_raises_keyerror(self):
        matrix = {"bull_hi": {"high_confidence": {}, "moderate_confidence": {}, "low_confidence": {}}}
        with pytest.raises(KeyError):
            recommend_structure("not_a_real_cell", 0.9, matrix, {"high": 0.75, "moderate": 0.55})


class TestPriceStructure:
    def test_no_trade_prices_to_zero(self):
        p = price_structure([], S0=100, ST=105, T0=21 / 365, sigma=0.25, r=0.03)
        assert p == 0.0

    def test_short_put_profits_when_price_stays_flat(self):
        legs = [{"cp": "put", "delta": -0.30, "pos": -1}]
        p = price_structure(legs, S0=100, ST=101, T0=21 / 365, sigma=0.25, r=0.03)
        assert p > 0, "a short put should profit (premium decays) when the underlying is flat/up"

    def test_short_put_loses_on_a_large_drop(self):
        legs = [{"cp": "put", "delta": -0.30, "pos": -1}]
        p = price_structure(legs, S0=100, ST=70, T0=21 / 365, sigma=0.25, r=0.03)
        assert p < 0, "a short put should lose money on a large drop in the underlying"

    def test_bull_call_debit_spread_has_defined_max_loss_and_gain(self):
        legs = [{"cp": "call", "delta": 0.55, "pos": 1}, {"cp": "call", "delta": 0.30, "pos": -1}]
        p_up = price_structure(legs, S0=100, ST=140, T0=21 / 365, sigma=0.25, r=0.03)
        p_down = price_structure(legs, S0=100, ST=60, T0=21 / 365, sigma=0.25, r=0.03)
        # max gain (spread width) should be modest relative to spot, and max loss bounded
        assert -0.10 < p_down < 0
        assert 0 < p_up < 0.10
