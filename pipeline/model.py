"""
Regime dashboard — market-layer model.

Ported from the validated step0 notebook (research/regime_dashboard_step0_validation.ipynb)
after six-plus rounds of live validation. Do not "simplify" the continuity-matching or
separation-diagnostic logic below without understanding why each piece exists -- both the
label-continuity bug (v1) and the separation-gate freeze bug (v2) were real, production-
breaking failures that this code specifically guards against. See
research/regime-dashboard-plan.md section 7b for the full history.

This module intentionally has NO data-acquisition code (see data_pull.py) and NO plotting --
pure, testable functions operating on pandas Series/DataFrames the caller supplies.
"""
from __future__ import annotations

from itertools import permutations
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm, t as student_t


# ---------------------------------------------------------------------------
# Direction engine: continuity-matched 3-state Gaussian HMM
# ---------------------------------------------------------------------------

def fit_hmm(
    r: np.ndarray,
    prev_means_ordered: Optional[np.ndarray] = None,
    n_states: int = 3,
    random_state: int = 7,
    min_covar: float = 1e-6,
) -> tuple[GaussianHMM, np.ndarray, np.ndarray, bool, float]:
    """Fit a Gaussian HMM and resolve state labels via continuity, not a fresh sort.

    v1 bug: sorting each refit's states independently by raw mean caused label swaps
    across refits when two state means were close (common -- regime mean gaps are
    inherently ~10x smaller than return vol). That spliced a real signal into
    backwards-looking noise (sign-inverted 21d forward-return effects).

    Fix: match each new fit's states to the PREVIOUS fit's canonical means via nearest
    assignment (permutation search over n_states!, cheap for n_states=3), so "bull"
    stays "bull" across windows regardless of separation quality.

    Returns
    -------
    model, order, raw_means, separated, z_min
        `order` maps raw HMM state index -> canonical (bear, neutral, bull) position.
        `separated` / `z_min` are DIAGNOSTIC ONLY (see fit note below) -- do not gate
        refit acceptance on them; v2 tried that and froze the model on a single stale
        pre-GFC fit for 17 years of subsequent data.
    """
    m = GaussianHMM(
        n_components=n_states, covariance_type="diag",
        n_iter=200, random_state=random_state, min_covar=min_covar,
    )
    m.fit(r.reshape(-1, 1))
    raw_means = m.means_.ravel()
    raw_vars = np.array([m.covars_[i].ravel()[0] for i in range(n_states)])
    state_seq = m.predict(r.reshape(-1, 1))
    n_i = np.array([max((state_seq == i).sum(), 1) for i in range(n_states)])

    ord_by_mean = np.argsort(raw_means)
    sm, sv, sn = raw_means[ord_by_mean], raw_vars[ord_by_mean], n_i[ord_by_mean]
    se = np.sqrt(sv[:-1] / sn[:-1] + sv[1:] / sn[1:])
    z = np.diff(sm) / np.maximum(se, 1e-12)
    separated = bool(np.all(z > 1.5))  # diagnostic threshold only, see config sep_z_thresh

    if prev_means_ordered is None:
        order = ord_by_mean
    else:
        best_perm, best_cost = None, np.inf
        for perm in permutations(range(n_states)):
            cost = sum(abs(raw_means[perm[i]] - prev_means_ordered[i]) for i in range(n_states))
            if cost < best_cost:
                best_cost, best_perm = cost, perm
        order = np.array(best_perm)
    return m, order, raw_means, separated, float(z.min())


def forward_filter(r: np.ndarray, m: GaussianHMM, order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Causal filtered posterior P(state_t | returns up to t), no lookahead."""
    n_states = len(order)
    means = m.means_.ravel()[order]
    stds = np.sqrt(np.array([m.covars_[i].ravel()[0] for i in range(n_states)]))[order]
    A = m.transmat_[np.ix_(order, order)]
    pi = m.startprob_[order]
    alphas = np.zeros((len(r), n_states))
    lik = norm.pdf(r[:, None], means[None, :], stds[None, :]) + 1e-300
    a = pi * lik[0]
    a /= a.sum()
    alphas[0] = a
    for t in range(1, len(r)):
        a = (A.T @ a) * lik[t]
        a /= a.sum()
        alphas[t] = a
    return alphas, A


def walk_forward_direction(
    returns: pd.Series,
    min_train: int = 756,
    refit_every: int = 63,
    n_states: int = 3,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict]:
    """Walk-forward direction posterior. ALWAYS accepts the continuity-matched refit
    (v3 fix) -- z-separation is tracked purely as a diagnostic, never blocks acceptance.
    """
    r = returns.values
    n = len(r)
    post = np.full((n, n_states), np.nan)
    transition_mats: dict[int, np.ndarray] = {}
    prev_means_ordered = None
    n_refit, n_below_thresh, z_hist = 0, 0, []

    for t in range(min_train, n, refit_every):
        m, order, raw_means, separated, zmin = fit_hmm(r[:t], prev_means_ordered, n_states=n_states)
        z_hist.append(zmin)
        n_refit += 1
        if not separated:
            n_below_thresh += 1
        prev_means_ordered = m.means_.ravel()[order]
        seg_end = min(t + refit_every, n)
        alphas, A = forward_filter(r[:seg_end], m, order)
        post[t:seg_end] = alphas[t:seg_end]
        transition_mats[t] = A

    dirpost = pd.DataFrame(
        post, index=returns.index, columns=["p_bear", "p_neut", "p_bull"]
    ).dropna()
    diagnostics = dict(n_refit=n_refit, n_below_z_thresh=n_below_thresh,
                       median_z_separation=float(np.median(z_hist)) if z_hist else None)
    return dirpost, transition_mats, diagnostics


# ---------------------------------------------------------------------------
# Vol layer (unconditional -- v5's direction-conditional variant was tested live and
# reverted: didn't shrink the bull_hi/bear_hi effect, cost backtest Sharpe 2.93->2.64)
# ---------------------------------------------------------------------------

def vol_layer(
    vix_pct: pd.Series, backwardation: pd.Series, rv_pct: pd.Series,
    weights: dict, logistic_scale: float = 4,
) -> pd.Series:
    z = (weights["vix_pct"] * vix_pct + weights["backwardation"] * backwardation
         + weights["rv_pct"] * rv_pct + weights["bias"])
    return 1 / (1 + np.exp(-logistic_scale * z))


# ---------------------------------------------------------------------------
# Curve-conditioned Bayesian drift model
# ---------------------------------------------------------------------------

def curve_conditioned_drift_posterior(
    slope_z: pd.Series, forward_returns: pd.Series, min_train: int = 252,
) -> tuple[pd.Series, pd.Series]:
    """mu_t = a + b*slope_z, conjugate Normal-Inverse-Gamma, walk-forward OOS.

    slope_z is the z-scored VX1-VX3 futures curve slope (vix-utils) -- NOT spot VIX3M/VIX
    ratios. This is Nikolas's researched conditioning variable.
    """
    X_all = np.column_stack([np.ones(len(slope_z)), slope_z.values])
    y_all = forward_returns.values
    p_up = pd.Series(np.nan, index=slope_z.index)
    beta_slope = pd.Series(np.nan, index=slope_z.index)

    for i in range(min_train, len(slope_z)):
        X, y = X_all[:i], y_all[:i]
        ok = ~np.isnan(y)
        X, y = X[ok], y[ok]
        if len(y) < 100:
            continue
        V0inv = np.eye(2) * 1e-4
        Vn = np.linalg.inv(V0inv + X.T @ X)
        bn = Vn @ (X.T @ y)
        a_n = 1.0 + len(y) / 2
        b_n = 1.0 + 0.5 * (y @ y - bn @ np.linalg.inv(Vn) @ bn)
        s2 = b_n / a_n
        x0 = X_all[i]
        mu_pred = x0 @ bn
        var_pred = s2 * (1 + x0 @ Vn @ x0)
        p_up.iloc[i] = 1 - student_t.cdf(0, df=2 * a_n, loc=mu_pred, scale=np.sqrt(var_pred))
        beta_slope.iloc[i] = bn[1]
    return p_up, beta_slope


# ---------------------------------------------------------------------------
# Smoothing / commitment layer
# ---------------------------------------------------------------------------

def commit_regime(
    cell_posterior: pd.DataFrame, commit_p: float = 0.70, commit_days: int = 2, min_dwell: int = 3,
) -> pd.Series:
    """Raw 6-cell posterior -> smoothed committed regime. Requires posterior of the new
    regime > commit_p for `commit_days` consecutive days, plus `min_dwell` days after any
    switch, before accepting a new committed cell.
    """
    raw_cell = cell_posterior.idxmax(axis=1)
    committed, cur, streak, dwell = [], None, 0, 0
    for t in range(len(cell_posterior)):
        top, ptop = raw_cell.iloc[t], cell_posterior.iloc[t].max()
        if cur is None:
            cur = top
        else:
            if top != cur and ptop >= commit_p:
                streak += 1
            else:
                streak = 0
            if streak >= commit_days and dwell >= min_dwell:
                cur, streak, dwell = top, 0, 0
        dwell += 1
        committed.append(cur)
    return pd.Series(committed, index=cell_posterior.index, name="regime")
