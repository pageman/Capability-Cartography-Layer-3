"""Real IV estimator implementations using NumPy and SciPy.

Each function takes instrument Z, treatment X, outcome Y (as numpy arrays)
and returns an estimate of the causal effect β together with diagnostics.

For the unpaired setting, X and Y come from separate samples that share
instruments Z.  The SplitUP estimator implements the cross-fold
denominator from Schur et al. (2026).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class IVResult:
    """Result from a single IV estimation."""
    name: str
    beta: np.ndarray
    se: Optional[np.ndarray] = None
    first_stage_F: float = 0.0
    residual_var: float = 0.0
    converged: bool = True

    def scalar(self) -> float:
        return float(self.beta.ravel()[0]) if self.beta.size > 0 else 0.0


def _safe_inv(A: np.ndarray) -> np.ndarray:
    """Invert with fallback to pseudoinverse."""
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def _safe_standard_errors(covariance: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix into finite standard errors."""
    diag = np.real(np.diag(covariance))
    diag = np.where(np.isfinite(diag), diag, 0.0)
    return np.sqrt(np.clip(diag, 0.0, None))


# ====================================================================
# OLS — biased baseline
# ====================================================================

def ols(X: np.ndarray, Y: np.ndarray) -> IVResult:
    """Ordinary least squares: β̂ = (X'X)^{-1} X'Y."""
    X = np.atleast_2d(X)
    if X.shape[0] < X.shape[1]:
        X = X.T
    Y = np.atleast_1d(Y).ravel()
    XtX = X.T @ X
    XtY = X.T @ Y
    beta = _safe_inv(XtX) @ XtY
    resid = Y - X @ beta
    sigma2 = float(np.sum(resid ** 2) / max(len(Y) - X.shape[1], 1))
    se = _safe_standard_errors(sigma2 * _safe_inv(XtX))
    return IVResult(name="OLS", beta=beta, se=se, residual_var=sigma2)


# ====================================================================
# 2SLS — two-stage least squares
# ====================================================================

def tsls(Z: np.ndarray, X: np.ndarray, Y: np.ndarray) -> IVResult:
    """Two-stage least squares.
    Stage 1: X̂ = Z (Z'Z)^{-1} Z'X
    Stage 2: β̂ = (X̂'X̂)^{-1} X̂'Y
    """
    Z = np.atleast_2d(Z); X = np.atleast_2d(X); Y = np.atleast_1d(Y).ravel()
    if Z.shape[0] < Z.shape[1]: Z = Z.T
    if X.shape[0] < X.shape[1]: X = X.T
    n = Z.shape[0]
    # Stage 1
    Pz = Z @ _safe_inv(Z.T @ Z) @ Z.T
    Xhat = Pz @ X
    # Stage 2
    beta = _safe_inv(Xhat.T @ Xhat) @ Xhat.T @ Y
    resid = Y - X @ beta
    sigma2 = float(np.sum(resid ** 2) / max(n - X.shape[1], 1))
    V = sigma2 * _safe_inv(Xhat.T @ Xhat)
    se = _safe_standard_errors(V)
    # First-stage F
    Xbar = np.mean(X, axis=0)
    ss_model = float(np.sum((Xhat - Xbar) ** 2))
    ss_resid = float(np.sum((X - Xhat) ** 2))
    k = Z.shape[1]
    F = (ss_model / max(k, 1)) / (ss_resid / max(n - k - 1, 1)) if ss_resid > 1e-12 else 0.0
    return IVResult(name="2SLS", beta=beta, se=se, first_stage_F=F, residual_var=sigma2)


# ====================================================================
# LIML — limited information maximum likelihood
# ====================================================================

def liml(Z: np.ndarray, X: np.ndarray, Y: np.ndarray) -> IVResult:
    """LIML estimator — Anderson & Rubin (1949).
    Finds the smallest eigenvalue kappa of (Y-Xβ)'Mz(Y-Xβ) / (Y-Xβ)'(Y-Xβ).
    """
    Z = np.atleast_2d(Z); X = np.atleast_2d(X); Y = np.atleast_1d(Y).ravel()
    if Z.shape[0] < Z.shape[1]: Z = Z.T
    if X.shape[0] < X.shape[1]: X = X.T
    n = Z.shape[0]
    Pz = Z @ _safe_inv(Z.T @ Z) @ Z.T
    Mz = np.eye(n) - Pz
    YX = np.column_stack([Y.reshape(-1, 1), X])
    W0 = YX.T @ Mz @ YX
    W1 = YX.T @ YX
    try:
        eigvals = np.linalg.eigvalsh(_safe_inv(W1) @ W0)
        kappa = float(np.min(eigvals[eigvals > -1e-10]))
    except Exception:
        kappa = 1.0
    # LIML: (X'(I - kappa*Mz)X)^{-1} X'(I - kappa*Mz)Y
    A = np.eye(n) - kappa * Mz
    beta = _safe_inv(X.T @ A @ X) @ (X.T @ A @ Y)
    resid = Y - X @ beta
    sigma2 = float(np.sum(resid ** 2) / max(n - X.shape[1], 1))
    se = _safe_standard_errors(sigma2 * _safe_inv(X.T @ A @ X)) if sigma2 > 0 else np.zeros(X.shape[1])
    return IVResult(name="LIML", beta=beta, se=se, residual_var=sigma2)


# ====================================================================
# Fuller-k — LIML with finite-sample correction
# ====================================================================

def fuller(Z: np.ndarray, X: np.ndarray, Y: np.ndarray, *, alpha: float = 1.0) -> IVResult:
    """Fuller (1977) estimator — LIML with kappa replaced by kappa - alpha/(n-k)."""
    Z = np.atleast_2d(Z); X = np.atleast_2d(X); Y = np.atleast_1d(Y).ravel()
    if Z.shape[0] < Z.shape[1]: Z = Z.T
    if X.shape[0] < X.shape[1]: X = X.T
    n = Z.shape[0]; k = Z.shape[1]
    Pz = Z @ _safe_inv(Z.T @ Z) @ Z.T
    Mz = np.eye(n) - Pz
    YX = np.column_stack([Y.reshape(-1, 1), X])
    W0 = YX.T @ Mz @ YX
    W1 = YX.T @ YX
    try:
        eigvals = np.linalg.eigvalsh(_safe_inv(W1) @ W0)
        kappa = float(np.min(eigvals[eigvals > -1e-10]))
    except Exception:
        kappa = 1.0
    kappa_f = kappa - alpha / max(n - k, 1)
    A = np.eye(n) - kappa_f * Mz
    beta = _safe_inv(X.T @ A @ X) @ (X.T @ A @ Y)
    resid = Y - X @ beta
    sigma2 = float(np.sum(resid ** 2) / max(n - X.shape[1], 1))
    se = _safe_standard_errors(sigma2 * _safe_inv(X.T @ A @ X)) if sigma2 > 0 else np.zeros(X.shape[1])
    return IVResult(name="Fuller_k", beta=beta, se=se, residual_var=sigma2)


# ====================================================================
# TS-IV — two-sample IV (for unpaired data)
# ====================================================================

def ts_iv(Z_x: np.ndarray, X: np.ndarray, Z_y: np.ndarray, Y: np.ndarray) -> IVResult:
    """Two-sample IV: β̂ = (Ĉov(Z,X)'Ĉov(Z,X))^{-1} Ĉov(Z,X)'Ĉov(Z,Y).
    
    Z_x, X are from sample A;  Z_y, Y are from sample B.
    """
    Z_x = np.atleast_2d(Z_x); X = np.atleast_2d(X)
    Z_y = np.atleast_2d(Z_y); Y = np.atleast_1d(Y).ravel()
    if Z_x.shape[0] < Z_x.shape[1]: Z_x = Z_x.T
    if X.shape[0] < X.shape[1]: X = X.T
    if Z_y.shape[0] < Z_y.shape[1]: Z_y = Z_y.T
    # Sample covariances
    Z_x_c = Z_x - Z_x.mean(axis=0)
    X_c = X - X.mean(axis=0)
    Z_y_c = Z_y - Z_y.mean(axis=0)
    Y_c = Y - Y.mean()
    cov_zx = (Z_x_c.T @ X_c) / Z_x.shape[0]
    cov_zy = (Z_y_c.T @ Y_c.reshape(-1, 1)) / Z_y.shape[0]
    beta = _safe_inv(cov_zx.T @ cov_zx) @ cov_zx.T @ cov_zy
    return IVResult(name="TS_IV", beta=beta.ravel())


# ====================================================================
# UP-GMM — unpaired GMM (Schur et al. 2026, Eq 4.2)
# ====================================================================

def up_gmm(Z_x: np.ndarray, X: np.ndarray, Z_y: np.ndarray, Y: np.ndarray,
           W: Optional[np.ndarray] = None) -> IVResult:
    """Unpaired GMM estimator."""
    Z_x = np.atleast_2d(Z_x); X = np.atleast_2d(X)
    Z_y = np.atleast_2d(Z_y); Y = np.atleast_1d(Y).ravel()
    if Z_x.shape[0] < Z_x.shape[1]: Z_x = Z_x.T
    if X.shape[0] < X.shape[1]: X = X.T
    if Z_y.shape[0] < Z_y.shape[1]: Z_y = Z_y.T
    n, m = Z_y.shape
    n_tilde = Z_x.shape[0]
    cov_zx = (Z_x - Z_x.mean(0)).T @ (X - X.mean(0)) / n_tilde
    cov_zy = (Z_y - Z_y.mean(0)).T @ (Y - Y.mean()).reshape(-1, 1) / n
    if W is None:
        W = np.eye(m)
    # β̂ = (cov_zx' W cov_zx)^{-1} cov_zx' W cov_zy
    A = cov_zx.T @ W @ cov_zx
    B = cov_zx.T @ W @ cov_zy
    beta = _safe_inv(A) @ B
    return IVResult(name="UP_GMM", beta=beta.ravel())


# ====================================================================
# SplitUP — cross-fold bias-corrected GMM (Schur et al. 2026, Thm 4.7)
# ====================================================================

def splitup(Z_x: np.ndarray, X: np.ndarray, Z_y: np.ndarray, Y: np.ndarray,
            *, K: int = 2, H: int = 50, W: Optional[np.ndarray] = None) -> IVResult:
    """SplitUP estimator with Monte Carlo averaging over H random splits.
    
    The key innovation: cross-fold denominator C_XX uses products of
    covariances estimated on independent folds, so E_A^T E_B → 0
    removing the measurement-error bias that afflicts naive TS-IV.
    """
    Z_x = np.atleast_2d(Z_x); X = np.atleast_2d(X)
    Z_y = np.atleast_2d(Z_y); Y = np.atleast_1d(Y).ravel()
    if Z_x.shape[0] < Z_x.shape[1]: Z_x = Z_x.T
    if X.shape[0] < X.shape[1]: X = X.T
    if Z_y.shape[0] < Z_y.shape[1]: Z_y = Z_y.T
    n_tilde = Z_x.shape[0]
    m = Z_x.shape[1]
    d = X.shape[1]
    n = Z_y.shape[0]

    # Numerator: C_XY = m * Cov(Z,X)' Cov(Z,Y) — same as UP-GMM
    cov_zy = (Z_y - Z_y.mean(0)).T @ (Y - Y.mean()).reshape(-1, 1) / n
    C_XY = m * ((Z_x - Z_x.mean(0)).T @ (X - X.mean(0)) / n_tilde).T @ cov_zy

    # Denominator: C_XX via cross-fold averaging
    rng = np.random.default_rng(42)
    C_XX_accum = np.zeros((d, d))
    for _h in range(H):
        perm = rng.permutation(n_tilde)
        half = n_tilde // 2
        A_idx = perm[:half]
        B_idx = perm[half:2 * half]
        Z_A = Z_x[A_idx]; X_A = X[A_idx]
        Z_B = Z_x[B_idx]; X_B = X[B_idx]
        cov_A = (Z_A - Z_A.mean(0)).T @ (X_A - X_A.mean(0)) / len(A_idx)
        cov_B = (Z_B - Z_B.mean(0)).T @ (X_B - X_B.mean(0)) / len(B_idx)
        C_XX_accum += m * cov_A.T @ cov_B
    C_XX = C_XX_accum / H

    if W is None:
        W = np.eye(d)
    beta = _safe_inv(C_XX.T @ W @ C_XX) @ C_XX.T @ W @ C_XY
    return IVResult(name="SplitUP", beta=beta.ravel())


# ====================================================================
# SplitUP analytic — closed-form infinite-split average
# ====================================================================

def splitup_analytic(Z_x: np.ndarray, X: np.ndarray, Z_y: np.ndarray, Y: np.ndarray,
                     *, W: Optional[np.ndarray] = None) -> IVResult:
    """SplitUP with the analytic infinite-split average (Schur et al. 2026, §4.3).
    
    C̄_XX = n/(n-1) Ĉov(Z,X)' Ĉov(Z,X)  -  1/(n(n-1)) Σ_i (Z̃_i X̃_i')' (Z̃_i X̃_i')
    """
    Z_x = np.atleast_2d(Z_x); X = np.atleast_2d(X)
    Z_y = np.atleast_2d(Z_y); Y = np.atleast_1d(Y).ravel()
    if Z_x.shape[0] < Z_x.shape[1]: Z_x = Z_x.T
    if X.shape[0] < X.shape[1]: X = X.T
    if Z_y.shape[0] < Z_y.shape[1]: Z_y = Z_y.T
    n_tilde = Z_x.shape[0]
    m = Z_x.shape[1]
    d = X.shape[1]
    n = Z_y.shape[0]

    Z_c = Z_x - Z_x.mean(0)
    X_c = X - X.mean(0)
    cov_zx = Z_c.T @ X_c / n_tilde  # m × d

    # Plug-in quadratic
    plug_in = m * cov_zx.T @ cov_zx  # d × d

    # Self-inner-product correction
    correction = np.zeros((d, d))
    for i in range(n_tilde):
        zi_xi = Z_c[i:i+1, :].T @ X_c[i:i+1, :]  # m × d
        correction += m * zi_xi.T @ zi_xi
    correction /= (n_tilde * (n_tilde - 1)) if n_tilde > 1 else 1.0

    C_XX = (n_tilde / max(n_tilde - 1, 1)) * plug_in - correction

    cov_zy = (Z_y - Z_y.mean(0)).T @ (Y - Y.mean()).reshape(-1, 1) / n
    C_XY = m * cov_zx.T @ cov_zy

    if W is None:
        W = np.eye(d)
    beta = _safe_inv(C_XX.T @ W @ C_XX) @ C_XX.T @ W @ C_XY
    return IVResult(name="SplitUP_analytic", beta=beta.ravel())


# ====================================================================
# IVW — inverse-variance weighted (standard MR estimator)
# ====================================================================

def ivw(beta_X: np.ndarray, beta_Y: np.ndarray, se_X: np.ndarray) -> IVResult:
    """Inverse-variance weighted MR: β̂ = Σ(w_j β̂_Yj / β̂_Xj) / Σ(w_j)
    where w_j = β̂_Xj² / se_Xj².
    """
    beta_X = np.atleast_1d(beta_X).ravel()
    beta_Y = np.atleast_1d(beta_Y).ravel()
    se_X = np.atleast_1d(se_X).ravel()
    # Wald ratios
    valid = np.abs(beta_X) > 1e-10
    ratios = np.zeros_like(beta_Y, dtype=float)
    np.divide(beta_Y, beta_X, out=ratios, where=valid)
    weights = np.where(valid, beta_X ** 2 / (se_X ** 2 + 1e-12), 0.0)
    if np.sum(weights) < 1e-12:
        return IVResult(name="IVW", beta=np.array([0.0]))
    beta_ivw = np.sum(weights * ratios) / np.sum(weights)
    se_ivw = 1.0 / np.sqrt(np.sum(weights) + 1e-12)
    return IVResult(name="IVW", beta=np.array([beta_ivw]), se=np.array([se_ivw]))


# ====================================================================
# MR-Egger
# ====================================================================

def mr_egger(beta_X: np.ndarray, beta_Y: np.ndarray, se_X: np.ndarray) -> IVResult:
    """MR-Egger regression: β̂_Y = α + β β̂_X + ε, weighted by 1/se_X²."""
    beta_X = np.atleast_1d(beta_X).ravel()
    beta_Y = np.atleast_1d(beta_Y).ravel()
    se_X = np.atleast_1d(se_X).ravel()
    if beta_X.size < 2:
        return IVResult(name="MR_Egger", beta=np.array([0.0]), converged=False)
    W = np.diag(1.0 / (se_X ** 2 + 1e-12))
    ones = np.ones_like(beta_X)
    D = np.column_stack([ones, beta_X])
    beta_eg = _safe_inv(D.T @ W @ D) @ D.T @ W @ beta_Y
    return IVResult(name="MR_Egger", beta=np.array([beta_eg[1]]))


# ====================================================================
# Convenience: generate DGP for testing
# ====================================================================

def generate_iv_data(
    n: int = 200,
    m: int = 10,
    d: int = 1,
    beta_true: float = 1.0,
    gamma: float = 0.5,
    instrument_strength: float = 0.5,
    seed: int = 42,
    unpaired: bool = False,
) -> dict:
    """Generate IV data: Z→X→Y with confounding U→X, U→Y."""
    rng = np.random.default_rng(seed)
    # Instruments (categorical via one-hot for m categories)
    if m <= n:
        labels = rng.integers(0, m, size=n)
        Z = np.zeros((n, m))
        Z[np.arange(n), labels] = 1.0
    else:
        Z = rng.normal(0, 1 / np.sqrt(m), size=(n, m))
    # First stage coefficients
    pi = rng.normal(instrument_strength, 0.1, size=(m, d))
    # Confounder
    U = rng.normal(0, 1, size=n)
    # Treatment
    X = Z @ pi + gamma * U.reshape(-1, 1) + rng.normal(0, 0.5, size=(n, d))
    # Outcome
    beta = np.full(d, beta_true)
    Y = X @ beta + gamma * U + rng.normal(0, 0.5, size=n)

    if unpaired:
        # Split: sample A gets (Z, X), sample B gets (Z, Y) with fresh draws
        n_a = n // 2
        n_b = n - n_a
        Z_a = Z[:n_a]; X_a = X[:n_a]
        # Redraw for sample B
        labels_b = rng.integers(0, m, size=n_b)
        Z_b = np.zeros((n_b, m))
        Z_b[np.arange(n_b), labels_b] = 1.0
        U_b = rng.normal(0, 1, size=n_b)
        X_b = Z_b @ pi + gamma * U_b.reshape(-1, 1) + rng.normal(0, 0.5, size=(n_b, d))
        Y_b = X_b @ beta + gamma * U_b + rng.normal(0, 0.5, size=n_b)
        return {"Z_x": Z_a, "X": X_a, "Z_y": Z_b, "Y": Y_b, "beta_true": beta}
    return {"Z": Z, "X": X, "Y": Y, "beta_true": beta}
