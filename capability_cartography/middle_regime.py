"""Middle-regime analysis using Schur et al. (2026) theory.

The "middle regime" is m → ∞ with n/m → r ∈ (0, ∞):
many environments/instruments, but few observations per environment.

In this regime:
  - Standard TS-IV is asymptotically biased: β̂ → Q/(Q + r̃b)
  - SplitUP removes the bias via cross-fold splitting
  - Sparse extensions work with ℓ₁ regularization

This module:
  1. Classifies each paper into a regime label.
  2. Computes the theoretical measurement-error bias magnitude.
  3. Detects boundary events where the regime transition occurs.
  4. Reports whether SplitUP is needed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from .schemas import BoundaryEvent, MiddleRegimeProfile


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def classify_regime(m: int, r: float, d: int, s_star: int) -> str:
    """Return a human-readable regime label."""
    if m <= d:
        return "under_identified"
    if r > 50:
        return "classical_large_sample"
    if m <= 2 * s_star:
        return "sparse_under_identified"
    if r <= 1:
        return "extreme_high_dim"
    if r <= 8:
        return "high_dim_moderate_r"
    return "moderate_high_dim"


def measurement_error_bias(Q: float, r_tilde: float, b: float) -> float:
    """Compute the TS-IV attenuation factor from Schur et al. Lemma 4.5.

    β̂_TS-IV  →  Q / (Q + r̃ · b)   ≠  β*

    Returns the attenuation factor (< 1 means attenuated toward zero).
    """
    denom = Q + r_tilde * b
    if abs(denom) < 1e-12:
        return 0.0
    return Q / denom


# ---------------------------------------------------------------------------
# Paper-type → default structural parameters
# ---------------------------------------------------------------------------

_PAPER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "architecture":   {"m": 5,   "r": 100.0, "d": 3,  "s_star": 2, "Q": 0.8, "b": 0.02},
    "regularization": {"m": 3,   "r": 150.0, "d": 5,  "s_star": 2, "Q": 0.6, "b": 0.03},
    "generative":     {"m": 4,   "r": 80.0,  "d": 4,  "s_star": 2, "Q": 0.7, "b": 0.02},
    "retrieval":      {"m": 20,  "r": 4.0,   "d": 5,  "s_star": 2, "Q": 0.3, "b": 0.10},
    "scaling":        {"m": 10,  "r": 30.0,  "d": 2,  "s_star": 1, "Q": 0.9, "b": 0.01},
    "theory":         {"m": 2,   "r": 50.0,  "d": 2,  "s_star": 1, "Q": 0.1, "b": 0.05},
    "systems":        {"m": 2,   "r": 50.0,  "d": 2,  "s_star": 1, "Q": 0.1, "b": 0.05},
}


class MiddleRegimeAnalyzer:
    """Classify papers into Schur et al. regimes and compute bias profiles."""

    def profile_paper(self, paper_id: int, paper_type: str) -> MiddleRegimeProfile:
        defs = _PAPER_DEFAULTS.get(paper_type, _PAPER_DEFAULTS["theory"])
        m = defs["m"]
        r = defs["r"]
        d = defs["d"]
        s_star = defs["s_star"]
        Q = defs["Q"]
        b = defs["b"]

        regime = classify_regime(m, r, d, s_star)
        is_high_dim = r < 50 and m > d
        attenuation = measurement_error_bias(Q, 1.0 / max(r, 1e-9), b)
        me_bias = 1.0 - attenuation if is_high_dim else 0.0
        splitup_needed = is_high_dim and me_bias > 0.01

        if is_high_dim and s_star < d:
            theorem = "Theorem 4.8 (high-dim sparse)"
        elif is_high_dim:
            theorem = "Theorem 4.7 (high-dim dense)"
        elif s_star < d:
            theorem = "Theorem 4.2 (finite-dim sparse)"
        else:
            theorem = "Proposition 4.1 (finite-dim dense)"

        return MiddleRegimeProfile(
            paper_id=paper_id,
            m=m,
            r=round(r, 2),
            d=d,
            s_star=s_star,
            is_high_dim=is_high_dim,
            measurement_error_bias=round(me_bias, 6),
            attenuation_factor=round(attenuation, 6),
            splitup_needed=splitup_needed,
            regime_label=regime,
            schur_theorem=theorem,
        )

    def profile_all(self, records: Sequence[Dict[str, Any]]) -> List[MiddleRegimeProfile]:
        return [
            self.profile_paper(int(rec.get("paper_id", 0)), str(rec.get("paper_type", "theory")))
            for rec in records
        ]

    def detect_regime_boundary(
        self,
        m_values: Sequence[int],
        r: float = 8.0,
        d: int = 3,
        s_star: int = 2,
        Q: float = 0.5,
        b: float = 0.05,
    ) -> List[BoundaryEvent]:
        """Sweep over m values and detect the boundary where bias becomes significant."""
        events: List[BoundaryEvent] = []
        prev_bias = 0.0
        for m in sorted(m_values):
            regime = classify_regime(m, r, d, s_star)
            is_hd = r < 50 and m > d
            atten = measurement_error_bias(Q, 1.0 / max(r, 1e-9), b)
            bias = (1.0 - atten) if is_hd else 0.0
            delta = bias - prev_bias
            if abs(delta) > 0.01 and prev_bias < 0.01 <= bias:
                events.append(BoundaryEvent(
                    metric="measurement_error_bias",
                    step=m,
                    value=round(bias, 6),
                    delta=round(delta, 6),
                    regime_before="consistent" if prev_bias < 0.01 else "biased",
                    regime_after="measurement_error_biased",
                ))
            prev_bias = bias
        return events

    def summary(self, profiles: Sequence[MiddleRegimeProfile]) -> Dict[str, Any]:
        regimes = [p.regime_label for p in profiles]
        from collections import Counter
        regime_counts = dict(Counter(regimes))
        n_splitup = sum(1 for p in profiles if p.splitup_needed)
        return {
            "record_count": len(profiles),
            "regime_counts": regime_counts,
            "n_splitup_needed": n_splitup,
            "mean_attenuation": round(float(np.mean([p.attenuation_factor for p in profiles])), 4),
            "mean_measurement_error_bias": round(float(np.mean([p.measurement_error_bias for p in profiles])), 4),
        }
