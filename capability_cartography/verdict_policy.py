"""Policy for turning estimator summaries into causal verdicts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class VerdictPolicy:
    """Explicit thresholds for causal verdict assignment."""

    min_confirmed_applicable: int = 20
    min_confirmed_consensus: float = 0.8
    min_confirmed_backend_evidence: int = 0
    min_conditional_consensus: float = 0.3

    def evaluate(
        self,
        *,
        paper_type: str,
        n_applicable: int,
        consensus: float,
        backend_evidence: int = 0,
    ) -> Dict[str, str]:
        if paper_type == "retrieval" and n_applicable < self.min_confirmed_applicable:
            if consensus >= self.min_conditional_consensus:
                return {"verdict": "CONDITIONAL", "reason": "retrieval regime has too few applicable estimators for confirmation"}
            return {"verdict": "UNCONFIRMED", "reason": "retrieval regime has too few applicable estimators and weak consensus"}

        if paper_type in {"theory", "systems"}:
            if consensus >= self.min_conditional_consensus:
                return {"verdict": "CONDITIONAL", "reason": "theory/systems claims remain conditional without direct experimental identification"}
            return {"verdict": "UNCONFIRMED", "reason": "theory/systems claim lacks enough consensus for even conditional support"}

        if (
            n_applicable >= self.min_confirmed_applicable
            and consensus >= self.min_confirmed_consensus
            and backend_evidence >= self.min_confirmed_backend_evidence
        ):
            return {"verdict": "CONFIRMED", "reason": "high consensus with enough applicable estimators"}

        if consensus >= self.min_conditional_consensus:
            return {"verdict": "CONDITIONAL", "reason": "partial support but below confirmation threshold"}

        return {"verdict": "UNCONFIRMED", "reason": "insufficient consensus"}
