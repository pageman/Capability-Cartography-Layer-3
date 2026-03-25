"""Identification-aware failure atlas.

Layer 2's failure_atlas classifies records into *symptoms*:
  collapse, generalization_risk, stable_reasoning.

Layer 3's causal_atlas classifies records into *causal pathologies*:
  unpaired_bias, weak_instrument, exclusion_violation,
  sparse_identification_failure, stable_identification.

This tells the user not just THAT a paper's mechanism fails to be
identified as causal, but WHY the identification fails.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .schemas import CausalRecord, EstimatorResult


# ---------------------------------------------------------------------------
# Causal pathology labels
# ---------------------------------------------------------------------------

PATHOLOGY_LABELS = [
    "stable_identification",
    "unpaired_bias",
    "weak_instrument",
    "sparse_identification_failure",
    "exclusion_violation_risk",
    "insufficient_environments",
]


class CausalAtlasClassifier:
    """Classify measured records by causal-identification pathology."""

    def __init__(self) -> None:
        self.centroids: Dict[str, np.ndarray] = {}
        self.feature_keys = [
            "estimator_consensus",
            "retrieval_dependence",
            "n_applicable",
            "n_consistent",
            "avg_estimator_bias",
        ]

    # ------------------------------------------------------------------
    # Label assignment (rule-based)
    # ------------------------------------------------------------------

    @staticmethod
    def label(record: Dict[str, Any]) -> str:
        consensus = float(record.get("estimator_consensus", 0))
        n_app = int(record.get("n_applicable", 0))
        n_con = int(record.get("n_consistent", 0))
        retrieval = float(record.get("retrieval_dependence", 0))
        avg_bias = float(record.get("avg_estimator_bias", 0))
        paper_type = str(record.get("paper_type", ""))

        # Stable: high consensus, many applicable
        if consensus >= 0.8 and n_app >= 20:
            return "stable_identification"

        # Unpaired bias: retrieval papers with low applicability
        if retrieval > 0.5 and n_app < 20:
            return "unpaired_bias"

        # Weak instrument: low consensus despite many applicable
        if n_app >= 20 and consensus < 0.5:
            return "weak_instrument"

        # Sparse ID failure: sparse paper type but low consistency
        if paper_type in ("regularization", "theory") and n_con < n_app * 0.5:
            return "sparse_identification_failure"

        # Insufficient environments: theory/systems with very few environments
        if paper_type in ("theory", "systems") and n_app < 22:
            return "insufficient_environments"

        # Exclusion violation risk: moderate bias across estimators
        if avg_bias > 0.04:
            return "exclusion_violation_risk"

        return "stable_identification"

    # ------------------------------------------------------------------
    # Train (centroid-based, as in Layer 2)
    # ------------------------------------------------------------------

    def train(self, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        labeled = [(rec, self.label(rec)) for rec in records]
        by_label: Dict[str, List[np.ndarray]] = {}
        for rec, lbl in labeled:
            by_label.setdefault(lbl, []).append(self._vector(rec))

        self.centroids = {
            lbl: np.mean(np.stack(vecs), axis=0)
            for lbl, vecs in by_label.items()
            if vecs
        }

        label_counts = Counter(lbl for _, lbl in labeled)
        predictions = []
        for rec, actual_lbl in labeled:
            pred = self.predict(rec)
            predictions.append({
                "paper_id": rec.get("paper_id", ""),
                "paper_name": rec.get("paper_name", ""),
                "actual_label": actual_lbl,
                "predicted_label": pred["label"],
                "distances": pred["distances"],
                "estimator_consensus": float(rec.get("estimator_consensus", 0)),
                "retrieval_dependence": float(rec.get("retrieval_dependence", 0)),
                "n_applicable": int(rec.get("n_applicable", 0)),
            })

        return {
            "labels": sorted(by_label.keys()),
            "label_counts": dict(sorted(label_counts.items())),
            "record_count": len(records),
            "feature_keys": list(self.feature_keys),
            "centroids": {l: c.tolist() for l, c in self.centroids.items()},
            "records": predictions,
        }

    def predict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if not self.centroids:
            return {"label": self.label(record), "distances": {}}
        vec = self._vector(record)
        distances = {
            lbl: float(np.linalg.norm(vec - c))
            for lbl, c in self.centroids.items()
        }
        best = min(distances, key=distances.get)
        return {"label": best, "distances": distances}

    def export(self, path: str | Path, summary: Dict[str, Any]) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))
        return str(path)

    def _vector(self, record: Dict[str, Any]) -> np.ndarray:
        return np.array([float(record.get(k, 0)) for k in self.feature_keys], dtype=float)
