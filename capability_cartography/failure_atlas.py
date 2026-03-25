"""Failure-atlas training over exported records."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


class FailureAtlasClassifier:
    """A simple centroid-based failure atlas over measured records."""

    def __init__(self):
        self.centroids: Dict[str, np.ndarray] = {}
        self.feature_keys = ["capability_score", "generalization_gap", "retrieval_dependence", "task_family_code", "scale", "data_tokens"]

    def train(self, records: Sequence[Dict[str, float]]) -> Dict[str, object]:
        labeled = [(record, self._label(record)) for record in records]
        by_label: Dict[str, List[np.ndarray]] = {}
        for record, label in labeled:
            by_label.setdefault(label, []).append(self._vector(record))
        self.centroids = {
            label: np.mean(np.stack(vectors, axis=0), axis=0)
            for label, vectors in by_label.items()
            if vectors
        }
        predictions = []
        label_counts = Counter()
        for record, label in labeled:
            prediction = self.predict(record)
            label_counts[label] += 1
            predictions.append(
                {
                    "experiment_id": record.get("experiment_id", ""),
                    "task_family": record.get("task_family", ""),
                    "actual_label": label,
                    "predicted_label": prediction["label"],
                    "distances": prediction["distances"],
                    "capability_score": float(record.get("capability_score", 0.0)),
                    "generalization_gap": float(record.get("generalization_gap", 0.0)),
                    "retrieval_dependence": float(record.get("retrieval_dependence", 0.0)),
                    "scale": float(record.get("scale", 0.0)),
                    "data_tokens": float(record.get("data_tokens", 0.0)),
                }
            )
        return {
            "labels": sorted(by_label.keys()),
            "label_counts": dict(sorted(label_counts.items())),
            "record_count": len(records),
            "feature_keys": list(self.feature_keys),
            "centroids": {label: centroid.tolist() for label, centroid in self.centroids.items()},
            "records": predictions,
        }

    def predict(self, record: Dict[str, float]) -> Dict[str, object]:
        if not self.centroids:
            raise RuntimeError("Failure atlas classifier has not been trained.")
        vector = self._vector(record)
        distances = {
            label: float(np.linalg.norm(vector - centroid))
            for label, centroid in self.centroids.items()
        }
        best = min(distances, key=distances.get)
        return {"label": best, "distances": distances}

    def fit_from_csv(self, path: str | Path) -> Dict[str, object]:
        rows = []
        with Path(path).open() as handle:
            for row in csv.DictReader(handle):
                rows.append({key: float(value) if key not in {"experiment_id", "task_family"} else value for key, value in row.items()})
        return self.train(rows)

    def export(self, path: str | Path, summary: Dict[str, object]) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))
        return str(path)

    def _vector(self, record: Dict[str, float]) -> np.ndarray:
        values = []
        for key in self.feature_keys:
            values.append(float(record.get(key, 0.0)))
        return np.array(values, dtype=float)

    @staticmethod
    def _label(record: Dict[str, float]) -> str:
        score = float(record.get("capability_score", 0.0))
        gap = float(record.get("generalization_gap", 0.0))
        retrieval = float(record.get("retrieval_dependence", 0.0))
        if score < 0.21:
            return "collapse"
        if retrieval > 0.5 and gap > 0.02:
            return "brittle_retrieval"
        if gap > 0.03:
            return "generalization_risk"
        return "stable_reasoning"
