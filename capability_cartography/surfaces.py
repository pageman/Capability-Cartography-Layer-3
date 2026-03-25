"""Response-surface fitting utilities for capability onset analysis."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


class CapabilitySurfaceFitter:
    """Fit lightweight predictive surfaces over sweep records."""

    def fit_linear_surface(
        self,
        records: Sequence[Dict[str, float]],
        *,
        feature_keys: Sequence[str],
        target_key: str = "capability_score",
    ) -> Dict[str, object]:
        if not records:
            return {"feature_keys": list(feature_keys), "coefficients": {}, "intercept": 0.0, "r2": 0.0}
        X = np.array([[float(record.get(key, 0.0)) for key in feature_keys] for record in records], dtype=float)
        y = np.array([float(record.get(target_key, 0.0)) for record in records], dtype=float)
        X_aug = np.column_stack([np.ones(len(X)), X])
        coefficients, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        predictions = X_aug @ coefficients
        residual = np.sum((y - predictions) ** 2)
        total = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1.0 - residual / total) if total > 0 else 1.0
        return {
            "feature_keys": list(feature_keys),
            "coefficients": {key: float(value) for key, value in zip(feature_keys, coefficients[1:])},
            "intercept": float(coefficients[0]),
            "r2": r2,
        }

    def onset_threshold_by_feature(
        self,
        records: Sequence[Dict[str, float]],
        *,
        feature_key: str,
        target_key: str = "capability_score",
        competence_threshold: float = 0.8,
    ) -> Dict[str, float]:
        if not records:
            return {"feature": feature_key, "threshold": 0.0, "competence_threshold": competence_threshold}
        filtered = sorted(records, key=lambda item: float(item.get(feature_key, 0.0)))
        for record in filtered:
            if float(record.get(target_key, 0.0)) >= competence_threshold:
                return {
                    "feature": feature_key,
                    "threshold": float(record.get(feature_key, 0.0)),
                    "competence_threshold": competence_threshold,
                }
        return {
            "feature": feature_key,
            "threshold": float(filtered[-1].get(feature_key, 0.0)),
            "competence_threshold": competence_threshold,
        }
