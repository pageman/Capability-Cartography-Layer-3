"""Predictive-law validation and bootstrap uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .surfaces import CapabilitySurfaceFitter


class PredictiveLawValidator:
    """Fit, validate, and formulate falsifiable predictive laws."""

    def __init__(self):
        self.surface_fitter = CapabilitySurfaceFitter()

    def split_holdout(self, records: Sequence[Dict[str, float]], holdout_ratio: float = 0.25):
        grouped: Dict[tuple, List[Dict[str, float]]] = {}
        for record in records:
            key = (
                float(record.get("scale", 0.0)),
                float(record.get("data_tokens", 0.0)),
                float(record.get("task_family_code", 0.0)),
                float(record.get("retrieval_dependence", 0.0)),
            )
            grouped.setdefault(key, []).append(record)
        train: List[Dict[str, float]] = []
        holdout: List[Dict[str, float]] = []
        for group in grouped.values():
            ordered = sorted(group, key=lambda record: float(record.get("seed", 0.0)))
            if len(ordered) == 1:
                train.extend(ordered)
            else:
                train.extend(ordered[:-1])
                holdout.append(ordered[-1])
        if not holdout:
            ordered = sorted(records, key=lambda record: str(record.get("experiment_id", "")))
            cutoff = max(1, int(len(ordered) * (1.0 - holdout_ratio)))
            return ordered[:cutoff], ordered[cutoff:]
        return train, holdout

    def fit_and_validate(
        self,
        records: Sequence[Dict[str, float]],
        *,
        feature_keys: Sequence[str],
        target_key: str = "capability_score",
        bootstrap_samples: int = 64,
    ) -> Dict[str, object]:
        train, holdout = self.split_holdout(records)
        model = self.surface_fitter.fit_linear_surface(train, feature_keys=feature_keys, target_key=target_key)
        validation = self._validate(model, holdout, feature_keys=feature_keys, target_key=target_key)
        intervals = self._bootstrap_intervals(train, feature_keys=feature_keys, target_key=target_key, n=bootstrap_samples)
        laws = self._formulate_laws(model, validation, feature_keys=feature_keys)
        return {
            "train_count": len(train),
            "holdout_count": len(holdout),
            "model": model,
            "validation": validation,
            "bootstrap_intervals": intervals,
            "laws": laws,
        }

    def _validate(self, model: Dict[str, object], records: Sequence[Dict[str, float]], *, feature_keys: Sequence[str], target_key: str) -> Dict[str, float]:
        if not records:
            return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
        y_true = np.array([float(record.get(target_key, 0.0)) for record in records], dtype=float)
        y_pred = np.array([self._predict(model, record, feature_keys=feature_keys) for record in records], dtype=float)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        total = np.sum((y_true - np.mean(y_true)) ** 2)
        residual = np.sum((y_true - y_pred) ** 2)
        r2 = float(1.0 - residual / total) if total > 1e-9 else float("nan")
        return {"mae": mae, "rmse": rmse, "r2": r2}

    def _bootstrap_intervals(self, records: Sequence[Dict[str, float]], *, feature_keys: Sequence[str], target_key: str, n: int) -> Dict[str, Dict[str, float]]:
        if not records:
            return {}
        rng = np.random.default_rng(42)
        coefficients: Dict[str, List[float]] = {key: [] for key in feature_keys}
        intercepts: List[float] = []
        for _ in range(n):
            sample = [records[idx] for idx in rng.integers(0, len(records), size=len(records))]
            model = self.surface_fitter.fit_linear_surface(sample, feature_keys=feature_keys, target_key=target_key)
            intercepts.append(float(model["intercept"]))
            for key in feature_keys:
                coefficients[key].append(float(model["coefficients"].get(key, 0.0)))
        intervals = {
            "intercept": {
                "low": float(np.quantile(intercepts, 0.025)),
                "high": float(np.quantile(intercepts, 0.975)),
            }
        }
        for key, values in coefficients.items():
            intervals[key] = {
                "low": float(np.quantile(values, 0.025)),
                "high": float(np.quantile(values, 0.975)),
            }
        return intervals

    @staticmethod
    def _predict(model: Dict[str, object], record: Dict[str, float], *, feature_keys: Sequence[str]) -> float:
        intercept = float(model.get("intercept", 0.0))
        coefficients = model.get("coefficients", {})
        value = intercept
        for key in feature_keys:
            value += float(coefficients.get(key, 0.0)) * float(record.get(key, 0.0))
        return value

    @staticmethod
    def _formulate_laws(model: Dict[str, object], validation: Dict[str, float], *, feature_keys: Sequence[str]) -> List[str]:
        coefficients = model.get("coefficients", {})
        terms = [f"{float(coefficients.get(key, 0.0)):.6f}*{key}" for key in feature_keys]
        equation = f"capability_score = {float(model.get('intercept', 0.0)):.6f} + " + " + ".join(terms)
        if np.isnan(validation["r2"]):
            criterion = f"holdout MAE remains <= {validation['mae']:.4f}"
        else:
            criterion = (
                f"holdout MAE remains <= {validation['mae']:.4f} "
                f"and holdout R^2 remains >= {validation['r2']:.4f}"
            )
        statement = f"Within the measured regime, {equation}. This law is supported only if {criterion} on new runs from the same regime."
        return [statement]
