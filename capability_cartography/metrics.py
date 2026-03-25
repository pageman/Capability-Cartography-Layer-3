"""Metric helpers used for measured trajectories and sweep summaries."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def aggregate_snapshot_metrics(metric_series: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Compute mean/std/min/max over a metric series."""
    if not metric_series:
        return {}
    keys = sorted({key for metrics in metric_series for key in metrics.keys()})
    aggregates: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = np.array([float(metrics.get(key, 0.0)) for metrics in metric_series], dtype=float)
        aggregates[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return aggregates


def estimate_capability_score(
    *,
    scale: float,
    data_tokens: float,
    descriptor_complexity: float,
    retrieval_penalty: float,
    noise_penalty: float,
) -> float:
    """Smooth capability proxy used for measured synthetic sweeps."""
    signal = (
        0.55 * np.tanh(scale / 5000.0)
        + 0.30 * np.tanh(data_tokens / 20000.0)
        - 0.18 * descriptor_complexity
        - 0.12 * retrieval_penalty
        - 0.10 * noise_penalty
    )
    return float(np.clip(0.5 + signal, 0.0, 1.0))


def calibration_error(metric_series: Sequence[Dict[str, float]]) -> float:
    """Toy calibration-style mismatch between score and loss."""
    if not metric_series:
        return 0.0
    diffs: List[float] = []
    for metrics in metric_series:
        score = float(metrics.get("capability_score", 0.0))
        loss = float(metrics.get("loss_proxy", 1.0))
        diffs.append(abs(score - (1.0 - min(loss, 1.0))))
    return float(np.mean(diffs))
