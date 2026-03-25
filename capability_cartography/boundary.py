"""Boundary fitting and changepoint utilities."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .schemas import BoundaryEvent, BoundaryFit, CapabilitySnapshot


class BoundaryAnalyzer:
    """Identify changepoints, thresholds, and phase-like regions."""

    def detect_events(
        self,
        snapshots: Sequence[CapabilitySnapshot],
        *,
        metric: str,
        min_delta: float = 0.15,
    ) -> List[BoundaryEvent]:
        if not snapshots:
            return []
        values = np.array([snapshot.metrics.get(metric, 0.0) for snapshot in snapshots], dtype=float)
        steps = np.array([snapshot.step for snapshot in snapshots], dtype=int)
        if values.size < 2:
            return []
        deltas = np.diff(values)
        events: List[BoundaryEvent] = []
        for index, delta in enumerate(deltas, start=1):
            if abs(delta) < min_delta:
                continue
            before = self._regime_label(values[index - 1])
            after = self._regime_label(values[index])
            if before == after:
                continue
            events.append(
                BoundaryEvent(
                    metric=metric,
                    step=int(steps[index]),
                    value=float(values[index]),
                    delta=float(delta),
                    regime_before=before,
                    regime_after=after,
                )
            )
        return events

    def fit_threshold(self, snapshots: Sequence[CapabilitySnapshot], *, metric: str) -> BoundaryFit:
        if not snapshots:
            return BoundaryFit(metric=metric, threshold_value=0.0, threshold_step=0, slope=0.0, lower_band=0.0, upper_band=0.0)
        values = np.array([snapshot.metrics.get(metric, 0.0) for snapshot in snapshots], dtype=float)
        steps = np.array([snapshot.step for snapshot in snapshots], dtype=float)
        threshold_value = float(np.median(values))
        threshold_index = int(np.argmin(np.abs(values - threshold_value)))
        if len(values) >= 2:
            slope, _ = np.polyfit(steps, values, deg=1)
        else:
            slope = 0.0
        lower_band = float(np.quantile(values, 0.25))
        upper_band = float(np.quantile(values, 0.75))
        return BoundaryFit(
            metric=metric,
            threshold_value=threshold_value,
            threshold_step=int(steps[threshold_index]),
            slope=float(slope),
            lower_band=lower_band,
            upper_band=upper_band,
        )

    def summarize_phase_region(self, snapshots: Sequence[CapabilitySnapshot], *, metric: str) -> Dict[str, float]:
        values = np.array([snapshot.metrics.get(metric, 0.0) for snapshot in snapshots], dtype=float)
        if values.size == 0:
            return {"stable_reasoning_fraction": 0.0, "brittle_fraction": 0.0, "collapse_fraction": 0.0}
        regimes = [self._regime_label(value) for value in values]
        total = float(len(regimes))
        return {
            "stable_reasoning_fraction": float(sum(regime == "stable_reasoning" for regime in regimes) / total),
            "brittle_fraction": float(sum(regime == "partial_competence" for regime in regimes) / total),
            "collapse_fraction": float(sum(regime == "collapse" for regime in regimes) / total),
        }

    @staticmethod
    def _regime_label(value: float) -> str:
        if value >= 0.8:
            return "stable_reasoning"
        if value >= 0.45:
            return "partial_competence"
        return "collapse"
