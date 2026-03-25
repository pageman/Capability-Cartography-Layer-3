"""Shared runner for capability cartography experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .adapters import AgentOverlayAdapter, GPT1WindTunnelAdapter, NotebookSubstrateAdapter
from .boundary import BoundaryAnalyzer
from .compressibility import CompressibilityStack
from .descriptors import TaskDescriptorExtractor
from .execution import MeasuredRunExecutor
from .metrics import aggregate_snapshot_metrics, calibration_error, estimate_capability_score
from .schemas import (
    ArtifactBundle,
    CapabilitySnapshot,
    CapabilityTrajectory,
    ExperimentSpec,
    InterventionConfig,
)


class CapabilityCartographyRunner:
    """Coordinates descriptor extraction, compression, and boundary analysis."""

    def __init__(
        self,
        *,
        substrate_adapter: NotebookSubstrateAdapter | None = None,
        wind_tunnel_adapter: GPT1WindTunnelAdapter | None = None,
        agent_adapter: AgentOverlayAdapter | None = None,
    ):
        self.substrate_adapter = substrate_adapter or NotebookSubstrateAdapter()
        self.wind_tunnel_adapter = wind_tunnel_adapter or GPT1WindTunnelAdapter()
        self.agent_adapter = agent_adapter or AgentOverlayAdapter()
        self.descriptor_extractor = TaskDescriptorExtractor()
        self.compressibility = CompressibilityStack()
        self.boundary = BoundaryAnalyzer()
        self.measured_executor = MeasuredRunExecutor(self.substrate_adapter, self.wind_tunnel_adapter)

    def run_text_experiment(
        self,
        spec: ExperimentSpec,
        intervention: InterventionConfig,
        *,
        text: str,
        retrieval_context: str = "",
        metric_series: Optional[Sequence[Dict[str, float]]] = None,
        export_dir: str | Path | None = None,
    ) -> ArtifactBundle:
        snapshots: List[CapabilitySnapshot] = []
        metric_series = metric_series or self._default_metric_series(text, intervention)
        for step, metrics in enumerate(metric_series, start=1):
            descriptor = self.descriptor_extractor.extract_text_descriptor(
                text,
                task_name=spec.task_name,
                benchmark_label=spec.benchmark_label,
                substrate=spec.substrate,
                realism_level=spec.realism_level,
                metadata=spec.metadata,
                retrieval_context=retrieval_context,
            )
            profile = self.compressibility.profile_text(
                text,
                predictive_loss=metrics.get("loss_proxy"),
            )
            snapshots.append(
                CapabilitySnapshot(
                    step=step,
                    metrics=metrics,
                    descriptor=descriptor,
                    compressibility=profile,
                    notes={"intervention": intervention.to_dict()},
                )
            )

        trajectory = CapabilityTrajectory(
            experiment_id=spec.experiment_id,
            substrate=spec.substrate,
            intervention_config=intervention.to_dict(),
            snapshots=snapshots,
        )
        trajectory.boundary_events = self.boundary.detect_events(snapshots, metric="capability_score")
        trajectory.fitted_boundaries = [self.boundary.fit_threshold(snapshots, metric="capability_score")]
        trajectory.aggregate_metrics = {
            "series_metrics": aggregate_snapshot_metrics(metric_series),
            "phase_summary": self.boundary.summarize_phase_region(snapshots, metric="capability_score"),
            "calibration_error": calibration_error(metric_series),
        }

        bundle = ArtifactBundle(
            spec=spec,
            trajectory=trajectory,
            linked_repositories={
                "substrate": self.substrate_adapter.link_metadata(),
                "agent": self.agent_adapter.link_metadata(),
                "wind_tunnel": self.wind_tunnel_adapter.link_metadata(),
            },
        )
        bundle.narrative = self.agent_adapter.narrate(bundle.to_dict())
        if export_dir is not None:
            bundle.export_path = self.export(bundle, export_dir=export_dir)
        return bundle

    def run_measured_experiment(
        self,
        spec: ExperimentSpec,
        intervention: InterventionConfig,
        *,
        task_family: str,
        seed: int,
        scale: int,
        data_tokens: int,
        train_steps: int = 4,
        export_dir: str | Path | None = None,
    ) -> ArtifactBundle:
        measured = self.measured_executor.run(
            task_family=task_family,
            seed=seed,
            scale=scale,
            data_tokens=data_tokens,
            num_layers=int(intervention.architecture.get("num_layers", 2)),
            train_steps=train_steps,
            seq_length=int(intervention.context_geometry.get("max_seq_len", 24)),
            learning_rate=float(intervention.objective.get("learning_rate", 1e-4)),
        )
        text = measured["train_text"]
        bundle = self.run_text_experiment(
            spec,
            intervention,
            text=text,
            retrieval_context=measured["holdout_text"],
            metric_series=measured["metric_series"],
            export_dir=export_dir,
        )
        bundle.trajectory.aggregate_metrics["generalization_gap"] = measured["generalization_gap"]
        bundle.trajectory.aggregate_metrics["measured_mode"] = True
        bundle.trajectory.aggregate_metrics["weight_compressibility"] = measured["weight_compressibility"]
        bundle.spec.metadata.update(
            {
                "seed": seed,
                "task_family": task_family,
                "task_family_code": measured["task_family_code"],
                "data_tokens": data_tokens,
                "scale": scale,
                "descriptor_hints": measured["descriptor_hints"],
            }
        )
        if export_dir is not None:
            bundle.export_path = self.export(bundle, export_dir=export_dir)
        return bundle

    def profile_gpt1_wind_tunnel(
        self,
        *,
        prompt: str,
        intervention: InterventionConfig,
        export_dir: str | Path | None = None,
    ) -> ArtifactBundle:
        architecture = intervention.architecture
        metrics = self.wind_tunnel_adapter.dry_run_metrics(
            prompt=prompt,
            vocab_size=int(architecture.get("vocab_size", 96)),
            d_model=int(architecture.get("d_model", 64)),
            num_heads=int(architecture.get("num_heads", 4)),
            num_layers=int(architecture.get("num_layers", 2)),
            d_ff=int(architecture.get("d_ff", 128)),
            max_seq_len=int(intervention.context_geometry.get("max_seq_len", 64)),
        )
        metric_series = [
            {
                "capability_score": float(np.tanh(metrics["capacity_proxy"] / 5000.0)),
                "loss_proxy": float(max(0.1, 1.2 - np.tanh(metrics["logit_std"]))),
                "retrieval_dependence": float(intervention.retrieval.get("enabled", False)),
            },
            {
                "capability_score": float(min(1.0, np.tanh(metrics["capacity_proxy"] / 3500.0) + 0.08)),
                "loss_proxy": float(max(0.05, 0.95 - np.tanh(metrics["logit_std"]))),
                "retrieval_dependence": float(intervention.retrieval.get("enabled", False)),
            },
        ]
        spec = ExperimentSpec(
            experiment_id="gpt1-wind-tunnel",
            substrate="gpt1-from-sutskever30",
            task_name="wind_tunnel_probe",
            benchmark_label="gpt1_dry_run",
            realism_level="semi_synthetic",
            objective_type=str(intervention.objective.get("loss_type", "next_token")),
            model_family="gpt1",
            intervention_axes=list(intervention.flattened().keys()),
            metadata={"prompt_length": len(prompt)},
        )
        return self.run_text_experiment(
            spec,
            intervention,
            text=prompt,
            retrieval_context="",
            metric_series=metric_series,
            export_dir=export_dir,
        )

    @staticmethod
    def export(bundle: ArtifactBundle, *, export_dir: str | Path) -> str:
        path = Path(export_dir)
        path.mkdir(parents=True, exist_ok=True)
        export_path = path / f"{bundle.spec.experiment_id}.json"
        export_path.write_text(json.dumps(bundle.to_dict(), indent=2))
        return str(export_path)

    @staticmethod
    def _default_metric_series(text: str, intervention: InterventionConfig) -> List[Dict[str, float]]:
        length_factor = min(len(text) / 200.0, 1.0)
        retrieval_penalty = float(intervention.retrieval.get("distractor_density", 0.0))
        context_bonus = min(float(intervention.context_geometry.get("answer_position", 0)) / 100.0, 0.2)
        scale = float(intervention.architecture.get("d_model", 64)) * float(intervention.architecture.get("num_layers", 2))
        data_tokens = float(intervention.data_regime.get("data_tokens", intervention.data_regime.get("dataset_size", 4096)))
        descriptor_complexity = max(0.0, 1.0 - length_factor - context_bonus)
        noise_penalty = float(intervention.data_regime.get("noise_level", 0.0))
        base = estimate_capability_score(
            scale=scale,
            data_tokens=data_tokens,
            descriptor_complexity=descriptor_complexity,
            retrieval_penalty=retrieval_penalty,
            noise_penalty=noise_penalty,
        )
        phases = (0.0, 0.08, 0.16, 0.22, 0.27)
        return [
            {
                "capability_score": float(min(base + phases[0], 1.0)),
                "loss_proxy": float(max(0.1, 1.05 - (base + phases[0]))),
                "retrieval_dependence": float(intervention.retrieval.get("enabled", False)),
                "data_tokens": data_tokens,
                "scale_proxy": scale,
            },
            {
                "capability_score": float(min(base + phases[1], 1.0)),
                "loss_proxy": float(max(0.08, 0.98 - (base + phases[1]))),
                "retrieval_dependence": float(intervention.retrieval.get("enabled", False)),
                "data_tokens": data_tokens,
                "scale_proxy": scale,
            },
            {
                "capability_score": float(min(base + phases[2], 1.0)),
                "loss_proxy": float(max(0.06, 0.88 - (base + phases[2]))),
                "retrieval_dependence": float(intervention.retrieval.get("enabled", False)),
                "data_tokens": data_tokens,
                "scale_proxy": scale,
            },
            {
                "capability_score": float(min(base + phases[3], 1.0)),
                "loss_proxy": float(max(0.04, 0.80 - (base + phases[3]))),
                "retrieval_dependence": float(intervention.retrieval.get("enabled", False)),
                "data_tokens": data_tokens,
                "scale_proxy": scale,
            },
            {
                "capability_score": float(min(base + phases[4], 1.0)),
                "loss_proxy": float(max(0.03, 0.74 - (base + phases[4]))),
                "retrieval_dependence": float(intervention.retrieval.get("enabled", False)),
                "data_tokens": data_tokens,
                "scale_proxy": scale,
            },
        ]
