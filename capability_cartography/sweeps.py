"""Sweep execution for scale/data/task experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence

from .runner import CapabilityCartographyRunner
from .schemas import ExperimentSpec, InterventionConfig
from .storage import RunStorage
from .surfaces import CapabilitySurfaceFitter
from .validation import PredictiveLawValidator


class SweepRunner:
    """Run intervention grids and export a sweep registry."""

    def __init__(self, runner: CapabilityCartographyRunner, storage_root: str | Path):
        self.runner = runner
        self.storage = RunStorage(storage_root)
        self.surface_fitter = CapabilitySurfaceFitter()
        self.validator = PredictiveLawValidator()

    def run_grid(
        self,
        *,
        base_spec: ExperimentSpec,
        base_intervention: InterventionConfig,
        text: str,
        retrieval_context: str,
        scale_values: Sequence[int],
        data_token_values: Sequence[int],
        task_family_values: Sequence[str],
        seeds: Sequence[int],
    ) -> Dict[str, object]:
        records: List[Dict[str, object]] = []
        for scale in scale_values:
            for data_tokens in data_token_values:
                for task_family in task_family_values:
                    for seed in seeds:
                        intervention = InterventionConfig(**deepcopy(base_intervention.to_dict()))
                        intervention.architecture["d_model"] = int(scale)
                        intervention.data_regime["data_tokens"] = int(data_tokens)
                        intervention.data_regime["task_family"] = task_family
                        spec = ExperimentSpec(
                            experiment_id=f"{base_spec.experiment_id}-scale{scale}-data{data_tokens}-{task_family}-seed{seed}",
                            substrate=base_spec.substrate,
                            task_name=task_family,
                            benchmark_label=base_spec.benchmark_label,
                            realism_level=base_spec.realism_level,
                            objective_type=base_spec.objective_type,
                            model_family=base_spec.model_family,
                            intervention_axes=base_spec.intervention_axes,
                            metadata={**base_spec.metadata, "seed": seed, "task_family": task_family},
                        )
                        bundle = self.runner.run_text_experiment(
                            spec,
                            intervention,
                            text=text,
                            retrieval_context=retrieval_context,
                        )
                        aggregates = bundle.trajectory.aggregate_metrics.get("series_metrics", {})
                        records.append(
                            {
                                "experiment_id": spec.experiment_id,
                                "scale": scale,
                                "data_tokens": data_tokens,
                                "task_family": task_family,
                                "seed": seed,
                                "capability_score": aggregates.get("capability_score", {}).get("mean", 0.0),
                                "loss_proxy": aggregates.get("loss_proxy", {}).get("mean", 0.0),
                                "retrieval_dependence": aggregates.get("retrieval_dependence", {}).get("mean", 0.0),
                                "calibration_error": bundle.trajectory.aggregate_metrics.get("calibration_error", 0.0),
                            }
                        )

        surface_fit = self.surface_fitter.fit_linear_surface(
            records,
            feature_keys=("scale", "data_tokens", "retrieval_dependence"),
        )
        onset_by_scale = self.surface_fitter.onset_threshold_by_feature(records, feature_key="scale")
        onset_by_data = self.surface_fitter.onset_threshold_by_feature(records, feature_key="data_tokens")
        summary = {
            "record_count": len(records),
            "surface_fit": surface_fit,
            "onset_by_scale": onset_by_scale,
            "onset_by_data": onset_by_data,
        }
        self.storage.save_records_csv("sweeps/sweep_records.csv", records)
        self.storage.save_json("sweeps/sweep_summary.json", summary)
        self.storage.append_jsonl("sweeps/sweep_records.jsonl", {"summary": summary})
        return {"records": records, "summary": summary}

    def run_measured_grid(
        self,
        *,
        base_spec: ExperimentSpec,
        base_intervention: InterventionConfig,
        task_family_values: Sequence[str],
        scale_values: Sequence[int],
        data_token_values: Sequence[int],
        seeds: Sequence[int],
        train_steps: int = 4,
    ) -> Dict[str, object]:
        records: List[Dict[str, object]] = []
        for task_family in task_family_values:
            for scale in scale_values:
                for data_tokens in data_token_values:
                    for seed in seeds:
                        intervention = InterventionConfig(**deepcopy(base_intervention.to_dict()))
                        intervention.architecture["d_model"] = int(scale)
                        intervention.data_regime["data_tokens"] = int(data_tokens)
                        spec = ExperimentSpec(
                            experiment_id=f"measured-{task_family}-scale{scale}-data{data_tokens}-seed{seed}",
                            substrate="gpt1-from-sutskever30",
                            task_name=task_family,
                            benchmark_label=base_spec.benchmark_label,
                            realism_level=base_spec.realism_level,
                            objective_type=base_spec.objective_type,
                            model_family="gpt1-measured",
                            intervention_axes=base_spec.intervention_axes,
                            metadata={"seed": seed, "task_family": task_family},
                        )
                        bundle = self.runner.run_measured_experiment(
                            spec,
                            intervention,
                            task_family=task_family,
                            seed=seed,
                            scale=scale,
                            data_tokens=data_tokens,
                            train_steps=train_steps,
                        )
                        aggregates = bundle.trajectory.aggregate_metrics["series_metrics"]
                        record = {
                            "experiment_id": spec.experiment_id,
                            "task_family": task_family,
                            "task_family_code": float(bundle.spec.metadata["task_family_code"]),
                            "seed": seed,
                            "scale": scale,
                            "data_tokens": data_tokens,
                            "capability_score": aggregates["capability_score"]["mean"],
                            "loss_proxy": aggregates["loss_proxy"]["mean"],
                            "generalization_gap": bundle.trajectory.aggregate_metrics["generalization_gap"],
                            "retrieval_dependence": aggregates["retrieval_dependence"]["mean"],
                        }
                        records.append(record)

        validation = self.validator.fit_and_validate(
            records,
            feature_keys=("scale", "data_tokens", "task_family_code", "retrieval_dependence"),
        )
        summary = {
            "record_count": len(records),
            "validation": validation,
        }
        self.storage.save_records_csv("measured/measured_records.csv", records)
        self.storage.save_json("measured/measured_summary.json", summary)
        self.storage.save_json("measured/measured_laws.json", {"laws": validation["laws"]})
        return {"records": records, "summary": summary}
