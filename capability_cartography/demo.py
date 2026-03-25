"""Runnable demo for Capability Cartography Layer 3.

Runs the full pipeline:
  Layer 2: text experiment → wind tunnel → sweep → measured grid
  Layer 3: paper registry → estimator sweep (real IV) → causal atlas →
           middle regime → transfer diagnostics → visualizations →
           full orchestration
"""

from __future__ import annotations

from pathlib import Path

from .causal_registry import CausalEstimatorRegistry
from .orchestration import FullStudyOrchestrator
from .paper_registry import PaperRegistry
from .runner import CapabilityCartographyRunner
from .schemas import ExperimentSpec, InterventionConfig


def main() -> None:
    runner = CapabilityCartographyRunner()

    intervention = InterventionConfig(
        architecture={"d_model": 64, "num_heads": 4, "num_layers": 2, "d_ff": 128, "vocab_size": 96},
        objective={"loss_type": "next_token"},
        data_regime={"dataset_type": "semi_synthetic", "compressibility_target": 0.72},
        retrieval={"enabled": True, "distractor_density": 0.35, "position": "middle"},
        context_geometry={"answer_position": 48, "max_seq_len": 64},
        interpretability={"activation_patching": False},
    )

    spec = ExperimentSpec(
        experiment_id="capability-cartography-layer3-demo",
        substrate="sutskever-30-implementations",
        task_name="causal_identification_probe",
        benchmark_label="layer3_demo",
        realism_level="semi_synthetic",
        objective_type="next_token",
        model_family="gpt1-compatible",
        intervention_axes=list(intervention.flattened().keys()),
        metadata={"source": "Capability Cartography Layer 3 demo"},
    )

    text = (
        "A proof sketch is placed in the middle of a long context with several distractor passages. "
        "The model must retrieve the relevant lemma, reason through linked steps, and identify the answer."
    )
    retrieval_context = (
        "Distractor alpha. Distractor beta. Relevant lemma about linked reasoning. "
        "Additional irrelevant passage about unrelated retrieval documents."
    )

    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = repo_root / "artifacts"

    # Lightweight single-run baseline artifacts.
    bundle = runner.run_text_experiment(
        spec, intervention, text=text, retrieval_context=retrieval_context, export_dir=artifacts_dir,
    )
    wind_tunnel = runner.profile_gpt1_wind_tunnel(
        prompt="the capability atlas predicts a threshold", intervention=intervention, export_dir=artifacts_dir,
    )

    print("Capability Cartography Layer 3 Demo")
    print(f"  Text experiment: {bundle.spec.experiment_id}")
    print(f"  Wind tunnel: {wind_tunnel.spec.experiment_id}")

    # Registry summaries are cheap and useful even before the full study runs.
    registry = CausalEstimatorRegistry()
    papers = PaperRegistry()
    print(f"\n  Estimator Registry: {registry.summary()['total']} estimators, "
          f"{len(registry.summary()['by_family'])} families")
    print(f"  Paper Registry: {papers.summary()['total']} papers")

    # The heavy baseline and causal pipelines are run exactly once here.
    print("\n  Running full Layer 3 orchestration...")
    orchestrator = FullStudyOrchestrator(runner, output_root=artifacts_dir / "layer3")
    orch = orchestrator.run(spec=spec, intervention=intervention)
    print(f"  Orchestration complete:")
    print(f"    Sweep: {orch['sweep_summary']['record_count']} records, "
          f"R²={orch['sweep_summary']['surface_fit']['r2']:.4f}")
    print(f"    Measured: {orch['measured_summary']['record_count']} records")
    print(f"    Holdout R²={orch['measured_summary']['validation']['validation']['r2']:.4f}, "
          f"MAE={orch['measured_summary']['validation']['validation']['mae']:.4f}")
    print(f"    Failure atlas: {orch['failure_atlas']['label_counts']}")
    print(f"    Causal atlas: {orch['causal_atlas']['label_counts']}")
    print(f"    Middle regime: {orch['middle_regime']}")
    print(f"    Transfer: {orch['transfer_diagnostics']['transfer_summary']}")
    print(f"    Plots: {len(orch['plots'])} generated")
    print(f"\n  All artifacts saved to {artifacts_dir / 'layer3'}")


if __name__ == "__main__":
    main()
