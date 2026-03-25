"""Runnable demo for Capability Cartography Layer 3.

Runs the full pipeline:
  Layer 2: text experiment → wind tunnel → sweep → measured grid
  Layer 3: paper registry → estimator sweep (real IV) → causal atlas →
           middle regime → transfer diagnostics → visualizations →
           full orchestration
"""

from __future__ import annotations

import json
from pathlib import Path

from .causal_atlas import CausalAtlasClassifier
from .causal_registry import CausalEstimatorRegistry
from .estimator_sweep import EstimatorSweepRunner
from .middle_regime import MiddleRegimeAnalyzer
from .orchestration import FullStudyOrchestrator
from .paper_registry import PaperRegistry
from .runner import CapabilityCartographyRunner
from .schemas import ExperimentSpec, InterventionConfig
from .sweeps import SweepRunner
from .transfer_diagnostics import TransferDiagnosticsRunner


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

    # ==============================================================
    # Layer 2 baseline
    # ==============================================================
    bundle = runner.run_text_experiment(
        spec, intervention, text=text, retrieval_context=retrieval_context, export_dir=artifacts_dir,
    )
    wind_tunnel = runner.profile_gpt1_wind_tunnel(
        prompt="the capability atlas predicts a threshold", intervention=intervention, export_dir=artifacts_dir,
    )

    print("Capability Cartography Layer 3 Demo")
    print(f"  Text experiment: {bundle.spec.experiment_id}")
    print(f"  Wind tunnel: {wind_tunnel.spec.experiment_id}")

    sweep_runner = SweepRunner(runner, artifacts_dir)
    sweep_result = sweep_runner.run_grid(
        base_spec=spec, base_intervention=intervention, text=text, retrieval_context=retrieval_context,
        scale_values=[32, 64, 128], data_token_values=[2048, 8192, 32768],
        task_family_values=["synthetic_reasoning", "retrieval_qa"], seeds=[1, 2],
    )
    print(f"\n  Sweep: {sweep_result['summary']['record_count']} records, "
          f"R²={sweep_result['summary']['surface_fit']['r2']:.4f}")

    measured_result = sweep_runner.run_measured_grid(
        base_spec=spec, base_intervention=intervention,
        task_family_values=["object_tracking", "pair_matching", "babi_simple", "retrieval_qa"],
        scale_values=[32, 64], data_token_values=[1024, 2048], seeds=[1, 2], train_steps=2,
    )
    v = measured_result["summary"]["validation"]
    print(f"  Measured: {measured_result['summary']['record_count']} records, "
          f"holdout R²={v['validation']['r2']:.4f}, MAE={v['validation']['mae']:.4f}")

    # ==============================================================
    # Layer 3: Causal analysis
    # ==============================================================

    # Registry
    registry = CausalEstimatorRegistry()
    papers = PaperRegistry()
    print(f"\n  Estimator Registry: {registry.summary()['total']} estimators, "
          f"{len(registry.summary()['by_family'])} families")
    print(f"  Paper Registry: {papers.summary()['total']} papers")

    # Estimator sweep with REAL IV implementations
    sweeper = EstimatorSweepRunner()
    print("\n  Running estimator sweep on all 30 papers...")
    all_results = sweeper.sweep_all_papers(n=200, beta_true=1.0)

    confirmed = 0
    conditional = 0
    for pid in papers.all_ids():
        paper = papers.get(pid)
        results = all_results[pid]
        cons = sweeper.consensus(results)
        verdict = "CONFIRMED" if cons["consensus"] > 0.6 else ("CONDITIONAL" if cons["consensus"] > 0.3 else "UNCONFIRMED")
        if verdict == "CONFIRMED":
            confirmed += 1
        else:
            conditional += 1
        sym = {"CONFIRMED": "✓", "CONDITIONAL": "~", "UNCONFIRMED": "✗"}[verdict]
        print(f"    P{pid:02d} {paper.title[:35]:35s} {sym} {verdict:12s} "
              f"consensus={cons['consensus']:.2f} best={cons['best_estimator'][:16]}")

    print(f"\n  Verdicts: {confirmed} CONFIRMED, {conditional} CONDITIONAL")

    # Causal atlas
    causal_records = []
    for pid in papers.all_ids():
        paper = papers.get(pid)
        cons = sweeper.consensus(all_results[pid])
        causal_records.append({
            "paper_id": pid, "paper_name": paper.title,
            "paper_type": paper.paper_type, "data_structure": paper.data_structure,
            "n_applicable": cons["n_applicable"], "n_consistent": cons["n_consistent"],
            "estimator_consensus": cons["consensus"],
            "avg_estimator_bias": cons["avg_bias"], "retrieval_dependence": 1.0 if paper.paper_type == "retrieval" else 0.0,
        })
    atlas_cls = CausalAtlasClassifier()
    atlas = atlas_cls.train(causal_records)
    print(f"\n  Causal Atlas: {atlas['label_counts']}")

    # Middle regime
    analyzer = MiddleRegimeAnalyzer()
    profiles = analyzer.profile_all(causal_records)
    regime = analyzer.summary(profiles)
    print(f"  Middle Regime: {regime['regime_counts']}, "
          f"{regime['n_splitup_needed']} papers need SplitUP")

    # Transfer diagnostics
    td = TransferDiagnosticsRunner()
    td_result = td.run()
    print(f"  Transfer: {td_result['transfer_summary']}")

    # ==============================================================
    # Full orchestration
    # ==============================================================
    print("\n  Running full Layer 3 orchestration...")
    orchestrator = FullStudyOrchestrator(runner, output_root=artifacts_dir / "layer3")
    orch = orchestrator.run(spec=spec, intervention=intervention)
    print(f"  Orchestration complete:")
    print(f"    Measured: {orch['measured_summary']['record_count']} records")
    print(f"    Failure atlas: {orch['failure_atlas']['label_counts']}")
    print(f"    Causal atlas: {orch['causal_atlas']['label_counts']}")
    print(f"    Middle regime: {orch['middle_regime']}")
    print(f"    Transfer: {orch['transfer_diagnostics']['transfer_summary']}")
    print(f"    Plots: {len(orch['plots'])} generated")
    print(f"\n  All artifacts saved to {artifacts_dir / 'layer3'}")


if __name__ == "__main__":
    main()
