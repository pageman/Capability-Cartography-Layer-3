"""Full Layer 3 orchestration.

Runs both the Layer 2 pipeline (sweeps, measured runs, failure atlas, plots,
notebook execution, agent export) and the Layer 3 pipeline (estimator sweep
over all 30 papers, causal atlas, middle-regime analysis, transfer
diagnostics, causal visualizations).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .agent_integration import SutskeverAgentWorkflowBridge
from .causal_atlas import CausalAtlasClassifier
from .causal_registry import BASELINE_SOURCES, estimator_aliases, estimator_display_name
from .causal_visualization import plot_estimator_heatmap, plot_regime_map, plot_verdict_dashboard
from .estimator_sweep import EstimatorSweepRunner
from .failure_atlas import FailureAtlasClassifier
from .middle_regime import MiddleRegimeAnalyzer
from .notebook_runner import NotebookExecutionWrapper
from .paper_registry import PaperRegistry
from .runner import CapabilityCartographyRunner
from .schemas import ExperimentSpec, InterventionConfig
from .storage import RunStorage
from .sweeps import SweepRunner
from .transfer_diagnostics import TransferDiagnosticsRunner
from .visualization import CartographyVisualizer


class FullStudyOrchestrator:
    """Run the full Layer 3 study stack."""

    def __init__(self, runner: CapabilityCartographyRunner, *, output_root: str | Path):
        self.runner = runner
        self.output_root = Path(output_root)
        self.sweep_runner = SweepRunner(runner, self.output_root)
        self.failure_atlas = FailureAtlasClassifier()
        self.visualizer = CartographyVisualizer()
        self.notebook_wrapper = NotebookExecutionWrapper(runner.substrate_adapter)
        self.agent_bridge = SutskeverAgentWorkflowBridge(runner.agent_adapter)
        # Layer 3
        self.paper_registry = PaperRegistry()
        self.estimator_sweep = EstimatorSweepRunner()
        self.causal_atlas = CausalAtlasClassifier()
        self.middle_regime = MiddleRegimeAnalyzer()
        self.transfer_diag = TransferDiagnosticsRunner()
        self.storage = RunStorage(self.output_root)

    def run(self, *, spec: ExperimentSpec, intervention: InterventionConfig) -> Dict[str, Any]:
        # ==============================================================
        # LAYER 2 PIPELINE
        # ==============================================================
        sweep_result = self.sweep_runner.run_grid(
            base_spec=spec, base_intervention=intervention,
            text="Capability formation depends on scale, task structure, and retrieval geometry.",
            retrieval_context="Linked substrate context and retrieval passages.",
            scale_values=[32, 64, 128], data_token_values=[2048, 8192, 32768],
            task_family_values=["synthetic_reasoning", "retrieval_qa"], seeds=[1, 2],
        )
        measured_result = self.sweep_runner.run_measured_grid(
            base_spec=spec, base_intervention=intervention,
            task_family_values=["object_tracking", "pair_matching", "babi_simple", "retrieval_qa"],
            scale_values=[32, 64], data_token_values=[1024, 2048], seeds=[1, 2], train_steps=2,
        )
        measured_records = measured_result["records"]
        failure_summary = self.failure_atlas.train(measured_records)
        self.failure_atlas.export(self.output_root / "failure_atlas" / "failure_atlas.json", failure_summary)
        onset_plot = self.visualizer.plot_onset_surface(measured_records, output_path=self.output_root / "plots" / "onset_surface.png")
        phase_plot = self.visualizer.plot_phase_regions(measured_records, output_path=self.output_root / "plots" / "phase_regions.png")

        try:
            notebook_report = self.notebook_wrapper.execute_notebook("22_scaling_laws", output_dir=self.output_root / "notebooks")
        except Exception as exc:
            notebook_report = {"notebook_name": "22_scaling_laws", "returncode": -1, "stderr": str(exc), "report_path": ""}

        # ==============================================================
        # LAYER 3 PIPELINE
        # ==============================================================

        # 3a. Estimator sweep over all 30 papers
        all_paper_results = self.estimator_sweep.sweep_all_papers(n=200, beta_true=1.0)

        # Build enriched records for each paper
        causal_records = []
        for pid in self.paper_registry.all_ids():
            paper = self.paper_registry.get(pid)
            results = all_paper_results.get(pid, [])
            cons = self.estimator_sweep.consensus(results)
            causal_records.append({
                "paper_id": pid,
                "paper_name": paper.title,
                "mechanism_X": paper.mechanism_X,
                "capability_Y": paper.capability_Y,
                "paper_type": paper.paper_type,
                "data_structure": paper.data_structure,
                "n_applicable": cons["n_applicable"],
                "n_consistent": cons["n_consistent"],
                "estimator_consensus": cons["consensus"],
                "avg_estimator_bias": cons["avg_bias"],
                "avg_estimator_mse": cons["avg_mse"],
                "best_estimator": cons["best_estimator"],
                "best_estimator_display": estimator_display_name(cons["best_estimator"]),
                "best_estimator_aliases": list(estimator_aliases(cons["best_estimator"])),
                "retrieval_dependence": 1.0 if paper.paper_type == "retrieval" else 0.0,
                "capability_score": max(0.1, 0.35 + 0.3 * cons["consensus"] - 0.15 * (1.0 if paper.paper_type == "retrieval" else 0.0)),
                "generalization_gap": cons["avg_bias"] - 0.05,
                "causality_verdict": (
                    "CONFIRMED" if cons["consensus"] > 0.6 else
                    "CONDITIONAL" if cons["consensus"] > 0.3 else "UNCONFIRMED"
                ),
                "causal_question": paper.causal_question,
            })

        # 3b. Causal atlas
        causal_atlas_summary = self.causal_atlas.train(causal_records)
        self.causal_atlas.export(self.output_root / "causal" / "causal_atlas.json", causal_atlas_summary)

        # 3c. Middle regime
        profiles = self.middle_regime.profile_all(causal_records)
        regime_summary = self.middle_regime.summary(profiles)
        self.storage.save_json("causal/middle_regime_summary.json", regime_summary)

        # 3d. Transfer diagnostics
        transfer_path = self.transfer_diag.export(self.output_root / "causal" / "transfer_diagnostics.json")
        transfer_result = self.transfer_diag.run()

        # 3e. Causal visualizations
        paper_names = {p.paper_id: p.title for p in self.paper_registry.papers.values()}
        est_names = self.estimator_sweep.registry.all_names()
        paper_result_dicts = {
            pid: [r.to_dict() for r in results] for pid, results in all_paper_results.items()
        }
        heatmap_path = plot_estimator_heatmap(
            paper_result_dicts, paper_names, est_names,
            output_path=self.output_root / "causal" / "estimator_heatmap.png",
        )
        verdict_path = plot_verdict_dashboard(
            causal_records, output_path=self.output_root / "causal" / "verdict_dashboard.png",
        )
        regime_path = plot_regime_map(
            [p.to_dict() for p in profiles], output_path=self.output_root / "causal" / "regime_map.png",
        )

        # Save records
        verdict_counts: Dict[str, int] = {}
        for record in causal_records:
            verdict = record["causality_verdict"]
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        self.storage.save_json("causal/causal_records.json", causal_records)
        self.storage.save_json("causal/estimator_sweep_summary.json", {
            "total_papers": len(causal_records),
            "total_combinations": sum(len(v) for v in all_paper_results.values()),
            "verdict_counts": verdict_counts,
            "best_estimator_display_map": {
                name: estimator_display_name(name) for name in self.estimator_sweep.registry.all_names()
            },
            "best_estimator_alias_map": {
                name: list(estimator_aliases(name)) for name in self.estimator_sweep.registry.all_names()
            },
            "baseline_sources": list(BASELINE_SOURCES),
            "comparison_note": (
                "The Downloads causality note is a historical baseline for estimator inventory and sources. "
                "Current Layer 3 outputs are defined by artifacts/layer3/causal/*.json."
            ),
        })

        # Agent brief
        brief = self.agent_bridge.build_agent_brief(
            measured_summary=measured_result["summary"],
            failure_atlas_summary=failure_summary,
            visualization_paths=[onset_plot, phase_plot, heatmap_path, verdict_path, regime_path, transfer_path],
        )
        agent_bundle = self.agent_bridge.export_workflow_bundle(output_dir=self.output_root / "agent", brief=brief)

        return {
            "sweep_summary": sweep_result["summary"],
            "measured_summary": measured_result["summary"],
            "failure_atlas": failure_summary,
            "notebook_report": notebook_report,
            "causal_records": causal_records,
            "causal_atlas": causal_atlas_summary,
            "middle_regime": regime_summary,
            "transfer_diagnostics": transfer_result,
            "plots": [onset_plot, phase_plot, heatmap_path, verdict_path, regime_path],
            "agent_bundle": agent_bundle,
        }
