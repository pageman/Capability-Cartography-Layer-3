"""Tests for Capability Cartography Layer 3.

Covers all five new modules, real IV estimators, paper registry,
causal visualization, plus inherited Layer 2 functionality.
"""

import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from capability_cartography.adapters import (
    AgentOverlayAdapter, BeyondNumpyAdapter, GPT1WindTunnelAdapter, NotebookSubstrateAdapter,
)
from capability_cartography.boundary import BoundaryAnalyzer
from capability_cartography.causal_atlas import CausalAtlasClassifier
from capability_cartography.causal_registry import CausalEstimatorRegistry
from capability_cartography.compressibility import CompressibilityStack
from capability_cartography.descriptors import TaskDescriptorExtractor
from capability_cartography.estimator_sweep import EstimatorSweepRunner
from capability_cartography.failure_atlas import FailureAtlasClassifier
from capability_cartography.iv_estimators import (
    fuller, generate_iv_data, ivw, liml, mr_egger, ols, splitup, splitup_analytic, ts_iv, tsls, up_gmm,
)
from capability_cartography.middle_regime import MiddleRegimeAnalyzer, classify_regime, measurement_error_bias
from capability_cartography.notebook_runner import NotebookExecutionWrapper
from capability_cartography.paper_registry import PaperRegistry
from capability_cartography.runner import CapabilityCartographyRunner
from capability_cartography.schemas import (
    CapabilitySnapshot, CausalEstimator, CausalRecord, CompressibilityProfile,
    EstimatorResult, ExperimentSpec, InterventionConfig, MiddleRegimeProfile,
    TaskDescriptor, TransferDiagnostic,
)
from capability_cartography.transfer_diagnostics import TransferDiagnosticsRunner
from capability_cartography.validation import PredictiveLawValidator
from capability_cartography.verdict_policy import VerdictPolicy


class RealIVEstimatorTests(unittest.TestCase):
    """Tests for iv_estimators.py — real numpy/scipy implementations."""

    def setUp(self):
        self.paired = generate_iv_data(n=200, m=10, d=1, beta_true=1.0, seed=42)
        self.unpaired = generate_iv_data(n=200, m=10, d=1, beta_true=1.0, seed=42, unpaired=True)

    def test_ols_runs(self):
        r = ols(self.paired["X"], self.paired["Y"])
        self.assertEqual(r.name, "OLS")
        self.assertIsNotNone(r.beta)
        self.assertGreater(abs(r.scalar()), 0)

    def test_ols_is_biased(self):
        """OLS should be biased toward beta + gamma (confounding)."""
        r = ols(self.paired["X"], self.paired["Y"])
        self.assertGreater(r.scalar(), 1.1)  # biased upward by gamma=0.5

    def test_tsls_runs(self):
        r = tsls(self.paired["Z"], self.paired["X"], self.paired["Y"])
        self.assertEqual(r.name, "2SLS")
        self.assertAlmostEqual(r.scalar(), 1.0, delta=0.5)

    def test_tsls_less_biased_than_ols(self):
        ols_r = ols(self.paired["X"], self.paired["Y"])
        tsls_r = tsls(self.paired["Z"], self.paired["X"], self.paired["Y"])
        self.assertLess(abs(tsls_r.scalar() - 1.0), abs(ols_r.scalar() - 1.0))

    def test_liml_runs(self):
        r = liml(self.paired["Z"], self.paired["X"], self.paired["Y"])
        self.assertEqual(r.name, "LIML")
        self.assertAlmostEqual(r.scalar(), 1.0, delta=0.5)

    def test_fuller_runs(self):
        r = fuller(self.paired["Z"], self.paired["X"], self.paired["Y"])
        self.assertEqual(r.name, "Fuller_k")
        self.assertAlmostEqual(r.scalar(), 1.0, delta=0.5)

    def test_ts_iv_runs(self):
        r = ts_iv(self.unpaired["Z_x"], self.unpaired["X"],
                   self.unpaired["Z_y"], self.unpaired["Y"])
        self.assertEqual(r.name, "TS_IV")
        self.assertGreater(abs(r.scalar()), 0)

    def test_up_gmm_runs(self):
        r = up_gmm(self.unpaired["Z_x"], self.unpaired["X"],
                    self.unpaired["Z_y"], self.unpaired["Y"])
        self.assertEqual(r.name, "UP_GMM")

    def test_splitup_runs(self):
        r = splitup(self.unpaired["Z_x"], self.unpaired["X"],
                     self.unpaired["Z_y"], self.unpaired["Y"], H=10)
        self.assertEqual(r.name, "SplitUP")
        self.assertGreater(abs(r.scalar()), 0)

    def test_splitup_analytic_runs(self):
        r = splitup_analytic(self.unpaired["Z_x"], self.unpaired["X"],
                              self.unpaired["Z_y"], self.unpaired["Y"])
        self.assertEqual(r.name, "SplitUP_analytic")

    def test_ivw_runs(self):
        r = ivw(np.array([0.5, 0.3, 0.7]), np.array([0.5, 0.3, 0.7]), np.array([0.1, 0.1, 0.1]))
        self.assertEqual(r.name, "IVW")
        self.assertAlmostEqual(r.scalar(), 1.0, delta=0.2)

    def test_mr_egger_runs(self):
        r = mr_egger(np.array([0.5, 0.3, 0.7, 0.4]),
                      np.array([0.5, 0.3, 0.7, 0.4]),
                      np.array([0.1, 0.1, 0.1, 0.1]))
        self.assertEqual(r.name, "MR_Egger")

    def test_tsls_no_runtime_warnings(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("error", RuntimeWarning)
            r = tsls(self.paired["Z"], self.paired["X"], self.paired["Y"])
        self.assertEqual(len(caught), 0)
        self.assertTrue(np.all(np.isfinite(r.se)))

    def test_ivw_zero_denominator_is_warning_free(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("error", RuntimeWarning)
            r = ivw(np.array([0.0, 0.3, 0.0]), np.array([0.5, 0.3, 0.7]), np.array([0.1, 0.1, 0.1]))
        self.assertEqual(len(caught), 0)
        self.assertTrue(np.isfinite(r.scalar()))

    def test_generate_paired_data(self):
        d = generate_iv_data(n=100, m=5, d=1)
        self.assertIn("Z", d)
        self.assertEqual(d["Z"].shape[0], 100)

    def test_generate_unpaired_data(self):
        d = generate_iv_data(n=100, m=5, d=1, unpaired=True)
        self.assertIn("Z_x", d)
        self.assertIn("Z_y", d)


class PaperRegistryTests(unittest.TestCase):
    def test_has_30_papers(self):
        reg = PaperRegistry()
        self.assertEqual(len(reg.all_ids()), 30)

    def test_retrieval_papers(self):
        reg = PaperRegistry()
        retrieval = reg.retrieval_papers()
        self.assertEqual(len(retrieval), 3)
        self.assertEqual({p.paper_id for p in retrieval}, {28, 29, 30})

    def test_theory_papers(self):
        reg = PaperRegistry()
        theory = reg.theory_papers()
        self.assertEqual({p.paper_id for p in theory}, {1, 19, 23, 24, 25})

    def test_paper_has_causal_question(self):
        reg = PaperRegistry()
        for pid in reg.all_ids():
            paper = reg.get(pid)
            self.assertTrue(len(paper.causal_question) > 0, f"P{pid} missing causal question")

    def test_summary(self):
        reg = PaperRegistry()
        s = reg.summary()
        self.assertEqual(s["total"], 30)


class EstimatorSweepTests(unittest.TestCase):
    def test_sweep_architecture_paper_real(self):
        sweeper = EstimatorSweepRunner()
        results = sweeper.sweep_paper(13)  # Transformer
        self.assertEqual(len(results), 27)
        applicable = [r for r in results if r.applicable]
        self.assertEqual(len(applicable), 27)

    def test_sweep_retrieval_paper_real(self):
        sweeper = EstimatorSweepRunner()
        results = sweeper.sweep_paper(30)  # Lost in Middle
        applicable = [r for r in results if r.applicable]
        self.assertLess(len(applicable), 27)

    def test_splitup_consistent_in_all_sweeps(self):
        sweeper = EstimatorSweepRunner()
        for pid in [2, 13, 22, 28, 30]:
            results = sweeper.sweep_paper(pid)
            splitup_results = [r for r in results if "SplitUP" in r.estimator]
            for r in splitup_results:
                self.assertTrue(r.applicable, f"SplitUP not applicable for P{pid}")
                self.assertTrue(r.consistent, f"SplitUP not consistent for P{pid}")

    def test_consensus_structure(self):
        sweeper = EstimatorSweepRunner()
        results = sweeper.sweep_paper(13)
        cons = sweeper.consensus(results)
        self.assertIn("n_applicable", cons)
        self.assertIn("consensus", cons)
        self.assertIn("best_estimator", cons)
        self.assertIn("best_estimator_mode", cons)
        self.assertGreater(cons["consensus"], 0.5)

    def test_best_estimator_prefers_exact_implementations(self):
        sweeper = EstimatorSweepRunner()
        cons = sweeper.consensus(sweeper.sweep_paper(13))
        self.assertEqual(cons["best_estimator_mode"], "exact")

    def test_sweep_all_30(self):
        sweeper = EstimatorSweepRunner()
        all_results = sweeper.sweep_all_papers(n=50)  # small n for speed
        self.assertEqual(len(all_results), 30)


class CausalRegistryTests(unittest.TestCase):
    def test_27_estimators(self):
        reg = CausalEstimatorRegistry()
        self.assertEqual(len(reg.all_names()), 27)

    def test_families(self):
        reg = CausalEstimatorRegistry()
        s = reg.summary()
        self.assertIn("splitUP", s["by_family"])
        self.assertEqual(s["by_family"]["splitUP"], 3)

    def test_splitup_universally_consistent(self):
        reg = CausalEstimatorRegistry()
        for name in ["SplitUP_dense", "SplitUP_L1", "SplitUP_analytic"]:
            est = reg.get(name)
            self.assertTrue(est.consistent_finite_m)
            self.assertTrue(est.consistent_high_dim_m)
            self.assertTrue(est.handles_unpaired)

    def test_unpaired_filtering(self):
        reg = CausalEstimatorRegistry()
        unpaired = reg.applicable_for(is_paired=False)
        for e in unpaired:
            self.assertFalse(e.requires_paired)


class CausalAtlasTests(unittest.TestCase):
    def test_retrieval_labeled_unpaired_bias(self):
        cls = CausalAtlasClassifier()
        label = cls.label({
            "estimator_consensus": 0.57, "retrieval_dependence": 1.0,
            "n_applicable": 14, "n_consistent": 8,
            "avg_estimator_bias": 0.06, "paper_type": "retrieval",
        })
        self.assertEqual(label, "unpaired_bias")

    def test_architecture_labeled_stable(self):
        cls = CausalAtlasClassifier()
        label = cls.label({
            "estimator_consensus": 0.96, "retrieval_dependence": 0.0,
            "n_applicable": 27, "n_consistent": 26,
            "avg_estimator_bias": 0.02, "paper_type": "architecture",
        })
        self.assertEqual(label, "stable_identification")

    def test_train_and_export(self):
        cls = CausalAtlasClassifier()
        records = [
            {"estimator_consensus": 0.96, "retrieval_dependence": 0.0, "n_applicable": 27, "n_consistent": 26, "avg_estimator_bias": 0.02, "paper_type": "architecture"},
            {"estimator_consensus": 0.57, "retrieval_dependence": 1.0, "n_applicable": 14, "n_consistent": 8, "avg_estimator_bias": 0.06, "paper_type": "retrieval"},
        ]
        summary = cls.train(records)
        self.assertEqual(summary["record_count"], 2)
        with TemporaryDirectory() as td:
            path = cls.export(Path(td) / "atlas.json", summary)
            self.assertTrue(Path(path).exists())

    def test_predict_matches_training_labels_on_simple_records(self):
        cls = CausalAtlasClassifier()
        records = [
            {"paper_id": 1, "paper_name": "a", "estimator_consensus": 0.96, "retrieval_dependence": 0.0, "n_applicable": 27, "n_consistent": 26, "avg_estimator_bias": 0.02, "paper_type": "architecture"},
            {"paper_id": 2, "paper_name": "b", "estimator_consensus": 0.45, "retrieval_dependence": 1.0, "n_applicable": 14, "n_consistent": 8, "avg_estimator_bias": 0.06, "paper_type": "retrieval"},
            {"paper_id": 3, "paper_name": "c", "estimator_consensus": 0.50, "retrieval_dependence": 0.0, "n_applicable": 13, "n_consistent": 7, "avg_estimator_bias": 0.03, "paper_type": "theory"},
        ]
        summary = cls.train(records)
        mismatches = [r for r in summary["records"] if r["actual_label"] != r["predicted_label"]]
        self.assertEqual(mismatches, [])


class VerdictPolicyTests(unittest.TestCase):
    def test_architecture_confirmed_requires_strong_consensus(self):
        p = VerdictPolicy()
        out = p.evaluate(paper_type="architecture", n_applicable=27, consensus=0.9)
        self.assertEqual(out["verdict"], "CONFIRMED")

    def test_theory_stays_conditional(self):
        p = VerdictPolicy()
        out = p.evaluate(paper_type="theory", n_applicable=13, consensus=1.0)
        self.assertEqual(out["verdict"], "CONDITIONAL")

    def test_retrieval_with_low_consensus_unconfirmed(self):
        p = VerdictPolicy()
        out = p.evaluate(paper_type="retrieval", n_applicable=13, consensus=0.2)
        self.assertEqual(out["verdict"], "UNCONFIRMED")


class MiddleRegimeTests(unittest.TestCase):
    def test_classify_classical(self):
        self.assertEqual(classify_regime(5, 100, 3, 2), "classical_large_sample")

    def test_classify_high_dim(self):
        self.assertIn("high_dim", classify_regime(200, 4, 5, 2))

    def test_bias_formula(self):
        result = measurement_error_bias(Q=0.5, r_tilde=0.25, b=0.1)
        self.assertAlmostEqual(result, 0.5 / (0.5 + 0.025), places=4)

    def test_retrieval_needs_splitup(self):
        analyzer = MiddleRegimeAnalyzer()
        profile = analyzer.profile_paper(30, "retrieval")
        self.assertTrue(profile.is_high_dim)
        self.assertTrue(profile.splitup_needed)

    def test_architecture_no_splitup(self):
        analyzer = MiddleRegimeAnalyzer()
        profile = analyzer.profile_paper(13, "architecture")
        self.assertFalse(profile.splitup_needed)

    def test_profile_all(self):
        analyzer = MiddleRegimeAnalyzer()
        records = [{"paper_id": i, "paper_type": "architecture"} for i in range(1, 4)]
        profiles = analyzer.profile_all(records)
        self.assertEqual(len(profiles), 3)

    def test_boundary_detection(self):
        analyzer = MiddleRegimeAnalyzer()
        events = analyzer.detect_regime_boundary([5, 10, 20, 50, 100, 500], r=4.0, d=3)
        self.assertIsInstance(events, list)


class TransferDiagnosticsTests(unittest.TestCase):
    def test_has_both_types(self):
        td = TransferDiagnosticsRunner()
        r = td.run()
        self.assertGreater(r["scale_invariant_count"], 0)
        self.assertGreater(r["scale_dependent_count"], 0)

    def test_export(self):
        td = TransferDiagnosticsRunner()
        with TemporaryDirectory() as tmpdir:
            path = td.export(Path(tmpdir) / "transfer.json")
            self.assertTrue(Path(path).exists())


class SchemaTests(unittest.TestCase):
    def test_causal_estimator(self):
        e = CausalEstimator(name="t", family="f", requires_paired=True, handles_sparsity=False,
                            consistent_finite_m=True, consistent_high_dim_m=False,
                            handles_unpaired=False, weak_iv_robust=False, bias_correction=False)
        self.assertEqual(e.to_dict()["name"], "t")

    def test_estimator_result(self):
        r = EstimatorResult(estimator="OLS", applicable=True, estimate=1.05, bias=0.05)
        self.assertAlmostEqual(r.to_dict()["estimate"], 1.05)

    def test_middle_regime_profile(self):
        p = MiddleRegimeProfile(paper_id=30, m=20, r=4.0, d=5, s_star=2, is_high_dim=True,
                                 measurement_error_bias=0.05, attenuation_factor=0.95,
                                 splitup_needed=True, regime_label="high_dim_moderate_r")
        self.assertTrue(p.to_dict()["splitup_needed"])

    def test_transfer_diagnostic(self):
        t = TransferDiagnostic(finding="test", scale_invariant=True, reason="math", confidence="high")
        self.assertTrue(t.to_dict()["scale_invariant"])


class InheritedLayerTwoTests(unittest.TestCase):
    def test_notebook_adapter_missing_dependency_message_is_actionable(self):
        adapter = NotebookSubstrateAdapter("/nonexistent")
        message = adapter.missing_dependency_message("22_scaling_laws")
        self.assertIn("SUTSKEVER30_ROOT", message)
        self.assertIn("22_scaling_laws.ipynb", message)
        self.assertIn("Canonical repository", message)

    def test_notebook_runner_falls_back_when_substrate_missing(self):
        wrapper = NotebookExecutionWrapper(NotebookSubstrateAdapter("/nonexistent"))
        with TemporaryDirectory() as td:
            report = wrapper.execute_notebook("22_scaling_laws", output_dir=td)
        self.assertFalse(report["executed"])
        self.assertIn("fallback_reason", report)
        self.assertEqual(report["substrate_diagnostics"]["env_var"], "SUTSKEVER30_ROOT")

    def test_notebook_runner_can_fail_strictly(self):
        wrapper = NotebookExecutionWrapper(NotebookSubstrateAdapter("/nonexistent"))
        with TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError) as ctx:
                wrapper.execute_notebook("22_scaling_laws", output_dir=td, allow_fallback=False)
        self.assertIn("SUTSKEVER30_ROOT", str(ctx.exception))

    def test_gpt1_adapter_missing_dependency_message_is_actionable(self):
        adapter = GPT1WindTunnelAdapter("/nonexistent")
        message = adapter.missing_dependency_message()
        self.assertIn("GPT1_WIND_TUNNEL_ROOT", message)
        self.assertIn("gpt1_complete_implementation.py", message)
        self.assertIn("Canonical repository", message)

    def test_runner_measured_experiment_falls_back_when_repo_missing(self):
        runner = CapabilityCartographyRunner(wind_tunnel_adapter=GPT1WindTunnelAdapter("/nonexistent"))
        intv = InterventionConfig(
            architecture={"d_model": 32, "num_layers": 2},
            objective={"loss_type": "next_token"},
            data_regime={},
            retrieval={},
            context_geometry={"max_seq_len": 24},
        )
        spec = ExperimentSpec(
            experiment_id="measured-fallback",
            substrate="gpt1-from-sutskever30",
            task_name="qa",
            benchmark_label="u",
            realism_level="semi_synthetic",
            objective_type="next_token",
            model_family="gpt1",
        )
        bundle = runner.run_measured_experiment(
            spec,
            intv,
            task_family="object_tracking",
            seed=1,
            scale=32,
            data_tokens=512,
            train_steps=2,
        )
        self.assertFalse(bundle.trajectory.aggregate_metrics["measured_mode"])
        self.assertIn("fallback_reason", bundle.trajectory.aggregate_metrics)
        self.assertFalse(bundle.trajectory.aggregate_metrics["wind_tunnel_available"])

    def test_runner_measured_experiment_can_fail_strictly(self):
        runner = CapabilityCartographyRunner(wind_tunnel_adapter=GPT1WindTunnelAdapter("/nonexistent"))
        intv = InterventionConfig(
            architecture={"d_model": 32, "num_layers": 2},
            objective={"loss_type": "next_token"},
            data_regime={},
            retrieval={},
            context_geometry={"max_seq_len": 24},
        )
        spec = ExperimentSpec(
            experiment_id="measured-strict",
            substrate="gpt1-from-sutskever30",
            task_name="qa",
            benchmark_label="u",
            realism_level="semi_synthetic",
            objective_type="next_token",
            model_family="gpt1",
        )
        with self.assertRaises(RuntimeError) as ctx:
            runner.run_measured_experiment(
                spec,
                intv,
                task_family="object_tracking",
                seed=1,
                scale=32,
                data_tokens=512,
                train_steps=2,
                allow_fallback=False,
            )
        self.assertIn("GPT1_WIND_TUNNEL_ROOT", str(ctx.exception))

    def test_descriptor_extraction(self):
        ext = TaskDescriptorExtractor()
        d = ext.extract_text_descriptor("Alice retrieves the passage.", task_name="qa",
                                         benchmark_label="u", substrate="t", retrieval_context="retrieves")
        self.assertGreaterEqual(d.retrieval_geometry["retrieval_dependency_score"], 0.0)

    def test_compressibility(self):
        stack = CompressibilityStack()
        p = stack.profile_text("Hello world, compressibility test.")
        self.assertIn("gzip_ratio", p.surface)

    def test_boundary_detection(self):
        analyzer = BoundaryAnalyzer()
        td = TaskDescriptor(task_name="t", benchmark_label="b", substrate="s", realism_level="r")
        cp = CompressibilityProfile(surface={}, predictive={}, structural={}, gaps={})
        snapshots = [CapabilitySnapshot(step=i, metrics={"cap": v}, descriptor=td, compressibility=cp)
                     for i, v in enumerate([0.1, 0.2, 0.8, 0.85])]
        events = analyzer.detect_events(snapshots, metric="cap")
        self.assertIsInstance(events, list)

    def test_failure_atlas(self):
        cls = FailureAtlasClassifier()
        s = cls.train([
            {"capability_score": 0.19, "generalization_gap": 0.05, "retrieval_dependence": 1.0,
             "task_family_code": 3.0, "scale": 32.0, "data_tokens": 1024.0},
            {"capability_score": 0.24, "generalization_gap": 0.01, "retrieval_dependence": 0.0,
             "task_family_code": 0.0, "scale": 64.0, "data_tokens": 2048.0},
        ])
        self.assertEqual(s["record_count"], 2)

    def test_runner(self):
        runner = CapabilityCartographyRunner()
        intv = InterventionConfig(architecture={"d_model": 32}, objective={"loss_type": "next_token"},
                                  data_regime={}, retrieval={}, context_geometry={"max_seq_len": 32})
        spec = ExperimentSpec(experiment_id="t", substrate="t", task_name="qa", benchmark_label="u",
                              realism_level="semi_synthetic", objective_type="next_token", model_family="t")
        b = runner.run_text_experiment(spec, intv, text="test", retrieval_context="ctx")
        self.assertIsNotNone(b.trajectory)

    def test_beyond_numpy_adapter(self):
        a = BeyondNumpyAdapter("/nonexistent")
        self.assertFalse(a.is_available())
        a2 = BeyondNumpyAdapter("/home/user/sutskever-30-beyond-numpy")
        if a2.is_available():
            self.assertEqual(len(a2.list_papers()), 30)


if __name__ == "__main__":
    unittest.main()
