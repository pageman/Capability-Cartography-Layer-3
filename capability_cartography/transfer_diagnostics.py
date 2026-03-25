"""Transfer diagnostics: which findings are scale-invariant?

Layer 3's transfer module checks each finding from the causal analysis
and flags it as:
  - scale_invariant: the finding is a mathematical property that holds
    regardless of model size (e.g. "SplitUP removes bias")
  - scale_dependent: the finding depends on the specific model/data
    regime and may not transfer (e.g. "coefficient = -0.0303")

This directly addresses Tao Mystery #2: "Whether these laws transfer
to GPT-4-scale models."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .schemas import TransferDiagnostic


def _build_diagnostics() -> List[TransferDiagnostic]:
    """Hardcoded diagnostics for the current analysis.  New ones should
    be added as the analysis evolves."""
    return [
        # ---- Scale-invariant ----
        TransferDiagnostic(
            finding="SplitUP removes measurement-error bias in unpaired high-dim IV",
            scale_invariant=True,
            reason="Mathematical property of cross-fold splitting (Schur et al. Theorem 4.7). "
                   "Does not depend on model size, only on the structure of the estimator.",
            confidence="high",
            evidence_at_scale="none (theorem is asymptotic; finite-sample verified at m=100-3200)",
        ),
        TransferDiagnostic(
            finding="Retrieval papers have fewer applicable estimators (14/27 vs 27/27)",
            scale_invariant=True,
            reason="Structural: retrieval data is unpaired by construction. "
                   "Paired-IV estimators (2SLS, LIML, etc.) cannot be applied regardless of scale.",
            confidence="high",
            evidence_at_scale="GPT-4 RAG architectures still separate retrieval from generation",
        ),
        TransferDiagnostic(
            finding="TS-IV is biased in high-dim instrument regime (β̂→Q/(Q+r̃b))",
            scale_invariant=True,
            reason="Mathematical property of the plug-in denominator (Schur et al. Lemma 4.5). "
                   "Holds for any model where n/m → r ∈ (0, ∞).",
            confidence="high",
            evidence_at_scale="none direct; bias formula is proven asymptotically",
        ),
        TransferDiagnostic(
            finding="Estimator taxonomy (27 methods with applicability conditions)",
            scale_invariant=True,
            reason="Each estimator's consistency/bias properties are asymptotic theorems "
                   "that hold regardless of the underlying generative model's size.",
            confidence="high",
            evidence_at_scale="theoretical",
        ),
        TransferDiagnostic(
            finding="Theory papers (P01, P19, P23-P25) are CONDITIONAL regardless of scale",
            scale_invariant=True,
            reason="Their causal questions involve uncomputability (Kolmogorov) or "
                   "non-measurability (intelligence proxy). Scale does not fix a fundamentally "
                   "untestable causal claim.",
            confidence="high",
            evidence_at_scale="inherent",
        ),
        TransferDiagnostic(
            finding="Causal atlas pathology 'unpaired_bias' for retrieval papers",
            scale_invariant=True,
            reason="The pathology is a property of the data structure (unpaired), "
                   "not of the model's parameter count.",
            confidence="high",
            evidence_at_scale="Lost-in-the-Middle documented at GPT-4 scale (Liu et al. 2024)",
        ),

        # ---- Scale-dependent ----
        TransferDiagnostic(
            finding="retrieval_dependence coefficient = -0.0303",
            scale_invariant=False,
            reason="Coefficient magnitude is fitted on GPT-1 (20K-83K params). "
                   "Direction likely transfers; magnitude probably does not.",
            confidence="low",
            evidence_at_scale="none at GPT-4 scale",
        ),
        TransferDiagnostic(
            finding="Onset threshold: scale=32, data=32768",
            scale_invariant=False,
            reason="These are regime-specific numbers from the toy GPT-1 wind tunnel. "
                   "Actual onset thresholds at frontier scale are unknown.",
            confidence="low",
            evidence_at_scale="none",
        ),
        TransferDiagnostic(
            finding="Failure atlas: 8/32 collapse, all in retrieval_qa",
            scale_invariant=False,
            reason="Collapse counts depend on model capacity. Larger models may solve "
                   "retrieval tasks that small models cannot. The proportion may change.",
            confidence="medium",
            evidence_at_scale="partial (retrieval remains hard even for GPT-4, per Liu et al.)",
        ),
        TransferDiagnostic(
            finding="CCL2 measured law R² = 0.93",
            scale_invariant=False,
            reason="R² is specific to the measured regime. A different model family "
                   "could produce different explanatory power.",
            confidence="low",
            evidence_at_scale="none",
        ),
        TransferDiagnostic(
            finding="task_family_code coefficient = 0.00264",
            scale_invariant=False,
            reason="Task ordering may differ at larger scale. Relative task difficulty "
                   "could change with model capacity.",
            confidence="low",
            evidence_at_scale="none",
        ),
    ]


class TransferDiagnosticsRunner:
    """Assess scale-transfer properties of the current analysis."""

    def __init__(self) -> None:
        self.diagnostics = _build_diagnostics()

    def run(self) -> Dict[str, Any]:
        scale_inv = [d for d in self.diagnostics if d.scale_invariant]
        scale_dep = [d for d in self.diagnostics if not d.scale_invariant]
        return {
            "total_findings": len(self.diagnostics),
            "scale_invariant_count": len(scale_inv),
            "scale_dependent_count": len(scale_dep),
            "scale_invariant": [d.to_dict() for d in scale_inv],
            "scale_dependent": [d.to_dict() for d in scale_dep],
            "transfer_summary": (
                f"{len(scale_inv)}/{len(self.diagnostics)} findings are scale-invariant "
                f"(mathematical properties of estimators/data structure). "
                f"{len(scale_dep)}/{len(self.diagnostics)} are scale-dependent "
                f"(specific coefficient values or threshold numbers)."
            ),
        }

    def export(self, path: str | Path) -> str:
        result = self.run()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        return str(path)
