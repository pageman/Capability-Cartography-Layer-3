"""Registry of 27 causal estimators from the IV/GMM/MR literature.

Each estimator is recorded with its applicability conditions so that
the estimator sweep can determine, for any given paper/task, which
estimators can be applied and which are theoretically consistent.

Estimator families:
    classical_IV        -- standard paired-data instrumental variables
    two_sample_IV       -- two-sample / unpaired IV
    unpaired_GMM        -- unpaired GMM (Schur et al. 2026)
    splitUP             -- SplitUP cross-fold bias correction
    sparse_regularized  -- L1/Lasso-based IV methods (paired)
    MR_robust           -- Mendelian randomization methods
    weak_IV_test        -- weak-instrument robust tests
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from .schemas import CausalEstimator


ESTIMATOR_DISPLAY_NAMES: Dict[str, str] = {
    "Naive_OLS": "Naive OLS",
    "2SLS": "2SLS",
    "LIML": "LIML",
    "Fuller_k": "Fuller-k",
    "JIVE": "JIVE",
    "RJIVE": "RJIVE",
    "SS_IV": "SS-IV",
    "TS_IV": "TS-IV",
    "TS_2SLS": "TS-2SLS",
    "UP_GMM": "UP-GMM",
    "UP_GMM_L1": "UP-GMM l1",
    "SplitUP_dense": "SplitUP (dense)",
    "SplitUP_L1": "SplitUP l1",
    "SplitUP_analytic": "SplitUP (analytic)",
    "L1_Reg_2SLS": "l1-Reg 2SLS",
    "Lasso_GMM": "Lasso-GMM",
    "GMM_Lasso": "GMM-Lasso",
    "FGMM": "FGMM",
    "Desparsified_GMM": "Desparsified GMM",
    "Post_Dbl_Selection": "Post-Double Selection",
    "spaceIV": "spaceIV",
    "spaceTSIV": "spaceTSIV",
    "IVW": "IVW",
    "MR_Egger": "MR-Egger",
    "Weighted_Median": "Weighted Median",
    "Mode_Based_MR": "Mode-Based MR",
    "MR_PRESSO": "MR-PRESSO",
}

ESTIMATOR_ALIASES: Dict[str, Sequence[str]] = {
    "UP_GMM_L1": ("UP-GMM l1", "UP-GMM ℓ1"),
    "SplitUP_L1": ("SplitUP l1", "SplitUP ℓ1"),
    "L1_Reg_2SLS": ("l1-Reg 2SLS", "ℓ1-Reg 2SLS"),
}

BASELINE_SOURCES: Sequence[str] = (
    "Schur et al. (2026). Many Experiments, Few Repetitions, Unpaired Data, and Sparse Effects: Is Causal Inference Possible? arXiv:2601.15254",
    "Pajo (2026). Finite-Sample Performance of SplitUP in Many-Environments Unpaired IV",
    "Pajo (2026). Capability Cartography Layer 2",
    "Pajo (2026). Sutskever 30 Beyond NumPy",
    "Pajo (2026). Sutskever 30 Implementations",
)


def build_registry() -> List[CausalEstimator]:
    """Return the canonical list of 27 estimators."""

    return [
        # ---- classical_IV (paired, standard) ----
        CausalEstimator(
            name="Naive_OLS",
            family="classical_IV",
            requires_paired=True,
            handles_sparsity=False,
            consistent_finite_m=False,   # biased under confounding
            consistent_high_dim_m=False,
            handles_unpaired=False,
            weak_iv_robust=False,
            bias_correction=False,
            reference="Angrist & Pischke 2009",
        ),
        CausalEstimator(
            name="2SLS",
            family="classical_IV",
            requires_paired=True,
            handles_sparsity=False,
            consistent_finite_m=True,
            consistent_high_dim_m=False,
            handles_unpaired=False,
            weak_iv_robust=False,
            bias_correction=False,
            reference="Angrist & Krueger 1991",
        ),
        CausalEstimator(
            name="LIML",
            family="classical_IV",
            requires_paired=True,
            handles_sparsity=False,
            consistent_finite_m=True,
            consistent_high_dim_m=False,
            handles_unpaired=False,
            weak_iv_robust=True,
            bias_correction=True,
            reference="Anderson & Rubin 1949",
        ),
        CausalEstimator(
            name="Fuller_k",
            family="classical_IV",
            requires_paired=True,
            handles_sparsity=False,
            consistent_finite_m=True,
            consistent_high_dim_m=False,
            handles_unpaired=False,
            weak_iv_robust=True,
            bias_correction=True,
            reference="Fuller 1977",
        ),
        CausalEstimator(
            name="JIVE",
            family="classical_IV",
            requires_paired=True,
            handles_sparsity=False,
            consistent_finite_m=True,
            consistent_high_dim_m=False,
            handles_unpaired=False,
            weak_iv_robust=True,
            bias_correction=True,
            reference="Angrist et al. 1999; Blomquist & Dahlberg 1999",
        ),
        CausalEstimator(
            name="RJIVE",
            family="classical_IV",
            requires_paired=True,
            handles_sparsity=False,
            consistent_finite_m=True,
            consistent_high_dim_m=True,
            handles_unpaired=False,
            weak_iv_robust=True,
            bias_correction=True,
            reference="Hansen & Kozbur 2014",
        ),
        CausalEstimator(
            name="SS_IV",
            family="classical_IV",
            requires_paired=True,
            handles_sparsity=False,
            consistent_finite_m=True,
            consistent_high_dim_m=False,
            handles_unpaired=False,
            weak_iv_robust=True,
            bias_correction=True,
            reference="Angrist & Krueger 1995",
        ),

        # ---- two_sample_IV ----
        CausalEstimator(
            name="TS_IV",
            family="two_sample_IV",
            requires_paired=False,
            handles_sparsity=False,
            consistent_finite_m=True,
            consistent_high_dim_m=False,   # biased: β̂ → Q/(Q+r̃b)
            handles_unpaired=True,
            weak_iv_robust=False,
            bias_correction=False,
            reference="Angrist & Krueger 1992; Inoue & Solon 2010",
        ),
        CausalEstimator(
            name="TS_2SLS",
            family="two_sample_IV",
            requires_paired=False,
            handles_sparsity=False,
            consistent_finite_m=True,
            consistent_high_dim_m=False,
            handles_unpaired=True,
            weak_iv_robust=False,
            bias_correction=False,
            reference="Inoue & Solon 2010",
        ),

        # ---- unpaired_GMM (Schur et al. 2026) ----
        CausalEstimator(
            name="UP_GMM",
            family="unpaired_GMM",
            requires_paired=False,
            handles_sparsity=False,
            consistent_finite_m=True,
            consistent_high_dim_m=False,   # same measurement-error bias as TS-IV
            handles_unpaired=True,
            weak_iv_robust=False,
            bias_correction=False,
            reference="Schur et al. 2026, Eq. 4.2",
        ),
        CausalEstimator(
            name="UP_GMM_L1",
            family="unpaired_GMM",
            requires_paired=False,
            handles_sparsity=True,
            consistent_finite_m=True,
            consistent_high_dim_m=False,
            handles_unpaired=True,
            weak_iv_robust=False,
            bias_correction=False,
            reference="Schur et al. 2026, Eq. 4.4",
        ),

        # ---- splitUP (Schur et al. 2026) ----
        CausalEstimator(
            name="SplitUP_dense",
            family="splitUP",
            requires_paired=False,
            handles_sparsity=False,
            consistent_finite_m=True,
            consistent_high_dim_m=True,    # cross-fold removes bias
            handles_unpaired=True,
            weak_iv_robust=True,
            bias_correction=True,
            reference="Schur et al. 2026, Theorem 4.7",
        ),
        CausalEstimator(
            name="SplitUP_L1",
            family="splitUP",
            requires_paired=False,
            handles_sparsity=True,
            consistent_finite_m=True,
            consistent_high_dim_m=True,
            handles_unpaired=True,
            weak_iv_robust=True,
            bias_correction=True,
            reference="Schur et al. 2026, Theorem 4.8",
        ),
        CausalEstimator(
            name="SplitUP_analytic",
            family="splitUP",
            requires_paired=False,
            handles_sparsity=True,
            consistent_finite_m=True,
            consistent_high_dim_m=True,
            handles_unpaired=True,
            weak_iv_robust=True,
            bias_correction=True,
            reference="Schur et al. 2026, Section 4.3 closed form",
        ),

        # ---- sparse_regularized (paired) ----
        CausalEstimator(
            name="L1_Reg_2SLS",
            family="sparse_regularized",
            requires_paired=True,
            handles_sparsity=True,
            consistent_finite_m=True,
            consistent_high_dim_m=False,
            handles_unpaired=False,
            weak_iv_robust=False,
            bias_correction=False,
            reference="Zhu 2018",
        ),
        CausalEstimator(
            name="Lasso_GMM",
            family="sparse_regularized",
            requires_paired=True,
            handles_sparsity=True,
            consistent_finite_m=True,
            consistent_high_dim_m=False,
            handles_unpaired=False,
            weak_iv_robust=False,
            bias_correction=False,
            reference="Caner 2009",
        ),
        CausalEstimator(
            name="GMM_Lasso",
            family="sparse_regularized",
            requires_paired=True,
            handles_sparsity=True,
            consistent_finite_m=True,
            consistent_high_dim_m=False,
            handles_unpaired=False,
            weak_iv_robust=False,
            bias_correction=False,
            reference="Shi 2016",
        ),
        CausalEstimator(
            name="FGMM",
            family="sparse_regularized",
            requires_paired=True,
            handles_sparsity=True,
            consistent_finite_m=True,
            consistent_high_dim_m=False,
            handles_unpaired=False,
            weak_iv_robust=False,
            bias_correction=False,
            reference="Fan & Liao 2014",
        ),
        CausalEstimator(
            name="Desparsified_GMM",
            family="sparse_regularized",
            requires_paired=True,
            handles_sparsity=True,
            consistent_finite_m=True,
            consistent_high_dim_m=False,
            handles_unpaired=False,
            weak_iv_robust=False,
            bias_correction=True,
            reference="Gold et al. 2020",
        ),
        CausalEstimator(
            name="Post_Dbl_Selection",
            family="sparse_regularized",
            requires_paired=True,
            handles_sparsity=True,
            consistent_finite_m=True,
            consistent_high_dim_m=False,
            handles_unpaired=False,
            weak_iv_robust=True,
            bias_correction=False,
            reference="Belloni et al. 2014",
        ),
        CausalEstimator(
            name="spaceIV",
            family="sparse_regularized",
            requires_paired=True,
            handles_sparsity=True,
            consistent_finite_m=True,
            consistent_high_dim_m=False,
            handles_unpaired=False,
            weak_iv_robust=False,
            bias_correction=False,
            reference="Pfister & Peters 2022",
        ),
        CausalEstimator(
            name="spaceTSIV",
            family="sparse_regularized",
            requires_paired=False,
            handles_sparsity=True,
            consistent_finite_m=True,
            consistent_high_dim_m=False,
            handles_unpaired=True,
            weak_iv_robust=False,
            bias_correction=False,
            reference="Huang et al. 2024",
        ),

        # ---- MR_robust (Mendelian randomization) ----
        CausalEstimator(
            name="IVW",
            family="MR_robust",
            requires_paired=False,
            handles_sparsity=False,
            consistent_finite_m=True,
            consistent_high_dim_m=False,
            handles_unpaired=True,
            weak_iv_robust=False,
            bias_correction=False,
            reference="Burgess et al. 2013",
        ),
        CausalEstimator(
            name="MR_Egger",
            family="MR_robust",
            requires_paired=False,
            handles_sparsity=False,
            consistent_finite_m=True,
            consistent_high_dim_m=False,
            handles_unpaired=True,
            weak_iv_robust=True,
            bias_correction=True,
            reference="Bowden et al. 2015",
        ),
        CausalEstimator(
            name="Weighted_Median",
            family="MR_robust",
            requires_paired=False,
            handles_sparsity=False,
            consistent_finite_m=True,
            consistent_high_dim_m=False,
            handles_unpaired=True,
            weak_iv_robust=True,
            bias_correction=True,
            reference="Hartwig et al. 2017",
        ),
        CausalEstimator(
            name="Mode_Based_MR",
            family="MR_robust",
            requires_paired=False,
            handles_sparsity=False,
            consistent_finite_m=True,
            consistent_high_dim_m=False,
            handles_unpaired=True,
            weak_iv_robust=True,
            bias_correction=True,
            reference="Hartwig et al. 2017",
        ),
        CausalEstimator(
            name="MR_PRESSO",
            family="MR_robust",
            requires_paired=False,
            handles_sparsity=False,
            consistent_finite_m=True,
            consistent_high_dim_m=False,
            handles_unpaired=True,
            weak_iv_robust=True,
            bias_correction=True,
            reference="Verbanck et al. 2018",
        ),
    ]


class CausalEstimatorRegistry:
    """Queryable registry of all 27 estimators."""

    def __init__(self) -> None:
        self.estimators = {e.name: e for e in build_registry()}

    def all_names(self) -> List[str]:
        return list(self.estimators.keys())

    def get(self, name: str) -> CausalEstimator:
        return self.estimators[name]

    def applicable_for(
        self,
        *,
        is_paired: bool,
        is_sparse: bool = False,
        is_high_dim: bool = False,
    ) -> List[CausalEstimator]:
        """Return estimators applicable to a given data structure."""
        out: List[CausalEstimator] = []
        for e in self.estimators.values():
            if e.requires_paired and not is_paired:
                continue
            out.append(e)
        return out

    def consistent_for(
        self,
        *,
        is_paired: bool,
        is_sparse: bool = False,
        is_high_dim: bool = False,
    ) -> List[CausalEstimator]:
        """Return estimators that are both applicable and consistent."""
        applicable = self.applicable_for(is_paired=is_paired, is_sparse=is_sparse, is_high_dim=is_high_dim)
        out: List[CausalEstimator] = []
        for e in applicable:
            if is_high_dim and not e.consistent_high_dim_m:
                continue
            if not e.consistent_finite_m and not is_high_dim:
                continue
            out.append(e)
        return out

    def by_family(self, family: str) -> List[CausalEstimator]:
        return [e for e in self.estimators.values() if e.family == family]

    def summary(self) -> Dict[str, int]:
        families: Dict[str, int] = {}
        for e in self.estimators.values():
            families[e.family] = families.get(e.family, 0) + 1
        return {"total": len(self.estimators), "by_family": families}


def estimator_display_name(name: str) -> str:
    """Return a human-readable label for an estimator identifier."""
    return ESTIMATOR_DISPLAY_NAMES.get(name, name)


def estimator_aliases(name: str) -> Sequence[str]:
    """Return known prose aliases for an estimator identifier."""
    return ESTIMATOR_ALIASES.get(name, ())
