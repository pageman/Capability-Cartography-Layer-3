"""Shared schemas for the Capability Cartography Layer.

Layer 3 extends the Layer 2 schema set with causal-identification dataclasses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# ====================================================================
# Layer 1 schemas (unchanged)
# ====================================================================

@dataclass
class TaskDescriptor:
    """Descriptor vector for a single task instance or task family."""

    task_name: str
    benchmark_label: str
    substrate: str
    realism_level: str
    surface_statistics: Dict[str, float] = field(default_factory=dict)
    latent_structure: Dict[str, float] = field(default_factory=dict)
    retrieval_geometry: Dict[str, float] = field(default_factory=dict)
    perturbation_profile: Dict[str, float] = field(default_factory=dict)
    cognitive_operations: Dict[str, float] = field(default_factory=dict)
    structural_complexity: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompressibilityProfile:
    """Surface, predictive, and structural compression proxies."""

    surface: Dict[str, float]
    predictive: Dict[str, float]
    structural: Dict[str, float]
    gaps: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CapabilitySnapshot:
    """A single checkpoint measurement."""

    step: int
    metrics: Dict[str, float]
    descriptor: TaskDescriptor
    compressibility: CompressibilityProfile
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "metrics": dict(self.metrics),
            "descriptor": self.descriptor.to_dict(),
            "compressibility": self.compressibility.to_dict(),
            "notes": dict(self.notes),
        }


@dataclass
class BoundaryEvent:
    """Detected qualitative shift in a metric series."""

    metric: str
    step: int
    value: float
    delta: float
    regime_before: str
    regime_after: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BoundaryFit:
    """Threshold summary across a sweep."""

    metric: str
    threshold_value: float
    threshold_step: int
    slope: float
    lower_band: float
    upper_band: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CapabilityTrajectory:
    """Time series of capability measurements."""

    experiment_id: str
    substrate: str
    intervention_config: Dict[str, Any]
    snapshots: List[CapabilitySnapshot]
    boundary_events: List[BoundaryEvent] = field(default_factory=list)
    fitted_boundaries: List[BoundaryFit] = field(default_factory=list)
    aggregate_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "substrate": self.substrate,
            "intervention_config": dict(self.intervention_config),
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
            "boundary_events": [event.to_dict() for event in self.boundary_events],
            "fitted_boundaries": [fit.to_dict() for fit in self.fitted_boundaries],
            "aggregate_metrics": dict(self.aggregate_metrics),
        }


@dataclass
class ExperimentSpec:
    """Single experiment contract."""

    experiment_id: str
    substrate: str
    task_name: str
    benchmark_label: str
    realism_level: str
    objective_type: str
    model_family: str
    intervention_axes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InterventionConfig:
    """One-factor-at-a-time or bundled interventions."""

    architecture: Dict[str, Any] = field(default_factory=dict)
    objective: Dict[str, Any] = field(default_factory=dict)
    data_regime: Dict[str, Any] = field(default_factory=dict)
    retrieval: Dict[str, Any] = field(default_factory=dict)
    context_geometry: Dict[str, Any] = field(default_factory=dict)
    interpretability: Dict[str, Any] = field(default_factory=dict)

    def flattened(self) -> Dict[str, Any]:
        flattened: Dict[str, Any] = {}
        for section_name, section_value in asdict(self).items():
            for key, value in section_value.items():
                flattened[f"{section_name}.{key}"] = value
        return flattened

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InterventionSweep:
    """Sweep definition for a single intervention axis."""

    axis: str
    values: List[Any]
    baseline: InterventionConfig

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axis": self.axis,
            "values": list(self.values),
            "baseline": self.baseline.to_dict(),
        }


@dataclass
class ArtifactBundle:
    """Serialized outputs for downstream narration and plotting."""

    spec: ExperimentSpec
    trajectory: CapabilityTrajectory
    narrative: Optional[str] = None
    export_path: Optional[str] = None
    linked_repositories: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "spec": self.spec.to_dict(),
            "trajectory": self.trajectory.to_dict(),
            "linked_repositories": dict(self.linked_repositories),
        }
        if self.narrative is not None:
            payload["narrative"] = self.narrative
        if self.export_path is not None:
            payload["export_path"] = self.export_path
        return payload


# ====================================================================
# Layer 3 schemas (new)
# ====================================================================

@dataclass
class CausalEstimator:
    """One of the 27 estimators from the IV/GMM/MR literature.

    Each estimator has a fixed set of applicability conditions that
    determine whether it can be used on a given paper/task.
    """

    name: str
    family: str
    requires_paired: bool
    handles_sparsity: bool
    consistent_finite_m: bool
    consistent_high_dim_m: bool
    handles_unpaired: bool
    weak_iv_robust: bool
    bias_correction: bool
    reference: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EstimatorResult:
    """Result of applying one estimator to one paper/task."""

    estimator: str
    applicable: bool
    estimate: float = 0.0
    standard_error: float = 0.0
    bias: float = 0.0
    mse: float = 0.0
    consistent: bool = False
    causal_detected: bool = False
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CausalRecord:
    """Extends a measured record with causal-identification metadata.

    This is the central data object for Layer 3.  One CausalRecord is
    produced per paper (or per paper × task family combination).
    """

    paper_id: int
    paper_name: str
    mechanism_X: str
    capability_Y: str
    paper_type: str
    data_structure: str
    n_environments: int
    instrument_strength: float
    numpy_loss: Optional[float] = None
    backend_parity_passed: int = 0
    backend_parity_failed: int = 0
    estimator_results: List[EstimatorResult] = field(default_factory=list)
    n_applicable: int = 0
    n_consistent: int = 0
    estimator_consensus: float = 0.0
    best_estimator: str = ""
    causality_verdict: str = ""
    causal_chain: str = ""
    capability_score: float = 0.0
    retrieval_dependence: float = 0.0
    generalization_gap: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["estimator_results"] = [er if isinstance(er, dict) else asdict(er) for er in (self.estimator_results or [])]
        return d


@dataclass
class MiddleRegimeProfile:
    """Characterizes where a task sits in the Schur et al. regime space.

    The "middle regime" is m -> inf with n/m -> r in (0, inf), i.e.
    many environments but few observations per environment.  Standard
    two-sample IV estimators are biased here; SplitUP is consistent.
    """

    paper_id: int
    m: int
    r: float
    d: int
    s_star: int
    is_high_dim: bool
    measurement_error_bias: float
    attenuation_factor: float
    splitup_needed: bool
    regime_label: str
    schur_theorem: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TransferDiagnostic:
    """Flags whether a finding is scale-invariant or scale-dependent."""

    finding: str
    scale_invariant: bool
    reason: str
    confidence: str
    evidence_at_scale: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
