"""Capability Cartography Layer 3 — From Classification to Causal Explanation.

Layer 1: Measurement (schemas, sweeps, surfaces, validation)
Layer 2: Classification (failure atlas, visualization, notebooks, agents)
Layer 3: Explanation (causal registry, IV estimators, estimator sweep,
                      causal atlas, middle regime, transfer diagnostics,
                      paper registry, causal visualization)
"""

# Layer 1
from .schemas import (
    ArtifactBundle,
    BoundaryEvent,
    BoundaryFit,
    CapabilitySnapshot,
    CapabilityTrajectory,
    CompressibilityProfile,
    ExperimentSpec,
    InterventionConfig,
    InterventionSweep,
    TaskDescriptor,
)

# Layer 3 schemas
from .schemas import (
    CausalEstimator,
    CausalRecord,
    EstimatorResult,
    MiddleRegimeProfile,
    TransferDiagnostic,
)

# Layer 1 modules
from .adapters import AgentOverlayAdapter, GPT1WindTunnelAdapter, NotebookSubstrateAdapter
from .boundary import BoundaryAnalyzer
from .compressibility import CompressibilityStack
from .descriptors import TaskDescriptorExtractor
from .metrics import aggregate_snapshot_metrics, calibration_error, estimate_capability_score
from .runner import CapabilityCartographyRunner
from .storage import RunStorage
from .surfaces import CapabilitySurfaceFitter
from .sweeps import SweepRunner
from .validation import PredictiveLawValidator

# Layer 2 modules
from .agent_integration import SutskeverAgentWorkflowBridge
from .failure_atlas import FailureAtlasClassifier
from .visualization import CartographyVisualizer

# Layer 3 modules
from .adapters import BeyondNumpyAdapter
from .causal_atlas import CausalAtlasClassifier
from .causal_registry import CausalEstimatorRegistry
from .causal_visualization import plot_estimator_heatmap, plot_regime_map, plot_verdict_dashboard
from .estimator_sweep import EstimatorSweepRunner
from .iv_estimators import fuller, generate_iv_data, ivw, liml, mr_egger, ols, splitup, splitup_analytic, ts_iv, tsls, up_gmm
from .middle_regime import MiddleRegimeAnalyzer, classify_regime, measurement_error_bias
from .orchestration import FullStudyOrchestrator
from .paper_registry import PaperEntry, PaperRegistry
from .transfer_diagnostics import TransferDiagnosticsRunner

__version__ = "0.3.0"
