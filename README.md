# Capability Cartography Layer 3

**Capability Cartography Layer 3** is the direct successor to `Capability-Cartography-Layer-2`. It keeps the full Layer 2 spine — measurement, classification, failure atlases, visualization, notebook execution, and agent integration — but extends it with **causal explanation**: estimator registries, identification-aware failure classification, middle-regime analysis, and transfer diagnostics.

This repository is designed to sit on top of four companion resources:

- `pageman/sutskever-30-implementations` as the experimental substrate
- `pageman/Sutskever-Agent` as the orchestration and explanatory layer
- `pageman/gpt1-from-sutskever30` as the first controlled transformer wind tunnel
- `pageman/sutskever-30-beyond-numpy` as the multi-backend triangulation layer **(new in Layer 3)**

These links are preserved explicitly in code and artifacts. The adapter layer records canonical repository URLs and configured local roots so every cartography export can retain its provenance.

## The Layer Progression

| Layer | Question | Verb | Key Addition |
|-------|----------|------|-------------|
| **1** | "What happened?" | **Measures** | Schemas, sweeps, surfaces, validation, falsifiable laws |
| **2** | "What kind of failure is this?" | **Classifies** | Failure atlas, visualization, notebook execution, agent briefs |
| **3** | "Why did it fail?" | **Explains** | Causal registry, estimator sweep, causal atlas, middle-regime analysis, transfer diagnostics |

## Reader Guide

If you want the shortest reader-friendly interpretation of what the current results do and do not establish, start with [`TAO_ASSESSMENT.md`](./TAO_ASSESSMENT.md). It evaluates the repository against three specific mysteries that Layer 2 left unresolved:

1. The causal explanation for why retrieval is harder
2. Whether these laws transfer to GPT-4-scale models
3. A mathematical theory of the "middle regime" itself

## What Layer 3 Adds

### Five New Modules

| Module | Purpose | Addresses |
|--------|---------|-----------|
| [`causal_registry.py`](./capability_cartography/causal_registry.py) | Registry of 27 IV/GMM/MR estimators with applicability conditions | The estimator landscape |
| [`estimator_sweep.py`](./capability_cartography/estimator_sweep.py) | Apply all applicable estimators to each measured record; compute consensus | "Which estimator works for which paper?" |
| [`causal_atlas.py`](./capability_cartography/causal_atlas.py) | Classify failures by *cause* (unpaired_bias, weak_instrument, etc.) not just *symptom* | Mystery #1: Why retrieval is harder |
| [`middle_regime.py`](./capability_cartography/middle_regime.py) | Schur et al. (2026) regime classification, bias computation, boundary detection | Mystery #3: Mathematical theory of the middle regime |
| [`transfer_diagnostics.py`](./capability_cartography/transfer_diagnostics.py) | Flag each finding as scale-invariant or scale-dependent | Mystery #2: Whether laws transfer |

### Extended Modules

| Module | What Changed |
|--------|-------------|
| [`schemas.py`](./capability_cartography/schemas.py) | Added `CausalEstimator`, `EstimatorResult`, `CausalRecord`, `MiddleRegimeProfile`, `TransferDiagnostic` |
| [`adapters.py`](./capability_cartography/adapters.py) | Added `BeyondNumpyAdapter` for the multi-backend substrate |
| [`orchestration.py`](./capability_cartography/orchestration.py) | Extended pipeline: L2 steps → estimator sweep → causal atlas → middle regime → transfer diagnostics |
| [`demo.py`](./capability_cartography/demo.py) | Extended demo with all Layer 3 steps |

### Inherited Modules (Unchanged from Layer 2)

`boundary.py`, `compressibility.py`, `datasets.py`, `descriptors.py`, `execution.py`, `failure_atlas.py`, `metrics.py`, `notebook_runner.py`, `provenance.py`, `runner.py`, `storage.py`, `surfaces.py`, `sweeps.py`, `validation.py`, `visualization.py`, `agent_integration.py`

## The 27 Estimators

The causal registry contains estimators from seven families:

| Family | Count | Key Members |
|--------|-------|-------------|
| classical_IV | 7 | OLS, 2SLS, LIML, Fuller-k, JIVE, RJIVE, SS-IV |
| two_sample_IV | 2 | TS-IV, TS-2SLS |
| unpaired_GMM | 2 | UP-GMM, UP-GMM ℓ₁ |
| splitUP | 3 | SplitUP (dense), SplitUP ℓ₁, SplitUP (analytic) |
| sparse_regularized | 8 | ℓ₁-Reg 2SLS, Lasso-GMM, GMM-Lasso, FGMM, Desparsified GMM, Post-Dbl-Selection, spaceIV, spaceTSIV |
| MR_robust | 5 | IVW, MR-Egger, Weighted Median, Mode-Based MR, MR-PRESSO |

SplitUP is the only family consistent across all regimes: finite-dimensional instruments, high-dimensional instruments, paired data, unpaired data, dense effects, and sparse effects.

## Causal Atlas Pathology Labels

Layer 2's failure atlas uses **symptom** labels: collapse, generalization_risk, stable_reasoning.

Layer 3's causal atlas uses **pathology** labels:

| Label | Meaning | Typical Papers |
|-------|---------|---------------|
| `stable_identification` | Mechanism → capability causal effect is identifiable by most estimators | Architecture papers (P02-P08, P10-P18, P20-P22, P26-P27) |
| `unpaired_bias` | Causal effect is attenuated because data is unpaired; most paired-IV estimators inapplicable | Retrieval papers (P28-P30) |
| `weak_instrument` | Instruments are too weak for reliable identification despite many applicable estimators | — |
| `sparse_identification_failure` | Sparse effects but restricted eigenvalue condition not met | — |
| `insufficient_environments` | Too few environments/instruments for high-dim methods | Theory papers (P01, P19, P23-P25) |
| `exclusion_violation_risk` | High average bias across estimators suggests exclusion restriction may be violated | — |

## Middle-Regime Theory

The `middle_regime.py` module implements the Schur et al. (2026) mathematical framework:

- **Regime classification**: `classify_regime(m, r, d, s_star)` → human-readable label
- **Bias computation**: `measurement_error_bias(Q, r_tilde, b)` → attenuation factor Q/(Q+r̃b)
- **Boundary detection**: identifies the m-value where TS-IV transitions from consistent to biased
- **Profile generation**: per-paper profiles with `splitup_needed` flag

## Transfer Diagnostics

Current analysis: **6/11 findings are scale-invariant**, **5/11 are scale-dependent**.

Scale-invariant (transfer to GPT-4):
- SplitUP removes measurement-error bias (mathematical theorem)
- Retrieval papers have fewer applicable estimators (structural property)
- TS-IV bias formula Q/(Q+r̃b) (asymptotic result)
- Estimator taxonomy (theoretical properties)
- Theory papers remain CONDITIONAL (uncomputability)
- Unpaired-bias pathology for retrieval (data structure property)

Scale-dependent (do NOT transfer without re-measurement):
- retrieval_dependence coefficient = -0.0303
- Onset thresholds (scale=32, data=32768)
- Failure atlas collapse counts
- CCL2 measured law R²
- Task family coefficient magnitude

## Installation

```bash
git clone https://github.com/pageman/Capability-Cartography-Layer-3
cd Capability-Cartography-Layer-3
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Optional companion repositories

```bash
export SUTSKEVER30_ROOT=/path/to/sutskever-30-implementations
export GPT1_WIND_TUNNEL_ROOT=/path/to/gpt1-from-sutskever30
export SUTSKEVER_AGENT_ROOT=/path/to/Sutskever-Agent/sutskever-agent
export BEYOND_NUMPY_ROOT=/path/to/sutskever-30-beyond-numpy
```

## Quick Start

```bash
# Run tests
python3 -m unittest discover -s tests -p 'test_*.py'

# Run demo
python3 -m capability_cartography.demo
```

## Layer 3 Artifact Tree

```text
artifacts/layer3/
├── causal/
│   ├── causal_atlas.json
│   ├── estimator_sweep_summary.json
│   ├── middle_regime_summary.json
│   └── transfer_diagnostics.json
├── failure_atlas/
│   └── failure_atlas.json
├── measured/
│   ├── measured_laws.json
│   ├── measured_records.csv
│   └── measured_summary.json
├── notebooks/
│   └── 22_scaling_laws.execution.json
├── plots/
│   ├── onset_surface.png
│   └── phase_regions.png
├── sweeps/
│   ├── sweep_records.csv
│   └── sweep_summary.json
└── agent/
    ├── agent_brief.json
    └── agent_workflow.yaml
```

## Citation

```bibtex
@misc{capability-cartography-layer-3-2026,
  author    = {Paul "The Pageman" Pajo, pageman@gmail.com},
  title     = {Capability-Cartography-Layer-3: from classification to causal explanation},
  year      = {2026},
  url       = {https://github.com/pageman/Capability-Cartography-Layer-3},
  note      = {Extends Layer 2 with causal estimator sweeps, identification-aware failure atlases,
               Schur et al. middle-regime analysis, and scale-transfer diagnostics.}
}
```

## License

This repository is released under the MIT License. See [`LICENSE`](./LICENSE).
