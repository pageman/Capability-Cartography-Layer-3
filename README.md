# Capability Cartography Layer 3

**Capability Cartography Layer 3** is the direct successor to `Capability-Cartography-Layer-2`. It keeps the full Layer 2 spine — measurement, classification, failure atlases, visualization, notebook execution, and agent integration — but extends it with **causal explanation**: estimator registries, identification-aware failure classification, middle-regime analysis, and transfer diagnostics.

This repository is designed to sit on top of four companion resources:

- `pageman/sutskever-30-implementations` as the experimental substrate
- `pageman/Sutskever-Agent` as the orchestration and explanatory layer
- `pageman/gpt1-from-sutskever30` as the first controlled transformer wind tunnel
- `pageman/sutskever-30-beyond-numpy` as the multi-backend triangulation layer **(new in Layer 3)**

These links are preserved explicitly in code and artifacts. The adapter layer records canonical repository URLs and configured local roots so every cartography export can retain its provenance.

## External Baseline Note

This repository should be read alongside the external note:

- `Downloads/In-Paper Causality Analysis- 30 Sutskever Papers × 27 Estimators × 6 Backends × CCL2.md`

That note is useful for the estimator inventory and source list, but it is not identical to the current Layer 3 outputs. In particular:

- the note reports `22 CONFIRMED / 8 CONDITIONAL / 0 UNCONFIRMED`
- the current Layer 3 artifacts report `27 CONFIRMED / 0 CONDITIONAL / 3 UNCONFIRMED`
- the note treats `SplitUP ℓ1` as the universal best estimator
- the current Layer 3 implementation usually selects `spaceTSIV`, `IVW`, `TS-IV`, or `SplitUP (dense)` as the current lowest-MSE estimator, depending on paper and regime

When these differ, treat `artifacts/layer3/causal/*.json` as the repository's current source of truth.

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

## Narrative Arc And Methodological Arc

### Narrative Arc

Layer 1 asked what happened. Layer 2 asked what kind of failure happened. Layer 3 asks why a mechanism does or does not causally explain a capability. The arc of the project is therefore a progression from measurement, to classification, to explanation, and finally to explicit criteria for when a causal claim is justified.

### Methodological Arc

The methodological arc mirrors that narrative progression: run the system end to end, preserve comparability with earlier layers, make estimator assumptions explicit, separate historical baseline notes from current outputs, and require verdicts to follow an explicit policy rather than a vague confidence threshold.

### Story Boxes

`Story Box 1: From Prototype To Instrument`
Layer 3 is not only meant to run; it is meant to support inspectable causal claims.

`Story Box 2: Baseline Is Not Ground Truth`
Historical notes and current artifacts can disagree; when they do, the current exported artifacts are the source of truth for the repository.

`Story Box 3: Retrieval Is The Boundary Case`
Retrieval papers remain the hardest regime because unpaired structure reduces estimator applicability and exposes the limits of naive causal claims.

`Story Box 4: The Framework Must Judge Itself`
The project now evaluates not only papers, but also the reliability of its own verdicting, ranking, and pathology labeling.

### Method Boxes

`Method Box 1: End-to-End Validation`
Re-run tests and demo after each structural change so the repo is treated as an empirical instrument, not static code.

`Method Box 2: Cross-Layer Consistency`
Compare Layer 3 against Layers 1 and 2 at the package, artifact, and documentation levels to preserve continuity of the research program.

`Method Box 3: Explicit Verdict Policy`
Assign `CONFIRMED`, `CONDITIONAL`, and `UNCONFIRMED` through a named policy with applicability and consensus thresholds.

`Method Box 4: Estimator Ranking Discipline`
Only rank estimators as “best” when the implementation is exact rather than proxy or fallback.

`Method Box 5: Normalized Pathology Prediction`
Predict causal-atlas labels in normalized feature space so large-magnitude features do not dominate the classification.

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
| classical_IV | 7 | Naive OLS, 2SLS, LIML, Fuller-k, JIVE, RJIVE, SS-IV |
| two_sample_IV | 2 | TS-IV, TS-2SLS |
| unpaired_GMM | 2 | UP-GMM, UP-GMM l1 |
| splitUP | 3 | SplitUP (dense), SplitUP l1, SplitUP (analytic) |
| sparse_regularized | 8 | l1-Reg 2SLS, Lasso-GMM, GMM-Lasso, FGMM, Desparsified GMM, Post-Double Selection, spaceIV, spaceTSIV |
| MR_robust | 5 | IVW, MR-Egger, Weighted Median, Mode-Based MR, MR-PRESSO |

SplitUP is the only family consistent across all regimes: finite-dimensional instruments, high-dimensional instruments, paired data, unpaired data, dense effects, and sparse effects.

### Naming Conventions

The code uses stable ASCII estimator identifiers in JSON and Python APIs:

- `SplitUP_L1` is rendered in prose as `SplitUP l1` and corresponds to the older note's `SplitUP ℓ1`
- `UP_GMM_L1` is rendered in prose as `UP-GMM l1` and corresponds to `UP-GMM ℓ1`
- `L1_Reg_2SLS` is rendered in prose as `l1-Reg 2SLS` and corresponds to `ℓ1-Reg 2SLS`

If you are comparing old figures or notes against the current repo, match on the code identifier first, then on the human-readable label.

### Source Baseline

The current estimator inventory and causal framing are grounded in:

- Schur, F. et al. (2026). *Many Experiments, Few Repetitions, Unpaired Data, and Sparse Effects: Is Causal Inference Possible?* arXiv:2601.15254
- Pajo, P. (2026). *Finite-Sample Performance of SplitUP in Many-Environments Unpaired IV*
- Pajo, P. (2026). *Capability Cartography Layer 2*
- Pajo, P. (2026). *Sutskever 30 Beyond NumPy*
- Pajo, P. (2026). *Sutskever 30 Implementations*

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
│   ├── causal_records.json
│   ├── estimator_heatmap.png
│   ├── estimator_sweep_summary.json
│   ├── middle_regime_summary.json
│   ├── regime_map.png
│   ├── transfer_diagnostics.json
│   └── verdict_dashboard.png
├── failure_atlas/
│   └── failure_atlas.json
├── measured/
│   ├── measured_laws.json
│   ├── measured_records.csv
│   └── measured_summary.json
├── notebooks/
│   ├── 22_scaling_laws.execution.json
│   └── 22_scaling_laws_figures/
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
