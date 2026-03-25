# Session Checkpoint

Date: 2026-03-26
Repo: `pageman/Capability-Cartography-Layer-3`
Branch: `main`

## Current HEAD

- `de31b6e` Layer 3 initial import
- `92ff297` Refactor Layer 3 demo and restore package compatibility
- `5dbbb8b` Document causal baseline and estimator aliases

## What Was Done This Session

- Unzipped the Layer 3 archive, initialized the repo, created the GitHub repository, and pushed `main`.
- Ran Layer 3 end to end:
  - `python3 -m unittest discover -s tests -p 'test_*.py'`
  - `python3 -m capability_cartography.demo`
- Compared Layer 3 structurally against Layer 2 and Layer 1.
- Refactored Layer 3 to:
  - restore Layer 1/2-style root package exports and explicit `__all__`
  - simplify the demo so the heavy sweep/measured/causal pipeline runs only once
  - harden IV estimator numerics to remove runtime warnings from successful test/demo runs
- Added tests for warning-prone estimator paths.
- Checked the external note:
  - `Downloads/In-Paper Causality Analysis- 30 Sutskever Papers × 27 Estimators × 6 Backends × CCL2.md`
- Updated Layer 3 docs and exports so the note is treated as a historical baseline, not the current source of truth.
- Added estimator display names and alias mappings such as:
  - `SplitUP_L1` -> `SplitUP l1` / `SplitUP ℓ1`
  - `UP_GMM_L1` -> `UP-GMM l1` / `UP-GMM ℓ1`
  - `L1_Reg_2SLS` -> `l1-Reg 2SLS` / `ℓ1-Reg 2SLS`

## Validation Status

- Tests: `52` passing
- Demo: passes end to end
- Current demo outputs write to:
  - `artifacts/layer3/`

## Current Layer 3 Behavior

The current Layer 3 artifacts differ from the external causality note.

External note baseline:
- `22 CONFIRMED / 8 CONDITIONAL / 0 UNCONFIRMED`
- treats `SplitUP ℓ1` as universal best estimator

Current Layer 3 artifacts:
- `27 CONFIRMED / 0 CONDITIONAL / 3 UNCONFIRMED`
- best estimator is usually one of:
  - `spaceTSIV`
  - `IVW`
  - `TS_IV`
  - `SplitUP_dense`

Current exported comparison metadata lives in:
- `artifacts/layer3/causal/estimator_sweep_summary.json`

## Important Files Changed

- `README.md`
- `capability_cartography/__init__.py`
- `capability_cartography/demo.py`
- `capability_cartography/iv_estimators.py`
- `capability_cartography/causal_registry.py`
- `capability_cartography/orchestration.py`
- `tests/test_capability_cartography.py`

## Open Issues / Next Steps

1. The current causal verdict logic and estimator ranking do not match the external note's claims.
   - If the note is intended to be normative, the implementation needs to be reconciled.
   - If the implementation is intended to be normative, the note should be rewritten as a historical comparison.

2. The causal atlas still looks statistically weak in its centroid behavior.
   - Some `stable_identification` records are closer to the `insufficient_environments` centroid because `avg_estimator_bias` magnitudes are poorly scaled.
   - Likely next fix: normalize or robust-scale causal-atlas features before centroid classification.

3. The current estimator sweep uses many proxy dispatches.
   - Several named estimators still route through shared approximations rather than distinct implementations.
   - If estimator-specific claims matter, the dispatch layer needs to become more faithful.

4. The demo output is still noisy because the linked GPT-1 implementation prints model summaries repeatedly.
   - That is not a repo-path leak, but it does make the demo output harder to scan.

## Suggested Next Work Item

If continuing from this checkpoint, the highest-value next task is:

- reconcile the implementation with the external causality note by deciding which of these should be authoritative:
  - the current Layer 3 artifact outputs
  - or the external note's `22/8/0` verdict regime and `SplitUP ℓ1` universal-best claim

Then update the estimator sweep, verdict thresholds, and README accordingly.
