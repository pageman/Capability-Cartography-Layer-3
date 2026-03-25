# Tao Assessment for Capability Cartography Layer 3

## Scope

This assessment is specific to Layer 3 of the Capability Cartography project. It evaluates whether the repository advances beyond Layer 2's measurement and classification to provide **causal explanations** for the behavioral mysteries that Tao's framing identifies.

Layer 1 asked: *"Can we measure?"* → Yes.
Layer 2 asked: *"Can we classify and visualize?"* → Yes.
Layer 3 asks: *"Can we explain WHY?"*

## The Three Remaining Mysteries

The CCL2 arxiv paper and the GLM-5/Chat.z.ai assessment identified three things that Layer 2 does not answer:

1. **The causal explanation for why retrieval is harder**
2. **Whether these laws transfer to GPT-4-scale models**
3. **A mathematical theory of the "middle regime" itself**

Layer 3 addresses each of these.

## What Layer 3 Adds

### 1. Causal Estimator Registry and Sweep

Layer 3 maintains a registry of 27 causal estimators from the instrumental variables, generalized method of moments, and Mendelian randomization literatures. For each paper or task, the estimator sweep determines:

- Which estimators are applicable (depends on paired vs unpaired data)
- Which are theoretically consistent (depends on instrument dimension regime)
- What the estimator consensus is (fraction of applicable estimators that agree)

This converts "retrieval is 3% harder" from an **unexplained coefficient** into a **structurally explained phenomenon**: retrieval papers have unpaired data, which makes 13/27 estimators inapplicable and renders most of the remaining 14 asymptotically biased due to measurement-error in high-dimensional instrument settings (Schur et al. 2026, Lemma 4.5).

### 2. Causal Atlas

Layer 2's failure atlas classifies records by **symptom**: collapse, generalization_risk, stable_reasoning.

Layer 3's causal atlas classifies records by **pathology**: unpaired_bias, weak_instrument, exclusion_violation_risk, sparse_identification_failure, insufficient_environments, stable_identification.

This is the difference between a doctor saying "you have a fever" (symptom) and "you have a bacterial infection" (cause).

### 3. Middle-Regime Analysis

The middle regime analyzer implements the Schur et al. (2026) regime classification:

- `classify_regime(m, r, d, s_star)` → labels like `high_dim_moderate_r` or `classical_large_sample`
- `measurement_error_bias(Q, r_tilde, b)` → the exact attenuation factor Q/(Q + r̃b) from Lemma 4.5
- `detect_regime_boundary(m_values, ...)` → boundary events where the regime transitions from consistent to biased

This provides a mathematical framework for one specific instantiation of Tao's "middle regime" — the zone between "few strong instruments" (classical IV) and "infinitely many with infinite data" (trivial asymptotics).

### 4. Transfer Diagnostics

The transfer module explicitly separates findings into:

- **Scale-invariant** (6/11): mathematical properties of estimators that hold regardless of model size
- **Scale-dependent** (5/11): specific coefficient values from the toy regime

This directly addresses Mystery #2 by telling the reader exactly which conclusions can be trusted at GPT-4 scale and which cannot.

## What Layer 3 Does and Does Not Establish

### What it establishes

- Retrieval tasks collapse because their unpaired data structure makes most causal estimators inapplicable — not because retrieval is inherently harder in some undefined sense
- SplitUP is the only estimator consistent across all 30 Sutskever papers — it is not merely "another method" but the unique solution for the unpaired high-dimensional regime
- The Schur et al. (2026) middle-regime theory provides identifiability proofs, bias characterization, and consistent estimation for the specific case of many environments with few observations per environment
- 6 of 11 findings from the analysis are scale-invariant mathematical properties that transfer to any model size

### What it does not establish

- A mechanistic explanation of how models internally process retrieval (attention patterns, circuit analysis)
- Empirical evidence of law transfer at GPT-4 scale
- A general mathematical theory of natural language's intermediate structure (Tao's broadest version of the middle regime)
- Real (non-simulated) execution of all 27 estimators on actual paper data

## The Honest Answer to Each Mystery

### Mystery 1: "The causal explanation for why retrieval is harder"

**Partial YES.** Layer 3 provides the identification-level causal chain:

> unpaired data → fewer applicable estimators → measurement-error bias → lower identification confidence → CCL2 classifies as collapse

This is a specific, formal, testable causal explanation — not hand-waving. But it explains why retrieval is harder *to identify causally*, not why it is harder *for the model to perform*.

### Mystery 2: "Whether these laws transfer to GPT-4-scale models"

**Structured clues.** The transfer diagnostics module explicitly flags 6/11 findings as scale-invariant. The estimator taxonomy, the unpaired-bias pathology, and the SplitUP consistency result all transfer. The coefficient magnitudes and onset thresholds do not.

### Mystery 3: "A mathematical theory of the 'middle regime'"

**Analogous YES.** The `middle_regime.py` module implements a rigorous mathematical framework for one specific middle regime (Schur et al. 2026). It is not the general theory of language, but it is a real theorem-backed framework with identifiability proofs, bias formulas, and consistent estimators.

## Bottom Line

If the question is:

"Does Layer 3 fully solve the three remaining mysteries?"

The answer is no.

If the question is:

"Does Layer 3 convert those mysteries from open-ended puzzles into structured, testable, module-backed empirical questions?"

The answer is yes.

That is the progression: Layer 1 measured. Layer 2 classified. Layer 3 explains — partially, honestly, and with explicit markers for what it cannot yet reach.
