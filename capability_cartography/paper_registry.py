"""Registry of the 30 Sutskever papers with causal structure.

Each paper has:
  - mechanism_X: the architectural/algorithmic intervention
  - capability_Y: the capability claimed to result
  - paper_type: determines which estimators are applicable
  - data_structure: paired vs unpaired
  - regime parameters: m (environments), r (obs/env), d, s_star
  - causal_question: what the paper implicitly asks about causality
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PaperEntry:
    paper_id: int
    slug: str
    title: str
    mechanism_X: str
    capability_Y: str
    paper_type: str
    data_structure: str
    n_environments: int
    obs_per_env: float
    treatment_dim: int
    sparsity: int
    instrument_strength: float
    causal_question: str

    def to_dict(self) -> Dict:
        from dataclasses import asdict
        return asdict(self)


_ENTRIES: List[PaperEntry] = [
    PaperEntry(1,  "complexodynamics",        "First Law of Complexodynamics",
               "state_evolution",        "complexity_score",        "theory",         "unpaired", 2,   50, 2, 1, 0.10,
               "Does the state evolution rule cause complexity to increase then decrease?"),
    PaperEntry(2,  "char_rnn",                "Unreasonable Effectiveness of RNNs",
               "recurrence",             "sequence_prediction",     "architecture",   "paired",   5,  100, 3, 2, 0.80,
               "Does the recurrent connection cause the model to predict sequences?"),
    PaperEntry(3,  "lstm",                    "Understanding LSTM Networks",
               "gating_mechanism",       "long_range_dependency",   "architecture",   "paired",   5,  100, 4, 3, 0.80,
               "Do forget/input/output gates cause long-range dependency learning?"),
    PaperEntry(4,  "rnn_regularization",      "RNN Regularization",
               "dropout_mask",           "generalization",          "regularization",  "paired",   3,  150, 5, 2, 0.70,
               "Does applying dropout between recurrent layers cause better generalization?"),
    PaperEntry(5,  "pruning",                 "Keeping Neural Networks Simple",
               "weight_pruning",         "model_simplicity",        "regularization",  "paired",   3,  150, 5, 2, 0.70,
               "Does pruning small weights cause the network to generalize with fewer parameters?"),
    PaperEntry(6,  "pointer_networks",        "Pointer Networks",
               "pointer_attention",      "combinatorial_accuracy",  "architecture",   "paired",   5,  100, 3, 2, 0.75,
               "Does the pointer mechanism cause combinatorial output accuracy?"),
    PaperEntry(7,  "alexnet",                 "ImageNet Classification with Deep CNNs",
               "conv_depth",             "classification_accuracy", "architecture",   "paired",   5,  100, 3, 2, 0.80,
               "Does deeper convolutional architecture cause higher classification accuracy?"),
    PaperEntry(8,  "seq2seq_sets",            "Order Matters: Seq2Seq for Sets",
               "set_encoding",           "order_invariance",        "architecture",   "paired",   5,  100, 3, 2, 0.75,
               "Does the set-encoding mechanism cause permutation-invariant outputs?"),
    PaperEntry(9,  "gpipe",                   "GPipe: Pipeline Parallelism",
               "pipeline_parallelism",   "training_throughput",     "systems",        "unpaired", 2,   50, 2, 1, 0.10,
               "Does pipeline-parallel splitting cause higher training throughput?"),
    PaperEntry(10, "resnet",                  "Deep Residual Learning",
               "residual_connection",    "deep_trainability",       "architecture",   "paired",   8,   80, 3, 2, 0.85,
               "Does the residual skip connection cause deep networks to remain trainable?"),
    PaperEntry(11, "dilated_convolutions",    "Dilated Convolutions",
               "dilation_rate",          "receptive_field_growth",  "architecture",   "paired",   5,  100, 3, 2, 0.80,
               "Does increasing dilation rate cause exponential receptive field growth?"),
    PaperEntry(12, "gnn",                     "Neural Message Passing for Graphs",
               "message_passing",        "graph_classification",    "architecture",   "paired",   6,   80, 4, 2, 0.75,
               "Does the message-passing mechanism cause graph-level prediction accuracy?"),
    PaperEntry(13, "transformer",             "Attention Is All You Need",
               "self_attention",         "sequence_modeling_loss",   "architecture",   "paired",   5,  100, 4, 3, 0.85,
               "Does multi-head self-attention cause better sequence modeling than recurrence?"),
    PaperEntry(14, "bahdanau_attention",      "Neural Machine Translation by Alignment",
               "alignment_score",        "translation_quality",     "architecture",   "paired",   5,  100, 3, 2, 0.80,
               "Does the learned alignment score cause better translation quality?"),
    PaperEntry(15, "identity_mappings",       "Identity Mappings in Deep ResNets",
               "preactivation_order",    "gradient_flow",           "architecture",   "paired",   4,  120, 2, 1, 0.85,
               "Does pre-activation batch norm ordering cause cleaner gradient flow?"),
    PaperEntry(16, "relational_reasoning",    "A Simple Neural Network Module for Relational Reasoning",
               "pairwise_composition",   "relational_accuracy",     "architecture",   "paired",   6,   80, 4, 2, 0.75,
               "Does explicit pairwise object comparison cause relational reasoning capability?"),
    PaperEntry(17, "vae",                     "Variational Lossy Autoencoder",
               "kl_regularization",      "reconstruction_quality",  "generative",     "paired",   4,   80, 6, 3, 0.65,
               "Does KL regularization of the latent space cause meaningful generation?"),
    PaperEntry(18, "relational_rnn",          "Relational Recurrent Neural Networks",
               "memory_attention",       "relational_memory",       "architecture",   "paired",   5,  100, 4, 2, 0.75,
               "Does multi-head attention over memory slots cause relational reasoning in RNNs?"),
    PaperEntry(19, "coffee_automaton",        "The Coffee Automaton",
               "automaton_rules",        "entropy_trajectory",      "theory",         "unpaired", 2,   50, 2, 1, 0.10,
               "Do the automaton update rules cause the characteristic entropy rise-then-fall?"),
    PaperEntry(20, "ntm",                     "Neural Turing Machines",
               "external_memory",        "algorithmic_generalization", "architecture", "paired",  5,  100, 5, 3, 0.70,
               "Does differentiable external memory cause algorithmic generalization?"),
    PaperEntry(21, "ctc",                     "Deep Speech 2 / CTC",
               "ctc_alignment",          "speech_recognition",      "architecture",   "paired",   5,  100, 3, 2, 0.75,
               "Does the CTC loss alignment cause accurate end-to-end speech recognition?"),
    PaperEntry(22, "scaling_laws",            "Scaling Laws for Neural Language Models",
               "parameter_scale",        "loss_reduction",          "scaling",        "paired",  10,   30, 2, 1, 0.90,
               "Does increasing parameter count cause predictable loss reduction?"),
    PaperEntry(23, "mdl",                     "MDL Principle",
               "description_length",     "model_selection",         "theory",         "unpaired", 2,   50, 2, 1, 0.10,
               "Does minimizing description length cause better model selection?"),
    PaperEntry(24, "machine_super_intelligence", "Machine Super Intelligence",
               "capability_aggregation", "intelligence_proxy",      "theory",         "unpaired", 2,   50, 2, 1, 0.10,
               "Does aggregating narrow capabilities cause general intelligence?"),
    PaperEntry(25, "kolmogorov",              "Kolmogorov Complexity",
               "program_length",         "compressibility",         "theory",         "unpaired", 2,   50, 2, 1, 0.10,
               "Does shorter program length cause higher compressibility?"),
    PaperEntry(26, "cs231n",                  "CS231n: CNNs for Visual Recognition",
               "conv_fundamentals",      "feature_extraction",      "architecture",   "paired",   5,  100, 3, 2, 0.80,
               "Do convolutional filters cause hierarchical feature extraction?"),
    PaperEntry(27, "multi_token_prediction",  "Better & Faster LLMs via Multi-Token Prediction",
               "multi_head_prediction",  "sample_efficiency",       "architecture",   "paired",   5,  100, 3, 2, 0.80,
               "Do multiple prediction heads cause improved sample efficiency?"),
    PaperEntry(28, "dpr",                     "Dense Passage Retrieval",
               "dual_encoder",           "retrieval_accuracy",      "retrieval",      "unpaired", 20,   4, 5, 2, 0.30,
               "Does a dual-encoder architecture cause accurate passage retrieval?"),
    PaperEntry(29, "rag",                     "Retrieval-Augmented Generation",
               "retrieval_conditioning", "generation_factuality",   "retrieval",      "unpaired", 20,   4, 5, 2, 0.30,
               "Does conditioning generation on retrieved documents cause more factual output?"),
    PaperEntry(30, "lost_in_middle",          "Lost in the Middle",
               "context_position",       "positional_accuracy",     "retrieval",      "unpaired", 20,   4, 5, 2, 0.30,
               "Does the position of relevant information in context cause accuracy variation?"),
]


class PaperRegistry:
    """Queryable registry of the 30 Sutskever papers."""

    def __init__(self) -> None:
        self.papers = {p.paper_id: p for p in _ENTRIES}

    def get(self, paper_id: int) -> PaperEntry:
        return self.papers[paper_id]

    def all_ids(self) -> List[int]:
        return sorted(self.papers.keys())

    def by_type(self, paper_type: str) -> List[PaperEntry]:
        return [p for p in self.papers.values() if p.paper_type == paper_type]

    def by_data_structure(self, structure: str) -> List[PaperEntry]:
        return [p for p in self.papers.values() if p.data_structure == structure]

    def retrieval_papers(self) -> List[PaperEntry]:
        return self.by_type("retrieval")

    def theory_papers(self) -> List[PaperEntry]:
        return self.by_type("theory")

    def summary(self) -> Dict:
        from collections import Counter
        types = Counter(p.paper_type for p in self.papers.values())
        structures = Counter(p.data_structure for p in self.papers.values())
        return {"total": len(self.papers), "by_type": dict(types), "by_data_structure": dict(structures)}
