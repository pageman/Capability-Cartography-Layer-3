"""Task descriptor extraction for the Capability Cartography Layer."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .schemas import TaskDescriptor


def _entropy(values: Sequence[Any]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(sum(counts.values()))
    probabilities = np.array([count / total for count in counts.values()], dtype=float)
    return float(-np.sum(probabilities * np.log2(probabilities + 1e-12)))


class TaskDescriptorExtractor:
    """Extracts structured task descriptors from text or arrays."""

    COGNITIVE_KEYWORDS = {
        "arithmetic": ("sum", "add", "subtract", "count", "number"),
        "logical_deduction": ("if", "then", "therefore", "because", "implies"),
        "analogical_reasoning": ("like", "similar", "analogy", "map"),
        "causal_inference": ("cause", "effect", "because", "due"),
        "counterfactual": ("would", "could", "if only", "counterfactual"),
        "retrieval": ("retrieve", "search", "passage", "document", "context"),
    }

    def extract_text_descriptor(
        self,
        text: str,
        *,
        task_name: str,
        benchmark_label: str,
        substrate: str,
        realism_level: str = "synthetic",
        metadata: Optional[Dict[str, Any]] = None,
        retrieval_context: Optional[str] = None,
    ) -> TaskDescriptor:
        tokens = re.findall(r"\w+|[^\w\s]", text.lower())
        token_count = len(tokens)
        chars = list(text)
        unique_tokens = len(set(tokens)) if tokens else 0
        sentences = [segment.strip() for segment in re.split(r"[.!?]+", text) if segment.strip()]
        avg_sentence_len = token_count / max(len(sentences), 1)
        punctuation_density = sum(1 for token in tokens if re.fullmatch(r"[^\w\s]+", token)) / max(token_count, 1)
        numbers = re.findall(r"\d+", text)
        uppercase_ratio = sum(1 for char in text if char.isupper()) / max(len(text), 1)

        retrieval_text = retrieval_context or ""
        retrieval_tokens = re.findall(r"\w+", retrieval_text.lower())
        shared_tokens = len(set(tokens).intersection(retrieval_tokens))

        cognitive = {
            name: float(any(keyword in text.lower() for keyword in keywords))
            for name, keywords in self.COGNITIVE_KEYWORDS.items()
        }
        cognitive["multi_hop_proxy"] = float(sum(text.lower().count(word) for word in ("and", "then", "after", "before")))
        cognitive["working_memory_load"] = float(avg_sentence_len / 10.0)

        latent = {
            "syntactic_depth_proxy": float(sum(text.count(symbol) for symbol in "(),") / max(len(sentences), 1)),
            "semantic_coherence_proxy": float(shared_tokens / max(unique_tokens, 1)),
            "logical_structure_score": float(sum(text.lower().count(word) for word in ("if", "then", "because", "therefore"))),
            "temporal_complexity": float(sum(text.lower().count(word) for word in ("before", "after", "while", "then"))),
        }

        retrieval = {
            "retrieval_dependency_score": float(shared_tokens / max(len(retrieval_tokens), 1)) if retrieval_tokens else 0.0,
            "distractor_density": float(max(len(retrieval_tokens) - shared_tokens, 0) / max(len(retrieval_tokens), 1)) if retrieval_tokens else 0.0,
            "context_window_utilization": float((token_count + len(retrieval_tokens)) / max(token_count + len(retrieval_tokens), 1)),
            "answer_position_bias": self._answer_position_bias(text),
        }

        perturbation = {
            "character_noise_level": float(sum(not char.isalnum() and not char.isspace() for char in text) / max(len(text), 1)),
            "negation_complexity": float(sum(text.lower().count(word) for word in ("not", "never", "no"))),
            "ambiguity_score": float(sum(text.lower().count(word) for word in ("maybe", "perhaps", "possibly"))),
            "paraphrase_distance_proxy": float(unique_tokens / max(token_count, 1)),
        }

        structural = {
            "token_entropy": _entropy(tokens),
            "character_entropy": _entropy(chars),
            "type_token_ratio": float(unique_tokens / max(token_count, 1)),
            "numeric_density": float(len(numbers) / max(token_count, 1)),
            "uppercase_ratio": float(uppercase_ratio),
            "avg_sentence_length": float(avg_sentence_len),
            "kolmogorov_proxy_seed": float(len(text.encode("utf-8")) / max(token_count, 1)),
        }

        return TaskDescriptor(
            task_name=task_name,
            benchmark_label=benchmark_label,
            substrate=substrate,
            realism_level=realism_level,
            surface_statistics={
                "token_count": float(token_count),
                "unique_tokens": float(unique_tokens),
                "punctuation_density": float(punctuation_density),
                "avg_sentence_length": float(avg_sentence_len),
            },
            latent_structure=latent,
            retrieval_geometry=retrieval,
            perturbation_profile=perturbation,
            cognitive_operations=cognitive,
            structural_complexity=structural,
            metadata=metadata or {},
        )

    def extract_array_descriptor(
        self,
        array: np.ndarray,
        *,
        task_name: str,
        benchmark_label: str,
        substrate: str,
        realism_level: str = "synthetic",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskDescriptor:
        flat = np.asarray(array, dtype=float).reshape(-1)
        rounded = np.round(flat, 3).tolist()
        finite = flat[np.isfinite(flat)]
        std = float(np.std(finite)) if finite.size else 0.0
        mean = float(np.mean(finite)) if finite.size else 0.0
        zero_fraction = float(np.mean(np.isclose(flat, 0.0))) if flat.size else 0.0
        sign_changes = float(np.mean(np.diff(np.signbit(flat).astype(int)) != 0)) if flat.size > 1 else 0.0

        metadata = metadata or {}
        seq_len = float(metadata.get("seq_len", array.shape[1] if array.ndim > 1 else array.shape[0]))

        return TaskDescriptor(
            task_name=task_name,
            benchmark_label=benchmark_label,
            substrate=substrate,
            realism_level=realism_level,
            surface_statistics={
                "element_count": float(flat.size),
                "mean": mean,
                "std": std,
                "zero_fraction": zero_fraction,
            },
            latent_structure={
                "relational_depth_proxy": float(metadata.get("relational_depth", math.log2(max(seq_len, 2)))),
                "temporal_complexity": float(seq_len / 4.0),
                "modularity_hint": float(metadata.get("n_objects", metadata.get("n_pairs", 1))),
            },
            retrieval_geometry={
                "retrieval_dependency_score": float(metadata.get("retrieval_dependency_score", 0.0)),
                "distractor_density": float(metadata.get("distractor_density", 0.0)),
                "answer_position_bias": float(metadata.get("answer_position_bias", 0.0)),
                "information_scattering": float(metadata.get("information_scattering", sign_changes)),
            },
            perturbation_profile={
                "noise_resistance_target": float(metadata.get("noise_level", std)),
                "shuffle_sensitivity_proxy": float(sign_changes),
            },
            cognitive_operations={
                "working_memory_load": float(seq_len / 8.0),
                "multi_hop_proxy": float(metadata.get("multi_hop", max(seq_len / 6.0, 1.0))),
                "mapping_operation": float(metadata.get("mapping_operation", 1.0)),
            },
            structural_complexity={
                "value_entropy": _entropy(rounded),
                "effective_dimensionality_proxy": float(np.linalg.matrix_rank(array.reshape(array.shape[0], -1))) if array.ndim > 1 else 1.0,
                "compressibility_seed": float(np.mean(np.abs(flat))) if flat.size else 0.0,
            },
            metadata=metadata,
        )

    @staticmethod
    def _answer_position_bias(text: str) -> float:
        lowered = text.lower()
        markers = ("answer", "therefore", "thus")
        positions = [lowered.find(marker) for marker in markers if lowered.find(marker) >= 0]
        if not positions:
            return 0.0
        average_position = sum(positions) / len(positions)
        return float(average_position / max(len(lowered), 1))
