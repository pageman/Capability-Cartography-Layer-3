"""Measured task-family dataset generation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .adapters import NotebookSubstrateAdapter


class TaskFamilyDatasetBuilder:
    """Builds distinct corpora for measured runs."""

    FAMILY_CODES = {
        "object_tracking": 0.0,
        "pair_matching": 1.0,
        "babi_simple": 2.0,
        "retrieval_qa": 3.0,
    }

    def __init__(self, substrate_adapter: NotebookSubstrateAdapter):
        self.substrate_adapter = substrate_adapter
        self.module = self._load_reasoning_tasks()

    def build_family_corpus(self, *, task_family: str, seed: int, target_tokens: int = 4096) -> Dict[str, Any]:
        text, descriptor_hints = self._family_text_and_hints(task_family=task_family, seed=seed, target_tokens=target_tokens)

        while len(text.split()) < target_tokens:
            text = text + "\n" + text
        tokens = text.split()
        clipped = " ".join(tokens[:target_tokens])
        midpoint = max(int(len(clipped) * 0.8), 1)
        train_text = clipped[:midpoint]
        val_text = clipped[midpoint:]
        holdout = self._build_holdout_variant(task_family=task_family, seed=seed + 101, target_tokens=max(target_tokens // 3, 512))
        return {
            "task_family": task_family,
            "task_family_code": self.FAMILY_CODES.get(task_family, -1.0),
            "train_text": train_text,
            "val_text": val_text,
            "holdout_text": holdout,
            "descriptor_hints": descriptor_hints,
        }

    def _load_reasoning_tasks(self):
        root = self.substrate_adapter.root
        if root is None:
            return None
        path = Path(root) / "reasoning_tasks.py"
        if not path.exists():
            return None
        spec = importlib.util.spec_from_file_location("reasoning_tasks", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _serialize_tracking(self, seed: int, sample_count: int) -> str:
        np.random.seed(seed)
        X, y, _ = self.module.generate_object_tracking(n_samples=sample_count, seq_len=10, n_objects=3, grid_size=5)
        lines = []
        for idx in range(sample_count):
            seq = " ".join(f"{value:.2f}" for value in X[idx].reshape(-1)[:30])
            target = " ".join(f"{value:.2f}" for value in y[idx])
            lines.append(f"TRACK sample={idx} seq={seq} target={target}")
        return "\n".join(lines)

    def _serialize_pair_matching(self, seed: int, sample_count: int) -> str:
        np.random.seed(seed)
        X, y, _ = self.module.generate_pair_matching(n_samples=sample_count, seq_len=12, vocab_size=10)
        lines = []
        for idx in range(sample_count):
            seq = " ".join(str(int(v * 9)) for v in X[idx].reshape(-1)[:40])
            target = " ".join(str(int(v)) for v in y[idx][:10])
            lines.append(f"PAIR sample={idx} seq={seq} target={target}")
        return "\n".join(lines)

    def _serialize_babi(self, seed: int, sample_count: int) -> str:
        np.random.seed(seed)
        X, y, _ = self.module.generate_babi_simple(n_samples=sample_count, max_facts=5, n_entities=5, n_locations=4)
        lines = []
        for idx in range(sample_count):
            seq = " ".join(str(int(v)) for v in X[idx].reshape(-1)[:50])
            target = " ".join(str(int(v)) for v in y[idx][:4])
            lines.append(f"BABI sample={idx} seq={seq} target={target}")
        return "\n".join(lines)

    def _build_retrieval_qa_text(self, seed: int, sample_count: int) -> str:
        rng = np.random.default_rng(seed)
        names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
        places = ["kitchen", "garden", "office", "library"]
        lines = []
        for idx in range(sample_count):
            answer_name = names[idx % len(names)]
            answer_place = places[idx % len(places)]
            distractor = names[(idx + 2) % len(names)]
            lines.append(
                f"RETRIEVAL sample={idx} context: {answer_name} stores the theorem in the {answer_place}. "
                f"{distractor} mentions an unrelated lemma in the archive. "
                f"question: where is the theorem? answer: {answer_place}"
            )
        return "\n".join(lines)

    def _build_holdout_variant(self, *, task_family: str, seed: int, target_tokens: int) -> str:
        text, _ = self._family_text_and_hints(task_family=task_family, seed=seed, target_tokens=target_tokens)
        while len(text.split()) < target_tokens:
            text = text + "\n" + text
        tokens = text.split()
        return " ".join(tokens[:target_tokens])

    def _family_text_and_hints(self, *, task_family: str, seed: int, target_tokens: int):
        if task_family == "object_tracking" and self.module is not None:
            return self._serialize_tracking(seed, sample_count=max(24, target_tokens // 120)), {
                "relational_depth": 3.0,
                "distractor_density": 0.1,
            }
        if task_family == "pair_matching" and self.module is not None:
            return self._serialize_pair_matching(seed, sample_count=max(24, target_tokens // 120)), {
                "relational_depth": 2.0,
                "distractor_density": 0.2,
            }
        if task_family == "babi_simple" and self.module is not None:
            return self._serialize_babi(seed, sample_count=max(24, target_tokens // 140)), {
                "relational_depth": 4.0,
                "distractor_density": 0.3,
            }
        return self._build_retrieval_qa_text(seed, sample_count=max(24, target_tokens // 140)), {
            "relational_depth": 2.5,
            "distractor_density": 0.45,
        }
