"""Measured execution using actual tiny GPT-1 training runs."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .adapters import GPT1WindTunnelAdapter, NotebookSubstrateAdapter
from .compressibility import CompressibilityStack
from .datasets import TaskFamilyDatasetBuilder


class MeasuredRunExecutor:
    """Run tiny but real measured experiments with the linked GPT-1 implementation."""

    def __init__(self, substrate_adapter: NotebookSubstrateAdapter, wind_tunnel_adapter: GPT1WindTunnelAdapter):
        self.substrate_adapter = substrate_adapter
        self.wind_tunnel_adapter = wind_tunnel_adapter
        self.dataset_builder = TaskFamilyDatasetBuilder(substrate_adapter)
        self.compressibility = CompressibilityStack()

    def run(
        self,
        *,
        task_family: str,
        seed: int,
        scale: int,
        data_tokens: int,
        num_layers: int = 2,
        train_steps: int = 4,
        seq_length: int = 24,
        learning_rate: float = 1e-4,
    ) -> Dict[str, Any]:
        if not self.wind_tunnel_adapter.is_available():
            raise RuntimeError("Measured execution requires the linked GPT-1 wind tunnel repository.")
        dataset = self.dataset_builder.build_family_corpus(task_family=task_family, seed=seed, target_tokens=data_tokens)
        module = self.wind_tunnel_adapter.module
        assert module is not None

        vocab = module.create_bpe_vocabulary(dataset["train_text"], num_merges=32)
        encoded_train = module.encode_text(dataset["train_text"], vocab)
        encoded_val = module.encode_text(dataset["val_text"], vocab)
        encoded_holdout = module.encode_text(dataset["holdout_text"], vocab)

        d_model = int(scale)
        num_heads = max(1, min(4, d_model // 16))
        while d_model % num_heads != 0:
            num_heads -= 1
        d_ff = max(d_model * 2, 64)

        np.random.seed(seed)
        model = module.GPT1(
            vocab_size=len(vocab),
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_len=seq_length,
        )
        optimizer = module.AdamOptimizer(lr=learning_rate, weight_decay=0.01)

        snapshots: List[Dict[str, float]] = []
        for step in range(1, train_steps + 1):
            train_loss = self._train_step(module, model, optimizer, encoded_train, seq_length=seq_length, step=step, seed=seed)
            val_loss = self._eval_loss(model, encoded_val, seq_length=seq_length)
            holdout_loss = self._eval_loss(model, encoded_holdout, seq_length=seq_length)
            capability = 1.0 / (1.0 + val_loss)
            snapshots.append(
                {
                    "capability_score": float(capability),
                    "loss_proxy": float(val_loss),
                    "train_loss": float(train_loss),
                    "holdout_loss": float(holdout_loss),
                    "retrieval_dependence": 1.0 if task_family == "retrieval_qa" else 0.0,
                    "data_tokens": float(data_tokens),
                    "scale_proxy": float(scale * num_layers),
                    "task_family_code": float(dataset["task_family_code"]),
                }
            )

        generalization_gap = snapshots[-1]["holdout_loss"] - snapshots[-1]["loss_proxy"]
        flat_params = module.flatten_params(model.get_all_params())
        weight_profile = self.compressibility.profile_model_weights(flat_params, predictive_loss=snapshots[-1]["loss_proxy"])
        return {
            "metric_series": snapshots,
            "train_text": dataset["train_text"],
            "val_text": dataset["val_text"],
            "holdout_text": dataset["holdout_text"],
            "descriptor_hints": dataset["descriptor_hints"],
            "task_family_code": dataset["task_family_code"],
            "generalization_gap": float(generalization_gap),
            "weight_compressibility": weight_profile.to_dict(),
        }

    def _train_step(self, module, model, optimizer, encoded_train: List[int], *, seq_length: int, step: int, seed: int) -> float:
        if len(encoded_train) <= seq_length + 1:
            return 0.0
        starts = [
            (step * 17 + seed * 7) % (len(encoded_train) - seq_length - 1),
            (step * 31 + seed * 11) % (len(encoded_train) - seq_length - 1),
        ]
        losses = []
        for start in starts:
            input_ids = encoded_train[start:start + seq_length]
            target_ids = encoded_train[start + 1:start + seq_length + 1]
            loss = model.compute_loss(input_ids, target_ids)
            dlogits = model.backward_from_loss()
            model.backward(dlogits)
            losses.append(loss)
        flat_params = module.flatten_params(model.get_all_params())
        flat_grads = {k: v for k, v in module.flatten_params(model.get_all_grads()).items() if v is not None}
        optimizer.step(flat_params, flat_grads)
        return float(np.mean(losses))

    @staticmethod
    def _eval_loss(model, encoded_text: List[int], *, seq_length: int) -> float:
        if len(encoded_text) <= seq_length + 1:
            return 0.0
        starts = [0, max((len(encoded_text) - seq_length - 1) // 2, 0)]
        losses = []
        for start in starts:
            input_ids = encoded_text[start:start + seq_length]
            target_ids = encoded_text[start + 1:start + seq_length + 1]
            losses.append(model.compute_loss(input_ids, target_ids))
        return float(np.mean(losses))
