"""Compressibility estimators for the Capability Cartography Layer."""

from __future__ import annotations

import bz2
import gzip
import json
import lzma
import math
import zlib
from typing import Dict

import numpy as np

from .schemas import CompressibilityProfile


class CompressibilityStack:
    """Compute surface, predictive, and structural compression proxies."""

    def profile_text(
        self,
        text: str,
        *,
        predictive_loss: float | None = None,
        weight_array: np.ndarray | None = None,
    ) -> CompressibilityProfile:
        payload = text.encode("utf-8")
        size = max(len(payload), 1)
        surface = {
            "gzip_ratio": float(len(gzip.compress(payload)) / size),
            "zlib_ratio": float(len(zlib.compress(payload)) / size),
            "bz2_ratio": float(len(bz2.compress(payload)) / size),
            "lzma_ratio": float(len(lzma.compress(payload)) / size),
        }
        predictive_loss = float(predictive_loss) if predictive_loss is not None else self._surprise_proxy(text)
        predictive = {
            "cross_entropy_proxy": predictive_loss,
            "perplexity_proxy": float(math.exp(min(predictive_loss, 10.0))),
            "bits_per_character_proxy": float(predictive_loss / math.log(2.0)),
        }
        structural = self._structural_proxy(weight_array, payload)
        gaps = self._gap_metrics(surface, predictive, structural)
        return CompressibilityProfile(surface=surface, predictive=predictive, structural=structural, gaps=gaps)

    def profile_model_weights(self, params: Dict[str, np.ndarray], *, predictive_loss: float | None = None) -> CompressibilityProfile:
        arrays = []
        for value in params.values():
            arrays.append(np.asarray(value, dtype=float).reshape(-1))
        if arrays:
            flat = np.concatenate(arrays)
        else:
            flat = np.zeros(1, dtype=float)
        payload = json.dumps(np.round(flat, 5).tolist()).encode("utf-8")
        size = max(len(payload), 1)
        quantized = np.round(flat, 2)
        delta = np.diff(flat[: min(len(flat), 2048)]) if len(flat) > 1 else np.array([0.0])
        surface = {
            "gzip_ratio": float(len(gzip.compress(payload)) / size),
            "zlib_ratio": float(len(zlib.compress(payload)) / size),
            "bz2_ratio": float(len(bz2.compress(payload)) / size),
            "lzma_ratio": float(len(lzma.compress(payload)) / size),
            "quantized_entropy": float(self._surprise_proxy(json.dumps(quantized[: min(len(quantized), 4096)].tolist()))),
            "delta_entropy": float(np.std(delta)),
        }
        if predictive_loss is None:
            predictive_loss = float(np.var(flat))
        predictive = {
            "cross_entropy_proxy": float(predictive_loss),
            "perplexity_proxy": float(math.exp(min(float(predictive_loss), 10.0))),
            "bits_per_weight_proxy": float(float(predictive_loss) / math.log(2.0)),
        }
        structural = self._structural_proxy(flat, payload)
        structural["l1_norm"] = float(np.sum(np.abs(flat)))
        structural["weight_mean"] = float(np.mean(flat))
        structural["weight_std"] = float(np.std(flat))
        gaps = self._gap_metrics(surface, predictive, structural)
        return CompressibilityProfile(surface=surface, predictive=predictive, structural=structural, gaps=gaps)

    def profile_array(
        self,
        array: np.ndarray,
        *,
        predictive_loss: float | None = None,
        weight_array: np.ndarray | None = None,
    ) -> CompressibilityProfile:
        arr = np.asarray(array, dtype=float)
        payload = json.dumps(np.round(arr, 4).tolist()).encode("utf-8")
        size = max(len(payload), 1)
        surface = {
            "gzip_ratio": float(len(gzip.compress(payload)) / size),
            "zlib_ratio": float(len(zlib.compress(payload)) / size),
            "bz2_ratio": float(len(bz2.compress(payload)) / size),
            "lzma_ratio": float(len(lzma.compress(payload)) / size),
        }
        if predictive_loss is None:
            predictive_loss = float(np.var(arr))
        predictive = {
            "cross_entropy_proxy": float(predictive_loss),
            "perplexity_proxy": float(math.exp(min(float(predictive_loss), 10.0))),
            "bits_per_value_proxy": float(float(predictive_loss) / math.log(2.0)),
        }
        structural = self._structural_proxy(weight_array if weight_array is not None else arr, payload)
        gaps = self._gap_metrics(surface, predictive, structural)
        return CompressibilityProfile(surface=surface, predictive=predictive, structural=structural, gaps=gaps)

    @staticmethod
    def _surprise_proxy(text: str) -> float:
        chars = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
        if chars.size == 0:
            return 0.0
        _, counts = np.unique(chars, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))
        return float(entropy)

    @staticmethod
    def _structural_proxy(weight_array: np.ndarray | None, payload: bytes) -> Dict[str, float]:
        if weight_array is None:
            description_length = float(len(payload) * 8)
            effective_params = float(len(payload))
            spectral_complexity = 0.0
            sparsity = 0.0
        else:
            arr = np.asarray(weight_array, dtype=float)
            description_length = float(np.sum(np.abs(arr)) + arr.size)
            effective_params = float(np.sum(~np.isclose(arr, 0.0)))
            if arr.ndim >= 2:
                matrix = arr.reshape(arr.shape[0], -1)
                singular_values = np.linalg.svd(matrix, compute_uv=False)
                normalized = singular_values / max(np.sum(singular_values), 1e-12)
                spectral_complexity = float(-np.sum(normalized * np.log2(normalized + 1e-12)))
            else:
                spectral_complexity = float(np.std(arr))
            sparsity = float(np.mean(np.isclose(arr, 0.0)))
        return {
            "description_length_proxy": description_length,
            "effective_params": effective_params,
            "spectral_complexity": spectral_complexity,
            "sparsity": sparsity,
        }

    @staticmethod
    def _gap_metrics(
        surface: Dict[str, float],
        predictive: Dict[str, float],
        structural: Dict[str, float],
    ) -> Dict[str, float]:
        surface_mean = float(np.mean(list(surface.values())))
        predictive_mean = float(np.mean(list(predictive.values())))
        effective_params = max(structural["effective_params"], 1.0)
        description_bits = structural["description_length_proxy"] / effective_params
        appearance_vs_structure = surface_mean - description_bits
        prediction_vs_structure = predictive_mean - description_bits
        if appearance_vs_structure > 0.25 and prediction_vs_structure > 0.25:
            regime = 2.0
        elif appearance_vs_structure < -0.25 and prediction_vs_structure < -0.25:
            regime = 0.0
        else:
            regime = 1.0
        return {
            "appearance_vs_structure": float(appearance_vs_structure),
            "prediction_vs_structure": float(prediction_vs_structure),
            "structure_vs_noise_gap": float(description_bits - surface_mean),
            "compressibility_regime_code": regime,
        }
