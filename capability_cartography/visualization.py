"""Visualization modules for onset surfaces and phase regions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


class CartographyVisualizer:
    """Create static plots from sweep and measured records."""

    def plot_onset_surface(self, records: Sequence[Dict[str, float]], *, output_path: str | Path) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        xs = [float(record.get("scale", 0.0)) for record in records]
        ys = [float(record.get("data_tokens", 0.0)) for record in records]
        zs = [float(record.get("capability_score", 0.0)) for record in records]
        colors = zs
        scatter = ax.scatter(xs, ys, zs, c=colors, cmap="viridis")
        ax.set_xlabel("Scale")
        ax.set_ylabel("Data Tokens")
        ax.set_zlabel("Capability")
        fig.colorbar(scatter, ax=ax, label="Capability")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return str(path)

    def plot_phase_regions(self, records: Sequence[Dict[str, float]], *, output_path: str | Path) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        task_names = sorted({str(record.get("task_family", "unknown")) for record in records})
        scale_values = sorted({float(record.get("scale", 0.0)) for record in records})
        matrix = np.zeros((len(task_names), len(scale_values)))
        for i, task in enumerate(task_names):
            for j, scale in enumerate(scale_values):
                matching = [
                    float(record.get("capability_score", 0.0))
                    for record in records
                    if str(record.get("task_family", "unknown")) == task and float(record.get("scale", 0.0)) == scale
                ]
                matrix[i, j] = np.mean(matching) if matching else 0.0
        image = ax.imshow(matrix, cmap="magma", aspect="auto")
        ax.set_xticks(range(len(scale_values)))
        ax.set_xticklabels([str(int(v)) for v in scale_values])
        ax.set_yticks(range(len(task_names)))
        ax.set_yticklabels(task_names)
        ax.set_xlabel("Scale")
        ax.set_ylabel("Task Family")
        fig.colorbar(image, ax=ax, label="Capability")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return str(path)
