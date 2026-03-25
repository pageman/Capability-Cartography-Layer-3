"""Causal-layer visualizations for Layer 3.

Produces:
  - 30×27 estimator heatmap
  - Causality verdicts dashboard
  - Middle-regime phase plot
  - Transfer diagnostics summary chart
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def plot_estimator_heatmap(
    paper_results: Dict[int, List[Dict[str, Any]]],
    paper_names: Dict[int, str],
    estimator_names: List[str],
    *,
    output_path: str | Path,
) -> str:
    """30-paper × 27-estimator heatmap: green=consistent, yellow=biased, gray=N/A."""
    n_papers = len(paper_results)
    n_est = len(estimator_names)
    matrix = np.zeros((n_papers, n_est))
    pids = sorted(paper_results.keys())

    for i, pid in enumerate(pids):
        results = paper_results[pid]
        est_map = {r["estimator"]: r for r in results}
        for j, ename in enumerate(estimator_names):
            r = est_map.get(ename, {})
            if not r.get("applicable", False):
                matrix[i, j] = 0.0
            elif r.get("consistent", False):
                matrix[i, j] = 1.0
            else:
                matrix[i, j] = 0.5

    cmap = mcolors.ListedColormap(["#e0e0e0", "#fff176", "#66bb6a"])
    bounds = [-0.25, 0.25, 0.75, 1.25]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(18, 12))
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(n_est))
    ax.set_xticklabels([e.replace("_", "\n") for e in estimator_names], fontsize=5, rotation=90)
    ax.set_yticks(range(n_papers))
    ax.set_yticklabels([f"P{pid:02d} {paper_names.get(pid, '')}" for pid in pids], fontsize=6)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.set_title("30 Papers × 27 Estimators: Causal Identification Matrix", fontsize=12, fontweight="bold", pad=20)

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#66bb6a", label="Consistent"),
        Patch(facecolor="#fff176", label="Applicable but biased"),
        Patch(facecolor="#e0e0e0", label="Not applicable"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=7)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_verdict_dashboard(
    records: Sequence[Dict[str, Any]],
    *,
    output_path: str | Path,
) -> str:
    """Per-paper causality verdict bar chart."""
    fig, ax = plt.subplots(figsize=(12, 9))
    color_map = {"CONFIRMED": "#2e7d32", "CONDITIONAL": "#f57f17", "UNCONFIRMED": "#c62828"}

    for i, rec in enumerate(records):
        v = rec.get("causality_verdict", "UNCONFIRMED")
        color = color_map.get(v, "#9e9e9e")
        score = float(rec.get("capability_score", 0))
        ax.barh(i, score, color=color, height=0.6, alpha=0.85, edgecolor="white")
        ax.text(-0.02, i, f"P{rec['paper_id']:02d} {rec.get('paper_name', '')}", ha="right", va="center", fontsize=6)
        ax.text(score + 0.01, i, f"{v} [{rec.get('best_estimator', '')}]", ha="left", va="center", fontsize=5, color=color)

    ax.set_xlim(-0.35, 0.85)
    ax.set_ylim(-0.5, len(records) - 0.5)
    ax.invert_yaxis()
    ax.set_xlabel("Capability Score")
    ax.set_title("Causality Verdicts: 30 Papers", fontsize=11, fontweight="bold")
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=c, label=l) for l, c in color_map.items()]
    ax.legend(handles=legend, loc="lower right", fontsize=7)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_regime_map(
    profiles: Sequence[Dict[str, Any]],
    *,
    output_path: str | Path,
) -> str:
    """Scatter plot of papers in (m, r) space colored by regime label."""
    fig, ax = plt.subplots(figsize=(9, 6))
    color_map = {
        "classical_large_sample": "#42a5f5",
        "high_dim_moderate_r": "#ef5350",
        "extreme_high_dim": "#c62828",
        "moderate_high_dim": "#ff7043",
        "under_identified": "#bdbdbd",
        "sparse_under_identified": "#9e9e9e",
    }
    for p in profiles:
        c = color_map.get(p.get("regime_label", ""), "#757575")
        ax.scatter(p["m"], p["r"], c=c, s=80, edgecolor="black", linewidth=0.5, zorder=3)
        ax.annotate(f"P{p['paper_id']:02d}", (p["m"], p["r"]), fontsize=6, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")
    ax.set_xlabel("Environments (m)")
    ax.set_ylabel("Observations per env (r = n/m)")
    ax.set_title("Middle-Regime Map: Papers in (m, r) Space", fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    legend = [Patch(facecolor=c, label=l) for l, c in color_map.items() if any(p.get("regime_label") == l for p in profiles)]
    if legend:
        ax.legend(handles=legend, fontsize=6, loc="upper right")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)
