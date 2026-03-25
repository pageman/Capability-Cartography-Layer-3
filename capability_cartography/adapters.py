"""Adapters for the Sutskever substrate, GPT-1 wind tunnel, and agent layer."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from .provenance import repository_provenance

def _env_or_default(env_var: str, default: str | None) -> Path | None:
    value = os.environ.get(env_var, default)
    return Path(value).expanduser() if value else None


def _resolve_root(explicit: Path | str | None, env_var: str, candidates: List[str]) -> Path | None:
    if explicit is not None:
        path = Path(explicit).expanduser()
        return path
    env_path = _env_or_default(env_var, None)
    if env_path is not None:
        return env_path
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return path
    return None


class NotebookSubstrateAdapter:
    """Metadata and task adapters for the Sutskever-30 substrate."""
    CANONICAL_REPOSITORY = "https://github.com/pageman/Sutskever-30-implementations"
    DEFAULT_CANDIDATES = [
        "~/sutskever-30-implementations",
        "~/Downloads/sutskever-30-implementations/sutskever-30-implementations-main",
        "~/Downloads/sutskever-30-implementations",
    ]

    PAPER_TRACKS = {
        "01_complexity_dynamics": "foundational",
        "05_neural_network_pruning": "foundational",
        "13_attention_is_all_you_need": "architecture",
        "16_relational_reasoning": "architecture",
        "18_relational_rnn": "advanced",
        "22_scaling_laws": "scaling",
        "23_mdl_principle": "theory",
        "25_kolmogorov_complexity": "theory",
        "28_dense_passage_retrieval": "retrieval",
        "29_rag": "retrieval",
        "30_lost_in_middle": "retrieval",
    }

    def __init__(self, root: Path | str | None = None):
        self.root = _resolve_root(root, "SUTSKEVER30_ROOT", self.DEFAULT_CANDIDATES)

    @classmethod
    def expected_paths(cls) -> List[str]:
        return [str(Path(candidate).expanduser()) for candidate in cls.DEFAULT_CANDIDATES]

    def notebook_path(self, notebook_name: str) -> Path | None:
        if self.root is None:
            return None
        return self.root / f"{notebook_name}.ipynb"

    def diagnostic_summary(self, notebook_name: str | None = None) -> Dict[str, Any]:
        notebook_path = self.notebook_path(notebook_name) if notebook_name is not None else None
        return {
            "env_var": "SUTSKEVER30_ROOT",
            "configured_root": str(self.root) if self.root is not None else None,
            "notebook_name": notebook_name,
            "notebook_path": str(notebook_path) if notebook_path is not None else None,
            "notebook_exists": bool(notebook_path and notebook_path.exists()),
            "searched_candidates": self.expected_paths(),
            "canonical_repository": self.CANONICAL_REPOSITORY,
            "available": bool(self.root and self.root.exists()),
        }

    def missing_dependency_message(self, notebook_name: str | None = None) -> str:
        diagnostic = self.diagnostic_summary(notebook_name)
        searched = "\n".join(f"  - {path}" for path in diagnostic["searched_candidates"])
        configured = diagnostic["configured_root"] or "<unset>"
        target = diagnostic["notebook_path"] or "<unresolved>"
        target_name = diagnostic["notebook_name"] or "<unspecified>"
        return (
            "Notebook execution requires the linked Sutskever-30 substrate repository.\n"
            f"Set {diagnostic['env_var']} to the repo root containing `{target_name}.ipynb`.\n"
            f"Configured root: {configured}\n"
            f"Expected notebook: {target}\n"
            f"Searched candidate roots:\n{searched}\n"
            f"Canonical repository: {diagnostic['canonical_repository']}"
        )

    def list_notebooks(self) -> List[Dict[str, str]]:
        if self.root is None or not self.root.exists():
            return []
        notebooks = []
        for path in sorted(self.root.glob("*.ipynb")):
            notebooks.append(
                {
                    "paper_id": path.stem.split("_", 1)[0],
                    "name": path.stem,
                    "track": self.PAPER_TRACKS.get(path.stem, "general"),
                    "path": str(path),
                }
            )
        return notebooks

    def describe_notebook(self, notebook_name: str) -> Dict[str, Any]:
        if self.root is None:
            raise FileNotFoundError(self.missing_dependency_message(notebook_name))
        path = self.notebook_path(notebook_name)
        assert path is not None
        if not path.exists():
            raise FileNotFoundError(self.missing_dependency_message(notebook_name))
        return {
            "name": notebook_name,
            "path": str(path),
            "track": self.PAPER_TRACKS.get(notebook_name, "general"),
            "size_bytes": path.stat().st_size,
        }

    def link_metadata(self) -> Dict[str, Any]:
        payload = repository_provenance(
            name="pageman/Sutskever-30-implementations",
            url=self.CANONICAL_REPOSITORY,
            root=self.root,
        )
        payload["notebook_count"] = len(self.list_notebooks())
        return payload


class GPT1WindTunnelAdapter:
    """Loads the GPT-1 implementation and exposes cartography-friendly hooks."""
    CANONICAL_REPOSITORY = "https://github.com/pageman/gpt1-from-Sutskever30"
    DEFAULT_CANDIDATES = [
        "~/gpt1-from-Sutskever30",
        "~/Downloads/GPT1_from_Sutskerver30/GPT1_from_Sutskever30",
        "~/Downloads/gpt1-from-Sutskever30",
    ]

    def __init__(self, root: Path | str | None = None):
        self.root = _resolve_root(root, "GPT1_WIND_TUNNEL_ROOT", self.DEFAULT_CANDIDATES)
        self.module = None
        if self.root is not None:
            path = self.root / "gpt1_complete_implementation.py"
            if path.exists():
                self.module = self._load_module(path)

    @classmethod
    def expected_paths(cls) -> List[str]:
        return [str(Path(candidate).expanduser()) for candidate in cls.DEFAULT_CANDIDATES]

    def implementation_path(self) -> Path | None:
        if self.root is None:
            return None
        return self.root / "gpt1_complete_implementation.py"

    def diagnostic_summary(self) -> Dict[str, Any]:
        implementation_path = self.implementation_path()
        return {
            "env_var": "GPT1_WIND_TUNNEL_ROOT",
            "configured_root": str(self.root) if self.root is not None else None,
            "implementation_path": str(implementation_path) if implementation_path is not None else None,
            "implementation_exists": bool(implementation_path and implementation_path.exists()),
            "searched_candidates": self.expected_paths(),
            "canonical_repository": self.CANONICAL_REPOSITORY,
            "available": self.is_available(),
        }

    def missing_dependency_message(self) -> str:
        diagnostic = self.diagnostic_summary()
        searched = "\n".join(f"  - {path}" for path in diagnostic["searched_candidates"])
        configured = diagnostic["configured_root"] or "<unset>"
        expected = diagnostic["implementation_path"] or "<unresolved>"
        return (
            "Measured execution requires the linked GPT-1 wind tunnel repository.\n"
            f"Set {diagnostic['env_var']} to the repo root containing `gpt1_complete_implementation.py`.\n"
            f"Configured root: {configured}\n"
            f"Expected file: {expected}\n"
            f"Searched candidate roots:\n{searched}\n"
            f"Canonical repository: {diagnostic['canonical_repository']}"
        )

    @staticmethod
    def _load_module(path: Path):
        spec = importlib.util.spec_from_file_location("gpt1_complete_implementation", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load GPT-1 module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def is_available(self) -> bool:
        return self.module is not None

    def instantiate(self, **config: Any):
        if self.module is None:
            raise RuntimeError(self.missing_dependency_message())
        model_cls = getattr(self.module, "GPT1")
        return model_cls(**config)

    def dry_run_metrics(self, *, prompt: str, vocab_size: int = 64, **config: Any) -> Dict[str, float]:
        if self.module is None:
            length = max(len(prompt), 1)
            width = int(config.get("d_model", 64))
            heads = int(config.get("num_heads", 4))
            layers = int(config.get("num_layers", 2))
            return {
                "logit_std": float(np.tanh(length / 50.0)),
                "attention_density_proxy": float(min(1.0, heads / max(width / 16.0, 1.0))),
                "capacity_proxy": float(width * heads * layers),
            }
        model = self.instantiate(vocab_size=vocab_size, **config)
        token_ids = list(range(min(len(prompt), model.max_seq_len)))
        if not token_ids:
            token_ids = [0, 1]
        logits = model.forward(token_ids)
        logits = np.asarray(logits, dtype=float)
        attention_density = float(np.mean(np.abs(logits) > np.mean(np.abs(logits))))
        return {
            "logit_std": float(np.std(logits)),
            "attention_density_proxy": attention_density,
            "capacity_proxy": float(model.num_layers * model.num_heads * model.d_model),
        }

    def link_metadata(self) -> Dict[str, Any]:
        payload = repository_provenance(
            name="pageman/gpt1-from-Sutskever30",
            url=self.CANONICAL_REPOSITORY,
            root=self.root,
        )
        payload["available"] = self.is_available()
        return payload


class AgentOverlayAdapter:
    """Reads the agent repo and produces cartography-oriented narratives."""
    CANONICAL_REPOSITORY = "https://github.com/pageman/Sutskever-Agent"
    DEFAULT_CANDIDATES = [
        "~/Sutskever-Agent/sutskever-agent",
        "~/Downloads/Sutskever-Agent/sutskever-agent",
        "~/Downloads/sutskever-agent",
    ]

    def __init__(self, root: Path | str | None = None):
        self.root = _resolve_root(root, "SUTSKEVER_AGENT_ROOT", self.DEFAULT_CANDIDATES)
        self.agent_config = {}
        if self.root is not None:
            agent_yaml = self.root / "agent.yaml"
            if agent_yaml.exists():
                self.agent_config = yaml.safe_load(agent_yaml.read_text()) or {}

    def available_skills(self) -> List[str]:
        return list(self.agent_config.get("skills", []))

    def link_metadata(self) -> Dict[str, Any]:
        payload = repository_provenance(
            name="pageman/Sutskever-Agent",
            url=self.CANONICAL_REPOSITORY,
            root=self.root,
        )
        payload["skill_count"] = len(self.available_skills())
        return payload

    def narrate(self, artifact: Dict[str, Any]) -> str:
        trajectory = artifact["trajectory"]
        boundaries = trajectory.get("boundary_events", [])
        fits = trajectory.get("fitted_boundaries", [])
        substrate = trajectory.get("substrate", "unknown")
        experiment_id = trajectory.get("experiment_id", "unknown")
        if boundaries:
            first_boundary = boundaries[0]
            boundary_clause = (
                f"{first_boundary['metric']} crossed into {first_boundary['regime_after']} "
                f"at step {first_boundary['step']} with delta {first_boundary['delta']:.3f}."
            )
        else:
            boundary_clause = "No abrupt regime transition cleared the changepoint threshold."
        fit_clause = ""
        if fits:
            fit = fits[0]
            fit_clause = (
                f" Median threshold for {fit['metric']} sat at {fit['threshold_value']:.3f} "
                f"(step {fit['threshold_step']})."
            )
        return (
            f"Sutskever-Agent cartography summary for {experiment_id} on {substrate}: "
            f"{boundary_clause}{fit_clause}"
        )


# ====================================================================
# Layer 3 addition: Beyond-NumPy multi-backend substrate adapter
# ====================================================================


class BeyondNumpyAdapter:
    """Adapter for the sutskever-30-beyond-numpy multi-backend repository.

    Links to: https://github.com/pageman/sutskever-30-beyond-numpy
    """

    CANONICAL_REPOSITORY = "https://github.com/pageman/sutskever-30-beyond-numpy"
    DEFAULT_CANDIDATES = [
        "~/sutskever-30-beyond-numpy",
        "~/Downloads/sutskever-30-beyond-numpy",
    ]

    BACKEND_ORDER = ["numpy", "sympy", "tinygrad", "torch", "jax", "cubical-agda"]

    def __init__(self, root: Path | str | None = None):
        self.root = _resolve_root(root, "BEYOND_NUMPY_ROOT", self.DEFAULT_CANDIDATES)

    def is_available(self) -> bool:
        return self.root is not None and self.root.exists()

    def list_papers(self) -> List[Dict[str, str]]:
        if not self.is_available():
            return []
        papers_dir = self.root / "papers"
        if not papers_dir.exists():
            return []
        results = []
        for d in sorted(papers_dir.iterdir()):
            if d.is_dir() and d.name[0].isdigit():
                pid = d.name.split("_")[0]
                results.append({
                    "paper_id": pid,
                    "directory": d.name,
                    "path": str(d),
                    "backends": self._detect_backends(d),
                })
        return results

    def paper_backends(self, paper_id: str) -> Dict[str, bool]:
        """Check which backends are present for a given paper."""
        if not self.is_available():
            return {}
        papers = self.list_papers()
        for p in papers:
            if p["paper_id"] == paper_id:
                return p["backends"]
        return {}

    def has_numpy_checks(self, paper_id: str) -> bool:
        if not self.is_available():
            return False
        for p in self.list_papers():
            if p["paper_id"] == paper_id:
                path = Path(p["path"]) / "numpy_checks.py"
                return path.exists()
        return False

    def _detect_backends(self, paper_dir: Path) -> Dict[str, bool]:
        backends = {}
        for b in self.BACKEND_ORDER:
            if b == "numpy":
                backends[b] = (paper_dir / "numpy_checks.py").exists()
            elif b == "cubical-agda":
                backends[b] = (paper_dir / "cubical-agda").is_dir()
            else:
                backends[b] = (paper_dir / b).is_dir()
        return backends

    def link_metadata(self) -> Dict[str, Any]:
        payload = repository_provenance(
            name="pageman/sutskever-30-beyond-numpy",
            url=self.CANONICAL_REPOSITORY,
            root=self.root,
        )
        payload["paper_count"] = len(self.list_papers())
        return payload
