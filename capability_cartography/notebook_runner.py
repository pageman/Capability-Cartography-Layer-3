"""Notebook wrapping and direct execution for the substrate repo."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List

from .adapters import NotebookSubstrateAdapter


class NotebookExecutionWrapper:
    """Extract and execute notebook code cells via a generated Python script."""

    def __init__(self, substrate_adapter: NotebookSubstrateAdapter):
        self.substrate_adapter = substrate_adapter

    def export_notebook_script(self, notebook_name: str, *, output_dir: str | Path) -> str:
        notebook_info = self.substrate_adapter.describe_notebook(notebook_name)
        notebook_path = Path(notebook_info["path"])
        notebook = json.loads(notebook_path.read_text())
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        figure_dir = output_dir / f"{notebook_name}_figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        script_lines: List[str] = [
            "# Auto-generated from notebook execution wrapper",
            "import os",
            "import warnings",
            "os.environ.setdefault('MPLBACKEND', 'Agg')",
            "warnings.filterwarnings('ignore', category=RuntimeWarning)",
            "import matplotlib",
            "matplotlib.use('Agg')",
            f"NOTEBOOK_OUTPUT_DIR = {str(output_dir)!r}",
            f"NOTEBOOK_FIGURE_DIR = {str(figure_dir)!r}",
            "NOTEBOOK_FIGURE_INDEX = 0",
        ]
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            cleaned = []
            for line in source.splitlines():
                stripped = line.lstrip()
                if stripped.startswith("%") or stripped.startswith("!"):
                    continue
                if "plt.show(" in stripped or stripped == "plt.show()":
                    cleaned.append(
                        "NOTEBOOK_FIGURE_INDEX += 1\n"
                        "plt.tight_layout()\n"
                        "plt.savefig(os.path.join(NOTEBOOK_FIGURE_DIR, f'figure_{NOTEBOOK_FIGURE_INDEX:02d}.png'))\n"
                        "plt.close()"
                    )
                    continue
                cleaned.append(line)
            if cleaned:
                script_lines.append("\n".join(cleaned))
                script_lines.append("")
        script_path = output_dir / f"{notebook_name}.py"
        script_path.write_text("\n".join(script_lines))
        return str(script_path)

    def execute_notebook(self, notebook_name: str, *, output_dir: str | Path, timeout_seconds: int = 60) -> Dict[str, object]:
        script_path = self.export_notebook_script(notebook_name, output_dir=output_dir)
        env = dict(os.environ)
        env.setdefault("MPLBACKEND", "Agg")
        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
        figure_dir = Path(output_dir) / f"{notebook_name}_figures"
        payload = {
            "notebook_name": notebook_name,
            "script_path": script_path,
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "figure_dir": str(figure_dir),
            "generated_figures": sorted(str(path) for path in figure_dir.glob("*.png")),
        }
        report_path = Path(output_dir) / f"{notebook_name}.execution.json"
        report_path.write_text(json.dumps(payload, indent=2))
        payload["report_path"] = str(report_path)
        return payload
