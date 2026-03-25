"""Stronger Sutskever-Agent workflow integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import yaml

from .adapters import AgentOverlayAdapter


class SutskeverAgentWorkflowBridge:
    """Generate agent-facing workflow payloads from cartography outputs."""

    def __init__(self, agent_adapter: AgentOverlayAdapter):
        self.agent_adapter = agent_adapter

    def build_agent_brief(
        self,
        *,
        measured_summary: Dict[str, object],
        failure_atlas_summary: Dict[str, object],
        visualization_paths: List[str],
    ) -> Dict[str, object]:
        return {
            "linked_agent": self.agent_adapter.link_metadata(),
            "measured_summary": measured_summary,
            "failure_atlas_summary": failure_atlas_summary,
            "visualizations": visualization_paths,
        }

    def export_workflow_bundle(
        self,
        *,
        output_dir: str | Path,
        brief: Dict[str, object],
    ) -> Dict[str, str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        brief_path = output_dir / "agent_brief.json"
        brief_path.write_text(json.dumps(brief, indent=2))
        workflow = {
            "name": "capability-cartography-layer-2-study",
            "description": "Read measured cartography outputs, failure atlas summaries, and plots; then narrate boundary shifts and failure risks.",
            "steps": [
                {"use_skill": "capability-cartography"},
                {"consume": str(brief_path)},
                {"summarize": "predictive_laws"},
                {"summarize": "failure_atlas"},
                {"summarize": "phase_regions"},
            ],
        }
        workflow_path = output_dir / "agent_workflow.yaml"
        workflow_path.write_text(yaml.safe_dump(workflow, sort_keys=False))
        return {"brief_path": str(brief_path), "workflow_path": str(workflow_path)}
