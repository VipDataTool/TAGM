"""TAGM analysis layer.

Analysis modules are distinct from measurement modules:

  - Measurement modules run per-prompt, consuming a RunResult and producing
    a MeasurementResult merged into the session for that prompt.

  - Analysis modules run post-session, consuming a whole session (many
    prompts' measurement results) and producing aggregate outputs.

Each analysis module returns a ModuleOutput (three compartments:
scalars/objects/per_prompt); the framework wraps that return value in a
four-compartment mailbox stored at session.record.analyses[name]. See
TAGM_analysis_layer_interface.md for the full contract.
"""
from tagm.analysis.base import AnalysisModule, ModuleOutput
from tagm.analysis.registry import (
    register_analysis,
    find_analysis,
    list_analyses,
)

# Concrete analyses register here as they are ported in.
from tagm.analysis import correction_field_topology  # noqa: F401

__all__ = [
    "AnalysisModule",
    "ModuleOutput",
    "register_analysis",
    "find_analysis",
    "list_analyses",
]
