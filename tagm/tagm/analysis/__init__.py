"""TAGM analysis layer.

Analysis modules are distinct from measurement modules:

  - Measurement modules run per-prompt, consuming a RunResult and producing
    a MeasurementResult merged into the session for that prompt.

  - Analysis modules run post-session, consuming a whole session (many
    prompts' measurement results) and producing aggregate outputs — cluster
    assignments, comparative plots, MI-readiness diagnostics, topology
    summaries, etc.

The base class here defines the contract; concrete analyses live in the
sibling modules.
"""
from tagm.analysis.base import AnalysisModule, AnalysisResult
from tagm.analysis.registry import (
    register_analysis,
    find_analysis,
    list_analyses,
)

# Auto-register all analysis modules on import
from tagm.analysis import comparative_analysis       # noqa: F401
from tagm.analysis import mi_readiness               # noqa: F401
from tagm.analysis import mi_instrumentation         # noqa: F401
from tagm.analysis import token_variance             # noqa: F401
from tagm.analysis import correction_heatmap         # noqa: F401
from tagm.analysis import correction_manifold        # noqa: F401
from tagm.analysis import correction_backscatter     # noqa: F401
from tagm.analysis import domain_surface             # noqa: F401
from tagm.analysis import correction_field_topology  # noqa: F401

__all__ = [
    "AnalysisModule",
    "AnalysisResult",
    "register_analysis",
    "find_analysis",
    "list_analyses",
]
