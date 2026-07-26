"""Shared helpers for the API routers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

# src/api/_util.py -> src/api -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_under(root: Union[str, Path],
                  filename: str) -> Optional[Path]:
    """Resolve ``filename`` against ``root`` and confirm it stays inside.

    Endpoints such as ``/api/probe_diagnostic`` and
    ``/api/modules/probe_generator/embed_active`` legitimately accept
    relative *sub-paths* ("probe_cache/foo.csv"), so ``safe_filename``
    (which flattens separators) is the wrong tool here. Instead the
    joined path is fully resolved — collapsing ``..`` segments and
    symlinks — and rejected unless it is still under ``root``.

    Failure mode this closes: a request-supplied
    ``../../../../etc/passwd`` (or an absolute path, which ``/`` joining
    silently honours) read anything the server process could read.

    Returns the resolved absolute Path, or None if the name escapes
    ``root`` / is empty.
    """
    if not filename:
        return None
    root_resolved = Path(root).resolve()
    try:
        candidate = (root_resolved / filename).resolve()
    except (OSError, ValueError, RuntimeError):
        return None
    if candidate == root_resolved or root_resolved in candidate.parents:
        return candidate
    return None
