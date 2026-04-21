"""Module runner: async dispatch for analysis modules.

Measurements are computed by the engine during /api/analyze — the runner
only handles post-session analysis modules. The list_modules() method
returns both built-in measurements (metadata only) and registered
analyses (runnable).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from tagm.analysis.registry import find_analysis, list_analyses
from tagm.measurement.registry import list_measurements
from tagm.measurement.parameters import resolve_parameters

logger = logging.getLogger("tagm")


@dataclass
class _ModuleState:
    name: str
    kind: str
    status: str = "idle"
    progress: str = ""
    results: Optional[dict] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    log_path: Optional[str] = None
    last_params: dict = field(default_factory=dict)


class ModuleRunner:
    def __init__(self):
        self._lock = threading.Lock()
        self._state: dict[str, _ModuleState] = {}

    def list_modules(self) -> list[dict]:
        modules = []
        for m in list_measurements():
            name = m["name"] if isinstance(m, dict) else m.get("name", "")
            display = m.get("display_name", name) if isinstance(m, dict) else name
            desc = m.get("description", "") if isinstance(m, dict) else ""
            ms = self._state.get(name)
            modules.append({
                "name": name, "display_name": display,
                "description": desc, "kind": "measurement",
                "status": ms.status if ms else "idle",
                "parameters": [],
            })
        for a in list_analyses():
            name = a["name"] if isinstance(a, dict) else getattr(a, "name", "")
            display = a.get("display_name", name) if isinstance(a, dict) else getattr(a, "display_name", name)
            desc = a.get("description", "") if isinstance(a, dict) else getattr(a, "description", "")
            params = a.get("parameters", []) if isinstance(a, dict) else getattr(a, "parameters", [])
            ms = self._state.get(name)
            modules.append({
                "name": name, "display_name": display,
                "description": desc, "kind": "analysis",
                "status": ms.status if ms else "idle",
                "parameters": [p.to_dict() if hasattr(p, "to_dict") else p for p in params],
            })
        return modules

    def run(self, name: str, session, orchestrator=None,
            params: dict = None, progress_fn=None) -> dict:
        """Run an analysis module asynchronously."""
        try:
            analysis_cls = find_analysis(name)
        except KeyError:
            return {"ok": False, "error": f"Unknown module: {name}"}

        with self._lock:
            ms = self._state.get(name)
            if ms and ms.status == "running":
                return {"ok": False, "error": f"{name} already running"}
            ms = _ModuleState(name=name, kind="analysis", status="running",
                              started_at=time.time(), last_params=params or {})
            self._state[name] = ms

        def _worker():
            try:
                module = analysis_cls()
                resolved = resolve_parameters(module.parameters, params or {})
                # Build session dict from flat results
                session_dict = {
                    "prompts": [{"prompt": r.get("prompt", ""),
                                 "category": r.get("category", ""),
                                 "measurements": r}
                                for r in session.results],
                    "model_pair": {"instruct": session.model_name},
                }
                result = module.run(session_dict, resolved)
                ms.results = result.to_dict() if hasattr(result, "to_dict") else result
                ms.status = "completed"
                ms.completed_at = time.time()
                if progress_fn:
                    progress_fn("ready", f"Module {name} complete")
            except Exception as e:
                logger.exception(f"Module {name} failed")
                ms.error = str(e)
                ms.status = "error"
                ms.completed_at = time.time()
                if progress_fn:
                    progress_fn("error", f"Module {name} failed: {e}")

        threading.Thread(target=_worker, daemon=True).start()
        return {"ok": True, "status": "running"}

    def get_status(self, name: str) -> dict:
        ms = self._state.get(name)
        if ms is None:
            return {"name": name, "status": "idle"}
        return {"name": name, "status": ms.status, "progress": ms.progress,
                "error": ms.error, "started_at": ms.started_at,
                "completed_at": ms.completed_at}

    def get_results(self, name: str) -> Optional[dict]:
        ms = self._state.get(name)
        return ms.results if ms else None

    def get_log_path(self, name: str) -> Optional[str]:
        ms = self._state.get(name)
        return ms.log_path if ms else None

    def reset(self, name: str) -> dict:
        with self._lock:
            if name in self._state:
                self._state[name] = _ModuleState(name=name, kind="analysis")
        return {"ok": True}
