"""Module runner for the Modules tab.

The Modules tab lists analyses — tertiary consumers that read the session
and produce aggregate outputs. Measurements are secondary producers that
run inside /api/analyze; they are not modules and are not exposed here.

An analysis module is a single file under `tagm/analysis/` that:
  - registers itself via @register_analysis
  - declares `parameters` (list of ModuleParameter)
  - implements `run(session_dict, params, probes=None) -> dict`

`run()` returns a plain dict whose keys are whatever the frontend reads
off the module result. No wrapper classes, no schema subkeys.

State is per-app singleton. `run()` starts a background thread; the UI
polls `get_status()` and fetches the final dict via `get_results()`.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from tagm.analysis.registry import find_analysis, list_analyses
from tagm.measurement.parameters import resolve_parameters

logger = logging.getLogger("tagm")


@dataclass
class _ModuleState:
    name: str
    status: str = "idle"         # "idle" | "running" | "completed" | "error"
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
        self._discover()

    def _discover(self) -> None:
        for meta in list_analyses():
            name = meta["name"]
            if name not in self._state:
                self._state[name] = _ModuleState(name=name)

    # ── Listing / introspection ────────────────────────────────────

    def list_modules(self) -> list[dict]:
        """One entry per registered analysis, with status + metadata."""
        self._discover()
        out = []
        for name, st in sorted(self._state.items()):
            meta = self._metadata_for(name)
            meta["name"] = name
            meta["kind"] = "analysis"
            meta["status"] = st.status
            meta["progress"] = st.progress
            meta["has_results"] = st.results is not None
            meta["has_log"] = st.log_path is not None
            if st.error:
                meta["error"] = st.error
            out.append(meta)
        return out

    def _metadata_for(self, name: str) -> dict:
        try:
            cls = find_analysis(name)
            return cls().metadata()
        except KeyError:
            return {}

    def get_status(self, name: str) -> dict:
        st = self._state.get(name)
        if st is None:
            return {"ok": False, "error": f"Module '{name}' not found."}
        return {
            "ok": True,
            "status": st.status,
            "progress": st.progress,
            "has_results": st.results is not None,
            "has_log": st.log_path is not None,
            "error": st.error,
            "started_at": st.started_at,
            "completed_at": st.completed_at,
        }

    def get_results(self, name: str) -> Optional[dict]:
        st = self._state.get(name)
        return st.results if st else None

    def get_log_path(self, name: str) -> Optional[str]:
        st = self._state.get(name)
        return st.log_path if st else None

    def reset(self, name: str) -> dict:
        st = self._state.get(name)
        if st is None:
            return {"ok": False, "error": f"Module '{name}' not found."}
        if st.status == "running":
            return {"ok": False,
                    "error": f"Module '{name}' is currently running."}
        st.status = "idle"
        st.progress = ""
        st.results = None
        st.error = None
        st.started_at = None
        st.completed_at = None
        st.log_path = None
        logger.info(f"[modules] {name} reset to idle")
        return {"ok": True}

    # ── Run dispatch ───────────────────────────────────────────────

    def run(self, name: str, session, params: dict,
            progress_fn=None) -> dict:
        """Start an analysis module in a background thread.

        The module receives the session dict and its resolved params; it
        must not mutate session data. Its return value (a plain dict) is
        stashed verbatim and is what the UI reads via
        /api/modules/.../results.
        """
        st = self._state.get(name)
        if st is None:
            return {"ok": False, "error": f"Module '{name}' not found."}

        try:
            find_analysis(name)
        except KeyError:
            return {"ok": False,
                    "error": f"No analysis registered as '{name}'"}

        with self._lock:
            if st.status == "running":
                return {"ok": False,
                        "error": f"Module '{name}' is already running."}
            st.status = "running"
            st.progress = "starting"
            st.started_at = time.time()
            st.completed_at = None
            st.error = None
            st.results = None
            st.last_params = dict(params or {})

        thread = threading.Thread(
            target=self._run_in_thread,
            args=(st, session, params or {}, progress_fn),
            daemon=True)
        thread.start()
        return {"ok": True, "started": True}

    def _run_in_thread(self, st: _ModuleState, session, params: dict,
                        progress_fn) -> None:
        def _prog(stage: str, msg: str) -> None:
            st.progress = msg
            if progress_fn:
                try:
                    progress_fn(stage, f"[{st.name}] {msg}")
                except Exception:
                    pass

        try:
            _prog("analysis", f"running {st.name}")
            cls = find_analysis(st.name)
            module = cls()
            resolved = resolve_parameters(module.parameters, params)
            session_dict = session.to_dict()

            errors = module.check_dependencies(session_dict)
            if errors:
                st.results = {"ok": False, "errors": errors}
            else:
                out = module.run(session_dict, resolved)
                # Modules should return a plain dict. Legacy modules
                # may return an AnalysisResult-style object — serialize
                # it via .to_dict() so the runner doesn't choke.
                if not isinstance(out, dict) and hasattr(out, "to_dict"):
                    out = out.to_dict()
                if not isinstance(out, dict):
                    out = {"_raw": str(out)}
                session.add_analysis(st.name, out)
                st.results = out

            _prog("done", "complete")
            st.status = "completed"
            st.completed_at = time.time()
            logger.info(f"[modules] {st.name} completed in "
                        f"{st.completed_at - st.started_at:.1f}s")
        except Exception as e:
            logger.exception(f"[modules] {st.name} failed")
            st.status = "error"
            st.error = str(e)
            st.completed_at = time.time()


# Module-level singleton, instantiated at first import.
runner = ModuleRunner()
