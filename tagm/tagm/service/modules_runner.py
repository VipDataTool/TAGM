"""TASM-compat module runner.

TASM's frontend treats every measurement and analysis as a "module" that
you configure, run asynchronously, poll for status, and fetch results
from. TAGM separates these concerns: measurements run as part of
`/api/analyze` (synchronous, against capture data); analyses run via
`/api/analysis/{name}` (synchronous, against session data). This module
provides a thin async wrapper that exposes both kinds through the
unified TASM-compat interface.

Module name conventions:
  - measurement modules: same names as TAGM measurements
    ("stress_score", "lateral_tension_profile", etc.)
  - analysis modules: same names as TAGM analyses
    ("comparative_analysis", "correction_heatmap", etc.)

When run_module() is called with a measurement name, it triggers an
analyze_batch over all session prompts using that measurement; when
called with an analysis name, it dispatches the analysis. In both
cases the results are stashed and retrievable via get_results().
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from tagm.analysis.registry import find_analysis, list_analyses
from tagm.measurement.registry import find_measurement, list_measurements
from tagm.measurement.parameters import resolve_parameters

logger = logging.getLogger("tagm")


@dataclass
class _ModuleState:
    name: str
    kind: str                   # "measurement" or "analysis"
    status: str = "idle"        # "idle" | "running" | "completed" | "error"
    progress: str = ""
    results: Optional[dict] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    log_path: Optional[str] = None
    last_params: dict = field(default_factory=dict)


class ModuleRunner:
    """Per-app singleton that owns module state and dispatches runs.

    All state lives in `_state`; `run()` mutates a state entry from
    a background thread. The frontend polls `get_status(name)` to
    observe progress.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state: dict[str, _ModuleState] = {}
        self._discover()

    def _discover(self) -> None:
        """Build the initial state map from the measurement and analysis
        registries. Called once at construction; safe to call again
        (idempotent — won't clobber existing state)."""
        for meta in list_measurements():
            name = meta["name"]
            if name not in self._state:
                self._state[name] = _ModuleState(name=name, kind="measurement")
        for meta in list_analyses():
            name = meta["name"]
            if name not in self._state:
                self._state[name] = _ModuleState(name=name, kind="analysis")

    # ── Listing / introspection ────────────────────────────────────

    def list_modules(self) -> list[dict]:
        """Return one dict per module with status + metadata for the UI."""
        self._discover()
        out = []
        for name, st in sorted(self._state.items()):
            base_meta = self._metadata_for(name, st.kind)
            base_meta["name"] = name
            base_meta["kind"] = st.kind
            base_meta["status"] = st.status
            base_meta["progress"] = st.progress
            base_meta["has_results"] = st.results is not None
            base_meta["has_log"] = st.log_path is not None
            if st.error:
                base_meta["error"] = st.error
            out.append(base_meta)
        return out

    def _metadata_for(self, name: str, kind: str) -> dict:
        """Return parameter declarations and description for a module."""
        try:
            if kind == "measurement":
                cls = find_measurement(name)
                inst = cls()
                return inst.metadata()
            else:
                cls = find_analysis(name)
                inst = cls()
                return inst.metadata()
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

    def run(self, name: str, session, orchestrator, params: dict,
            progress_fn=None) -> dict:
        """Start a module in a background thread.

        Returns immediately with {ok, started}. Caller polls get_status()
        until status is 'completed' or 'error'.

        For measurement modules: requires orchestrator with capture set;
        runs analyze on session prompts (or raises if no prompts).
        For analysis modules: dispatches the analysis against the session.
        """
        st = self._state.get(name)
        if st is None:
            return {"ok": False, "error": f"Module '{name}' not found."}

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
            args=(st, session, orchestrator, params or {}, progress_fn),
            daemon=True)
        thread.start()
        return {"ok": True, "started": True}

    def _run_in_thread(self, st: _ModuleState, session, orchestrator,
                        params: dict, progress_fn) -> None:
        """Background worker. Updates state on completion or failure."""
        def _prog(stage: str, msg: str) -> None:
            st.progress = msg
            if progress_fn:
                try:
                    progress_fn(stage, f"[{st.name}] {msg}")
                except Exception:
                    pass

        try:
            if st.kind == "analysis":
                _prog("analysis", f"running {st.name}")
                cls = find_analysis(st.name)
                module = cls()
                resolved = resolve_parameters(module.parameters, params)
                session_dict = session.to_dict()
                errors = module.check_dependencies(session_dict)
                if errors:
                    st.results = {"ok": False, "errors": errors}
                else:
                    # Inject pipeline for modules that need live model access
                    if hasattr(module, 'set_pipeline') and orchestrator is not None:
                        module.set_pipeline(orchestrator.pipeline)
                    if hasattr(module, 'set_probe_store') and orchestrator is not None:
                        module.set_probe_store(orchestrator.probe_store)
                    if hasattr(module, 'set_progress'):
                        module.set_progress(lambda msg: _prog("running", msg))
                    result = module.run(session_dict, resolved)
                    rdict = result.to_dict() if hasattr(result, "to_dict") else result
                    session.add_analysis(st.name, rdict)
                    st.results = rdict
                _prog("done", "complete")
            else:
                # Measurement module — run over all session prompts via
                # the orchestrator. Adds this measurement to the active
                # selection for the duration of the run, then restores.
                if orchestrator is None or orchestrator.capture_config is None:
                    raise RuntimeError(
                        "No CaptureConfig set. Configure capture before "
                        "running measurement modules.")
                if not session.record.prompts:
                    raise RuntimeError(
                        "No session prompts to re-measure. Analyze prompts "
                        "first, or use this module via /api/analyze.")

                _prog("configure", f"reconfiguring with {st.name}")
                # Snapshot existing selection so we can restore it
                prior_selection = list(orchestrator._selected)
                report = orchestrator.configure_measurements(
                    [(st.name, params)])
                if report.get("violations") or report.get("errors"):
                    raise RuntimeError(
                        f"Measurement '{st.name}' cannot run against "
                        f"current capture: {report.get('violations') or report.get('errors')}")

                try:
                    _prog("running", f"re-measuring {len(session.record.prompts)} prompts")
                    rerun_prompts = [
                        {"prompt": p.prompt, "category": p.category}
                        for p in list(session.record.prompts)
                    ]
                    # Replace prompts: rerun strategy mirrors /api/session/rerun
                    session.record.prompts = []
                    orchestrator.analyze_batch(
                        rerun_prompts, session=session,
                        progress=lambda s, m: _prog(s, m))
                finally:
                    # Restore prior selection on the orchestrator
                    orchestrator._selected = prior_selection
                # Aggregate this measurement's outputs across all prompts
                # for the module results view
                per_prompt = []
                for p in session.record.prompts:
                    m = p.measurements.get(st.name)
                    if m:
                        per_prompt.append({
                            "prompt": p.prompt,
                            "category": p.category,
                            "result": m,
                        })
                st.results = {
                    "ok": True,
                    "name": st.name,
                    "n_prompts": len(per_prompt),
                    "per_prompt": per_prompt,
                }
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


# Module-level singleton, instantiated at first import. The runner
# discovers measurements and analyses from their registries, so as long
# as those have been imported by the time this is created (which is
# guaranteed by `app.py`'s import order), every registered module is
# visible here too.
runner = ModuleRunner()
