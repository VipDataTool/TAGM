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
                    # Build a context dict of runtime-only resources that
                    # aren't captured in the session snapshot. Analyses that
                    # need them (correction_heatmap needs probe embeddings
                    # from the active set) pull from here; analyses that
                    # don't ignore the kwarg.
                    ctx: dict = {}
                    try:
                        # Imported lazily so tagm.service.modules_runner
                        # doesn't gain a hard dependency on the app module.
                        from tagm import app as _app_mod  # type: ignore
                        ctx["probe_store"] = getattr(
                            _app_mod.state, "probe_store", None)
                        ctx["pipeline"] = getattr(
                            _app_mod.state, "pipeline", None)
                        ctx["active_probe_template"] = getattr(
                            _app_mod, "_active_probe_template", None)
                    except Exception:
                        # Unit tests and standalone invocations can run
                        # without an app module. Analyses that need
                        # context resources will no-op gracefully.
                        pass
                    aresult = module.run(session_dict, resolved, context=ctx)
                    rdict = aresult.to_dict()
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
                    # Snapshot each prompt's existing measurements dict BEFORE
                    # the rerun. analyze_batch replaces session.record.prompts
                    # entirely with freshly-computed PromptRecords, each of
                    # which contains only the single selected measurement.
                    # Without this snapshot, re-running one measurement from
                    # the Modules tab destroys every other measurement on
                    # every prompt in the session — the user would lose
                    # stress_score, LTP, SFD, etc. any time they tweaked a
                    # parameter on one module and clicked Run.
                    prior_by_key = []  # list of (prompt_text, category, measurements_dict)
                    for p in session.record.prompts:
                        prior_by_key.append((
                            p.prompt, p.category, dict(p.measurements),
                        ))
                    rerun_prompts = [
                        {"prompt": p_text, "category": p_cat}
                        for (p_text, p_cat, _) in prior_by_key
                    ]
                    session.record.prompts = []
                    orchestrator.analyze_batch(
                        rerun_prompts, session=session,
                        progress=lambda s, m: _prog(s, m))

                    # Merge: for each re-analyzed prompt, overlay the new
                    # measurement on top of the prior measurements (which
                    # includes every other module's data). Matching is
                    # positional since analyze_batch preserves input order.
                    for new_prec, (_, _, prior_meas) in zip(
                            session.record.prompts, prior_by_key):
                        # New prec now contains only {st.name: ...}. Merge it
                        # onto the prior dict so we end up with all prior
                        # measurements, with st.name updated/added.
                        merged = dict(prior_meas)
                        for k, v in (new_prec.measurements or {}).items():
                            merged[k] = v
                        new_prec.measurements = merged
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
