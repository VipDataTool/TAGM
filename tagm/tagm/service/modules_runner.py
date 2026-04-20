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


# ── Mailbox receipt builders (framework-owned, no module input) ────
# These assemble the three framework-owned compartments of an analysis
# mailbox. Kept at module scope so they can be called from both the
# success path and the error path.

def _module_receipt(cls) -> dict:
    """The `module` compartment — identity lifted from class attrs.

    Read-only from the module's perspective (the module can't
    override these; we read straight from the registered class)."""
    return {
        "name": cls.name,
        "display_name": cls.display_name,
        "description": cls.description,
        "version": cls.version,
    }


def _sources_receipt(cls, resources: dict, session: dict) -> dict:
    """The `sources` compartment — a receipt, not a copy.

    Records identifiers only: measurement names (not measurement
    data), probe set id (not probes), prompt ids (not prompts),
    resource signatures (not the resources themselves). Zero bytes
    of primary data are duplicated into the mailbox.

    To retrieve what the module read, look up measurement names in
    session.prompts[i].measurements[name] using the prompt_ids in
    this receipt.
    """
    prompts = session.get("prompts") or []

    probes = resources.get("probes")
    probe_set_id = (
        getattr(probes, "set_id", None) if probes is not None else None)

    delta = resources.get("delta_store")
    delta_sig = None
    if delta is not None:
        # Prefer an explicit signature method if the delta store has one;
        # otherwise fall back to the class name for provenance.
        sig_fn = getattr(delta, "signature", None)
        if callable(sig_fn):
            try:
                delta_sig = str(sig_fn())
            except Exception:
                delta_sig = type(delta).__name__
        else:
            delta_sig = type(delta).__name__

    pipeline = resources.get("pipeline")
    pipeline_sig = None
    if pipeline is not None:
        mp = (session.get("model_pair") or {})
        instruct = mp.get("instruct") or ""
        base = mp.get("base") or ""
        pipeline_sig = f"{instruct}+{base}" if (instruct or base) else None

    return {
        "measurements": list(cls.depends_on_measurements),
        "probe_set_id": probe_set_id,
        "delta_store_sig": delta_sig,
        "pipeline_sig": pipeline_sig,
        "n_prompts": len(prompts),
        "prompt_ids": [p.get("prompt_id") or f"p{i:04d}"
                       for i, p in enumerate(prompts)],
    }


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

    def _dispatch_analysis(self, st: _ModuleState, session,
                           user_params: dict, prog) -> None:
        """Dispatch one analysis and build its mailbox entry.

        Interface contract: TAGM_analysis_layer_interface.md §8.

        Mailbox layout: TAGM_analysis_layer_interface.md §2.
        Four compartments — three framework-owned (module, run,
        sources), one module-owned (output). We never let the module
        write outside its compartment: its return value populates
        output verbatim; we fill the other three from ground truth
        (the registered class, dispatch state, resolved resources).
        """
        prog("analysis", f"running {st.name}")
        cls = find_analysis(st.name)
        module = cls()

        # 1. Resolve parameters (may raise ValueError for validation errors).
        try:
            resolved = resolve_parameters(module.parameters, user_params)
        except ValueError as exc:
            return self._write_error_mailbox(
                st, session, cls, dict(user_params or {}), str(exc))

        # 2. Snapshot the session once so everything downstream sees
        #    the same view.
        session_dict = session.to_dict()

        # 3. Pre-dispatch gates. Any failure here means the module
        #    never runs; we write an error mailbox and bail.
        if len(session_dict.get("prompts") or []) < cls.min_prompts:
            return self._write_error_mailbox(
                st, session, cls, resolved,
                f"analysis '{st.name}' needs >={cls.min_prompts} prompts, "
                f"session has {len(session_dict.get('prompts') or [])}")

        dep_errors = module.check_dependencies(session_dict)
        if dep_errors:
            return self._write_error_mailbox(
                st, session, cls, resolved, "; ".join(dep_errors))

        # 4. Resolve resource requirements into kwargs.
        try:
            resources = self._resolve_resources(cls)
        except RuntimeError as exc:
            return self._write_error_mailbox(
                st, session, cls, resolved, str(exc))

        # 5. Dispatch. The module gets exactly what the spec promises.
        warnings: list[str] = []

        def _module_progress(msg: str, *, level: str = "info") -> None:
            # Modules call `progress("...")` — single string arg.
            # We also accept a `level` kwarg internally, but don't
            # require modules to use it. Warnings bubble into the
            # mailbox's run.warnings list.
            if level == "warning":
                warnings.append(msg)
            st.progress = msg

        started_at = st.started_at or time.time()
        try:
            output = module.run(
                session_dict, resolved,
                progress=_module_progress, **resources)
        except Exception as exc:
            logger.exception(f"[modules] analysis {st.name} raised")
            return self._write_error_mailbox(
                st, session, cls, resolved, str(exc),
                started_at=started_at)

        # 6. Validate the return value shape defensively. If a module
        #    returns None or the wrong type, treat it as an error
        #    rather than storing garbage.
        from tagm.analysis.base import ModuleOutput
        if not isinstance(output, ModuleOutput):
            return self._write_error_mailbox(
                st, session, cls, resolved,
                f"analysis '{st.name}' returned "
                f"{type(output).__name__}, expected ModuleOutput",
                started_at=started_at)

        # 7. Build the mailbox and store.
        completed_at = time.time()
        mailbox = {
            "module": _module_receipt(cls),
            "run": {
                "status": "warnings" if warnings else "completed",
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_s": completed_at - started_at,
                "params": resolved,
                "warnings": warnings,
                "error": None,
            },
            "sources": _sources_receipt(cls, resources, session_dict),
            "output": output.to_dict(),
        }
        session.add_analysis(st.name, mailbox)
        st.results = mailbox
        prog("done", "complete")

    def _write_error_mailbox(self, st: _ModuleState, session,
                             cls, resolved_params: dict, error: str,
                             *, started_at: Optional[float] = None) -> None:
        """Write an error-status mailbox entry.

        Called on any pre-dispatch gate failure or raised exception
        during the module's run(). Populates module/run/sources with
        whatever ground truth we have; output is an empty shell.
        """
        started_at = started_at or st.started_at or time.time()
        completed_at = time.time()
        session_dict = session.to_dict()
        mailbox = {
            "module": _module_receipt(cls),
            "run": {
                "status": "error",
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_s": completed_at - started_at,
                "params": resolved_params,
                "warnings": [],
                "error": error,
            },
            "sources": _sources_receipt(cls, resources={}, session=session_dict),
            "output": {"scalars": {}, "objects": {}, "per_prompt": {}},
        }
        session.add_analysis(st.name, mailbox)
        st.results = mailbox
        st.error = error
        # Surface the error through the state record too so get_status
        # callers see it without having to read the mailbox.
        logger.warning(f"[modules] {st.name} error: {error}")

    def _resolve_resources(self, cls) -> dict:
        """Resolve declared resource requirements into kwargs.

        Raises RuntimeError if a required resource isn't available.
        Imports the app module lazily because modules_runner is
        imported during app startup.
        """
        out: dict = {}
        if not (cls.requires_probe_set or cls.requires_delta_store
                or cls.requires_pipeline):
            return out

        try:
            from tagm import app as _app_mod  # type: ignore
        except Exception as e:
            raise RuntimeError(
                f"cannot resolve resources for '{cls.name}': "
                f"app module not importable ({e})") from e

        state = getattr(_app_mod, "state", None)
        if cls.requires_probe_set:
            active = getattr(_app_mod, "_active_probe_template", None)
            probe_store = getattr(state, "probe_store", None) if state else None
            if not active or probe_store is None:
                raise RuntimeError(
                    f"analysis '{cls.name}' requires an active probe set; "
                    f"apply a probe template via the Configuration tab first")
            set_id = active.get("set_id")
            probe_set = probe_store.get(set_id) if set_id else None
            if probe_set is None:
                raise RuntimeError(
                    f"analysis '{cls.name}' requires a probe set, but the "
                    f"active template's set_id '{set_id}' is not in the "
                    f"probe store")
            out["probes"] = probe_set

        if cls.requires_delta_store:
            pipeline = getattr(state, "pipeline", None) if state else None
            delta_store = (
                getattr(pipeline, "delta_store", None) if pipeline else None)
            if delta_store is None:
                raise RuntimeError(
                    f"analysis '{cls.name}' requires a delta store; "
                    f"load a model pair first")
            out["delta_store"] = delta_store

        if cls.requires_pipeline:
            pipeline = getattr(state, "pipeline", None) if state else None
            if pipeline is None or not getattr(pipeline, "loaded", False):
                raise RuntimeError(
                    f"analysis '{cls.name}' requires a loaded pipeline; "
                    f"load a model pair first")
            out["pipeline"] = pipeline

        return out

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
                self._dispatch_analysis(st, session, params, _prog)
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

            st.completed_at = time.time()
            # If _dispatch_analysis wrote an error mailbox, it set
            # st.error; honor that and surface "error" to the UI.
            # Otherwise we're in the success path -> "completed".
            if st.error:
                st.status = "error"
            else:
                st.status = "completed"
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
