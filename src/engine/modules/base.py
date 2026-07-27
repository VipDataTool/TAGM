"""
Base module class and runner for TASM extensible analysis framework.

Design principles:
  - Modules operate on collected session data, not on inference.
  - Each module runs in its own thread. If it crashes, nothing else dies.
  - Modules declare their parameters as structured metadata so the UI
    can render controls without knowing anything about the module.
  - Results are JSON-serializable dicts stored in the session directory.
"""

import os
import json
import time
import logging
import threading
import traceback
import importlib
import inspect
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Callable

logger = logging.getLogger("tasm")


def classify_category(category: Optional[str],
                      harm_override: Optional[set] = None,
                      safe_override: Optional[set] = None) -> Optional[str]:
    """Canonical harm/safe classification, with optional per-module overrides.

    WAS WRONG: six modules each carried their own harm/safe vocabulary and the
    copies disagreed (``unknown`` was harmful in two and dropped in four;
    ``mild`` was safe in three and dropped in one), so two modules reading the
    same session reported contradictory ``n_harm``.  The canonical taxonomy now
    lives in ``src.engine.metrics``; this wrapper exists only so modules that
    expose user-editable category lists can layer an explicit override on top
    of it.  Returns "harm", "safe", or None (= exclude from the contrast).
    """
    from src.engine.metrics import category_class

    if category is None:
        return None
    c = category.strip().lower()
    if harm_override and c in {x.strip().lower() for x in harm_override}:
        return "harm"
    if safe_override and c in {x.strip().lower() for x in safe_override}:
        return "safe"
    return category_class(c)


@dataclass
class ModuleParameter:
    """A parameter that the user can configure before running a module."""
    name: str
    display_name: str
    description: str
    type: str  # "int", "float", "bool", "select"
    default: Any
    options: list = field(default_factory=list)   # for "select" type
    min_val: Optional[float] = None               # for numeric types
    max_val: Optional[float] = None
    group: str = ""                               # UI section label
    advanced: bool = False                        # collapsed by default

    def to_dict(self):
        d = asdict(self)
        # Strip None optional fields for clean JSON
        return {k: v for k, v in d.items() if v is not None or k in ("default",)}


class TASMModule:
    """Base class for all TASM analysis modules.

    Subclass this and implement run() to create a module.
    Place the file in engine/modules/ and it will be auto-discovered.
    """

    # ── Identity (override in subclass) ──
    name: str = "unnamed"
    display_name: str = "Unnamed Module"
    description: str = ""
    version: str = "0.1.0"

    # ── Requirements ──
    min_results: int = 1          # minimum session results needed
    requires_sfd: bool = False    # needs SFD data in results
    requires_ltp: bool = False    # needs LTP data in results
    requires_rd: bool = False     # needs RD data in results

    # ── Parameters (override in subclass) ──
    parameters: list = []  # list of ModuleParameter

    def validate(self, session_results: list, params: dict) -> tuple:
        """Check whether the module can run.

        Args:
            session_results: list of result dicts from the session.
            params: user-supplied parameter values.

        Returns:
            (ok: bool, message: str)
        """
        if len(session_results) < self.min_results:
            return False, f"Need at least {self.min_results} results, have {len(session_results)}."

        if self.requires_sfd:
            has_sfd = sum(1 for r in session_results if r.get("sfd"))
            if has_sfd == 0:
                return False, "This module requires SFD data. Re-run with SFD enabled."

        if self.requires_ltp:
            has_ltp = sum(1 for r in session_results if r.get("ltp"))
            if has_ltp == 0:
                return False, "This module requires LTP data. Re-run with LTP enabled."

        if self.requires_rd:
            has_rd = sum(1 for r in session_results if r.get("rank_displacement"))
            if has_rd == 0:
                return False, "This module requires Rank Displacement data."

        return True, "OK"

    def run(self, session_results: list, params: dict,
            progress: Callable[[str], None] = None) -> dict:
        """Execute the module analysis.

        Args:
            session_results: list of result dicts from the session.
            params: user-supplied parameter values.
            progress: callback for status updates, e.g. progress("Processing token 50/200").
                **Calling this is also the cancellation checkpoint**: it raises
                ModuleCancelled if the user has requested a stop. Modules that
                already report progress inside their loops therefore become
                cancellable with no further changes.

        A module may optionally declare a ``should_cancel`` keyword parameter::

            def run(self, session_results, params, progress=None, should_cancel=None):

        The runner detects it by signature and passes a zero-argument predicate.
        Use it in hot loops that do not emit progress often enough (checking a
        threading.Event is far cheaper than a progress update).

        Cancellation unwinds by exception, so `finally` blocks still run —
        weights get restored and hooks removed on the way out.

        Returns:
            dict with module-specific results. Must be JSON-serializable.
        """
        raise NotImplementedError("Subclass must implement run()")

    def get_metadata(self) -> dict:
        """Return module metadata for the UI."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "version": self.version,
            "min_results": self.min_results,
            "requires_sfd": self.requires_sfd,
            "requires_ltp": self.requires_ltp,
            "requires_rd": self.requires_rd,
            "parameters": [p.to_dict() if isinstance(p, ModuleParameter)
                           else p for p in self.parameters],
        }


class ModuleCancelled(Exception):
    """Raised inside a module's thread when the user requests cancellation.

    Modules do not need to catch this — the runner treats it as the
    "cancelled" terminal state rather than an error.  Any `finally` blocks in
    the module still run, which is what makes cancelling an abliteration or a
    hooked forward pass safe: weights are restored and hooks removed on the
    way out.
    """


class _ModuleState:
    """Runtime state for a single module."""
    def __init__(self):
        # idle | running | completed | partial | cancelled | error
        #   partial   = returned a usable result set AND an "error" key, i.e.
        #               an optional stage failed after the main analysis
        #               succeeded.  The UI renders the results but flags it.
        #   cancelled = the user stopped the run; any partial results are
        #               discarded because they are not a complete analysis.
        self.status = "idle"
        self.progress = ""
        self.results = None
        self.error = None
        self.started_at = None
        self.completed_at = None
        self.thread = None
        self.log_path = None
        # Set by cancel_module(); polled cooperatively by the running module.
        # There is no safe way to kill a Python thread mid-torch-op, so
        # cancellation is cooperative by construction.
        self.cancel_event = threading.Event()


class ModuleRunner:
    """Discovers, manages, and runs TASM modules.

    Modules are auto-discovered from the engine/modules/ directory.
    Each module runs in its own thread with crash isolation.
    """

    def __init__(self, project_root=None, event_hook=None):
        self._modules: dict[str, TASMModule] = {}
        self._state: dict[str, _ModuleState] = {}
        self._lock = threading.Lock()
        self._project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent.parent
        self._event_hook: Optional[Callable] = event_hook
        self._discover_modules()

    def _discover_modules(self):
        """Scan engine/modules/ for TASMModule subclasses."""
        modules_dir = Path(__file__).parent
        for filepath in modules_dir.glob("*.py"):
            if filepath.name.startswith("_") or filepath.name == "base.py":
                continue
            module_name = filepath.stem
            try:
                # Import relative to tagm.engine.modules
                mod = importlib.import_module(f".{module_name}", package="src.engine.modules")
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if (isinstance(attr, type)
                            and issubclass(attr, TASMModule)
                            and attr is not TASMModule):
                        instance = attr()
                        # Let modules discover project-root-relative resources
                        if hasattr(instance, 'set_project_root'):
                            try:
                                instance.set_project_root(str(self._project_root))
                            except Exception as e:
                                logger.warning(f"[MODULES] {instance.name} set_project_root failed: {e}")
                        self._modules[instance.name] = instance
                        self._state[instance.name] = _ModuleState()
                        logger.info(f"[MODULES] Discovered: {instance.display_name} "
                                    f"v{instance.version} ({instance.name})")
            except Exception as e:
                logger.warning(f"[MODULES] Failed to load {filepath.name}: {e}")

    def list_modules(self) -> list:
        """Return metadata for all discovered modules.

        Modules are returned in a deliberate display order rather than
        whatever filesystem-walk order Python produced at discovery
        time. The frontend renders top-to-bottom in the order received.

        Three loose tiers, signalled by the `tier` field:
          - "infrastructure": utility modules (probe gen, dialogue)
          - "showpiece":      featured analysis modules (bordered in UI)
          - "mi":             mechanistic-interpretability work
          - "routing":        routing research (SFD, harm direction, ablation)
          - "standard":       everything else
        """
        DISPLAY_ORDER = [
            # Infrastructure
            ("model_dialogue",                "infrastructure"),
            ("roundtable_lma",                "infrastructure"),
            ("probe_generator",               "infrastructure"),
            # Showpieces (cyan border in UI)
            ("domain_surface",                "showpiece"),
            ("correction_field_topology",     "showpiece"),
            ("correction_prism",              "showpiece"),
            # Mechanistic-interpretability work (lime-green border in UI)
            ("mechanistic_interpretability",  "mi"),
            ("mi_instrumentation",            "mi"),
            # Routing research (pink border in UI)
            ("arditi_benchmarks",             "routing"),
            ("harm_direction",               "routing"),
            ("harm_trajectory",              "routing"),
            ("syco_signature",               "routing"),
            ("concept_atoms",                "routing"),
            ("routing_ablation",             "routing"),
            # ECM analysis (blue border in UI)
            ("ecm",                          "ecm"),
            # Legacy (purple border in UI)
            ("comparative_analysis",          "legacy"),
        ]

        # Build ordered list using DISPLAY_ORDER as the spine. Any
        # discovered module not in DISPLAY_ORDER is appended at the end
        # with tier "standard" — keeps newly-added modules visible
        # without forcing a registry edit before they show up.
        seen = set()
        result = []
        for name, tier in DISPLAY_ORDER:
            if name not in self._modules:
                continue
            mod = self._modules[name]
            meta = mod.get_metadata()
            state = self._state[name]
            meta["status"] = state.status
            meta["progress"] = state.progress
            meta["has_results"] = state.results is not None
            if state.error:
                meta["error"] = state.error
            meta["tier"] = tier
            result.append(meta)
            seen.add(name)
        for name, mod in self._modules.items():
            if name in seen:
                continue
            meta = mod.get_metadata()
            state = self._state[name]
            meta["status"] = state.status
            meta["progress"] = state.progress
            meta["has_results"] = state.results is not None
            if state.error:
                meta["error"] = state.error
            meta["tier"] = "standard"
            result.append(meta)
        return result

    def get_module(self, name: str) -> Optional[TASMModule]:
        return self._modules.get(name)

    def set_pipeline(self, pipeline) -> None:
        """Propagate pipeline reference to modules that need model access.

        Called after model load. Modules that support set_pipeline() get
        the pipeline (model, tokenizer, adapter, delta_store).
        """
        for name, mod in self._modules.items():
            if hasattr(mod, 'set_pipeline'):
                try:
                    mod.set_pipeline(pipeline)
                    logger.info(f"[MODULES] {name}: pipeline connected")
                except Exception as e:
                    logger.warning(f"[MODULES] {name}: set_pipeline failed: {e}")

    def get_status(self, name: str) -> dict:
        state = self._state.get(name)
        if not state:
            return {"error": f"Module '{name}' not found."}
        return {
            "status": state.status,
            "progress": state.progress,
            "has_results": state.results is not None,
            "has_log": state.log_path is not None,
            "error": state.error,
            "started_at": state.started_at,
            "completed_at": state.completed_at,
        }

    def get_results(self, name: str) -> Optional[dict]:
        state = self._state.get(name)
        return state.results if state else None

    def collect_results(self, skip: set = None) -> dict:
        """Return {name: results} for all modules that have results.

        Args:
            skip: set of module names to exclude (e.g. {'comparative_analysis'})

        Returns:
            dict mapping module name to its results dict.
        """
        skip = skip or set()
        out = {}
        for name, st in self._state.items():
            if name in skip:
                continue
            if st.results is not None:
                out[name] = st.results
        return out

    def get_log_path(self, name: str) -> Optional[str]:
        state = self._state.get(name)
        return state.log_path if state else None

    def reset_module(self, name: str) -> dict:
        """Reset a module to idle state, clearing results and errors."""
        state = self._state.get(name)
        if not state:
            return {"ok": False, "error": f"Module '{name}' not found."}
        if state.status == "running":
            return {"ok": False, "error": f"Module '{name}' is currently running."}
        state.status = "idle"
        state.progress = ""
        state.results = None
        state.error = None
        state.started_at = None
        state.completed_at = None
        state.log_path = None
        logger.info(f"[MODULES] {name} reset to idle")
        return {"ok": True}

    def cancel_module(self, name: str) -> dict:
        """Request cancellation of a running module.

        Cooperative: sets a flag the module notices at its next progress
        report (or its next should_cancel() poll).  A module in the middle of
        a single long torch op will not stop until that op returns — there is
        no safe way to interrupt one — so the response says "requested", not
        "stopped".
        """
        state = self._state.get(name)
        if not state:
            return {"ok": False, "error": f"Module '{name}' not found."}
        if state.status != "running":
            return {"ok": False,
                    "error": f"Module '{name}' is not running "
                             f"(status: {state.status})."}
        state.cancel_event.set()
        state.progress = "Cancelling — waiting for the current step to finish..."
        logger.info(f"[MODULES] cancellation requested for {name}")
        if self._event_hook:
            try:
                self._event_hook("module_status", {
                    "name": name, "status": "running",
                    "progress": state.progress,
                })
            except Exception:
                pass
        return {"ok": True, "message": f"Cancellation requested for '{name}'.",
                "cancelling": True}

    def run_module(self, name: str, session_results: list,
                   params: dict, session_dir: Path = None) -> dict:
        """Start a module in a background thread.

        Returns immediately with status. Poll get_status() for completion.
        """
        mod = self._modules.get(name)
        if not mod:
            return {"ok": False, "error": f"Module '{name}' not found."}

        state = self._state[name]

        with self._lock:
            if state.status == "running":
                return {"ok": False, "error": f"Module '{name}' is already running."}

        # Validate
        ok, msg = mod.validate(session_results, params)
        if not ok:
            return {"ok": False, "error": msg}

        # Reset state
        state.status = "running"
        state.progress = "Starting..."
        state.results = None
        state.error = None
        state.started_at = time.time()
        state.completed_at = None
        # Fresh token per run — a cancel from a previous run must not abort
        # the next one.
        state.cancel_event = threading.Event()
        cancel_event = state.cancel_event

        def _should_cancel() -> bool:
            return cancel_event.is_set()

        def _progress(message: str):
            # Progress reporting doubles as the cancellation checkpoint: the
            # modules already call this inside their long loops, so raising
            # here makes them cancellable without touching each loop.
            if cancel_event.is_set():
                raise ModuleCancelled(name)
            state.progress = message
            if self._event_hook:
                now = time.time()
                if not hasattr(_progress, '_last') or now - _progress._last > 0.5:
                    _progress._last = now
                    try:
                        self._event_hook("module_status", {
                            "name": name, "status": "running",
                            "progress": message,
                        })
                    except Exception:
                        pass

        def _run():
            try:
                # Set session directory on module if supported
                if session_dir and hasattr(mod, 'set_session_dir'):
                    try:
                        mod.set_session_dir(str(session_dir))
                    except Exception:
                        pass

                # Pass should_cancel only to modules that declare it, so
                # existing three-argument run() signatures keep working.
                kwargs = {"progress": _progress}
                try:
                    if "should_cancel" in inspect.signature(mod.run).parameters:
                        kwargs["should_cancel"] = _should_cancel
                except (TypeError, ValueError):
                    pass

                results = mod.run(session_results, params, **kwargs)
                # A module may also return without raising after noticing the
                # cancel flag; treat that as cancelled too.
                if cancel_event.is_set():
                    raise ModuleCancelled(name)
                state.results = results
                # ERROR CONTRACT (was wrong): this used to set
                # status="completed", has_results=True for anything that
                # RETURNED, so the ~10 places that return {"error": ...} as a
                # normal result were reported to the user as successful runs.
                # One convention now: a returned dict carrying a truthy
                # "error" key is a failure; raising is the other failure path.
                returned_error = (results.get("error")
                                  if isinstance(results, dict) else None)
                # A module may return an error alongside a usable result set
                # (e.g. an optional late stage failed after the main analysis
                # succeeded).  Distinguish that from a total failure, so the
                # UI does not show "error" over a panel full of valid results
                # — or "completed" over nothing.
                other_keys = [k for k in results
                              if k != "error"] if isinstance(results, dict) else []
                if returned_error and other_keys:
                    state.status = "partial"
                elif returned_error:
                    state.status = "error"
                else:
                    state.status = "completed"
                state.error = str(returned_error) if returned_error else None
                state.completed_at = time.time()
                elapsed = state.completed_at - state.started_at
                if returned_error:
                    logger.error(f"[MODULES] {name} finished with status "
                                 f"'{state.status}' after {elapsed:.1f}s: "
                                 f"{returned_error}")
                else:
                    logger.info(f"[MODULES] {name} completed in {elapsed:.1f}s")

                # Persist results to session directory
                if session_dir:
                    out_path = session_dir / f"module_{name}.json"
                    try:
                        with open(out_path, "w") as f:
                            json.dump(results, f, indent=1, default=str)
                        logger.info(f"[MODULES] Saved results to {out_path}")
                    except Exception as e:
                        logger.warning(f"[MODULES] Failed to save results: {e}")

                # Always save a standalone log. Runtime output, so it goes
                # under ~/.tagm with the rest of the persistent data — not
                # the repo root, where it dirtied the checkout.
                log_dir = Path.home() / ".tagm" / "module_logs"
                log_path = log_dir / f"module_{name}_log.json"
                try:
                    log_dir.mkdir(parents=True, exist_ok=True)
                    with open(log_path, "w") as f:
                        json.dump(results, f, indent=1, default=str)
                    state.log_path = str(log_path)
                    logger.info(f"[MODULES] Saved log to {log_path}")
                except Exception as e:
                    logger.warning(f"[MODULES] Failed to save log: {e}")

                # Emit completion event AFTER all persistence work
                if self._event_hook:
                    try:
                        evt = {
                            "name": name, "status": state.status,
                            "elapsed": round(elapsed, 1),
                            "has_log": state.log_path is not None,
                        }
                        if returned_error:
                            evt["error"] = str(returned_error)
                        self._event_hook("module_status", evt)
                    except Exception:
                        pass

            except ModuleCancelled:
                # Not an error: the user asked for this.  Any partial results
                # are dropped — a half-finished analysis is not a result, and
                # keeping it invites citing numbers from an aborted run.
                state.status = "cancelled"
                state.results = None
                state.error = None
                state.completed_at = time.time()
                elapsed = state.completed_at - state.started_at
                state.progress = f"Cancelled after {elapsed:.1f}s"
                logger.info(f"[MODULES] {name} cancelled by user after "
                            f"{elapsed:.1f}s")
                if self._event_hook:
                    try:
                        self._event_hook("module_status", {
                            "name": name, "status": "cancelled",
                            "elapsed": round(elapsed, 1),
                        })
                    except Exception:
                        pass

            except Exception as e:
                state.status = "error"
                state.error = str(e)
                state.completed_at = time.time()
                logger.error(f"[MODULES] {name} crashed: {traceback.format_exc()}")

                if self._event_hook:
                    try:
                        self._event_hook("module_status", {
                            "name": name, "status": "error",
                            "error": str(e),
                        })
                    except Exception:
                        pass

        thread = threading.Thread(target=_run, name=f"tasm-module-{name}", daemon=True)
        state.thread = thread
        thread.start()

        return {"ok": True, "message": f"Module '{mod.display_name}' started."}

    def reload_modules(self):
        """Re-scan for modules. Does not interrupt running modules."""
        running = {n for n, s in self._state.items() if s.status == "running"}
        if running:
            logger.warning(f"[MODULES] Cannot reload while modules are running: {running}")
            return

        self._modules.clear()
        self._state.clear()
        self._discover_modules()
        logger.info(f"[MODULES] Reloaded. Found {len(self._modules)} modules.")
