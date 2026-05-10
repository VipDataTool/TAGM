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
            progress: callback for status updates, e.g. progress("Processing token 50/200")

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


class _ModuleState:
    """Runtime state for a single module."""
    def __init__(self):
        self.status = "idle"       # idle | running | completed | error
        self.progress = ""
        self.results = None
        self.error = None
        self.started_at = None
        self.completed_at = None
        self.thread = None
        self.log_path = None


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
          - "standard":       everything else
        """
        DISPLAY_ORDER = [
            # Infrastructure
            ("model_dialogue",                "infrastructure"),
            ("probe_generator",               "infrastructure"),
            # Showpieces (cyan border in UI)
            ("domain_surface",                "showpiece"),
            ("correction_field_topology",     "showpiece"),
            ("correction_prism",              "showpiece"),
            # Mechanistic-interpretability work (lime-green border in UI)
            ("mechanistic_interpretability",  "mi"),
            ("mi_instrumentation",            "mi"),
            ("arditi_benchmarks",             "mi"),
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

        def _progress(message: str):
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

                results = mod.run(session_results, params, progress=_progress)
                state.results = results
                state.status = "completed"
                state.completed_at = time.time()
                elapsed = state.completed_at - state.started_at
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

                # Always save standalone log to project root
                log_path = self._project_root / f"module_{name}_log.json"
                try:
                    with open(log_path, "w") as f:
                        json.dump(results, f, indent=1, default=str)
                    state.log_path = str(log_path)
                    logger.info(f"[MODULES] Saved log to {log_path}")
                except Exception as e:
                    logger.warning(f"[MODULES] Failed to save log: {e}")

                # Emit completion event AFTER all persistence work
                if self._event_hook:
                    try:
                        self._event_hook("module_status", {
                            "name": name, "status": "completed",
                            "elapsed": round(elapsed, 1),
                            "has_log": state.log_path is not None,
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
