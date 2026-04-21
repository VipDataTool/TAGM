"""Module runner: async dispatch for analysis modules.

Measurements are computed by the engine during /api/analyze — the runner
only handles post-session analysis modules. list_modules() returns both
built-in measurements (metadata only) and registered analyses (runnable).
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

        # Built-in measurements (computed via /api/analyze, not runnable here)
        for m in list_measurements():
            name = m["name"] if isinstance(m, dict) else ""
            ms = self._state.get(name)
            modules.append({
                "name": name,
                "display_name": m.get("display_name", name),
                "description": m.get("description", ""),
                "kind": "measurement",
                "version": "1.0.0",
                "status": ms.status if ms else "idle",
                "has_results": ms is not None and ms.results is not None,
                "min_results": 1,
                "requires_sfd": False,
                "requires_ltp": False,
                "requires_rd": False,
                "parameters": [],
            })

        # Registered analysis modules (runnable)
        for a_info in list_analyses():
            if isinstance(a_info, dict):
                name = a_info.get("name", "")
                display = a_info.get("display_name", name)
                desc = a_info.get("description", "")
                version = a_info.get("version", "0.1.0")
                params = a_info.get("parameters", [])
                deps = a_info.get("depends_on_measurements", ())
            else:
                # It's an AnalysisModule class
                try:
                    inst = a_info()
                    name = inst.name
                    display = inst.display_name
                    desc = inst.description
                    version = inst.version
                    params = inst.parameters
                    deps = inst.depends_on_measurements
                except Exception:
                    continue

            ms = self._state.get(name)
            modules.append({
                "name": name,
                "display_name": display,
                "description": desc,
                "kind": "analysis",
                "version": version,
                "status": ms.status if ms else "idle",
                "has_results": ms is not None and ms.results is not None,
                "min_results": 2,  # most analyses need multiple prompts
                "requires_sfd": "spectral_field_density" in deps,
                "requires_ltp": "lateral_tension_profile" in deps,
                "requires_rd": "rank_displacement" in deps,
                "parameters": [p.to_dict() if hasattr(p, "to_dict") else p
                               for p in params],
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

                # Build session dict from flat results for the analysis module.
                # Analysis modules expect session.prompts[i].measurements.X
                # but our results are flat. We adapt here.
                session_dict = {
                    "prompts": [_flat_result_to_analysis_prompt(r)
                                for r in session.results],
                    "model_pair": {"instruct": session.model_name},
                    "structure": {},
                }

                errors = module.check_dependencies(session_dict)
                if errors:
                    ms.results = {"ok": False, "errors": errors}
                    ms.status = "error"
                    ms.error = "; ".join(errors)
                    ms.completed_at = time.time()
                    return

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


def _flat_result_to_analysis_prompt(r: dict) -> dict:
    """Adapt a flat result dict into the format analysis modules expect.

    Analysis modules read session_dict["prompts"][i]["measurements"]["X"].
    Our flat results have those fields at the top level. We project
    the relevant fields into a nested measurements dict so the analysis
    modules' check_dependencies() and run() work without modification.
    """
    measurements = {}

    # Map flat fields back to measurement names
    if r.get("stress_score") is not None:
        measurements["stress_score"] = {
            "scalars": {"stress_mean": r.get("stress_score")},
            "per_token": {"stress": r.get("per_token_stress", [])},
        }
    if r.get("signed_attr") is not None:
        measurements["last_position_attribution"] = {
            "scalars": {
                "net_correction_to_last": r.get("net_correction"),
                "entropy": r.get("entropy"),
                "top2_share": r.get("top2_share"),
                "middle_share": r.get("middle_share"),
                "interior_cv": r.get("interior_cv"),
                "n_negative_tokens": r.get("n_negative_tokens"),
                "has_negative_tokens": r.get("has_negative_tokens"),
            },
            "per_token": {"signed_attribution_to_last": r.get("signed_attr", [])},
        }
    if r.get("ltp") is not None:
        measurements["lateral_tension_profile"] = {
            "scalars": {
                "mean_M": r["ltp"].get("mean_M"),
                "mean_V": r["ltp"].get("mean_V"),
                "max_prc": r["ltp"].get("max_prc"),
                "n_directional": r["ltp"].get("n_directional"),
            },
            "per_token": {
                "tension_magnitude": r["ltp"].get("tension_magnitudes", []),
            },
            "objects": r["ltp"],
        }
    if r.get("sfd") is not None:
        measurements["spectral_field_density"] = {
            "scalars": {
                "density_mean": r["sfd"].get("density_mean"),
                "density_max": r["sfd"].get("density_max"),
            },
            "per_token": {
                "density": r["sfd"].get("per_token_density", []),
            },
        }
    if r.get("rank_displacement") is not None:
        measurements["rank_displacement"] = {
            "scalars": {
                "mean_tau": r["rank_displacement"].get("mean_tau"),
                "mean_overlap": r["rank_displacement"].get("mean_overlap"),
            },
            "objects": r["rank_displacement"],
        }
    if r.get("amplitude_trajectory"):
        measurements["amplitude_trajectory"] = {
            "objects": {
                "amplitude_raw": r.get("amplitude_trajectory", []),
                "amplitude_normalized": r.get("amplitude_normalized", []),
                "heatmap": r.get("heatmap", []),
            },
        }
    if r.get("per_token_domain_emb") is not None:
        measurements["per_token_embedding"] = {
            "objects": {
                "per_token_embeddings": {
                    "subject": r.get("per_token_domain_emb"),
                    "escalation": r.get("per_token_escalation_emb"),
                    "final": r.get("per_token_final_emb"),
                },
            },
        }

    return {
        "prompt": r.get("prompt", ""),
        "category": r.get("category", ""),
        "tokens": r.get("tokens", []),
        "seq_len": r.get("seq_len", 0),
        "measurements": measurements,
    }
