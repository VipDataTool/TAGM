"""Orchestrator: wire Pipeline + measurements + analyses together.

Flow:
  1. The user sets a CaptureConfig on the pipeline (independent of which
     measurements will run). The orchestrator records it.
  2. The user selects a set of measurements with their parameters. The
     orchestrator validates each measurement's CaptureExpectation against
     the active CaptureConfig; measurements whose expectations are not
     satisfied are rejected with actionable diagnostics.
  3. On analyze(), the orchestrator runs the Pipeline exactly once per
     prompt per the (user-set) CaptureConfig — concurrent or sequential
     base/instruct depending on whether any active measurement needs
     base logits.
  4. The orchestrator dispatches the RunResult to each measurement's
     compute() in topological order (measurement dependencies).
  5. Each MeasurementResult is validated against the per-token alignment
     contract and merged into the Session.

The orchestrator never modifies the CaptureConfig based on measurement
selection. The CaptureConfig is a pipeline-level choice the user makes
up front; measurements consume whatever is captured. See
NOTES.md (correction pass) for design rationale.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from tagm.core.capture.config import CaptureConfig
from tagm.core.pipeline import Pipeline, RunResult
from tagm.measurement.base import MeasurementModule
from tagm.measurement.parameters import resolve_parameters
from tagm.measurement.registry import find_measurement
from tagm.measurement.requirements import (
    CaptureExpectation, ExpectationViolation, ProbeNotAvailableError,
    ProbeRequirement, validate_expectation,
)
from tagm.measurement.result import MeasurementResult
from tagm.probes.artifact import ProbeSet
from tagm.probes.store import ProbeStore
from tagm.service.session import PromptRecord, Session
from tagm.analysis.registry import find_analysis

logger = logging.getLogger("tagm")


class Orchestrator:
    """Orchestrates a TAGM analysis run.

    Usage:
        orch = Orchestrator(pipeline, probe_store)
        orch.set_capture_config(user_set_cfg)                 # user-set up front
        orch.configure_measurements([                          # selections + scope params
            ("stress_score", {"layers": [10, 11, 12]}),
            ("last_position_attribution", {"layers": []}),
        ])
        orch.analyze_prompt("What is recursion?", category="benign", session=session)
    """

    def __init__(self, pipeline: Pipeline, probe_store: ProbeStore):
        self.pipeline = pipeline
        self.probe_store = probe_store
        self._capture_config: Optional[CaptureConfig] = None
        self._base_capture_config: Optional[CaptureConfig] = None
        self._selected: list[tuple[MeasurementModule, dict]] = []

    # ── Capture configuration ──────────────────────────────────────
    def set_capture_config(self, capture_config: CaptureConfig,
                           base_capture_config: Optional[CaptureConfig] = None
                           ) -> None:
        """Set the CaptureConfig the pipeline will use for forward passes.

        This is the user's choice, made independently of measurement
        selection. A base_capture_config can be supplied for paired runs
        where base-side activations are also needed; most measurements
        only need base logits, which are always captured on a paired run
        regardless of base_capture_config.
        """
        self._capture_config = capture_config
        self._base_capture_config = base_capture_config

    @property
    def capture_config(self) -> Optional[CaptureConfig]:
        return self._capture_config

    # ── Measurement selection ──────────────────────────────────────
    def configure_measurements(
        self,
        selections: list[tuple[str, dict]],
    ) -> dict[str, Any]:
        """Select measurements with their (scope) parameter values.

        For each entry (measurement_name, user_params_dict):
          - Look up the measurement class in the registry.
          - Resolve parameters (fill defaults, validate).
          - Validate the measurement's CaptureExpectation against the
            active CaptureConfig. If not satisfied, reject with reasons.

        Returns a dict with keys:
          'errors':     parameter- or lookup-level errors per name
          'skipped':    measurements rejected for any reason
          'violations': expectation violations per name (list of reasons)
          'selected':   [(name, resolved_params), ...] for measurements accepted
        """
        if self._capture_config is None:
            raise RuntimeError(
                "Orchestrator: set_capture_config() must be called before "
                "configure_measurements(). CaptureConfig is a user-level "
                "choice that configures the pipeline; it is not derived "
                "from measurement selection.")

        report: dict[str, Any] = {
            "errors": [],
            "skipped": [],
            "violations": {},
            "selected": [],
        }
        accepted: list[tuple[MeasurementModule, dict]] = []

        for name, user_params in selections:
            try:
                cls = find_measurement(name)
            except KeyError as e:
                report["errors"].append(f"{name}: {e}")
                report["skipped"].append(name)
                continue
            module = cls()

            try:
                resolved = resolve_parameters(module.parameters, user_params)
            except ValueError as e:
                report["errors"].append(f"{name}: {e}")
                report["skipped"].append(name)
                continue

            expectation = module.capture_expectation(resolved)
            violations = validate_expectation(expectation, self._capture_config)
            if violations:
                report["violations"][name] = violations
                report["skipped"].append(name)
                continue

            accepted.append((module, resolved))
            report["selected"].append((name, resolved))

        self._selected = accepted
        return report

    @property
    def selected(self) -> list[tuple[str, dict]]:
        return [(m.name, p) for m, p in self._selected]

    # ── Needs-base-model check ─────────────────────────────────────
    def _needs_base_model(self) -> bool:
        """True if any selected measurement needs base logits."""
        for module, params in self._selected:
            exp = module.capture_expectation(params)
            if exp.needs_base_logits:
                return True
        return False

    # ── Dependency ordering ────────────────────────────────────────
    def _ordered_modules(self) -> list[tuple[MeasurementModule, dict]]:
        """Topological sort of selected measurements by `depends_on`."""
        by_name = {m.name: (m, p) for m, p in self._selected}
        in_degree = {n: 0 for n in by_name}
        edges: dict[str, list[str]] = {n: [] for n in by_name}
        for name, (module, _) in by_name.items():
            for dep in getattr(module, "depends_on", ()) or ():
                if dep in by_name:
                    edges[dep].append(name)
                    in_degree[name] += 1
        queue = [n for n, d in in_degree.items() if d == 0]
        ordered: list[str] = []
        while queue:
            n = queue.pop(0)
            ordered.append(n)
            for succ in edges[n]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
        if len(ordered) != len(by_name):
            raise ValueError(
                f"Cyclic dependency among selected measurements: "
                f"{set(by_name) - set(ordered)}")
        return [by_name[n] for n in ordered]

    # ── Probe lookup ────────────────────────────────────────────────
    def _resolve_probes(self, module: MeasurementModule, params: dict
                         ) -> Optional[dict]:
        req: Optional[ProbeRequirement] = module.probe_requirements(params)
        if req is None:
            return None
        model_pair_id = (f"{self.pipeline.instruct_model_id}|"
                         f"{self.pipeline.base_model_id}")
        probe_set: Optional[ProbeSet] = self.probe_store.get(
            template_id=req.template_id,
            capture_signature=req.capture_signature,
            model_pair_id=model_pair_id,
            parameters={},
        )
        if probe_set is None:
            available = [d.get("template_name", d.get("set_id", "?"))
                         for d in self.probe_store.list()]
            raise ProbeNotAvailableError(req, available_sets=available)
        if req.subclass_filter or req.class_filter:
            probe_set = probe_set.filtered(
                class_filter=req.class_filter,
                subclass_filter=req.subclass_filter,
            )
        return {"probe_set": probe_set, "requirement": req}

    # ── Single-prompt analysis ──────────────────────────────────────
    def analyze_prompt(
        self,
        prompt: str,
        category: str = "",
        session: Optional[Session] = None,
    ) -> PromptRecord:
        """Run all selected measurements on a single prompt.

        Uses the user-set CaptureConfig. Chooses concurrent or sequential
        base/instruct dispatch based on whether any selected measurement
        needs base logits AND whether the base model is loaded.
        """
        if self._capture_config is None:
            raise RuntimeError("Orchestrator: no CaptureConfig set")

        needs_base = self._needs_base_model()
        adapter = self.pipeline.adapter
        delta_store = self.pipeline.delta_store

        # Record session-level configuration on first prompt
        if session is not None and not session.record.model_pair:
            session.set_model_pair({
                "instruct": self.pipeline.instruct_model_id,
                "base": self.pipeline.base_model_id,
                "adapter_family": adapter.family_id,
                "device": self.pipeline.device,
                "dtype": str(self.pipeline.dtype).replace("torch.", ""),
            })
            session.set_capture_config(self._capture_config.to_dict())
            session.set_measurements_config({
                m.name: p for m, p in self._selected
            })

        # Select dispatch path
        if needs_base and self.pipeline.base_model is not None:
            pair = self.pipeline.run_pair(
                prompt, self._capture_config, self._base_capture_config)
            run_result = RunResult.from_pair(pair)
            base_cache: dict = {}
            for module, params in self._selected:
                exp = module.capture_expectation(params)
                if exp.needs_base_logits and getattr(module, "base_extract", None):
                    extracted = module.base_extract(
                        self.pipeline, prompt, pair.base_logits, params) or {}
                    base_cache.update(extracted)
        elif needs_base:
            def _base_extractor(pipeline, p_text, base_logits):
                merged: dict = {}
                for module, params in self._selected:
                    exp = module.capture_expectation(params)
                    if exp.needs_base_logits and getattr(module, "base_extract", None):
                        merged.update(
                            module.base_extract(pipeline, p_text, base_logits, params)
                            or {})
                return merged

            results = self.pipeline.run_pair_batch(
                [prompt],
                instruct_capture=self._capture_config,
                base_extractor=_base_extractor,
                base_capture=self._base_capture_config,
            )
            run_result, bcache = results[0]
            base_cache = {
                "per_position_base_alts": bcache.per_position_base_alts,
                "base_counterfactual_tokens": bcache.base_counterfactual_tokens,
                "base_topk": bcache.base_topk,
                "base_log_softmax": bcache.base_log_softmax,
            }
        else:
            run_result = self.pipeline.run(prompt, self._capture_config)
            base_cache = {}

        # Record structure on first prompt
        if session is not None and not session.record.structure:
            s = run_result.structure
            session.set_structure({
                "n_layers": s.n_layers, "hidden_size": s.hidden_size,
                "n_attention_heads": s.n_attention_heads,
                "n_kv_heads": s.n_kv_heads, "head_dim": s.head_dim,
                "vocab_size": s.vocab_size,
            })

        prec = self._dispatch_measurements(
            run_result, adapter, delta_store, base_cache, prompt, category)
        if session is not None:
            session.add_prompt(prec)
        return prec

    # ── Batch analysis ──────────────────────────────────────────────
    def analyze_batch(
        self,
        prompts: list[dict],
        session: Session,
        progress=None,
    ) -> list[PromptRecord]:
        """Analyze multiple prompts. Uses sequential base-phase if base model
        isn't loaded and any measurement needs base logits; otherwise iterates
        analyze_prompt."""
        if self._capture_config is None:
            raise RuntimeError("Orchestrator: no CaptureConfig set")

        needs_base = self._needs_base_model()
        adapter = self.pipeline.adapter
        delta_store = self.pipeline.delta_store

        # Record session config on first prompt
        if not session.record.model_pair:
            session.set_model_pair({
                "instruct": self.pipeline.instruct_model_id,
                "base": self.pipeline.base_model_id,
                "adapter_family": adapter.family_id,
                "device": self.pipeline.device,
                "dtype": str(self.pipeline.dtype).replace("torch.", ""),
            })
            session.set_capture_config(self._capture_config.to_dict())
            session.set_measurements_config({
                m.name: p for m, p in self._selected
            })

        prompt_texts = [p["prompt"] for p in prompts]
        records: list[PromptRecord] = []

        if needs_base and self.pipeline.base_model is None:
            def _base_extractor(pipeline, p_text, base_logits):
                merged: dict = {}
                for module, params in self._selected:
                    exp = module.capture_expectation(params)
                    if exp.needs_base_logits and getattr(module, "base_extract", None):
                        merged.update(
                            module.base_extract(pipeline, p_text, base_logits, params)
                            or {})
                return merged

            paired = self.pipeline.run_pair_batch(
                prompt_texts,
                instruct_capture=self._capture_config,
                base_extractor=_base_extractor,
                base_capture=self._base_capture_config,
                progress=progress,
            )
            for (run_result, bcache), p in zip(paired, prompts):
                base_cache = {
                    "per_position_base_alts": bcache.per_position_base_alts,
                    "base_counterfactual_tokens": bcache.base_counterfactual_tokens,
                    "base_topk": bcache.base_topk,
                    "base_log_softmax": bcache.base_log_softmax,
                }

                if not session.record.structure:
                    s = run_result.structure
                    session.set_structure({
                        "n_layers": s.n_layers, "hidden_size": s.hidden_size,
                        "n_attention_heads": s.n_attention_heads,
                        "n_kv_heads": s.n_kv_heads, "head_dim": s.head_dim,
                        "vocab_size": s.vocab_size,
                    })

                prec = self._dispatch_measurements(
                    run_result, adapter, delta_store, base_cache,
                    p["prompt"], p.get("category", ""))
                records.append(prec)
                session.add_prompt(prec)
        else:
            for p in prompts:
                records.append(self.analyze_prompt(
                    p["prompt"], p.get("category", ""), session=session,
                ))

        return records

    # ── Dispatch (shared by single and batch paths) ─────────────────
    def _dispatch_measurements(
        self, run_result: RunResult, adapter, delta_store,
        base_cache: dict, prompt: str, category: str,
    ) -> PromptRecord:
        """Run all selected measurements on a single RunResult.

        Single dispatch path used by both analyze_prompt and analyze_batch.
        Handles dependency ordering, probe resolution, compute(), and
        per-token contract validation uniformly.
        """
        ordered = self._ordered_modules()
        results_by_name: dict[str, MeasurementResult] = {}
        errors: list[str] = []

        for module, params in ordered:
            try:
                probes = self._resolve_probes(module, params)
            except ProbeNotAvailableError as e:
                errors.append(f"{module.name}: probe not available: {e}")
                continue

            deps = {dep: results_by_name.get(dep)
                    for dep in getattr(module, "depends_on", ()) or ()}
            effective_params = dict(params)
            if deps:
                effective_params["_dependencies"] = deps

            try:
                mresult = module.compute(
                    run_result, adapter, delta_store,
                    effective_params, probes=probes, base_cache=base_cache,
                )
            except Exception as e:
                logger.exception(f"[orchestrator] {module.name} compute failed")
                errors.append(f"{module.name}: {type(e).__name__}: {e}")
                continue

            validation_errors = mresult.validate(run_result.seq_len)
            if validation_errors:
                errors.append(
                    f"{module.name}: contract violation — {validation_errors}")
                continue

            results_by_name[module.name] = mresult

        # Build the record with flat field mapping. This replaces the old
        # nested `measurements` serialization plus the tasm_compat translator
        # that used to un-nest on the way out. Mapping lives here, once, in
        # the writer.
        fields: dict[str, Any] = {}
        for name, mresult in results_by_name.items():
            _record_measurement(fields, name, mresult)

        # Pass-throughs from base_cache that the UI reads at the record root.
        if base_cache:
            bct = base_cache.get("base_counterfactual_tokens")
            if bct:
                fields["base_counterfactual_tokens"] = bct
            # per_token_kl is optional; reserved for a future KL measurement.
            ptk = base_cache.get("per_token_kl")
            if ptk:
                fields["per_token_kl"] = ptk

        if errors:
            fields["_errors"] = errors

        return PromptRecord(
            prompt=prompt,
            category=category,
            tokens=list(run_result.tokens),
            seq_len=int(run_result.seq_len),
            fields=fields,
        )

    # ── Analysis dispatch ───────────────────────────────────────────
    def run_analysis(self, name: str, session: Session,
                      params: Optional[dict] = None,
                      probes: Optional[dict] = None) -> dict:
        """Run a post-session analysis module.

        Validates that the analysis's declared measurement dependencies
        are present; returns a dict representation of the AnalysisResult.
        """
        cls = find_analysis(name)
        module = cls()
        resolved = resolve_parameters(module.parameters, params)

        session_dict = session.to_dict()
        errors = module.check_dependencies(session_dict)
        if errors:
            return {"ok": False, "errors": errors, "warnings": []}

        # Analysis modules return a plain dict keyed with whatever fields
        # the frontend reads directly. No schema wrapping.
        rdict = module.run(session_dict, resolved, probes=probes)
        if not isinstance(rdict, dict):
            rdict = {"_raw": rdict}
        session.add_analysis(name, rdict)
        return {"ok": True, "errors": [], **rdict}


# ── Measurement → record field mapping ──────────────────────────────
#
# One place, one function. Given a measurement name and its MeasurementResult,
# write the keys the frontend reads onto the prompt record's `fields` dict.
# When a new measurement is added, add its case here.
#
# The frontend key names come from TASM's `main.js` / `_terrainTransformData`
# / `renderCFTResults` etc. — i.e. what the UI actually looks for.

def _record_measurement(fields: dict, name: str, mresult) -> None:
    """Map a MeasurementResult onto root-level fields on a prompt record."""
    scalars = getattr(mresult, "scalars", {}) or {}
    per_tok = getattr(mresult, "per_token", {}) or {}
    per_lyr = getattr(mresult, "per_layer", {}) or {}
    objects = getattr(mresult, "objects", {}) or {}

    if name == "stress_score":
        fields["stress_score"] = scalars.get("stress_mean")
        if per_tok.get("stress"):
            fields["per_token_stress"] = per_tok["stress"]

    elif name == "last_position_attribution":
        if per_tok.get("signed_attribution_to_last"):
            fields["signed_attr"] = per_tok["signed_attribution_to_last"]
        for k_out, k_in in (
            ("net_correction", "net_correction_to_last"),
            ("entropy", "entropy"),
            ("top2_share", "top2_share"),
            ("middle_share", "middle_share"),
            ("interior_cv", "interior_cv"),
            ("n_negative_tokens", "n_negative_tokens"),
            ("has_negative_tokens", "has_negative_tokens"),
        ):
            if k_in in scalars:
                fields[k_out] = scalars[k_in]
        if objects.get("proof1_checks"):
            fields["proof1_checks"] = objects["proof1_checks"]

    elif name == "amplitude_trajectory":
        fields["amplitude_trajectory"] = {
            "raw": objects.get("amplitude_raw") or [],
            "normalized": objects.get("amplitude_normalized") or [],
            "heatmap": objects.get("heatmap") or [],
            "heatmap_shape": objects.get("heatmap_shape") or [0, 0],
            "sublayer_labels": objects.get("sublayer_labels") or [],
            "mean_raw": scalars.get("trajectory_mean_raw"),
            "mean_normalized": scalars.get("trajectory_mean_normalized"),
        }
        # Legacy alias some UI sites read at root.
        if objects.get("amplitude_normalized"):
            fields["amplitude_normalized"] = objects["amplitude_normalized"]

    elif name == "amplitude_derived_metrics":
        if per_tok.get("attn_frac"):
            fields["per_token_attn_frac"] = per_tok["attn_frac"]
        if per_tok.get("coherence"):
            fields["per_token_coherence"] = per_tok["coherence"]
        if per_tok.get("sublayer_rank"):
            fields["per_token_sublayer_rank"] = per_tok["sublayer_rank"]
        if objects.get("token_similarity"):
            fields["token_similarity"] = objects["token_similarity"]

    elif name == "lateral_tension_profile":
        fields["ltp"] = {
            "mean_M": scalars.get("mean_M"),
            "mean_V": scalars.get("mean_V"),
            "mean_L": scalars.get("mean_L"),
            "max_prc": scalars.get("max_prc"),
            "n_directional": scalars.get("n_directional"),
            "n_layers_used": scalars.get("n_layers_used"),
            "tension_magnitudes": per_tok.get("tension_magnitude") or [],
            "prc_per_token": per_tok.get("prc") or [],
            "offset_magnitude": per_lyr.get("offset_magnitude") or {},
            "offset_variance": per_lyr.get("offset_variance") or {},
            "lateral_coverage": per_lyr.get("lateral_coverage") or {},
            "profiles": objects.get("profiles") or [],
            "base_profiles": objects.get("base_profiles") or [],
            "profile_shapes": objects.get("profile_shapes") or [],
            "counterfactual_tokens": objects.get("counterfactual_tokens") or [],
            "semantic_trajectory_2d": objects.get("semantic_trajectory_2d") or [],
            "tension_trajectory_2d": objects.get("tension_trajectory_2d") or [],
        }

    elif name == "spectral_field_density":
        fields["sfd"] = {
            "density_mean": scalars.get("density_mean"),
            "density_max": scalars.get("density_max"),
            "density_var": scalars.get("density_var"),
            "density_p90": scalars.get("density_p90"),
            "global_erank": scalars.get("global_erank"),
            "n_layers_used": scalars.get("n_layers_used"),
            "per_token_density": per_tok.get("density") or [],
        }

    elif name == "rank_displacement":
        fields["rank_displacement"] = {
            "mean_disp_per_token": scalars.get("mean_disp_per_token"),
            "mean_replacement": scalars.get("mean_replacement"),
            "mean_tau": scalars.get("mean_tau"),
            "mean_overlap": scalars.get("mean_overlap"),
            "total_displacement": scalars.get("total_displacement"),
            "n_positions": scalars.get("n_positions"),
            "per_position": objects.get("per_position") or [],
            "per_token_disp": per_tok.get("total_disp") or [],
            "per_token_replacement": per_tok.get("replacement_ratio") or [],
            "instruct_disp_profiles": objects.get("instruct_disp_profiles") or [],
            "base_disp_profiles": objects.get("base_disp_profiles") or [],
            "per_position_tau": objects.get("per_position_tau") or [],
            "per_position_overlap": objects.get("per_position_overlap") or [],
        }

    elif name == "probe_projection":
        fields["probe_projection"] = {
            "best_class_idx": per_tok.get("best_class_idx") or [],
            "best_score": per_tok.get("best_score") or [],
            "score_matrix": objects.get("score_matrix") or [],
            "probe_labels": objects.get("probe_labels") or [],
            "per_token_assignment": objects.get("per_token_assignment") or [],
        }

    elif name == "per_token_embedding":
        if objects.get("per_token_embeddings"):
            fields["per_token_embeddings"] = objects["per_token_embeddings"]
        # Alias used by correction_heatmap / correction_backscatter viz sites.
        if objects.get("per_token_final_emb"):
            fields["per_token_final_emb"] = objects["per_token_final_emb"]

    elif name == "backscatter_projection":
        fields["backscatter"] = {
            "magnitude_matrix": objects.get("magnitude_matrix") or [],
            "probe_labels": objects.get("probe_labels") or [],
            "sublayer_labels": objects.get("sublayer_labels") or [],
            "n_probes": scalars.get("n_probes"),
            "n_sublayers": scalars.get("n_sublayers"),
            "mean_magnitude": scalars.get("mean_magnitude"),
        }

    else:
        # Unknown measurement: dump its serialized result under its own key
        # so data isn't lost. Frontend won't know how to read it, but it's
        # preserved for inspection.
        if hasattr(mresult, "to_dict"):
            fields[name] = mresult.to_dict()
        else:
            fields[name] = mresult
