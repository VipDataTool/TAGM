"""Correction Prism — measures which probe-lattice directions are
excited or opposed by a prompt's correction-induced field.

Architecture (see correction_prism_spec.md for full derivation):

    1. Capture H^{ℓ_low}_p — the prompt's contextualized residual stream
       at the probe's lower depth — by re-forwarding through the
       instruct model with a hook.
    2. Reduce to a single beam vector b_p (mean / last / rowspace).
    3. Project the beam through the model-side weight delta ΔW,
       optionally restricted to a circuit (q, k, v, o, qk, vo, full).
    4. For each probe in the lattice, form the prism direction
       Δp_i = p_i^{L_high} - p_i^{L_low}, then compute the signed
       cosine between Δp_i and the projected beam.
    5. Aggregate per cell (subject × level), per prompt, per layer.

The signed scalar tells you, per cell, whether the correction pushed
the prompt's field along (+) or against (−) the natural depth-trajectory
of the probes in that cell. Magnitudes-only would lose this; the prism
interpretation makes the sign meaningful.

This module is the "what does the correction excite for this prompt"
question. Magnitudes-only analyses lose the sign of probe-prompt
alignment; the prism interpretation makes that sign meaningful.
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict

import numpy as np
import torch

from .base import TASMModule, ModuleParameter
from src.engine.hooks import ActivationCapture
from src.probes.io import (
    detect_level_cols,
    load_probe_cache,
    load_probes,
    get_active_probe_set,
)

logger = logging.getLogger("tasm")


CIRCUIT_ORDER = ["full", "q", "k", "v", "o", "qk", "vo"]
CIRCUIT_LABELS = {
    "full": "Full ΔW (Q+K+V+O)",
    "q":    "Q only",
    "k":    "K only",
    "v":    "V only",
    "o":    "O only",
    "qk":   "QK first-order (routing)",
    "vo":   "VO first-order (content)",
}


class CorrectionPrismModule(TASMModule):
    name = "correction_prism"
    display_name = "Correction Prism"
    description = (
        "Captures the prompt's contextualized residual at the probe's "
        "lower depth, projects it through the model-side weight delta "
        "(optionally restricted to a circuit: Q, K, V, O, QK, VO, or full), "
        "and decomposes the resulting beam against probe-side deltas "
        "Δp_i = p_i^L_high − p_i^L_low. Signed cell values distinguish "
        "concepts the correction excites (+) from concepts it opposes (−)."
    )
    version = "0.1.0"

    min_results = 1
    requires_sfd = False
    requires_ltp = False
    requires_rd = False

    parameters = [
        ModuleParameter(
            name="circuit",
            display_name="Circuit",
            description=(
                "Which sub-circuit of ΔW to push the prompt's beam through. "
                "'full' uses all four projections summed; 'qk' is the first-"
                "order routing operator; 'vo' is the first-order content-"
                "flow operator; q/k/v/o restrict to a single projection."
            ),
            type="select",
            default="full",
            options=["full", "q", "k", "v", "o", "qk", "vo"],
        ),
        ModuleParameter(
            name="beam_reduction",
            display_name="Beam Reduction",
            description=(
                "How to reduce the prompt's per-token L_low residual to a "
                "single beam vector. 'mean' = average over tokens (canonical "
                "default). 'last' = take the last token. 'rowspace' = keep "
                "the full T×d matrix and ask which probes lie in its rowspace."
            ),
            type="select",
            default="mean",
            options=["mean", "last", "rowspace"],
        ),
        ModuleParameter(
            name="prism_metric",
            display_name="Prism Metric",
            description=(
                "Signed cosine normalizes per probe so cells are comparable; "
                "signed dot product preserves raw magnitudes."
            ),
            type="select",
            default="signed_cosine",
            options=["signed_cosine", "signed_dot"],
        ),
        ModuleParameter(
            name="layer_aggregation",
            display_name="Layer Aggregation",
            description=(
                "How to combine per-layer responses into a single per-prompt-"
                "per-probe number."
            ),
            type="select",
            default="mean",
            options=["mean", "max"],
        ),
        ModuleParameter(
            name="cell_aggregation",
            display_name="Cell Aggregation",
            description=(
                "How to collapse the signed responses of probes within a "
                "(subject, level) cell into one cell value. 'sum' treats "
                "each probe as an independent detector — the cell reads "
                "as the net pull on the field, with magnitude scaling by "
                "the number of probes firing. 'mean' size-normalizes "
                "(useful for irregular lattices). 'median' is robust to "
                "outlier probes within a cell."
            ),
            type="select",
            default="sum",
            options=["sum", "mean", "median"],
        ),
        ModuleParameter(
            name="prompt_aggregation",
            display_name="Prompt Aggregation",
            description=(
                "How to collapse per-prompt cell values into the heatmap. "
                "'mean' is canonical — across these prompts, what's the "
                "typical cell value. 'sum' would say 'net pull integrated "
                "across the session.'"
            ),
            type="select",
            default="mean",
            options=["mean", "sum"],
        ),
        ModuleParameter(
            name="include_baseline",
            display_name="Include Baseline",
            description=(
                "Also compute responses with ΔW = I — the prompt's beam "
                "under the prism with no correction lens applied. Lets users "
                "see whether bright cells are correction-induced or were "
                "already aligned in the base model."
            ),
            type="bool",
            default=True,
        ),
        ModuleParameter(
            name="prompt_circuit",
            display_name="Prompt Circuit (grafting)",
            description=(
                "Override the circuit operator for the prompt beam only. "
                "When set to a value different from 'circuit', the prism "
                "grafts: prompt is projected through this operator while "
                "probes are projected through 'probe_circuit'. Set both to "
                "'(same as circuit)' to disable grafting. Typical graft: "
                "prompt_circuit='qk', probe_circuit='vo'."
            ),
            type="select",
            default="(same as circuit)",
            options=["(same as circuit)", "full", "q", "k", "v", "o", "qk", "vo"],
        ),
        ModuleParameter(
            name="probe_circuit",
            display_name="Probe Circuit (grafting)",
            description=(
                "Override the circuit operator for probe directions only. "
                "When set to a value different from 'circuit', the prism "
                "grafts: probes are projected through this operator while "
                "the prompt beam uses 'prompt_circuit'. Typical graft: "
                "prompt_circuit='qk', probe_circuit='vo'."
            ),
            type="select",
            default="(same as circuit)",
            options=["(same as circuit)", "full", "q", "k", "v", "o", "qk", "vo"],
        ),
    ]

    def __init__(self):
        super().__init__()
        self._project_root = None
        self._pipeline = None

    def set_project_root(self, root):
        self._project_root = root

    def set_pipeline(self, pipeline):
        self._pipeline = pipeline

    # ── Validation ─────────────────────────────────────────────
    def validate(self, session_results, params):
        ok, msg = super().validate(session_results, params)
        if not ok:
            return ok, msg

        if self._pipeline is None or not self._pipeline.loaded:
            return False, "Model not loaded. Prism requires ΔW + forward access."

        active = get_active_probe_set(self._project_root)
        if active is None:
            return False, ("No probe set active. Apply one in "
                           "Configuration → Probe Set.")
        ok, msg = active.validate_against(self._pipeline)
        if not ok:
            return False, msg

        if len(active.depths) < 2:
            return False, (
                f"Probe set {active.probe_file!r} has only "
                f"{len(active.depths)} depth(s) cached; the prism needs "
                f"both L_low and L_high to form the probe-side delta. "
                f"Re-Apply the probe set with two depths configured.")

        for frac in (active.subject_layer_frac(), active.escalation_layer_frac()):
            cp = active.cache_path(self._project_root, frac)
            if not os.path.exists(cp):
                return False, (
                    f"Probe cache missing at L{int(frac*100)} "
                    f"({os.path.basename(cp)}). Re-Apply the probe set.")

        if not any(r.get("prompt") for r in session_results):
            return False, "No prompts in session results."

        return True, "OK"

    # ── Helpers ────────────────────────────────────────────────
    def _make_circuit_operator(self, circuit, layer_idx):
        """Build the circuit-restricted weight-delta operator at one layer.

        Returns a (d_in, d_out) numpy array, or None if the operator can't
        be formed. The beam will be projected as `beam @ op` (so the
        operator's "input" axis is d_model and its "output" axis depends
        on the circuit). For first-order qk/vo, the operator has shape
        (d_model, d_model) because the bilinear form folds the kv-grouped
        axis away.
        """
        adapter = self._pipeline.adapter
        store = self._pipeline.delta_store
        d_model = adapter.hidden_size(self._pipeline.instruct_model)

        def _delta(role):
            t = store.get_or_none(layer_idx, role)
            return t.float().cpu().numpy() if t is not None else None

        def _instruct_weight(role):
            try:
                key = adapter.projection_weight_key(role, layer_idx)
                state = self._pipeline.instruct_model.state_dict()
                w = state.get(key)
                if w is None:
                    return None
                return w.float().cpu().numpy()
            except (KeyError, AttributeError):
                return None

        def _to_full_kv(mat):
            """Lift a (kv_dim, d_model) GQA matrix to (d_model, d_model)
            by repeating rows. If already (d_model, d_model), return as is."""
            if mat.shape[0] == d_model:
                return mat
            if d_model % mat.shape[0] != 0:
                return None
            return np.repeat(mat, d_model // mat.shape[0], axis=0)

        if circuit == "full":
            dq, dk, dv, do = (_delta(r) for r in ("q", "k", "v", "o"))
            if all(x is None for x in (dq, dk, dv, do)):
                return None
            acc = np.zeros((d_model, d_model), dtype=np.float32)
            for d in (dq, do):
                if d is not None and d.shape == (d_model, d_model):
                    acc = acc + d
            for d in (dk, dv):
                if d is None:
                    continue
                full = _to_full_kv(d)
                if full is not None and full.shape == (d_model, d_model):
                    acc = acc + full
            return acc

        if circuit in ("q", "k", "v", "o"):
            return _delta(circuit)

        if circuit == "qk":
            # First-order routing: ΔWQ · WK^{instruct}^T + WQ^{base} · ΔWK^T
            dq = _delta("q")
            dk = _delta("k")
            wq_inst = _instruct_weight("q")
            wk_inst = _instruct_weight("k")
            if dq is None or dk is None or wq_inst is None or wk_inst is None:
                return None
            wq_base = wq_inst - dq
            dk_full = _to_full_kv(dk)
            wk_inst_full = _to_full_kv(wk_inst)
            if dk_full is None or wk_inst_full is None:
                return None
            return dq @ wk_inst_full.T + wq_base @ dk_full.T

        if circuit == "vo":
            # First-order content-flow: ΔWV · WO^{instruct} + WV^{base} · ΔWO
            dv = _delta("v")
            do = _delta("o")
            wv_inst = _instruct_weight("v")
            wo_inst = _instruct_weight("o")
            if dv is None or do is None or wv_inst is None or wo_inst is None:
                return None
            dv_full = _to_full_kv(dv)
            wv_inst_full = _to_full_kv(wv_inst)
            if dv_full is None or wv_inst_full is None:
                return None
            wv_base = wv_inst_full - dv_full
            return dv_full @ wo_inst + wv_base @ do

        return None

    def _capture_beam(self, prompt_text, layer_low, beam_reduction):
        """Re-forward one prompt and return its beam at layer_low.

        For 'mean' / 'last': returns a 1D vector of length d_model.
        For 'rowspace':       returns the full (T, d_model) matrix.
        """
        cap = ActivationCapture()
        try:
            cap.install(
                self._pipeline.instruct_model,
                self._pipeline.adapter,
                signal_layers=[layer_low],
                full_trajectory=False,
            )
            tokens, _inputs, _out = cap.forward(
                self._pipeline.instruct_model,
                self._pipeline.tokenizer,
                prompt_text,
            )
            h = cap.activations.get(f"layer_{layer_low}_h")
            if h is None:
                return None, []
            h = h[0].float().cpu().numpy()
            if beam_reduction == "rowspace":
                return h, tokens
            if beam_reduction == "last":
                return h[-1], tokens
            return h.mean(axis=0), tokens
        finally:
            cap.remove()

    def _prism_response(self, beam, prism_dirs, op_prompt, metric,
                         op_probe=None):
        """Project beam through op_prompt and probes through op_probe,
        then compute the prism decomposition.

        beam:        (d_model,) for mean/last, or (T, d_model) for rowspace
        prism_dirs:  (n_probes, d_model)  — Δp_i directions, L2-normalized
        op_prompt:   (d_in, d_out)  — circuit operator for the prompt beam
        op_probe:    (d_in, d_out)  — circuit operator for the probes
                     (defaults to op_prompt when None — existing behavior)

        Returns (n_probes,) signed responses.
        """
        if op_probe is None:
            op_probe = op_prompt

        if beam.ndim == 1:
            field = beam @ op_prompt                    # (d_out,)
            probes_field = prism_dirs @ op_probe        # (n_probes, d_out)
            dots = probes_field @ field                 # (n_probes,)
            if metric == "signed_dot":
                return dots
            pf_norms = np.linalg.norm(probes_field, axis=1)
            f_norm = np.linalg.norm(field)
            denom = pf_norms * f_norm
            out = np.zeros_like(dots)
            mask = denom > 1e-12
            out[mask] = dots[mask] / denom[mask]
            return out

        # rowspace: beam is (T, d_model); the field is a subspace.
        field_mat = beam @ op_prompt                    # (T, d_out)
        probes_field = prism_dirs @ op_probe            # (n_probes, d_out)
        sims = probes_field @ field_mat.T               # (n_probes, T)
        if metric == "signed_dot":
            idx = np.argmax(np.abs(sims), axis=1)
            return sims[np.arange(sims.shape[0]), idx]
        pf_norms = np.linalg.norm(probes_field, axis=1, keepdims=True)
        rm_norms = np.linalg.norm(field_mat, axis=1, keepdims=True)
        denom = pf_norms * rm_norms.T                   # (n_probes, T)
        cosines = np.zeros_like(sims)
        mask = denom > 1e-12
        cosines[mask] = sims[mask] / denom[mask]
        idx = np.argmax(np.abs(cosines), axis=1)
        return cosines[np.arange(cosines.shape[0]), idx]

    # ── Main ───────────────────────────────────────────────────
    def run(self, session_results, params, progress=None):
        def prog(msg):
            if progress:
                progress(msg)
            logger.info(f"[PRISM] {msg}")

        circuit = params.get("circuit", "full")
        beam_reduction = params.get("beam_reduction", "mean")
        prism_metric = params.get("prism_metric", "signed_cosine")
        layer_agg = params.get("layer_aggregation", "mean")
        cell_agg = params.get("cell_aggregation", "sum")
        prompt_agg = params.get("prompt_aggregation", "mean")
        include_baseline = bool(params.get("include_baseline", True))

        # Grafting: separate circuit operators for prompt and probes
        _pc = params.get("prompt_circuit", "(same as circuit)")
        _prc = params.get("probe_circuit", "(same as circuit)")
        prompt_circuit = circuit if _pc == "(same as circuit)" else _pc
        probe_circuit = circuit if _prc == "(same as circuit)" else _prc
        grafting = (prompt_circuit != probe_circuit)

        active = get_active_probe_set(self._project_root)
        if active is None:
            raise RuntimeError(
                "No probe set active. Apply one in Configuration → Probe Set.")
        ok, msg = active.validate_against(self._pipeline)
        if not ok:
            raise RuntimeError(msg)

        L_low_frac = active.subject_layer_frac()
        L_high_frac = active.escalation_layer_frac()

        adapter = self._pipeline.adapter
        n_layers = adapter.n_layers(self._pipeline.instruct_model)
        d_model = adapter.hidden_size(self._pipeline.instruct_model)
        layer_low = max(0, min(n_layers - 1, int(L_low_frac * n_layers)))

        # ── Probe lattice ──
        prog("Loading probe structure...")
        probe_file = active.probe_file
        csv_path = os.path.join(self._project_root, probe_file)
        level_cols, level_names = detect_level_cols(csv_path)
        if not level_cols:
            raise RuntimeError(f"No level columns in {probe_file}")
        raw_probes = load_probes(csv_path)

        subjects = []
        for p in raw_probes:
            if p["subject"] not in subjects:
                subjects.append(p["subject"])
        n_subj = len(subjects)
        n_levels = len(level_cols)
        subj_idx = {s: i for i, s in enumerate(subjects)}
        probe_cells = [(subj_idx.get(p["subject"], 0), p["level"])
                       for p in raw_probes]

        # ── Probe caches → Δp ──
        prog(f"Loading probe caches at L{int(L_low_frac*100)} and "
             f"L{int(L_high_frac*100)}...")
        cp_low = active.cache_path(self._project_root, L_low_frac)
        cp_high = active.cache_path(self._project_root, L_high_frac)
        d_low = load_probe_cache(cp_low)
        d_high = load_probe_cache(cp_high)
        if not d_low or not d_high:
            raise RuntimeError(
                "Probe caches missing at L_low or L_high. Re-Apply the "
                "probe set in Configuration → Probe Set.")
        embs_low = np.array(d_low["embeddings"], dtype=np.float32)
        embs_high = np.array(d_high["embeddings"], dtype=np.float32)
        if embs_low.shape != embs_high.shape:
            raise RuntimeError(
                f"Probe cache shape mismatch: L_low {embs_low.shape} vs "
                f"L_high {embs_high.shape}. Re-Apply the probe set.")
        if embs_low.shape[0] != len(raw_probes):
            raise RuntimeError(
                f"Probe count mismatch: caches have {embs_low.shape[0]}, "
                f"CSV has {len(raw_probes)}. Re-Apply the probe set.")

        delta_p = embs_high - embs_low                   # (n_probes, d_model)
        delta_p_norms = np.linalg.norm(delta_p, axis=1, keepdims=True)
        active_mask = (delta_p_norms[:, 0] > 1e-6)
        n_active = int(active_mask.sum())
        n_degenerate = int(len(raw_probes) - n_active)
        if n_active == 0:
            raise RuntimeError(
                "All probe-side deltas are zero (L_low ≈ L_high). The "
                "prism cannot decompose anything. Are L_low and L_high "
                "actually different layers?")
        prism_dirs = np.zeros_like(delta_p)
        prism_dirs[active_mask] = (
            delta_p[active_mask] / delta_p_norms[active_mask])

        # ── Capture per-prompt beams ──
        prog(f"Capturing prompt beams at L{int(L_low_frac*100)} "
             f"({beam_reduction} reduction)...")
        beams = []
        cats_list = []
        prompts_list = []
        n_prompts = len(session_results)
        for pi, r in enumerate(session_results):
            prompt_text = r.get("prompt", "") or ""
            if not prompt_text:
                beams.append(None)
                cats_list.append("u")
                prompts_list.append("")
                continue
            try:
                beam, _toks = self._capture_beam(
                    prompt_text, layer_low, beam_reduction)
            except Exception as e:
                logger.warning(
                    f"[PRISM] Beam capture failed for prompt {pi}: {e}")
                beam = None
            beams.append(beam)
            cat = (r.get("category", "") or "unknown")[:1]
            cats_list.append(cat)
            prompts_list.append(prompt_text)
            if progress and (pi + 1) % 5 == 0:
                progress(f"Captured beams: {pi+1}/{n_prompts}")

        n_projected = sum(1 for b in beams if b is not None)
        if n_projected == 0:
            raise RuntimeError("No prompt beams captured.")

        signal_layers = list(range(n_layers))

        # ── For each circuit, accumulate per-prompt-per-probe responses ──
        circuits_to_run = [circuit]
        if include_baseline:
            circuits_to_run = [circuit, "_baseline"]

        decomposition = {}

        for c in circuits_to_run:
            if grafting and c != "_baseline":
                prog(f"Computing prism response: {c} "
                     f"(grafting: prompt={prompt_circuit}, "
                     f"probe={probe_circuit})")
            else:
                prog(f"Computing prism response: {c}")
            per_prompt_response = np.zeros((n_prompts, len(raw_probes)),
                                            dtype=np.float64)
            per_prompt_layer_count = np.zeros(n_prompts, dtype=np.int64)

            if c == "_baseline":
                op = np.eye(d_model, dtype=np.float32)
                for pi, beam in enumerate(beams):
                    if beam is None:
                        continue
                    resp = self._prism_response(
                        beam, prism_dirs, op, prism_metric)
                    per_prompt_response[pi] = resp
                    per_prompt_layer_count[pi] = 1
            else:
                layer_responses = [[] for _ in range(n_prompts)]
                for ℓ in signal_layers:
                    if grafting:
                        op_p = self._make_circuit_operator(prompt_circuit, ℓ)
                        op_pr = self._make_circuit_operator(probe_circuit, ℓ)
                        if op_p is None or op_pr is None:
                            continue
                        # Lift GQA-sized operators (kv_dim, d_model) to
                        # (d_model, d_model) so matmul with beam/prism_dirs works.
                        for label, op in [("prompt", op_p), ("probe", op_pr)]:
                            if op.shape[0] != d_model and d_model % op.shape[0] == 0:
                                expanded = np.repeat(
                                    op, d_model // op.shape[0], axis=0)
                                if label == "prompt":
                                    op_p = expanded
                                else:
                                    op_pr = expanded
                    else:
                        op_p = self._make_circuit_operator(c, ℓ)
                        if op_p is None:
                            continue
                        op_pr = None  # _prism_response will use op_p for both
                    for pi, beam in enumerate(beams):
                        if beam is None:
                            continue
                        resp = self._prism_response(
                            beam, prism_dirs, op_p, prism_metric,
                            op_probe=op_pr)
                        layer_responses[pi].append(resp)

                for pi, layer_list in enumerate(layer_responses):
                    if not layer_list:
                        continue
                    stacked = np.stack(layer_list, axis=0)
                    if layer_agg == "max":
                        idx = np.argmax(np.abs(stacked), axis=0)
                        agg_resp = stacked[idx, np.arange(stacked.shape[1])]
                    else:
                        agg_resp = stacked.mean(axis=0)
                    per_prompt_response[pi] = agg_resp
                    per_prompt_layer_count[pi] = len(layer_list)

            # ── Build cell heatmap for this circuit ──
            agg_grid = np.zeros((n_subj, n_levels))
            cnt_grid = np.zeros((n_subj, n_levels))
            cell_values_per_cell = defaultdict(list)
            cat_accum = defaultdict(lambda: {
                "grid": np.zeros((n_subj, n_levels)),
                "count": 0})

            for pi in range(n_prompts):
                if beams[pi] is None or per_prompt_layer_count[pi] == 0:
                    continue
                resp = per_prompt_response[pi]
                # Group responses by cell so we can apply the chosen
                # aggregation. Per-probe responses in (si,li) → list.
                cell_buckets = defaultdict(list)
                for probe_i, (si, li) in enumerate(probe_cells):
                    if not active_mask[probe_i]:
                        continue
                    cell_buckets[(si, li)].append(resp[probe_i])

                cell_grid = np.zeros((n_subj, n_levels))
                cell_count = np.zeros((n_subj, n_levels))
                for (si, li), vals in cell_buckets.items():
                    if not vals:
                        continue
                    arr = np.asarray(vals, dtype=np.float64)
                    if cell_agg == "mean":
                        cell_grid[si, li] = float(arr.mean())
                    elif cell_agg == "median":
                        cell_grid[si, li] = float(np.median(arr))
                    else:  # sum (default)
                        cell_grid[si, li] = float(arr.sum())
                    cell_count[si, li] = len(vals)

                # Cross-prompt accumulation respects prompt_aggregation:
                #   'mean' divides by n_prompts at the end (current path)
                #   'sum'  just accumulates without dividing
                agg_grid += cell_grid
                cnt_grid += (cell_count > 0).astype(np.float64)
                cat = cats_list[pi]
                cat_accum[cat]["grid"] += cell_grid
                cat_accum[cat]["count"] += 1

                for si in range(n_subj):
                    for li in range(n_levels):
                        if cell_count[si, li] > 0:
                            cell_values_per_cell[(si, li)].append(
                                cell_grid[si, li])

            if prompt_agg == "mean":
                m = cnt_grid > 0
                agg_grid[m] /= cnt_grid[m]

            variance = np.zeros((n_subj, n_levels))
            for (si, li), vals in cell_values_per_cell.items():
                if len(vals) > 1:
                    variance[si, li] = float(np.var(vals))

            per_category = {}
            for cat, info in cat_accum.items():
                if prompt_agg == "mean" and info["count"] > 0:
                    info["grid"] /= info["count"]
                per_category[cat] = {
                    "aggregate": info["grid"].tolist(),
                    "n_prompts": info["count"],
                }

            decomposition[c] = {
                "aggregate": agg_grid.tolist(),
                "variance": variance.tolist(),
                "per_category": per_category,
                "per_prompt_response": per_prompt_response.tolist(),
            }

        # ── Cell details for the primary circuit ──
        primary = circuit
        primary_data = decomposition.get(primary, {})
        primary_agg = (np.array(primary_data.get("aggregate"))
                       if primary_data else np.zeros((n_subj, n_levels)))
        primary_var = (np.array(primary_data.get("variance"))
                       if primary_data else np.zeros((n_subj, n_levels)))
        primary_per_prompt = (np.array(primary_data.get("per_prompt_response"))
                              if primary_data
                              else np.zeros((n_prompts, len(raw_probes))))

        cell_details = {}
        signed_thresh = 0.05
        for si in range(n_subj):
            for li in range(n_levels):
                key = f"{si}_{li}"
                cell_probe_idx = [pi for pi, (s, l) in enumerate(probe_cells)
                                   if s == si and l == li and active_mask[pi]]
                cell_probe_idx_all = [
                    pi for pi, (s, l) in enumerate(probe_cells)
                    if s == si and l == li]
                probes_info = []
                probe_responses = []
                for probe_i in cell_probe_idx:
                    col = primary_per_prompt[:, probe_i]
                    valid = [v for pi, v in enumerate(col)
                              if beams[pi] is not None]
                    mean_resp = float(np.mean(valid)) if valid else 0.0
                    probe_responses.append(mean_resp)
                    direction = ("excited" if mean_resp > signed_thresh
                                 else "opposed" if mean_resp < -signed_thresh
                                 else "orthogonal")
                    probes_info.append({
                        "probe_idx": probe_i,
                        "text": raw_probes[probe_i].get(
                            "text",
                            raw_probes[probe_i].get("anchor_id", "")),
                        "response": round(mean_resp, 6),
                        "direction": direction,
                    })

                # Per-cell composition record. Active = probes with
                # non-degenerate Δp (those that actually contribute to
                # the cell value); n_probes_total includes degenerate
                # ones so users see what the lattice declared.
                arr = (np.asarray(probe_responses, dtype=np.float64)
                       if probe_responses else np.zeros(0))
                cell_details[key] = {
                    "probes":               probes_info,
                    "n_probes":             len(probes_info),
                    "n_probes_total":       len(cell_probe_idx_all),
                    "n_probes_degenerate":  len(cell_probe_idx_all) - len(probes_info),
                    "n_excited":  int(np.sum(arr > signed_thresh)) if arr.size else 0,
                    "n_opposed":  int(np.sum(arr < -signed_thresh)) if arr.size else 0,
                    "n_neutral":  int(np.sum(np.abs(arr) <= signed_thresh)) if arr.size else 0,
                    "cell_value":     float(primary_agg[si, li]),
                    "cell_variance":  float(primary_var[si, li]),
                    "probe_response_min":     float(arr.min()) if arr.size else 0.0,
                    "probe_response_max":     float(arr.max()) if arr.size else 0.0,
                    "probe_response_mean":    float(arr.mean()) if arr.size else 0.0,
                    "probe_response_median":  float(np.median(arr)) if arr.size else 0.0,
                }

        # ── Per-subject summary ──
        per_subject = {}
        signed_thresh = 0.05
        for si, subj in enumerate(subjects):
            row = primary_agg[si]
            n_excited = int(np.sum(row > signed_thresh))
            n_opposed = int(np.sum(row < -signed_thresh))
            per_subject[subj] = {
                "mean_signed": float(np.mean(row)),
                "mean_abs":    float(np.mean(np.abs(row))),
                "max_abs":     float(np.max(np.abs(row))),
                "n_excited":   n_excited,
                "n_opposed":   n_opposed,
            }

        cat_names = {"b": "benign", "m": "mild", "h": "harmful",
                     "j": "jailbreak", "a": "adversarial",
                     "d": "dual-use", "u": "unknown"}
        all_cats_seen = sorted(set(cats_list))

        return {
            "version": self.version,
            "module": "correction_prism",
            "probe_file": probe_file,
            "subjects": subjects,
            "subj_short": [s.replace("_", " ").title()[:14] for s in subjects],
            "levels": level_names,
            "n_prompts": n_prompts,
            "n_prompts_projected": n_projected,
            "n_probes": len(raw_probes),
            "n_probes_active": n_active,

            "primary_circuit": primary,
            "aggregate": primary_agg.tolist(),
            "variance": primary_var.tolist(),
            "cell_details": cell_details,

            "decomposition": decomposition,
            "circuit_labels": CIRCUIT_LABELS,
            "circuit_order": [c for c in CIRCUIT_ORDER if c in decomposition],

            "categories": {k: cat_names.get(k, k) for k in all_cats_seen},
            "prompts": [
                {"idx": pi,
                 "category": cats_list[pi],
                 "text": (prompts_list[pi] or "")[:200],
                 "projected": beams[pi] is not None}
                for pi in range(n_prompts)
            ],

            "per_subject": per_subject,

            "config": {
                "circuit":              circuit,
                "beam_reduction":       beam_reduction,
                "prism_metric":         prism_metric,
                "layer_aggregation":    layer_agg,
                "cell_aggregation":     cell_agg,
                "prompt_aggregation":   prompt_agg,
                "include_baseline":     include_baseline,
                "L_low":                L_low_frac,
                "L_high":               L_high_frac,
                "n_signal_layers":      len(signal_layers),
                "model_dim":            d_model,
                "n_probes_degenerate":  n_degenerate,
                "probe_cache_used": {
                    "L_low":  os.path.basename(cp_low),
                    "L_high": os.path.basename(cp_high),
                },
                "active_probe_model_id": active.model_id,
            },

            "diagnostic": {"ok": True, "cause": None, "message": "OK"},
        }
