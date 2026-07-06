"""ECM Module — batch A/B instrumentation for Entropic Cascade Mitigation.

Runs a prompt set through paired generations:

  CONTROL   — plain sampling at base_temperature (no ECM), seeded.
  TREATMENT — ECMProcessorV4 actuating (temperature law, loop guard),
              same seed, same top_p, warper temperature 1.0 with the
              processor supplying the effective temperature.

Because the processor divides logits by base_temperature whenever the
fused signal is zero, the two arms present *identical* distributions to
the sampler until the first intervention. With matched RNG seeds the
sampled paths are therefore token-identical up to that point, and the
first differing token — the DIVERGENCE INDEX — is directly attributable
to the ECM: the exact place the actuator rearranged the generation
path, measured rather than eyeballed.

Parity invariant (checked per record): divergence_idx >= first
actuation step. A violation means something other than the ECM moved
the path (uneven sampler constraints, nondeterministic kernels) and
the record is flagged (parity_ok=False) rather than silently trusted.

Per record the module collects:
  - both response texts and token counts
  - treatment diagnostics from ECMProcessorV4.diagnostics_to_dict():
    per-token temperature, fused signal, per-channel value/signal
    traces, interventions, loop releases
  - divergence index and first-actuation index
  - optional post-hoc analyzer scalars on BOTH responses
    (stress_score, kl_divergence, sfd density_mean) so actuation's
    effect on the delta-referenced measures is itself measured

Aggregates per category and overall (treatment−control deltas). Full
records also stream to <session_dir>/ecm_ab_results.jsonl (append,
flush-per-record — crash-safe, benchmark-harness style).

The processor is built from the module's own params, never from
global engine config, so a batch cannot disturb a live chat session's
ECM settings — and vice versa.

Frontend: parameter widgets auto-render from the schema below.
static/js/ecm_module_ui.js provides the results view (generation-path
strip: temperature line, fused-signal columns, first-actuation tick,
divergence dot, side-by-side texts, measure deltas).

Original ECM concept and v2 formulation: Ostrander (2026).
"""
from __future__ import annotations

import csv
import json
import logging
import math
import time
from pathlib import Path
from typing import Optional

from .base import TASMModule, ModuleParameter

logger = logging.getLogger("tasm")

_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"


# ─────────────────────────────────────────────────────────────────
# Framework-independent core
# ─────────────────────────────────────────────────────────────────

def _load_prompts(params: dict) -> list[dict]:
    """Resolve the prompt set from the uploaded CSV or the textarea."""
    fname = (params.get("prompts_file") or "").strip()
    if fname:
        path = _TEMPLATES_DIR / Path(fname).name   # sanitized upload dir
        if not path.exists():
            raise FileNotFoundError(f"Uploaded prompt file not found: {path}")
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "prompt" not in row:
                    raise ValueError(
                        "Prompt CSV needs a 'prompt' column. "
                        f"Found: {list(row.keys())}")
                p = (row.get("prompt") or "").strip()
                if p:
                    rows.append({
                        "prompt": p,
                        "category": (row.get("category") or "").strip()
                                    or "uncategorized",
                    })
        return rows

    text = (params.get("prompts_text") or "").strip()
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        category = "uncategorized"
        if line.startswith("[") and "]" in line[:64]:
            close = line.index("]")
            category = line[1:close].strip() or "uncategorized"
            line = line[close + 1:].strip()
        if line:
            rows.append({"prompt": line, "category": category})
    return rows


def _build_processor(analyzer, params: dict):
    """Build ECMProcessorV4 directly from module params (not global
    config). Mirrors ecm_v4.build_processor_from_config otherwise."""
    from src.engine.ecm_v4 import (
        CascadeDetector, Channel, DensitySignal, ECMProcessorV4,
        EntropySignal,
    )

    n_scales = int(params["n_scales"])
    deadband = float(params["deadband"])
    agreement = int(params["agreement"])

    def _detector():
        return CascadeDetector(n_scales=n_scales, deadband=deadband,
                               agreement=agreement)

    requested = [s.strip() for s in str(params["channels"]).split(",")
                 if s.strip()]
    channels = []
    for name in requested:
        if name == "entropy":
            channels.append(Channel(EntropySignal(), _detector(),
                                    weight=float(params["entropy_weight"])))
        elif name == "density":
            if analyzer._sfd_cache is None:
                from src.engine.sfd import precompute_sfd_cache
                analyzer._sfd_cache = precompute_sfd_cache(analyzer)
            src = DensitySignal(analyzer._sfd_cache)
            n = src.install(analyzer.pipeline.active_model, analyzer.adapter)
            w = float(params["density_weight"])
            logger.info(f"[ECM module] density channel: {n} layers hooked, "
                        f"weight={w}{' (record-only)' if w == 0.0 else ''}")
            channels.append(Channel(src, _detector(), weight=w))
    if not channels:
        channels = [Channel(EntropySignal(), _detector(), weight=1.0)]

    return ECMProcessorV4(
        channels=channels,
        temperature=float(params["base_temperature"]),
        gain=float(params["gain"]),
        floor=float(params["floor"]),
        fusion=str(params["fusion"]),
    )


def _prepare_inputs(pipeline, prompt_text: str):
    """Chat-templated (instruct) or raw (base) tokenization — the
    chat.py convention, so batch results are comparable to live chat."""
    tokenizer = pipeline.tokenizer
    if pipeline.inference_class == "base":
        text = prompt_text
    else:
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"user: {prompt_text}\nassistant:"
    return tokenizer(text, return_tensors="pt").to(pipeline.device)


def _generate_arm(pipeline, inputs, params: dict, seed: int,
                  processor=None) -> tuple[list[int], str]:
    """One seeded generation. processor=None → control arm."""
    import torch
    from src.core.locks import MODEL_LOCK

    tokenizer = pipeline.tokenizer
    model = pipeline.active_model

    kwargs = dict(
        **inputs,
        max_new_tokens=int(params["max_tokens"]),
        do_sample=True,
        top_p=float(params["top_p"]),
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    nrn = int(params.get("no_repeat_ngram") or 0)
    if nrn > 0:
        kwargs["no_repeat_ngram_size"] = nrn   # BOTH arms — parity-safe

    if processor is None:
        kwargs["temperature"] = float(params["base_temperature"])
    else:
        # ECM contract: warper temperature 1.0; the processor divides
        # by its effective temperature (== base while quiescent).
        kwargs["temperature"] = 1.0
        kwargs["logits_processor"] = [processor]

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    with MODEL_LOCK, torch.no_grad():
        out = model.generate(**kwargs)

    new_ids = out[0][inputs["input_ids"].shape[1]:].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return new_ids, text


def _divergence_index(a: list[int], b: list[int]) -> Optional[int]:
    """First position where token paths differ; None if identical."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def _first_actuation_index(diag: dict) -> Optional[int]:
    for i, v in enumerate(diag.get("per_token_fused_signal") or []):
        if v and v > 0:
            return i
    return None


def _posthoc_scalars(analyzer, text: str) -> dict:
    """Analyzer pass over a generated response: the delta-referenced
    scalars, so actuation's effect on the measures is itself measured."""
    out = {"stress_score": None, "kl_divergence": None, "density_mean": None}
    if not text or not text.strip():
        return out
    try:
        result = analyzer.analyze_prompt(
            text, category="ecm_module",
            compute_kl=True, compute_sfd=True, compute_ltp=False,
            compute_full_trajectory=False, full_capture=False)
        out["stress_score"] = _f(getattr(result, "stress_score", None))
        out["kl_divergence"] = _f(getattr(result, "kl_divergence", None))
        sfd = getattr(result, "sfd", None)
        if sfd is not None:
            out["density_mean"] = _f(getattr(sfd, "density_mean", None))
    except Exception as e:
        logger.warning(f"[ECM module] post-hoc analysis failed: {e}")
    return out


def _f(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _round_trace(trace, nd=4):
    return [round(v, nd) if isinstance(v, (int, float)) and v == v else None
            for v in (trace or [])]


def _mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
    return round(sum(vals) / len(vals), 4) if vals else None


def _summarize(records: list[dict]) -> dict:
    ok_recs = [r for r in records if not r.get("error")]
    by_cat: dict[str, list] = {}
    for r in ok_recs:
        by_cat.setdefault(r.get("category", "uncategorized"), []).append(r)

    def _cat_stats(recs):
        n_div = sum(1 for r in recs if r.get("divergence_idx") is not None)
        stats = {
            "n": len(recs),
            "intervention_rate": _mean(
                [r.get("intervention_rate") for r in recs]),
            "mean_interventions": _mean(
                [r.get("n_interventions") for r in recs]),
            "diverged_frac": round(n_div / len(recs), 4) if recs else None,
            "mean_divergence_idx": _mean(
                [r.get("divergence_idx") for r in recs
                 if r.get("divergence_idx") is not None]),
            "mean_first_actuation_idx": _mean(
                [r.get("first_actuation_idx") for r in recs
                 if r.get("first_actuation_idx") is not None]),
            "mean_min_temperature": _mean(
                [r.get("min_temperature") for r in recs]),
            "parity_violations": sum(
                1 for r in recs if r.get("parity_ok") is False),
            "loop_releases": sum(
                r.get("n_loop_releases", 0) for r in recs),
        }
        for key in ("stress_score", "kl_divergence", "density_mean"):
            deltas = []
            for r in recs:
                c = (r.get("control_measures") or {}).get(key)
                t = (r.get("treatment_measures") or {}).get(key)
                if c is not None and t is not None:
                    deltas.append(t - c)
            stats[f"delta_{key}"] = _mean(deltas)
        return stats

    return {
        "overall": _cat_stats(ok_recs),
        "categories": {cat: _cat_stats(recs)
                       for cat, recs in sorted(by_cat.items())},
    }


# ─────────────────────────────────────────────────────────────────
# Module
# ─────────────────────────────────────────────────────────────────

class EcmModule(TASMModule):
    """Paired A/B batches over the ECM actuator."""

    name = "ecm"
    display_name = "ECM — Entropic Cascade Mitigation"
    description = ("Paired A/B batches over the ECM actuator: control vs. "
                   "ECM-enabled generation with matched seeds, per-token "
                   "temperature and channel traces, divergence-point "
                   "attribution, and post-hoc stress/KL/density on both "
                   "arms.")
    version = "1.0.0"

    # Generates its own data — needs a loaded pair, not session results.
    min_results = 0
    requires_sfd = False   # density channel precomputes its own cache
    requires_ltp = False
    requires_rd = False

    parameters = [
        ModuleParameter(
            name="prompts_file", display_name="Prompt CSV",
            description=("CSV with a 'prompt' column and optional "
                         "'category' column (benchmark-harness format). "
                         "Takes precedence over the text box below."),
            type="file", default=""),
        ModuleParameter(
            name="prompts_text", display_name="Prompts (one per line)",
            description=("Used when no CSV is uploaded. Optional category "
                         "prefix in brackets: [harmful] how do I ..."),
            type="textarea", default=""),
        ModuleParameter(
            name="runs_per_prompt", display_name="Runs per prompt",
            description=("Seeds are seed+run_idx; each run is a matched "
                         "control/treatment pair."),
            type="int", default=1, min_val=1, max_val=20),
        ModuleParameter(
            name="max_tokens", display_name="Max new tokens",
            description="Generation budget per arm.",
            type="int", default=256, min_val=8, max_val=2048),
        ModuleParameter(
            name="base_temperature", display_name="Base temperature",
            description=("Control-arm sampling temperature and the ECM's "
                         "quiescent temperature in the treatment arm."),
            type="float", default=0.7, min_val=0.05, max_val=2.0),
        ModuleParameter(
            name="top_p", display_name="Top-p",
            description="Applied identically to both arms (parity-safe).",
            type="float", default=0.9, min_val=0.1, max_val=1.0),
        ModuleParameter(
            name="seed", display_name="Seed",
            description=("Both arms of a pair share the seed, so paths "
                         "match token-for-token until the first "
                         "intervention."),
            type="int", default=42, min_val=0, max_val=999999),
        ModuleParameter(
            name="channels", display_name="ECM channels",
            description=("Signal sources. Density requires the SFD cache "
                         "(precomputed on first use) and layer hooks."),
            type="select", default="entropy,density",
            options=["entropy", "entropy,density", "density"]),
        ModuleParameter(
            name="entropy_weight", display_name="Entropy weight",
            description="0 = record-only: detected and logged, never "
                        "actuates.",
            type="float", default=1.0, min_val=0.0, max_val=5.0),
        ModuleParameter(
            name="density_weight", display_name="Density weight",
            description=("Defaults to 0 (record-only) per the "
                         "measure-before-actuate discipline. Raise it to "
                         "grant density the actuator for this batch only."),
            type="float", default=0.0, min_val=0.0, max_val=5.0),
        ModuleParameter(
            name="gain", display_name="Gain",
            description="Temperature reduction per σ of excess signal.",
            type="float", default=0.5, min_val=0.0, max_val=2.0),
        ModuleParameter(
            name="floor", display_name="Temperature floor",
            description="Hard lower bound on effective temperature.",
            type="float", default=0.1, min_val=0.01, max_val=1.0),
        ModuleParameter(
            name="deadband", display_name="Deadband (σ)",
            description="Slope in σ-units ignored as ordinary jitter.",
            type="float", default=0.75, min_val=0.0, max_val=3.0),
        ModuleParameter(
            name="agreement", display_name="Agreement (scales)",
            description=("EWMA scales that must corroborate before the "
                         "signal fires."),
            type="int", default=2, min_val=1, max_val=8),
        ModuleParameter(
            name="n_scales", display_name="EWMA scales",
            description="Dyadic windows ≈2,4,8,… tokens.",
            type="int", default=5, min_val=2, max_val=8),
        ModuleParameter(
            name="fusion", display_name="Fusion",
            description=("How weighted channel signals combine. 'sum' "
                         "compounds — retune gain if you switch."),
            type="select", default="max", options=["max", "sum"]),
        ModuleParameter(
            name="no_repeat_ngram", display_name="No-repeat n-gram",
            description=("0 (default) preserves strict A/B parity. If >0 "
                         "it applies to BOTH arms so divergence stays "
                         "attributable to the ECM."),
            type="int", default=0, min_val=0, max_val=12),
        ModuleParameter(
            name="analyze_responses", display_name="Analyze responses",
            description=("Post-hoc analyzer pass on both arms' outputs: "
                         "stress, KL(instruct‖base), SFD density. Roughly "
                         "doubles per-pair cost."),
            type="bool", default=True),
        ModuleParameter(
            name="include_traces", display_name="Include per-token traces",
            description=("Keep temperature/signal traces in the results "
                         "for the in-app path viewer. Disable for very "
                         "large batches (traces always reach the JSONL)."),
            type="bool", default=True),
    ]

    # ── Wiring ──────────────────────────────────────────────────

    def __init__(self):
        super().__init__()
        self._pipeline = None
        self._session_dir = None

    def set_pipeline(self, pipeline):
        """Receive the loaded pipeline (model pair, tokenizer, adapter)."""
        self._pipeline = pipeline

    def set_session_dir(self, session_dir: str):
        self._session_dir = session_dir

    def _analyzer(self):
        """Prefer the app's live analyzer (shared SFD cache); fall back
        to a private instance for headless use."""
        try:
            from src.engine.app_core import state
            if state.analyzer is not None:
                return state.analyzer
        except Exception:
            pass
        from src.engine.analyzer import Analyzer
        return Analyzer(self._pipeline)

    # ── Validation ──────────────────────────────────────────────

    def validate(self, session_results, params):
        ok, msg = super().validate(session_results, params)
        if not ok:
            return ok, msg
        if self._pipeline is None or not getattr(self._pipeline,
                                                 "loaded", False):
            return False, ("ECM module requires a loaded model pair. "
                           "Load a model first.")
        if not (params.get("prompts_file") or "").strip() and \
           not (params.get("prompts_text") or "").strip():
            return False, ("No prompts. Upload a CSV or enter prompts in "
                           "the text box.")
        return True, "OK"

    # ── Run ─────────────────────────────────────────────────────

    def run(self, session_results, params, progress=None):
        def prog(msg):
            if progress:
                progress(msg)

        pipeline = self._pipeline
        analyzer = self._analyzer()

        # defaults from the schema for anything the UI omitted
        merged = {p.name: p.default for p in self.parameters}
        merged.update({k: v for k, v in (params or {}).items()
                       if v is not None})
        params = merged

        prompts = _load_prompts(params)
        if not prompts:
            return {"ok": False,
                    "error": "Prompt source resolved to zero prompts."}

        runs = int(params["runs_per_prompt"])
        total = len(prompts) * runs
        records = []

        jsonl_path = None
        jsonl_f = None
        if self._session_dir:
            jsonl_path = Path(self._session_dir) / "ecm_ab_results.jsonl"
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_f = open(jsonl_path, "a", encoding="utf-8")

        done = 0
        t_batch = time.time()
        prog(f"Starting {total} control/treatment pairs "
             f"({len(prompts)} prompts × {runs} runs)")
        try:
            for prompt_idx, row in enumerate(prompts):
                inputs = _prepare_inputs(pipeline, row["prompt"])
                for run_idx in range(runs):
                    seed = int(params["seed"]) + run_idx
                    t0 = time.time()
                    rec = {
                        "prompt": row["prompt"],
                        "category": row["category"],
                        "prompt_idx": prompt_idx,
                        "run_idx": run_idx,
                        "seed": seed,
                    }
                    try:
                        # ── CONTROL ──
                        ctrl_ids, ctrl_text = _generate_arm(
                            pipeline, inputs, params, seed, processor=None)

                        # ── TREATMENT ── fresh processor per generation:
                        # detector state must not leak across prompts.
                        processor = _build_processor(analyzer, params)
                        try:
                            trt_ids, trt_text = _generate_arm(
                                pipeline, inputs, params, seed,
                                processor=processor)
                            diag = processor.diagnostics_to_dict()
                        finally:
                            processor.close()   # hooks must never leak

                        div_idx = _divergence_index(ctrl_ids, trt_ids)
                        act_idx = _first_actuation_index(diag)
                        parity_ok = (div_idx is None or act_idx is None
                                     or div_idx >= act_idx)

                        rec.update({
                            "control_text": ctrl_text,
                            "treatment_text": trt_text,
                            "control_n_tokens": len(ctrl_ids),
                            "treatment_n_tokens": len(trt_ids),
                            "divergence_idx": div_idx,
                            "first_actuation_idx": act_idx,
                            "parity_ok": parity_ok,
                            "n_interventions":
                                diag.get("n_interventions", 0),
                            "n_loop_releases":
                                diag.get("n_loop_releases", 0),
                            "intervention_rate":
                                diag.get("intervention_rate", 0.0),
                            "max_fused_signal":
                                diag.get("max_cascade_signal", 0.0),
                            "min_temperature": _f(min(
                                diag.get("per_token_temperature") or
                                [params["base_temperature"]])),
                            "elapsed_s": round(time.time() - t0, 2),
                        })

                        rec["channels"] = {
                            cname: {
                                "n_interventions":
                                    ch.get("n_interventions", 0),
                                "max_signal": ch.get("max_signal", 0.0),
                                "weight": ch.get("weight"),
                                "record_only":
                                    ch.get("record_only", False),
                            }
                            for cname, ch in
                            (diag.get("channels") or {}).items()
                        }

                        if bool(params["include_traces"]):
                            rec["trace"] = {
                                "temperature": _round_trace(
                                    diag.get("per_token_temperature")),
                                "fused": _round_trace(
                                    diag.get("per_token_fused_signal")),
                                "channels": {
                                    cname: {
                                        "value": _round_trace(
                                            ch.get("per_token_value")),
                                        "signal": _round_trace(
                                            ch.get("per_token_signal")),
                                    }
                                    for cname, ch in
                                    (diag.get("channels") or {}).items()
                                },
                            }

                        if bool(params["analyze_responses"]):
                            prog(f"{done + 1}/{total} analyzing responses "
                                 f"({row['category']})")
                            rec["control_measures"] = _posthoc_scalars(
                                analyzer, ctrl_text)
                            rec["treatment_measures"] = _posthoc_scalars(
                                analyzer, trt_text)

                    except Exception as e:
                        logger.exception("[ECM module] pair failed")
                        rec["error"] = str(e)

                    records.append(rec)
                    if jsonl_f is not None:
                        jsonl_f.write(json.dumps(rec, default=str) + "\n")
                        jsonl_f.flush()

                    done += 1
                    prog(f"{done}/{total} pairs done "
                         f"({row['category']})")
        finally:
            if jsonl_f is not None:
                jsonl_f.close()

        return {
            "ok": True,
            "config": {p.name: params[p.name] for p in self.parameters
                       if p.name not in ("prompts_text",)},
            "n_prompts": len(prompts),
            "runs_per_prompt": runs,
            "n_pairs": len(records),
            "n_errors": sum(1 for r in records if r.get("error")),
            "elapsed_s": round(time.time() - t_batch, 1),
            "jsonl_path": str(jsonl_path) if jsonl_path else None,
            "summary": _summarize(records),
            "records": records,
        }
