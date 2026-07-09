"""ECM harvest — generate a short response and return it as text.

Called from _analyze_prompt_list when ECM is active and
ecm_harvest_tokens > 0. Generates through the live ECM processor,
captures diagnostics, returns the response text for re-analysis.
"""
from __future__ import annotations

import logging
import torch
import numpy as np

from src.engine import config as engine_config

logger = logging.getLogger("src")


def _set_seed(seed: int = 42):
    """Reset RNG state for reproducible sampling."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_harvest_response(
    analyzer,
    prompt: str,
    max_new_tokens: int = 64,
    seed: int = 42,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> dict:
    """Generate a short ECM-regulated response and return it with diagnostics.

    Called inside _analyze_prompt_list, which already holds MODEL_LOCK
    (aliased as _analysis_lock).  threading.Lock is NOT reentrant, so
    this function must NOT acquire MODEL_LOCK itself — doing so would
    deadlock immediately.

    Returns
    -------
    dict with keys:
        response_text : str
        n_tokens : int
        ecm_diagnostics : dict   (per-token temp, signals, interventions)
        seed : int
    """
    pipeline = analyzer.pipeline
    model = pipeline.active_model
    tokenizer = pipeline.tokenizer
    device = pipeline.device
    model_class = pipeline.inference_class

    # ── Tokenize (same convention as chat.py) ──────────────
    if model_class == "base":
        text = prompt
    else:
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"user: {prompt}\nassistant:"

    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # ── Build ECM processor ────────────────────────────────
    ecm_version = str(engine_config.get("ecm_version") or "v2")

    if ecm_version == "v4":
        from src.engine.ecm_v4 import build_processor_from_config
        ecm_proc = build_processor_from_config(analyzer, temperature=temperature)
    else:
        from src.engine.ecm import ECMProcessor
        ecm_proc = ECMProcessor(
            temperature=temperature,
            n_scales=int(engine_config.get("ecm_n_scales")),
            gain=float(engine_config.get("ecm_gain")),
            floor=float(engine_config.get("ecm_floor")),
            deadband=float(engine_config.get("ecm_deadband")),
            agreement=int(engine_config.get("ecm_agreement")),
        )

    # ── Generate ───────────────────────────────────────────
    _set_seed(seed)
    generate_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,          # ECM owns temperature
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        logits_processor=[ecm_proc],
    )
    nrn = int(engine_config.get("ecm_no_repeat_ngram") or 0)
    if nrn > 0:
        generate_kwargs["no_repeat_ngram_size"] = nrn

    # NOTE: MODEL_LOCK is NOT acquired here.  The caller
    # (_analyze_prompt_list) already holds it via _analysis_lock,
    # which IS MODEL_LOCK (line 92 of app_core.py).  Acquiring it
    # again would deadlock (threading.Lock is not reentrant).
    try:
        with torch.no_grad():
            output_ids = model.generate(**generate_kwargs)
    finally:
        if hasattr(ecm_proc, "close"):
            ecm_proc.close()

    new_ids = output_ids[0, input_len:]
    response_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    ecm_diag = ecm_proc.diagnostics_to_dict()

    return {
        "response_text": response_text,
        "n_tokens": len(new_ids),
        "ecm_diagnostics": ecm_diag,
        "seed": seed,
    }
