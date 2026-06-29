"""Chat service — Model Dialogue Interface.

Uses pipeline.active_model for generation. The active model is
determined by pipeline.inference_class, which is set via the
/api/set_inference_model endpoint. Base model loading happens at
toggle time, not per-message.

Chat interactions can optionally be analyzed and recorded into the
session for studying conversational dynamics.

ECM (Entropic Cascade Mitigation) can be toggled via engine config.
When active, a multi-scale entropy tracker modulates sampling
parameters during generation to dampen cascade-prone trajectories.
ECM diagnostics are returned in the response for downstream analysis.

Generation is streamed via SSE to prevent proxy timeouts on long
generations. Tokens are pushed as they're generated; metadata
(ECM diagnostics, analysis status) is sent as a final event.
"""
from __future__ import annotations

import json
import logging
import threading

import torch

from src.core.locks import MODEL_LOCK
from src.engine import config as engine_config

logger = logging.getLogger("src")


def generate_chat_response_streaming(
    pipeline,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Generate a chat response, yielding SSE events as tokens stream.

    Yields:
        SSE-formatted strings: "data: {json}\n\n"
        Event types:
          - {"type":"token", "text":"..."} — incremental token text
          - {"type":"done",  ...}          — final result with metadata
          - {"type":"error", "error":"..."}
    """
    from transformers import TextIteratorStreamer

    if not pipeline or not pipeline.loaded:
        yield _sse({"type": "error", "error": "No model loaded"})
        return
    if not messages:
        yield _sse({"type": "error", "error": "No messages provided"})
        return

    model = pipeline.active_model
    if model is None:
        yield _sse({"type": "error", "error": "No active model available"})
        return

    tokenizer = pipeline.tokenizer
    device = pipeline.device
    model_class = pipeline.inference_class
    ecm_active = engine_config.get("ecm_active")

    # ── Setup (no lock needed — tokenizer is thread-safe) ────────
    try:
        # Base models lack chat templates — use raw text
        if model_class == "base":
            text = messages[-1].get("content", "")
        else:
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = ""
                for m in messages:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    text += f"{role}: {content}\n"
                text += "assistant:"

        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        # Streamer: yields decoded text chunks as tokens are generated
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id
                          or tokenizer.eos_token_id,
            streamer=streamer,
        )

        # ── ECM setup ──────────────────────────────────────────
        ecm_processor = None

        if ecm_active:
            from src.engine.ecm import ECMProcessor

            ecm_processor = ECMProcessor(
                temperature=temperature,
                n_scales=int(engine_config.get("ecm_n_scales")),
                gain=float(engine_config.get("ecm_gain")),
                floor=float(engine_config.get("ecm_floor")),
            )
            generate_kwargs["temperature"] = 1.0
            generate_kwargs["logits_processor"] = [ecm_processor]

        # ── Launch generation in a thread ──────────────────────
        # MODEL_LOCK is acquired inside the thread, not here.
        # This ensures the lock is held for the duration of
        # model.generate() and released when generation completes,
        # while the main thread streams tokens from the queue.
        gen_error = [None]

        def _generate():
            try:
                with MODEL_LOCK, torch.no_grad():
                    model.generate(**generate_kwargs)
            except Exception as e:
                gen_error[0] = e

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()

    except Exception as e:
        logger.exception("[chat] generation setup failed")
        yield _sse({"type": "error", "error": str(e)})
        return

    # ── Stream tokens ──────────────────────────────────────────
    # TextIteratorStreamer blocks on __next__ until a token is ready
    # or generation completes. We yield SSE events as they arrive.
    full_response = []
    try:
        for chunk in streamer:
            if chunk:
                full_response.append(chunk)
                yield _sse({"type": "token", "text": chunk})
    except Exception as e:
        logger.exception("[chat] streaming error")
        yield _sse({"type": "error", "error": str(e)})
        return

    # Wait for the generation thread to finish
    thread.join(timeout=10)

    if gen_error[0]:
        yield _sse({"type": "error", "error": str(gen_error[0])})
        return

    response = "".join(full_response).strip()

    # ── Build final result ─────────────────────────────────────
    result = {
        "type": "done",
        "ok": True,
        "response": response,
        "model_class": model_class,
        "n_input_tokens": int(input_len),
        "n_output_tokens": len(full_response),
        "ecm_active": bool(ecm_active),
    }

    if ecm_processor is not None:
        result["ecm_diagnostics"] = ecm_processor.diagnostics_to_dict()
        diag = ecm_processor.get_diagnostics()
        n_total = len(diag.per_token_entropy)
        logger.info(
            f"[ECM] {diag.n_interventions}/{n_total} tokens "
            f"intervened, max_signal="
            f"{diag.max_cascade_signal:.4f}"
        )

    yield _sse(result)


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


# ── Legacy non-streaming interface (kept for compatibility) ────

def generate_chat_response(
    pipeline,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> dict:
    """Non-streaming generation. Used by analysis paths that need
    the full response as a dict."""
    result = {}
    for sse_line in generate_chat_response_streaming(
        pipeline, messages, max_tokens, temperature, top_p
    ):
        # Parse the SSE data back out
        if sse_line.startswith("data: "):
            try:
                evt = json.loads(sse_line[6:].strip())
                if evt.get("type") == "done":
                    result = evt
                elif evt.get("type") == "error":
                    result = {"ok": False, "error": evt.get("error", "Unknown")}
            except Exception:
                pass
    return result or {"ok": False, "error": "No response generated"}
