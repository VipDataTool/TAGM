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
import os
import queue
import threading

import torch

from src.core.locks import MODEL_LOCK
from src.engine import config as engine_config

logger = logging.getLogger("src")

# Wall-clock budget for a single streamed token, and for the generation
# thread to finish after the stream ends.  TextIteratorStreamer defaults
# to timeout=None, i.e. an unbounded blocking get() on its queue: if the
# generation thread dies without putting the sentinel (OOM killer, a C-level
# abort inside model.generate), the reader blocks forever and — before this
# generator was moved off the event loop — took the whole server with it.
_STREAM_TIMEOUT = float(os.environ.get("TAGM_CHAT_STREAM_TIMEOUT", "300"))


def generate_chat_response_streaming(
    pipeline,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    analyzer=None,
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
            tokenizer, skip_prompt=True, skip_special_tokens=True,
            timeout=_STREAM_TIMEOUT)

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
            ecm_version = str(engine_config.get("ecm_version") or "v2")

            if ecm_version == "v4" and analyzer is not None:
                from src.engine.ecm_v4 import build_processor_from_config
                ecm_processor = build_processor_from_config(
                    analyzer, temperature=temperature)
            else:
                if ecm_version == "v4":
                    logger.warning(
                        "[ECM] v4 requested but no analyzer available — "
                        "falling back to v2")
                from src.engine.ecm import ECMProcessor
                ecm_processor = ECMProcessor(
                    temperature=temperature,
                    n_scales=int(engine_config.get("ecm_n_scales")),
                    gain=float(engine_config.get("ecm_gain")),
                    floor=float(engine_config.get("ecm_floor")),
                    deadband=float(engine_config.get("ecm_deadband")),
                    agreement=int(engine_config.get("ecm_agreement")),
                )
            generate_kwargs["temperature"] = 1.0
            generate_kwargs["logits_processor"] = [ecm_processor]

            # Backstop against n-gram loops seeded during cooled steps —
            # now cooling-gated and generation-scoped inside the
            # processor (see ecm_guard). The old generate-kwargs
            # constraint was unconditional and included the prompt in
            # its window.
            nrn = int(engine_config.get("ecm_no_repeat_ngram") or 0)
            ecm_processor.configure_no_repeat(nrn, input_len)

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
            finally:
                # v4 density hooks must never outlive the generation —
                # closing here (inside the thread, lock still relevant)
                # covers success, error, AND client-disconnect paths
                # where the streaming generator is abandoned.
                if ecm_processor is not None and hasattr(ecm_processor, "close"):
                    try:
                        ecm_processor.close()
                    except Exception:
                        logger.exception("[ECM] processor close failed")

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()

    except Exception as e:
        logger.exception("[chat] generation setup failed")
        yield _sse({"type": "error", "error": str(e)})
        return

    # ── Stream tokens ──────────────────────────────────────────
    # TextIteratorStreamer blocks on __next__ until a token is ready,
    # generation completes, or _STREAM_TIMEOUT elapses (queue.Empty).
    # This whole generator is driven from a worker thread by the caller
    # (starlette.concurrency.iterate_in_threadpool), never the event loop.
    full_response = []
    try:
        for chunk in streamer:
            if chunk:
                full_response.append(chunk)
                yield _sse({"type": "token", "text": chunk})
    except queue.Empty:
        # Wedged generation thread: report it instead of hanging. The
        # thread is a daemon and its finally-block still closes any ECM
        # hooks, so abandoning it here is safe.
        logger.error(
            f"[chat] generation stalled: no token in {_STREAM_TIMEOUT:.0f}s")
        yield _sse({"type": "error",
                    "error": f"Generation stalled (no token in "
                             f"{_STREAM_TIMEOUT:.0f}s)"})
        return
    except Exception as e:
        logger.exception("[chat] streaming error")
        yield _sse({"type": "error", "error": str(e)})
        return

    # Wait for the generation thread to finish. Bounded for the same
    # reason as the streamer timeout — an unbounded join() on a wedged
    # thread never returns.
    thread.join(timeout=_STREAM_TIMEOUT)
    if thread.is_alive():
        logger.error("[chat] generation thread did not exit after streaming")
        yield _sse({"type": "error",
                    "error": "Generation thread did not exit."})
        return

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
        diag_dict = ecm_processor.diagnostics_to_dict()
        result["ecm_diagnostics"] = diag_dict
        logger.info(
            f"[ECM {diag_dict.get('version', 'v2')}] "
            f"{diag_dict.get('n_interventions', 0)}/"
            f"{diag_dict.get('n_tokens', 0)} tokens intervened, "
            f"max_signal={diag_dict.get('max_cascade_signal', 0.0):.4f}"
        )

    yield _sse(result)


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"
