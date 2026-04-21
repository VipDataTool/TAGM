"""Chat service.

Wraps the loaded pipeline's instruct (or base) model with a generation
endpoint that mirrors TASM's /api/chat surface. The TASM frontend
(chat.html) sends a list of {role, content} messages, an optional
analyze flag, and expects {ok, response, model_class, ...} back.

Streaming is intentionally NOT enabled in this version — TASM's chat
returned the full response in one JSON payload, and chat.html doesn't
implement streaming-receive. Adding streaming later would require both
ends.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import torch

logger = logging.getLogger("tagm")

# Single global lock — generation and analysis must not interleave on
# the same model; both call into torch on shared parameters.
_chat_lock = threading.Lock()


def generate_chat_response(
    pipeline,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_base: bool = False,
) -> dict:
    """Generate a chat response from the loaded pipeline's model.

    Args:
        pipeline:    Loaded TAGM Pipeline.
        messages:    [{"role": "user"|"assistant"|"system", "content": "..."}, ...]
        max_tokens:  Max new tokens to generate.
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        use_base:    If True, generate with the base model; else instruct.

    Returns:
        {"ok": True, "response": str, "model_class": "instruct"|"base"}
        or {"ok": False, "error": str}
    """
    if not pipeline or not pipeline.loaded:
        return {"ok": False, "error": "No model loaded"}
    if not messages:
        return {"ok": False, "error": "No messages provided"}

    model = pipeline.base_model if use_base else pipeline.instruct_model
    if model is None:
        return {"ok": False,
                "error": ("Base model not loaded" if use_base
                          else "Instruct model not loaded")}

    tokenizer = pipeline.tokenizer
    device = pipeline.device
    model_class = "base" if use_base else "instruct"

    with _chat_lock:
        try:
            # Base models lack chat templates — fall back to raw last message.
            if use_base:
                text = messages[-1].get("content", "")
            else:
                # Try chat template; fall back gracefully if unavailable
                try:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    # Roll our own minimal format
                    text = ""
                    for m in messages:
                        role = m.get("role", "user")
                        content = m.get("content", "")
                        text += f"{role}: {content}\n"
                    text += "assistant:"

            inputs = tokenizer(text, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id
                                  or tokenizer.eos_token_id,
                )

            generated = outputs[0][input_len:]
            response = tokenizer.decode(
                generated, skip_special_tokens=True).strip()

            return {
                "ok": True,
                "response": response,
                "model_class": model_class,
                "n_input_tokens": int(input_len),
                "n_output_tokens": int(generated.shape[0]),
            }
        except Exception as e:
            logger.exception("[chat] generation failed")
            return {"ok": False, "error": str(e)}
