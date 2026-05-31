"""Chat service — Model Dialogue Interface.

Uses pipeline.active_model for generation. The active model is
determined by pipeline.inference_class, which is set via the
/api/set_inference_model endpoint. Base model loading happens at
toggle time, not per-message.

Chat interactions can optionally be analyzed and recorded into the
session for studying conversational dynamics.
"""
from __future__ import annotations

import logging
import threading

import torch

logger = logging.getLogger("src")

_chat_lock = threading.Lock()


def generate_chat_response(
    pipeline,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> dict:
    """Generate a chat response from the active model.

    The active model is selected by pipeline.inference_class.
    Base model is loaded at toggle time via set_inference_model.
    """
    if not pipeline or not pipeline.loaded:
        return {"ok": False, "error": "No model loaded"}
    if not messages:
        return {"ok": False, "error": "No messages provided"}

    model = pipeline.active_model
    if model is None:
        return {"ok": False, "error": "No active model available"}

    tokenizer = pipeline.tokenizer
    device = pipeline.device
    model_class = pipeline.inference_class

    with _chat_lock:
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
