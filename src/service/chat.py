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
"""
from __future__ import annotations

import logging

import torch

from src.core.locks import MODEL_LOCK
from src.engine import config as engine_config

logger = logging.getLogger("src")


def generate_chat_response(
    pipeline,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> dict:
    """Generate a chat response from the active model.

    When engine_config 'ecm_active' is True, generation is modulated
    by the ECM entropy tracker. The processor handles all temperature
    scaling; HuggingFace's TemperatureLogitsWarper is bypassed by
    setting temperature=1.0 (the multiplicative identity, which causes
    the warper to be skipped entirely). Processors fire before warpers
    in the generate() pipeline, so ECM's temperature-scaled logits
    flow directly into top-p filtering.

    Diagnostics are attached to the response when ECM is active.
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
    ecm_active = engine_config.get("ecm_active")

    with MODEL_LOCK:
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

            # ── Build generate kwargs ──────────────────────────────
            generate_kwargs = dict(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id
                              or tokenizer.eos_token_id,
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

                # Delegate temperature control to ECM. Setting
                # temperature=1.0 causes HuggingFace to skip its
                # TemperatureLogitsWarper (1.0 is the identity).
                # ECM divides logits by its adaptive temperature
                # in the processor step, before top-p runs.
                generate_kwargs["temperature"] = 1.0
                generate_kwargs["logits_processor"] = [ecm_processor]

            # ── Generate ───────────────────────────────────────────
            with torch.no_grad():
                outputs = model.generate(**generate_kwargs)

            generated = outputs[0][input_len:]
            response = tokenizer.decode(
                generated, skip_special_tokens=True).strip()

            result = {
                "ok": True,
                "response": response,
                "model_class": model_class,
                "n_input_tokens": int(input_len),
                "n_output_tokens": int(generated.shape[0]),
                "ecm_active": bool(ecm_active),
            }

            # ── Attach ECM diagnostics ─────────────────────────────
            if ecm_processor is not None:
                result["ecm_diagnostics"] = ecm_processor.diagnostics_to_dict()

                diag = ecm_processor.get_diagnostics()
                n_total = len(diag.per_token_entropy)
                logger.info(
                    f"[ECM] {diag.n_interventions}/{n_total} tokens "
                    f"intervened, max_signal="
                    f"{diag.max_cascade_signal:.4f}"
                )

            return result

        except Exception as e:
            logger.exception("[chat] generation failed")
            return {"ok": False, "error": str(e)}
