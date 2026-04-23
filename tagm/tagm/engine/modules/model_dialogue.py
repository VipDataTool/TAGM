"""
Model Dialogue Interface — interactive chat module.

Provides a conversational interface to the loaded model pair.
Configurable parameters control generation behavior. User prompts
and model responses are optionally analyzed and recorded into the
session for studying conversational alignment dynamics.

Run launches the chat window. Interactions flow into the session
with role tags ('user' / 'assistant') for downstream analysis.
"""

import logging
from .base import TASMModule, ModuleParameter

logger = logging.getLogger("tagm")


class ModelDialogueModule(TASMModule):
    name = "model_dialogue"
    display_name = "Model Dialogue Interface"
    description = (
        "Interactive chat with the loaded model. Configure generation "
        "parameters below, then click Run to open the chat window. "
        "User prompts and model responses are analyzed and recorded "
        "into the session for studying conversational alignment dynamics."
    )
    version = "1.0.0"

    min_results = 0
    requires_sfd = False
    requires_ltp = False
    requires_rd = False

    parameters = [
        ModuleParameter(
            name="temperature",
            display_name="Temperature",
            description=(
                "Sampling temperature. Lower values (0.1–0.5) produce "
                "focused, deterministic responses. Higher values (0.7–1.5) "
                "increase diversity and creativity. Default 0.7."
            ),
            type="float",
            default=0.7,
        ),
        ModuleParameter(
            name="top_p",
            display_name="Top-P (nucleus)",
            description=(
                "Nucleus sampling threshold. Only tokens whose cumulative "
                "probability mass falls within top_p are considered. "
                "Lower values sharpen the distribution. Default 0.9."
            ),
            type="float",
            default=0.9,
        ),
        ModuleParameter(
            name="max_tokens",
            display_name="Max Tokens",
            description=(
                "Maximum number of tokens the model can generate per "
                "response. Longer responses use more memory and take "
                "longer. Default 256."
            ),
            type="int",
            default=256,
        ),
        ModuleParameter(
            name="analyze_prompts",
            display_name="Analyze Prompts",
            description=(
                "Run the alignment stress analyzer on each user prompt "
                "and record the result into the session. Adds ~1s per "
                "turn on CPU."
            ),
            type="bool",
            default=True,
        ),
        ModuleParameter(
            name="analyze_responses",
            display_name="Analyze Responses",
            description=(
                "Also analyze the model's generated responses. Captures "
                "how the model's own output looks under the correction "
                "field — useful for studying self-consistency."
            ),
            type="bool",
            default=False,
        ),
        ModuleParameter(
            name="compute_ltp",
            display_name="LTP in Chat",
            description="Compute Lateral Tension Profile for chat turns.",
            type="bool",
            default=True,
        ),
        ModuleParameter(
            name="compute_sfd",
            display_name="SFD in Chat",
            description="Compute Spectral Field Density for chat turns.",
            type="bool",
            default=True,
        ),
    ]

    def validate(self, session_results, params):
        # No session data required — chat creates it
        return True, "OK"

    def run(self, session_results, params, progress=None):
        """Save chat configuration. The frontend opens the chat window."""
        def prog(msg):
            if progress:
                progress(msg)

        config = {
            "temperature": float(params.get("temperature", 0.7)),
            "top_p": float(params.get("top_p", 0.9)),
            "max_tokens": int(params.get("max_tokens", 256)),
            "analyze_prompts": bool(params.get("analyze_prompts", True)),
            "analyze_responses": bool(params.get("analyze_responses", False)),
            "compute_ltp": bool(params.get("compute_ltp", True)),
            "compute_sfd": bool(params.get("compute_sfd", True)),
        }

        # Store config for the chat endpoint to read
        import json
        from pathlib import Path
        config_path = Path(__file__).parent.parent.parent.parent / "chat_config.json"
        try:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            prog(f"Chat configured: temp={config['temperature']}, "
                 f"top_p={config['top_p']}, max_tokens={config['max_tokens']}")
        except Exception as e:
            logger.warning(f"Failed to save chat config: {e}")

        prog("Ready — open chat window")

        return {
            "config": config,
            "chat_url": "/chat",
            "message": "Chat configured. Click 'Open Chat' to start.",
        }
