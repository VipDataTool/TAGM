"""Process-wide locks for shared mutable state.

MODEL_LOCK serializes ALL access to the loaded model objects — forward
passes, generation, and any code path that installs forward hooks.

Why one lock: ``Analyzer.analyze_prompt`` installs ActivationCapture
hooks on the shared model modules and holds them across its forward
pass and extraction. Any concurrent forward (chat generation, probe
embedding, probe-generator sampling, roundtable turns, ablation runs)
fires those hooks and overwrites the capture dict mid-extraction —
silently corrupting measurements. The corruption requires no error to
surface, which is exactly why it must be structurally impossible.

Rules:
  * Acquire MODEL_LOCK around every ``model(...)`` forward and every
    ``model.generate(...)`` call outside the analyzer.
  * The analyzer acquires it once per ``analyze_prompt`` (covering
    hook install → forward → extraction → hook removal).
  * Never hold it across model load/unload progress waits — load and
    reset are instead guarded at the job level in app_core.
"""
from __future__ import annotations

import threading

MODEL_LOCK = threading.Lock()
