"""TAGM engine: TASM-compatible computation layer.

Uses TAGM's adapter/pipeline/delta infrastructure to produce TASM's
native result shapes. No translation shims. The adapter tells us
where to hook; the delta store tells us what changed; the extraction
functions read both and write to a flat PromptResult.
"""
