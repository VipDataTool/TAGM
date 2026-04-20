"""Capture subsystem: config, activation store, hook installer.

A CaptureConfig is a flat specification of what to record during a forward
pass. The hook installer translates it into registered torch hooks that
write into an ActivationStore. The store is per-prompt; the installer
returns handles the caller must remove after the forward pass completes.
"""
from tagm.core.capture.config import (
    CapturePoint,
    CaptureConfig,
    VALID_CAPTURE_TYPES,
    VALID_PRECISIONS,
    VALID_REDUCTIONS,
)
from tagm.core.capture.store import ActivationStore
from tagm.core.capture.installer import install_hooks, remove_hooks

__all__ = [
    "CapturePoint",
    "CaptureConfig",
    "ActivationStore",
    "install_hooks",
    "remove_hooks",
    "VALID_CAPTURE_TYPES",
    "VALID_PRECISIONS",
    "VALID_REDUCTIONS",
]
