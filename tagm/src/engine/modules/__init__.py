"""
TASM Module Framework

Extensible post-collection analysis modules for TASM.
Each module operates on session data (results.json) and produces
structured output without affecting the core instrument pipeline.

Modules are auto-discovered from this directory. Any Python file
that defines a class inheriting from TASMModule will be registered.
"""

from .base import TASMModule, ModuleParameter, ModuleRunner

__all__ = ["TASMModule", "ModuleParameter", "ModuleRunner"]
