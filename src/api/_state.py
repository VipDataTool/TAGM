"""Process-wide singletons shared by app.py and the API routers.

Lives here rather than in app.py so the routers can import it without
importing the app (which would be circular). Import side effects are
limited to constructing the ModuleRunner, exactly as app.py used to.
"""
from __future__ import annotations

from pathlib import Path

from src.engine.modules import ModuleRunner
from src.service.events import broker

# src/api/_state.py -> src/api -> src
PACKAGE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_DIR.parent

# The single ModuleRunner instance. app.py wires it to state.on_model_loaded.
module_runner = ModuleRunner(event_hook=broker.publish)
