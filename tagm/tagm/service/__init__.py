"""TAGM service layer.

Orchestrates the instrument, measurement, and analysis layers behind the
FastAPI surface in `tagm/app.py`. Handles session records, result
merging, export, and the sequential run_pair_batch pattern the
measurement spec described.
"""
from tagm.service.orchestration import Orchestrator
from tagm.service.session import Session, SessionRecord
from tagm.service.export import export_session, load_session

__all__ = [
    "Orchestrator",
    "Session",
    "SessionRecord",
    "export_session",
    "load_session",
]
