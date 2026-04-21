"""Session records: per-prompt container and per-session aggregator.

A Session holds:
  - model_pair info (instruct, base, adapter family)
  - structure snapshot (n_layers, hidden_size, ...)
  - union CaptureConfig used for the session's runs
  - list of per-prompt records (each with its measurement results)
  - merged analysis results
  - probe set references (by set_id, if used)

Persistence is JSON (gzipped on write), stdlib only. No HDF5, no parquet.
The schema is the dict produced by `to_dict`; `from_dict` reconstructs.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


# Session schema version. Bump when the dict shape changes in a
# backward-incompatible way; add migration code paths when it does.
SCHEMA_VERSION = 1


@dataclass
class PromptRecord:
    """One prompt's entry in a session.

    The record is a flat dict. The UI reads fields like `r.stress_score`,
    `r.rank_displacement.instruct_disp_profiles`, `r.ltp.counterfactual_tokens`
    directly off the root. Anything a measurement produces lands in `fields`
    and is spread onto the record root at serialize time. No nested
    `measurements` sub-dict, no translator layer. The orchestrator's merge
    step writes `fields` directly (see `orchestration._record_measurement`).
    """
    prompt: str
    category: str = ""
    tokens: list[str] = field(default_factory=list)
    seq_len: int = 0
    prompt_id: str = ""
    # Free-form dict of keys the UI reads at the record root. Populated by
    # the orchestrator per measurement. Example keys: "stress_score",
    # "per_token_stress", "rank_displacement", "ltp", "sfd",
    # "base_counterfactual_tokens", "per_token_kl", "signed_attr", ...
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "category": self.category,
            "tokens": list(self.tokens),
            "seq_len": int(self.seq_len),
        }
        # Spread fields at the root, not nested.
        for k, v in self.fields.items():
            if k in out:
                # Don't clobber the base keys
                continue
            out[k] = v
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "PromptRecord":
        base_keys = {"prompt_id", "prompt", "category", "tokens", "seq_len"}
        fields = {k: v for k, v in d.items() if k not in base_keys}
        return cls(
            prompt_id=d.get("prompt_id", ""),
            prompt=d.get("prompt", ""),
            category=d.get("category", ""),
            tokens=list(d.get("tokens") or []),
            seq_len=int(d.get("seq_len", 0)),
            fields=fields,
        )


@dataclass
class SessionRecord:
    """A complete session.

    Written to disk as <session_id>.json.gz. Contains everything needed
    to reproduce measurements computed in the session (capture config,
    parameters, model pair) plus the measurement and analysis results.
    """
    session_id: str
    created_at: float                                # unix timestamp
    model_pair: dict                                  # {instruct, base, adapter_family, ...}
    structure: dict                                   # ModelStructure dict
    capture_config: dict                              # unioned CaptureConfig dict
    measurements_config: dict                         # {measurement_name: resolved_params}
    prompts: list[PromptRecord] = field(default_factory=list)
    analyses: dict[str, dict] = field(default_factory=dict)
    probe_sets: list[dict] = field(default_factory=list)
    # Each entry: {set_id, template_id, template_name, capture_signature, ...}
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "model_pair": dict(self.model_pair),
            "structure": dict(self.structure),
            "capture_config": dict(self.capture_config),
            "measurements_config": dict(self.measurements_config),
            "prompts": [p.to_dict() for p in self.prompts],
            "analyses": dict(self.analyses),
            "probe_sets": list(self.probe_sets),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionRecord":
        return cls(
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            session_id=d["session_id"],
            created_at=float(d.get("created_at", 0.0)),
            model_pair=dict(d.get("model_pair") or {}),
            structure=dict(d.get("structure") or {}),
            capture_config=dict(d.get("capture_config") or {}),
            measurements_config=dict(d.get("measurements_config") or {}),
            prompts=[PromptRecord.from_dict(p) for p in d.get("prompts") or []],
            analyses=dict(d.get("analyses") or {}),
            probe_sets=list(d.get("probe_sets") or []),
        )


class Session:
    """Active (in-memory) session, populated as prompts are analyzed.

    The service layer holds a single Session at a time. Multi-user
    scenarios are handled by running multiple TAGM instances ("seats"),
    not by multi-tenancy in one instance — per the "toaster, not spa"
    architectural principle.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.record = SessionRecord(
            session_id=session_id or uuid.uuid4().hex[:12],
            created_at=time.time(),
            model_pair={},
            structure={},
            capture_config={},
            measurements_config={},
        )

    @property
    def session_id(self) -> str:
        return self.record.session_id

    @property
    def prompts(self) -> list[PromptRecord]:
        return self.record.prompts

    # ── Population ─────────────────────────────────────────────────
    def set_model_pair(self, pair: dict) -> None:
        self.record.model_pair = dict(pair)

    def set_structure(self, structure: dict) -> None:
        self.record.structure = dict(structure)

    def set_capture_config(self, config_dict: dict) -> None:
        self.record.capture_config = dict(config_dict)

    def set_measurements_config(self, measurements_config: dict) -> None:
        self.record.measurements_config = dict(measurements_config)

    def add_prompt(self, prompt_record: PromptRecord) -> None:
        if not prompt_record.prompt_id:
            prompt_record.prompt_id = f"p{len(self.record.prompts):04d}"
        self.record.prompts.append(prompt_record)

    def add_analysis(self, name: str, result_dict: dict) -> None:
        self.record.analyses[name] = result_dict

    def add_probe_set_reference(self, ref: dict) -> None:
        self.record.probe_sets.append(dict(ref))

    # ── Queries ─────────────────────────────────────────────────────
    def categories(self) -> list[str]:
        seen: set[str] = set()
        for p in self.record.prompts:
            seen.add(p.category or "uncategorized")
        return sorted(seen)

    def measurement_names(self) -> list[str]:
        # Known top-level keys a measurement may publish. If any of these
        # keys are present on any prompt record, that measurement ran.
        _KNOWN = {
            "stress_score": "stress_score",
            "per_token_stress": "stress_score",
            "signed_attr": "last_position_attribution",
            "net_correction": "last_position_attribution",
            "amplitude_trajectory": "amplitude_trajectory",
            "per_token_attn_frac": "amplitude_derived_metrics",
            "ltp": "lateral_tension_profile",
            "sfd": "spectral_field_density",
            "rank_displacement": "rank_displacement",
            "probe_projection": "probe_projection",
            "per_token_embeddings": "per_token_embedding",
            "backscatter": "backscatter_projection",
        }
        seen: set[str] = set()
        for p in self.record.prompts:
            for k in p.fields:
                if k in _KNOWN:
                    seen.add(_KNOWN[k])
        return sorted(seen)

    def to_dict(self) -> dict:
        return self.record.to_dict()
