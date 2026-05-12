"""Roundtable LMA — Configurable Language Model Array Pipeline.

A data-transformation pipeline built on the ClownCar.AI Language Model
Array concept.  The workflow is a sequence of *stages*, each defined by
its type.  Two core stage types ship with the module:

  **PANEL**    — a roundtable of N agents, each with individual seed
                 parameters, generating responses sequentially.  Each
                 agent sees the *accumulating* transcript of the current
                 panel (deliberative pressure) but *never* sees prior
                 panel transcripts (blank-canvas isolation).

  **ANALYSIS** — an intermediate processing node that transforms the
                 prior stage's output.  The ``method`` field selects
                 from a registry of built-in methods (synthesize,
                 analyze, evaluate, extract, aggregate, passthrough)
                 or a user-defined system prompt.  This is the point
                 where the recursive roundtable cycle breaks and
                 new workflows commence.

Workflow definition
-------------------
Three tiers, highest-priority first:

  1. **UI parameters** — global overrides (temperature, max_tokens).
  2. **CSV template**  — full topology + per-agent seeds.  Columns are
     stages, rows are agents, cells contain JSON seed dicts.  Column
     headers use ``TYPE:Label`` syntax (e.g. ``PANEL:Ethics Review``).
  3. **Internal defaults** — built-in method prompts and fallback
     generation settings.

CSV template format
-------------------
::

    PANEL:Round 1,ANALYSIS:Synthesize,PANEL:Round 2,ANALYSIS:Final Report
    {"name":"Dr.X","system_prompt":"...","temperature":0.7},{"method":"synthesize","system_prompt":"..."},{"ref":"ethicist"},{"method":"evaluate","system_prompt":"..."}
    {"name":"Prof.Z","temperature":0.9},,,
    {"ref":"skeptic","max_tokens":512},,,

- **Header row**: ``TYPE:Label`` per column.  TYPE dispatches execution.
- **Data rows**: JSON dicts (agent seeds) or empty cells.
- **Reference cells**: ``{"ref":"<id_or_name>", ...}`` merges with the
  persistent participant registry entry, with cell overrides winning.
- Empty cells are skipped; a PANEL column's agent count = its non-empty
  cell count.  An ANALYSIS column typically has one cell.

Extending
---------
Add a new analysis method::

    @RoundtableLMAModule.register_method("my_method",
        description="Does something novel",
        default_prompt="You are a specialist who...")
    def _method_mine(module, prior_output, seed, context, progress):
        return _generate(module._pipeline, seed["system_prompt"],
                         prior_output, seed.get("temperature", 0.7), ...)

Add a new stage type::

    @RoundtableLMAModule.register_stage("VOTE")
    def _stage_vote(module, stage, context, progress):
        ...
        return {"type": "VOTE", "output": ..., "n_generations": 0}

Persistent state
----------------
``~/.tagm/roundtable_config.json`` — participant registry + config.
``~/.tagm/roundtable_transcripts/`` — per-run transcript archives.

Original concept: Ostrander (2024) — ClownCar.AI / alice.ipynb LMA.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import time
import threading
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

from src.engine.modules.base import TASMModule, ModuleParameter

try:
    from src.engine import config as engine_config
except ImportError:
    engine_config = None

logger = logging.getLogger("src")

# ── Persistent storage ──────────────────────────────────────────

_CONFIG_DIR = Path.home() / ".tagm"
_CONFIG_PATH = _CONFIG_DIR / "roundtable_config.json"
_TRANSCRIPT_DIR = _CONFIG_DIR / "roundtable_transcripts"

_gen_lock = threading.Lock()

# ────────────────────────────────────────────────────────────────
#  Data structures
# ────────────────────────────────────────────────────────────────

@dataclass
class Participant:
    """A reusable participant identity in the persistent registry."""
    id: str
    name: str
    role: str
    system_prompt: str
    active: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Participant":
        return cls(
            id=d.get("id", str(uuid4())[:8]),
            name=d.get("name", "Unnamed"),
            role=d.get("role", "Participant"),
            system_prompt=d.get("system_prompt",
                                "You are a helpful roundtable participant."),
            active=d.get("active", True),
        )


@dataclass
class WorkflowStage:
    """One column in the template: a typed stage with agent seeds."""
    stage_type: str                    # PANEL, ANALYSIS, etc.
    label: str                         # human-readable label
    seeds: list[dict] = field(default_factory=list)


@dataclass
class Workflow:
    """A complete pipeline parsed from a template or built programmatically."""
    stages: list[WorkflowStage] = field(default_factory=list)
    topic: str = ""

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "stages": [
                {
                    "stage_type": s.stage_type,
                    "label": s.label,
                    "n_agents": len(s.seeds),
                    "seeds": s.seeds,
                }
                for s in self.stages
            ],
        }


@dataclass
class RoundtableConfig:
    """Persistent configuration: participant registry + defaults."""
    participants: list[Participant] = field(default_factory=list)
    default_topic: str = ""

    def to_dict(self) -> dict:
        return {
            "participants": [p.to_dict() for p in self.participants],
            "default_topic": self.default_topic,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RoundtableConfig":
        return cls(
            participants=[Participant.from_dict(p)
                          for p in d.get("participants", [])],
            default_topic=d.get("default_topic", ""),
        )


# ────────────────────────────────────────────────────────────────
#  Config persistence + participant CRUD
# ────────────────────────────────────────────────────────────────

def load_config() -> RoundtableConfig:
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH) as f:
                return RoundtableConfig.from_dict(json.load(f))
        except Exception as e:
            logger.warning(f"[ROUNDTABLE] Config load failed: {e}")
    return RoundtableConfig()

def save_config(cfg: RoundtableConfig) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_PATH, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

def list_participants() -> list[dict]:
    return [p.to_dict() for p in load_config().participants]

def get_participant(pid: str) -> Optional[dict]:
    for p in load_config().participants:
        if p.id == pid:
            return p.to_dict()
    return None

def upsert_participant(data: dict) -> dict:
    cfg = load_config()
    pid = data.get("id")
    if pid:
        for i, p in enumerate(cfg.participants):
            if p.id == pid:
                cfg.participants[i] = Participant.from_dict(
                    {**p.to_dict(), **data})
                save_config(cfg)
                return cfg.participants[i].to_dict()
    if not pid:
        data["id"] = str(uuid4())[:8]
    cfg.participants.append(Participant.from_dict(data))
    save_config(cfg)
    return cfg.participants[-1].to_dict()

def remove_participant(pid: str) -> bool:
    cfg = load_config()
    before = len(cfg.participants)
    cfg.participants = [p for p in cfg.participants if p.id != pid]
    if len(cfg.participants) < before:
        save_config(cfg)
        return True
    return False

def reorder_participants(ordered_ids: list[str]) -> list[dict]:
    cfg = load_config()
    by_id = {p.id: p for p in cfg.participants}
    reordered = [by_id[pid] for pid in ordered_ids if pid in by_id]
    seen = set(ordered_ids)
    for p in cfg.participants:
        if p.id not in seen:
            reordered.append(p)
    cfg.participants = reordered
    save_config(cfg)
    return [p.to_dict() for p in cfg.participants]

def update_default_topic(topic: str) -> str:
    cfg = load_config()
    cfg.default_topic = topic
    save_config(cfg)
    return cfg.default_topic


# ────────────────────────────────────────────────────────────────
#  Generation
# ────────────────────────────────────────────────────────────────

def _generate(pipeline, system_prompt: str, user_content: str,
              temperature: float = 0.7, top_p: float = 0.9,
              max_tokens: int = 256) -> str:
    """Single generation via the loaded instruct model."""
    import torch

    model = pipeline.instruct_model
    tokenizer = pipeline.tokenizer
    device = pipeline.device

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    with _gen_lock:
        try:
            inputs = tokenizer.apply_chat_template(
                messages, return_tensors="pt",
                add_generation_prompt=True, return_dict=True,
            )
        except Exception:
            text = f"system: {system_prompt}\nuser: {user_content}\nassistant:"
            inputs = tokenizer(
                text, return_tensors="pt",
                add_special_tokens=(engine_config.get("add_special_tokens")
                                    if engine_config else False),
            )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=max(temperature, 0.01),
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

        reply = tokenizer.decode(out[0, prompt_len:],
                                 skip_special_tokens=True).strip()
    return reply


# ────────────────────────────────────────────────────────────────
#  CSV template parser
# ────────────────────────────────────────────────────────────────

def parse_template(csv_text: str, registry: list[Participant] = None
                   ) -> Workflow:
    """Parse a CSV template string into a Workflow.

    Header format: ``TYPE:Label`` or just ``TYPE``.
    Cell format:   JSON dict, or empty to skip.
    Reference:     ``{"ref": "<id_or_name>", ...}`` merges with registry.
    """
    registry = registry or []
    reg_by_id = {p.id: p for p in registry}
    reg_by_name = {p.name.lower(): p for p in registry}

    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)
    if len(rows) < 2:
        raise ValueError("Template must have a header row and at least "
                         "one data row.")

    # ── Parse header ────────────────────────────────────────────
    headers = rows[0]
    stages: list[WorkflowStage] = []
    for h in headers:
        h = h.strip()
        if ":" in h:
            stype, label = h.split(":", 1)
        else:
            stype, label = h, h
        stages.append(WorkflowStage(
            stage_type=stype.strip().upper(),
            label=label.strip(),
        ))

    # ── Parse data rows ─────────────────────────────────────────
    for row in rows[1:]:
        for col_idx, cell in enumerate(row):
            if col_idx >= len(stages):
                break
            cell = cell.strip()
            if not cell:
                continue

            try:
                seed = json.loads(cell)
            except json.JSONDecodeError:
                logger.warning(f"[ROUNDTABLE] Skipping malformed cell: "
                               f"{cell[:80]}")
                continue

            if not isinstance(seed, dict):
                continue

            # Resolve references
            ref = seed.pop("ref", None)
            if ref:
                participant = reg_by_id.get(ref)
                if not participant:
                    participant = reg_by_name.get(ref.lower())
                if participant:
                    base = participant.to_dict()
                    base.update(seed)
                    seed = base
                else:
                    logger.warning(f"[ROUNDTABLE] Ref '{ref}' not found "
                                   f"in registry, using cell as-is.")

            stages[col_idx].seeds.append(seed)

    return Workflow(stages=stages)


def workflow_to_csv(workflow: Workflow) -> str:
    """Serialize a Workflow back to CSV template text."""
    if not workflow.stages:
        return ""

    headers = []
    for s in workflow.stages:
        if s.label and s.label != s.stage_type:
            headers.append(f"{s.stage_type}:{s.label}")
        else:
            headers.append(s.stage_type)

    max_rows = max((len(s.seeds) for s in workflow.stages), default=0)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)

    for row_idx in range(max_rows):
        row = []
        for stage in workflow.stages:
            if row_idx < len(stage.seeds):
                row.append(json.dumps(stage.seeds[row_idx],
                                      ensure_ascii=False))
            else:
                row.append("")
        writer.writerow(row)

    return buf.getvalue()


# ────────────────────────────────────────────────────────────────
#  Default prompts for built-in analysis methods
# ────────────────────────────────────────────────────────────────

_DEFAULT_PROMPTS = {
    "synthesize": (
        "You are a skilled moderator and analyst.  You have just observed "
        "a roundtable discussion.  Synthesize the participants' "
        "contributions into a coherent analysis that identifies key "
        "themes, areas of agreement and disagreement, novel insights, "
        "and open questions.  Be thorough but concise."
    ),
    "analyze": (
        "You are a qualitative research analyst.  Examine the following "
        "discussion transcript and perform a systematic qualitative "
        "coding analysis.  Identify emergent codes, group them into "
        "themes, note frequency and co-occurrence patterns, and "
        "summarize your findings."
    ),
    "evaluate": (
        "You are a critical evaluator.  Review the following discussion "
        "and assess the quality, rigor, and completeness of each "
        "contribution.  Rate each participant's argument on clarity, "
        "evidence, and originality.  Summarize overall strengths and "
        "weaknesses."
    ),
    "extract": (
        "You are an information extraction specialist.  From the "
        "following discussion, extract: (1) key claims and assertions, "
        "(2) supporting evidence cited, (3) areas of consensus, "
        "(4) unresolved questions, (5) actionable recommendations.  "
        "Present each category clearly."
    ),
    "report": (
        "You are a technical writer.  Produce a structured report from "
        "the following material.  Include an executive summary, key "
        "findings, detailed analysis, and conclusions.  Use clear "
        "headings and professional tone."
    ),
}


# ────────────────────────────────────────────────────────────────
#  Module
# ────────────────────────────────────────────────────────────────

class RoundtableLMAModule(TASMModule):
    name = "roundtable_lma"
    display_name = "Roundtable LMA"
    description = (
        "Configurable Language Model Array pipeline.  Define workflows "
        "via CSV templates: columns are stages (PANEL or ANALYSIS), "
        "rows are agents, cells contain JSON seed parameters.  Each "
        "panel is a fresh roundtable; intermediate analysis stages "
        "transform and route data between panels.  Manage participant "
        "identities and templates via the Roundtable Configuration panel."
    )
    version = "2.0.0"

    min_results = 0
    requires_sfd = False
    requires_ltp = False
    requires_rd = False

    parameters = [
        ModuleParameter(
            name="n_roundtables",
            display_name="Roundtables (no-template mode)",
            description=(
                "Number of PANEL stages when running without a CSV "
                "template.  Ignored when a template is provided."
            ),
            type="int",
            default=2,
            min_val=1,
            max_val=20,
        ),
        ModuleParameter(
            name="participants_per_roundtable",
            display_name="Participants per Roundtable",
            description=(
                "Agents per panel in no-template mode.  Draws from "
                "active participant registry entries."
            ),
            type="int",
            default=3,
            min_val=1,
            max_val=20,
        ),
        ModuleParameter(
            name="temperature",
            display_name="Temperature (global override)",
            description=(
                "Global temperature applied to all agents unless a "
                "per-agent seed specifies its own.  0 = use per-seed "
                "values only."
            ),
            type="float",
            default=0.0,
        ),
        ModuleParameter(
            name="max_tokens",
            display_name="Max Tokens (global override)",
            description=(
                "Global max-tokens override.  0 = use per-seed values."
            ),
            type="int",
            default=0,
        ),
        ModuleParameter(
            name="save_transcripts",
            display_name="Save Transcripts to Disk",
            description="Persist transcripts to ~/.tagm/roundtable_transcripts/.",
            type="bool",
            default=True,
        ),
    ]

    # ── Registries (class-level, populated by decorators below) ─
    _stage_handlers: dict[str, Callable] = {}
    _analysis_methods: dict[str, dict] = {}

    def __init__(self):
        self._pipeline = None
        self._project_root = None

    def set_pipeline(self, pipeline):
        self._pipeline = pipeline
        _interactive_manager.set_pipeline(pipeline)

    def set_project_root(self, root):
        self._project_root = root

    # ── Extension decorators ────────────────────────────────────

    @classmethod
    def register_stage(cls, stage_type: str):
        """Decorator: register a handler for a new stage type.

        Handler signature::

            (module, stage: WorkflowStage, context: dict,
             progress: Callable) -> dict

        The returned dict must include ``"output"`` (str) and
        ``"n_generations"`` (int).
        """
        def decorator(fn):
            cls._stage_handlers[stage_type.upper()] = fn
            return fn
        return decorator

    @classmethod
    def register_method(cls, method_name: str, description: str = "",
                        default_prompt: str = ""):
        """Decorator: register a new analysis method.

        Handler signature::

            (module, prior_output: str, seed: dict,
             context: dict, progress: Callable) -> str
        """
        def decorator(fn):
            cls._analysis_methods[method_name] = {
                "handler": fn,
                "description": description,
                "default_prompt": default_prompt,
            }
            return fn
        return decorator

    # ── Seed resolution ─────────────────────────────────────────

    def _resolve_seed(self, seed: dict, global_overrides: dict) -> dict:
        """Merge: seed > global overrides > defaults."""
        resolved = dict(seed)

        g_temp = global_overrides.get("temperature", 0)
        g_maxt = global_overrides.get("max_tokens", 0)

        if "temperature" not in resolved and g_temp > 0:
            resolved["temperature"] = g_temp
        if "max_tokens" not in resolved and g_maxt > 0:
            resolved["max_tokens"] = g_maxt

        resolved.setdefault("temperature", 0.7)
        resolved.setdefault("max_tokens", 256)
        resolved.setdefault("top_p", 0.9)
        resolved.setdefault("name", "Agent")
        resolved.setdefault("role", "")
        resolved.setdefault("system_prompt",
                            "You are a helpful roundtable participant.")
        return resolved

    # ── Validate ────────────────────────────────────────────────

    def validate(self, session_results, params):
        if self._pipeline is None or not self._pipeline.loaded:
            return False, "No model loaded.  Load a model pair first."

        mode = params.get("mode", "interactive")

        if mode == "interactive":
            # Interactive mode: no topic or template needed upfront
            return True, "OK"

        # Batch mode: need template or participants + topic
        template_csv = params.get("template_csv", "")
        if not template_csv:
            cfg = load_config()
            active = [p for p in cfg.participants if p.active]
            if not active:
                return False, (
                    "No CSV template provided and no active participants "
                    "in the registry.  Upload a template or add "
                    "participants first."
                )

        topic = params.get("topic") or load_config().default_topic
        if not topic or not topic.strip():
            return False, "No discussion topic set."

        return True, "OK"

    # ── Run ─────────────────────────────────────────────────────

    def run(self, session_results, params, progress=None):
        mode = params.get("mode", "interactive")

        if mode == "interactive":
            return self._run_interactive(params, progress)
        else:
            return self._run_batch(params, progress)

    def _run_interactive(self, params, progress=None):
        """Configure and open the interactive roundtable chat window."""
        def prog(msg):
            if progress:
                progress(msg)

        config = {
            "temperature": float(params.get("temperature", 0.7)),
            "max_tokens": int(params.get("max_tokens", 256)),
            "top_p": float(params.get("top_p", 0.9)),
        }

        prog("Interactive roundtable configured — open chat window")

        return {
            "mode": "interactive",
            "config": config,
            "chat_url": "/roundtable-chat",
            "participants": list_participants(),
            "methods": self.list_methods(),
            "message": "Interactive roundtable ready.  Open the chat "
                       "window and type your inquiry to begin.",
        }

    def _run_batch(self, params, progress=None):
        """Execute a full batch pipeline (CSV template or auto-built)."""
        def prog(msg):
            if progress:
                progress(msg)

        t0 = time.time()
        cfg = load_config()

        # ── Resolve topic ───────────────────────────────────────
        topic = (params.get("topic") or cfg.default_topic).strip()
        if topic:
            cfg.default_topic = topic
            save_config(cfg)

        # ── Build workflow ──────────────────────────────────────
        template_csv = params.get("template_csv", "")
        if template_csv:
            prog("Parsing CSV template")
            workflow = parse_template(template_csv, cfg.participants)
        else:
            prog("Building workflow from active participants")
            workflow = self._workflow_from_params(params, cfg)

        workflow.topic = topic

        if not workflow.stages:
            return {"error": "Workflow has no stages."}

        prog(f"Workflow: {len(workflow.stages)} stages — "
             + " → ".join(f"{s.stage_type}:{s.label}"
                          for s in workflow.stages))

        # ── Global overrides ────────────────────────────────────
        global_overrides = {
            "temperature": float(params.get("temperature", 0)),
            "max_tokens": int(params.get("max_tokens", 0)),
        }
        save_transcripts = bool(params.get("save_transcripts", True))

        # ── Execute stages ──────────────────────────────────────
        run_id = time.strftime("%Y%m%d_%H%M%S")
        context = {
            "topic": topic,
            "prior_output": None,
            "prior_type": None,
            "run_id": run_id,
            "stage_index": 0,
            "global_overrides": global_overrides,
        }

        stage_results = []
        for idx, stage in enumerate(workflow.stages):
            context["stage_index"] = idx
            prog(f"Stage {idx + 1}/{len(workflow.stages)}: "
                 f"{stage.stage_type}:{stage.label}")

            handler = self._stage_handlers.get(stage.stage_type)
            if handler is None:
                msg = (f"Unknown stage type '{stage.stage_type}'.  "
                       f"Registered: {list(self._stage_handlers.keys())}")
                logger.error(f"[ROUNDTABLE] {msg}")
                stage_results.append({
                    "stage_index": idx,
                    "stage_type": stage.stage_type,
                    "label": stage.label,
                    "error": msg,
                    "output": context.get("prior_output", ""),
                    "n_generations": 0,
                })
                continue

            result = handler(self, stage, context, prog)
            stage_results.append(result)

            context["prior_output"] = result.get("output", "")
            context["prior_type"] = stage.stage_type

            if save_transcripts:
                _TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
                fname = (f"rt_{run_id}_{idx:02d}"
                         f"_{stage.stage_type.lower()}.json")
                try:
                    with open(_TRANSCRIPT_DIR / fname, "w") as f:
                        json.dump(result, f, indent=2)
                except Exception as e:
                    logger.warning(
                        f"[ROUNDTABLE] Transcript save failed: {e}")

        # ── Assemble result ─────────────────────────────────────
        elapsed = time.time() - t0
        prog(f"Pipeline complete in {elapsed:.1f}s")

        n_gen = sum(r.get("n_generations", 0) for r in stage_results)

        return {
            "run_id": run_id,
            "topic": topic,
            "workflow": workflow.to_dict(),
            "stages": stage_results,
            "final_output": context.get("prior_output", ""),
            "n_stages": len(workflow.stages),
            "n_total_generations": n_gen,
            "elapsed_seconds": round(elapsed, 2),
            "template_csv": template_csv or None,
        }

    # ── Workflow from UI params (no template) ───────────────────

    def _workflow_from_params(self, params, cfg):
        n_rt = int(params.get("n_roundtables", 2))
        n_per = int(params.get("participants_per_roundtable", 3))
        active = [p for p in cfg.participants if p.active]
        if n_per > len(active):
            n_per = len(active)

        synth_prompt = params.get("intermediate_prompt",
                                  _DEFAULT_PROMPTS["synthesize"])
        stages = []
        for i in range(n_rt):
            seeds = [p.to_dict() for p in active[:n_per]]
            stages.append(WorkflowStage("PANEL", f"Round {i + 1}", seeds))
            if i < n_rt - 1:
                stages.append(WorkflowStage("ANALYSIS", f"Synthesis {i + 1}", [{
                    "method": "synthesize",
                    "system_prompt": synth_prompt,
                }]))

        return Workflow(stages=stages)

    # ── Transcript management ───────────────────────────────────

    @staticmethod
    def list_transcripts() -> list[dict]:
        if not _TRANSCRIPT_DIR.exists():
            return []
        result = []
        for p in sorted(_TRANSCRIPT_DIR.glob("rt_*.json"), reverse=True):
            try:
                with open(p) as f:
                    data = json.load(f)
                result.append({
                    "filename": p.name,
                    "stage_type": data.get("stage_type"),
                    "label": data.get("label"),
                    "n_agents": len(data.get("responses", [])),
                })
            except Exception:
                result.append({"filename": p.name, "error": True})
        return result

    @staticmethod
    def get_transcript(filename: str) -> Optional[dict]:
        path = _TRANSCRIPT_DIR / filename
        if not path.exists() or not path.name.startswith("rt_"):
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def clear_transcripts() -> int:
        if not _TRANSCRIPT_DIR.exists():
            return 0
        count = 0
        for p in _TRANSCRIPT_DIR.glob("rt_*.json"):
            try:
                p.unlink()
                count += 1
            except Exception:
                pass
        return count

    @staticmethod
    def export_full_chain(run_id: str) -> Optional[dict]:
        if not _TRANSCRIPT_DIR.exists():
            return None
        parts = sorted(_TRANSCRIPT_DIR.glob(f"rt_{run_id}_*.json"))
        if not parts:
            return None
        stages = []
        for p in parts:
            try:
                with open(p) as f:
                    stages.append(json.load(f))
            except Exception:
                pass
        return {"run_id": run_id, "n_stages": len(stages), "stages": stages}

    @staticmethod
    def list_methods() -> list[dict]:
        """Return metadata for all registered analysis methods."""
        return [
            {
                "name": name,
                "description": info.get("description", ""),
                "has_default_prompt": bool(info.get("default_prompt")),
            }
            for name, info in RoundtableLMAModule._analysis_methods.items()
        ]

    @staticmethod
    def list_tools() -> list[dict]:
        """Return metadata for all registered TOOL functions."""
        return [
            {
                "name": name,
                "description": getattr(fn, "_tool_meta", {}).get(
                    "description", ""),
            }
            for name, fn in _TOOL_REGISTRY.items()
        ]


# ────────────────────────────────────────────────────────────────
#  Built-in stage handlers
# ────────────────────────────────────────────────────────────────

@RoundtableLMAModule.register_stage("PANEL")
def _stage_panel(module, stage: WorkflowStage, context: dict,
                 progress: Callable) -> dict:
    """Execute a roundtable panel.

    Each agent sees the *accumulating* transcript of this panel only.
    Prior panel transcripts are never exposed — blank-canvas isolation.
    """
    topic = context["topic"]
    prior = context.get("prior_output")
    overrides = context.get("global_overrides", {})
    stage_idx = context["stage_index"]

    # Panel input: prior stage output if it exists, else raw topic
    panel_input = prior if (prior and stage_idx > 0) else topic

    transcript_parts = []
    responses = []
    n_gen = 0

    for a_idx, raw_seed in enumerate(stage.seeds):
        seed = module._resolve_seed(raw_seed, overrides)
        agent_label = seed["name"]
        agent_role = seed["role"]
        progress(f"  {stage.label} — {agent_label} "
                 f"({a_idx + 1}/{len(stage.seeds)})")

        # Build user content: panel input + transcript so far
        if transcript_parts:
            user_content = (
                f"Discussion topic:\n{panel_input}\n\n"
                f"Transcript so far:\n"
                + "\n".join(transcript_parts)
                + "\n\nPlease contribute your perspective."
            )
        else:
            user_content = panel_input

        try:
            response = _generate(
                module._pipeline,
                system_prompt=seed["system_prompt"],
                user_content=user_content,
                temperature=seed["temperature"],
                top_p=seed["top_p"],
                max_tokens=seed["max_tokens"],
            )
            n_gen += 1
        except Exception as e:
            logger.error(f"[ROUNDTABLE] {agent_label} failed: {e}")
            response = f"[Generation error: {e}]"

        responses.append({
            "agent_index": a_idx,
            "name": agent_label,
            "role": agent_role,
            "response": response,
            "seed": {k: v for k, v in seed.items()
                     if k != "system_prompt"},
        })

        role_tag = (f"{agent_label} — {agent_role}"
                    if agent_role else agent_label)
        transcript_parts.append(f"[{role_tag}]\n{response}\n")

    transcript = "\n".join(transcript_parts)

    return {
        "stage_index": stage_idx,
        "stage_type": "PANEL",
        "label": stage.label,
        "input_preview": panel_input[:500],
        "responses": responses,
        "transcript": transcript,
        "output": transcript,
        "n_generations": n_gen,
    }


@RoundtableLMAModule.register_stage("ANALYSIS")
def _stage_analysis(module, stage: WorkflowStage, context: dict,
                    progress: Callable) -> dict:
    """Execute an analysis / transformation stage.

    Dispatches to the method named in each seed's ``method`` field.
    Multiple seeds in one ANALYSIS column are chained: each processes
    the prior's output in sequence.
    """
    prior = context.get("prior_output", "")
    overrides = context.get("global_overrides", {})

    if not stage.seeds:
        return {
            "stage_index": context["stage_index"],
            "stage_type": "ANALYSIS",
            "label": stage.label,
            "method_chain": ["passthrough"],
            "output": prior,
            "n_generations": 0,
        }

    outputs = []
    current = prior
    n_gen = 0

    for s_idx, raw_seed in enumerate(stage.seeds):
        seed = module._resolve_seed(raw_seed, overrides)
        method_name = raw_seed.get("method", "custom")
        progress(f"  {stage.label} — method:{method_name} "
                 f"({s_idx + 1}/{len(stage.seeds)})")

        method_info = module._analysis_methods.get(method_name)
        if method_info is None:
            logger.warning(f"[ROUNDTABLE] Unknown method '{method_name}', "
                           f"falling back to 'custom'.")
            method_info = module._analysis_methods.get("custom")

        handler = method_info["handler"]

        # Apply default prompt if seed doesn't specify one
        if ("system_prompt" not in raw_seed
                and method_info.get("default_prompt")):
            seed["system_prompt"] = method_info["default_prompt"]

        result_text = handler(module, current, seed, context, progress)
        outputs.append({"method": method_name, "output": result_text})

        # Count model-based methods as generations
        if method_name not in ("passthrough", "aggregate"):
            n_gen += 1

        current = result_text

    return {
        "stage_index": context["stage_index"],
        "stage_type": "ANALYSIS",
        "label": stage.label,
        "method_chain": [o["method"] for o in outputs],
        "outputs": outputs,
        "output": current,
        "n_generations": n_gen,
    }


@RoundtableLMAModule.register_stage("TOOL")
def _stage_tool(module, stage: WorkflowStage, context: dict,
                progress: Callable) -> dict:
    """Execute a programmatic tool stage — no model generation.

    Dispatches to registered tool functions by the ``tool`` field in
    the seed dict.  Tools operate on the pipeline state (prior output,
    context, run_id) and produce structured output or side effects
    (file export, external API calls, data transforms).
    """
    prior = context.get("prior_output", "")
    run_id = context.get("run_id", "unknown")
    stage_idx = context["stage_index"]

    if not stage.seeds:
        return {
            "stage_index": stage_idx,
            "stage_type": "TOOL",
            "label": stage.label,
            "tool_chain": ["passthrough"],
            "output": prior,
            "n_generations": 0,
        }

    outputs = []
    current = prior

    for s_idx, seed in enumerate(stage.seeds):
        tool_name = seed.get("tool", "export_json")
        progress(f"  {stage.label} — tool:{tool_name} "
                 f"({s_idx + 1}/{len(stage.seeds)})")

        handler = _TOOL_REGISTRY.get(tool_name)
        if handler is None:
            msg = (f"Unknown tool '{tool_name}'.  "
                   f"Available: {list(_TOOL_REGISTRY.keys())}")
            logger.warning(f"[ROUNDTABLE] {msg}")
            outputs.append({"tool": tool_name, "error": msg})
            continue

        try:
            result = handler(current, seed, context, progress)
        except Exception as e:
            logger.error(f"[ROUNDTABLE] Tool {tool_name} failed: {e}")
            result = {"error": str(e)}

        outputs.append({"tool": tool_name, "result": result})

        # Tools that produce text output update current for downstream
        if isinstance(result, dict) and "output" in result:
            current = result["output"]
        elif isinstance(result, str):
            current = result

    return {
        "stage_index": stage_idx,
        "stage_type": "TOOL",
        "label": stage.label,
        "tool_chain": [o["tool"] for o in outputs],
        "outputs": outputs,
        "output": current,
        "n_generations": 0,
    }


# ── Tool registry ──────────────────────────────────────────────

_TOOL_REGISTRY: dict[str, Callable] = {}


def register_tool(name: str, description: str = ""):
    """Decorator to register a TOOL-stage function.

    Handler signature::
        (prior_output: str, seed: dict, context: dict,
         progress: Callable) -> dict | str
    """
    def decorator(fn):
        _TOOL_REGISTRY[name] = fn
        fn._tool_meta = {"name": name, "description": description}
        return fn
    return decorator


@register_tool("export_json", "Export current context as a JSON file")
def _tool_export_json(prior_output, seed, context, progress):
    """Write the accumulated pipeline state to a JSON file."""
    run_id = context.get("run_id", "export")
    fname = seed.get("filename", f"roundtable_{run_id}.json")
    path = _TRANSCRIPT_DIR / fname
    _TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "topic": context.get("topic", ""),
        "stage_index": context.get("stage_index"),
        "output": prior_output,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    progress(f"    Exported to {path}")
    return {"output": prior_output, "file": str(path)}


@register_tool("snapshot", "Save a named checkpoint of the current output")
def _tool_snapshot(prior_output, seed, context, progress):
    """Save current output as a named snapshot for later reference."""
    label = seed.get("label", f"snapshot_{context.get('stage_index', 0)}")
    snapshots = context.setdefault("_snapshots", {})
    snapshots[label] = prior_output
    progress(f"    Snapshot saved: {label}")
    return {"output": prior_output, "snapshot": label}


@register_tool("truncate", "Trim the output to a max character length")
def _tool_truncate(prior_output, seed, context, progress):
    max_chars = int(seed.get("max_chars", 2000))
    if len(prior_output) > max_chars:
        trimmed = prior_output[:max_chars] + "\n\n[... truncated]"
        progress(f"    Truncated {len(prior_output)} → {max_chars} chars")
        return trimmed
    return prior_output


@register_tool("prepend", "Prepend fixed text to the output")
def _tool_prepend(prior_output, seed, context, progress):
    text = seed.get("text", "")
    return text + "\n\n" + prior_output


@register_tool("append", "Append fixed text to the output")
def _tool_append(prior_output, seed, context, progress):
    text = seed.get("text", "")
    return prior_output + "\n\n" + text


@register_tool("word_count", "Count words and basic stats (like ClownCar.AI)")
def _tool_word_count(prior_output, seed, context, progress):
    """ClownCar.AI-style word counting on the current output."""
    words = prior_output.lower().split()
    freq = Counter(words).most_common(30)
    lines = [
        f"Total words: {len(words)}",
        f"Unique words: {len(set(words))}",
        f"Top 30: " + ", ".join(f"{w}({c})" for w, c in freq),
    ]
    report = "\n".join(lines)
    return {"output": report, "total": len(words), "unique": len(set(words))}


# ────────────────────────────────────────────────────────────────
#  Built-in analysis methods
# ────────────────────────────────────────────────────────────────

@RoundtableLMAModule.register_method(
    "synthesize",
    description="Model-based synthesis of prior stage output",
    default_prompt=_DEFAULT_PROMPTS["synthesize"],
)
def _method_synthesize(module, prior_output, seed, context, progress):
    return _generate(
        module._pipeline, seed["system_prompt"], prior_output,
        seed["temperature"], seed["top_p"], seed["max_tokens"],
    )


@RoundtableLMAModule.register_method(
    "analyze",
    description="Qualitative coding analysis of discussion",
    default_prompt=_DEFAULT_PROMPTS["analyze"],
)
def _method_analyze(module, prior_output, seed, context, progress):
    return _generate(
        module._pipeline, seed["system_prompt"], prior_output,
        seed["temperature"], seed["top_p"], seed["max_tokens"],
    )


@RoundtableLMAModule.register_method(
    "evaluate",
    description="Critical evaluation and scoring of contributions",
    default_prompt=_DEFAULT_PROMPTS["evaluate"],
)
def _method_evaluate(module, prior_output, seed, context, progress):
    return _generate(
        module._pipeline, seed["system_prompt"], prior_output,
        seed["temperature"], seed["top_p"], seed["max_tokens"],
    )


@RoundtableLMAModule.register_method(
    "extract",
    description="Extract key themes, claims, and evidence",
    default_prompt=_DEFAULT_PROMPTS["extract"],
)
def _method_extract(module, prior_output, seed, context, progress):
    return _generate(
        module._pipeline, seed["system_prompt"], prior_output,
        seed["temperature"], seed["top_p"], seed["max_tokens"],
    )


@RoundtableLMAModule.register_method(
    "report",
    description="Generate a structured report from accumulated material",
    default_prompt=_DEFAULT_PROMPTS["report"],
)
def _method_report(module, prior_output, seed, context, progress):
    return _generate(
        module._pipeline, seed["system_prompt"], prior_output,
        seed["temperature"], seed["top_p"], seed["max_tokens"],
    )


@RoundtableLMAModule.register_method(
    "custom",
    description="Generation with a user-defined system prompt",
    default_prompt="You are a helpful analyst.  Process the following input.",
)
def _method_custom(module, prior_output, seed, context, progress):
    return _generate(
        module._pipeline, seed["system_prompt"], prior_output,
        seed["temperature"], seed["top_p"], seed["max_tokens"],
    )


@RoundtableLMAModule.register_method(
    "passthrough",
    description="Forward prior output without modification",
)
def _method_passthrough(module, prior_output, seed, context, progress):
    return prior_output


@RoundtableLMAModule.register_method(
    "aggregate",
    description="Structural aggregation: word counts, response stats "
                "(no model generation)",
)
def _method_aggregate(module, prior_output, seed, context, progress):
    """Non-model processing: structural statistics."""
    lines = prior_output.strip().split("\n")

    # Identify agent blocks by [Name] headers
    blocks = []
    current_block = None
    for line in lines:
        if line.startswith("[") and "]" in line:
            if current_block:
                blocks.append(current_block)
            current_block = {"header": line, "text": ""}
        elif current_block is not None:
            current_block["text"] += line + "\n"
    if current_block:
        blocks.append(current_block)

    # Per-block stats
    block_stats = []
    all_words = []
    for b in blocks:
        words = b["text"].lower().split()
        all_words.extend(words)
        block_stats.append({
            "agent": b["header"],
            "word_count": len(words),
            "char_count": len(b["text"]),
            "sentence_count": (b["text"].count(".")
                               + b["text"].count("!")
                               + b["text"].count("?")),
        })

    word_freq = Counter(all_words).most_common(20)

    report_lines = [
        "=== Aggregation Report ===",
        f"Total responses: {len(blocks)}",
        f"Total words: {len(all_words)}",
        f"Average words per response: "
        f"{len(all_words) / max(len(blocks), 1):.0f}",
        "",
        "--- Per-Agent Statistics ---",
    ]
    for bs in block_stats:
        report_lines.append(
            f"  {bs['agent']}: {bs['word_count']} words, "
            f"{bs['sentence_count']} sentences"
        )
    report_lines.extend([
        "",
        "--- Top 20 Words ---",
        ", ".join(f"{w}({c})" for w, c in word_freq),
    ])

    return "\n".join(report_lines)


# ────────────────────────────────────────────────────────────────
#  Interactive Session — user-driven roundtable from a chat window
# ────────────────────────────────────────────────────────────────

@dataclass
class InteractiveTurn:
    """One turn in the interactive transcript."""
    turn_type: str              # "user", "persona", "method"
    name: str                   # user name or persona name or method name
    role: str                   # persona role or "" for user/method
    content: str                # the message text
    timestamp: float = 0.0     # epoch seconds
    seed: Optional[dict] = None # persona seed used (sans system_prompt)
    stage_label: str = ""       # which stage this turn belongs to

    def to_dict(self) -> dict:
        d = {
            "turn_type": self.turn_type,
            "name": self.name,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "stage_label": self.stage_label,
        }
        if self.seed:
            d["seed"] = self.seed
        return d


@dataclass
class InteractiveSession:
    """In-memory state for a user-driven roundtable session."""
    session_id: str
    topic: str
    turns: list[InteractiveTurn] = field(default_factory=list)
    stages: list[dict] = field(default_factory=list)
    current_stage_label: str = "Panel 1"
    current_stage_type: str = "PANEL"
    stage_counter: int = 1
    gen_config: dict = field(default_factory=lambda: {
        "temperature": 0.7,
        "max_tokens": 256,
        "top_p": 0.9,
    })

    def transcript_text(self, current_stage_only: bool = True) -> str:
        """Build readable transcript.

        If current_stage_only=True, returns only the current stage's
        turns (blank-canvas isolation between panels).
        """
        turns = self.turns
        if current_stage_only:
            turns = [t for t in turns
                     if t.stage_label == self.current_stage_label]

        lines = []
        for t in turns:
            if t.turn_type == "user":
                lines.append(f"[User]")
            elif t.turn_type == "persona":
                tag = f"{t.name} — {t.role}" if t.role else t.name
                lines.append(f"[{tag}]")
            elif t.turn_type == "method":
                lines.append(f"[{t.name} — analysis]")
            lines.append(t.content)
            lines.append("")
        return "\n".join(lines)

    def full_transcript_text(self) -> str:
        """Full transcript across all stages, with stage headers."""
        if not self.turns:
            return ""
        lines = []
        current_label = None
        for t in self.turns:
            if t.stage_label != current_label:
                current_label = t.stage_label
                lines.append(f"\n{'='*40}")
                lines.append(f"  {current_label}")
                lines.append(f"{'='*40}\n")
            if t.turn_type == "user":
                lines.append("[User]")
            elif t.turn_type == "persona":
                tag = f"{t.name} — {t.role}" if t.role else t.name
                lines.append(f"[{tag}]")
            elif t.turn_type == "method":
                lines.append(f"[{t.name} — analysis]")
            lines.append(t.content)
            lines.append("")
        return "\n".join(lines)

    def mark_new_stage(self, stage_type: str = "PANEL",
                       label: str = "") -> dict:
        """Close the current stage and start a new one."""
        # Save current stage summary
        current_turns = [t for t in self.turns
                         if t.stage_label == self.current_stage_label]
        self.stages.append({
            "stage_type": self.current_stage_type,
            "label": self.current_stage_label,
            "n_turns": len(current_turns),
            "transcript": self.transcript_text(current_stage_only=True),
        })

        self.stage_counter += 1
        if not label:
            label = f"{stage_type.title()} {self.stage_counter}"
        self.current_stage_label = label
        self.current_stage_type = stage_type.upper()

        return {
            "stage_index": len(self.stages),
            "stage_type": self.current_stage_type,
            "label": self.current_stage_label,
        }

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "current_stage": {
                "type": self.current_stage_type,
                "label": self.current_stage_label,
            },
            "n_turns": len(self.turns),
            "n_stages_completed": len(self.stages),
            "turns": [t.to_dict() for t in self.turns],
            "stages": self.stages,
            "gen_config": self.gen_config,
        }

    def export(self) -> dict:
        """Full export for download/archival."""
        # Close current stage into the summary
        current_turns = [t for t in self.turns
                         if t.stage_label == self.current_stage_label]
        all_stages = list(self.stages) + [{
            "stage_type": self.current_stage_type,
            "label": self.current_stage_label,
            "n_turns": len(current_turns),
            "transcript": self.transcript_text(current_stage_only=True),
        }]

        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "stages": all_stages,
            "full_transcript": self.full_transcript_text(),
            "turns": [t.to_dict() for t in self.turns],
            "gen_config": self.gen_config,
            "n_total_turns": len(self.turns),
        }


class InteractiveSessionManager:
    """Thread-safe manager for the active interactive session.

    Only one interactive session exists at a time (matches the
    single-model, single-user TAGM deployment model).
    """

    def __init__(self):
        self._session: Optional[InteractiveSession] = None
        self._lock = threading.Lock()
        self._pipeline = None

    def set_pipeline(self, pipeline):
        self._pipeline = pipeline

    @property
    def active(self) -> bool:
        return self._session is not None

    def start(self, topic: str, gen_config: dict = None) -> dict:
        """Start a new interactive session."""
        with self._lock:
            sid = time.strftime("%Y%m%d_%H%M%S")
            self._session = InteractiveSession(
                session_id=sid,
                topic=topic,
            )
            if gen_config:
                self._session.gen_config.update(gen_config)

            # Record the user's opening inquiry as the first turn
            self._session.turns.append(InteractiveTurn(
                turn_type="user",
                name="User",
                role="",
                content=topic,
                timestamp=time.time(),
                stage_label=self._session.current_stage_label,
            ))

            return self._session.to_dict()

    def get_session(self) -> Optional[dict]:
        with self._lock:
            if self._session is None:
                return None
            return self._session.to_dict()

    def send_user_message(self, message: str) -> dict:
        """Add a user message to the transcript."""
        with self._lock:
            if self._session is None:
                return {"ok": False, "error": "No active session."}
            self._session.turns.append(InteractiveTurn(
                turn_type="user",
                name="User",
                role="",
                content=message,
                timestamp=time.time(),
                stage_label=self._session.current_stage_label,
            ))
            return {"ok": True, "n_turns": len(self._session.turns)}

    def apply_persona(self, participant_id: str = None,
                      inline_seed: dict = None) -> dict:
        """Generate a response from a specific persona.

        Either provide a participant_id to look up from registry,
        or an inline_seed dict with name/role/system_prompt.
        """
        if self._pipeline is None:
            return {"ok": False, "error": "No model loaded."}

        with self._lock:
            if self._session is None:
                return {"ok": False, "error": "No active session."}
            session = self._session

        # Resolve persona seed
        if inline_seed:
            seed = dict(inline_seed)
        elif participant_id:
            p = get_participant(participant_id)
            if p is None:
                return {"ok": False,
                        "error": f"Participant '{participant_id}' not found."}
            seed = dict(p)
        else:
            return {"ok": False, "error": "No persona specified."}

        # Apply session gen_config as defaults
        seed.setdefault("temperature", session.gen_config.get("temperature", 0.7))
        seed.setdefault("max_tokens", session.gen_config.get("max_tokens", 256))
        seed.setdefault("top_p", session.gen_config.get("top_p", 0.9))
        seed.setdefault("system_prompt", "You are a helpful roundtable participant.")
        seed.setdefault("name", "Agent")
        seed.setdefault("role", "")

        # Build user content: topic + current stage transcript
        transcript = session.transcript_text(current_stage_only=True)
        if transcript.strip():
            user_content = (
                f"Discussion topic:\n{session.topic}\n\n"
                f"Transcript so far:\n{transcript}\n\n"
                f"Please contribute your perspective."
            )
        else:
            user_content = session.topic

        # Generate
        try:
            response = _generate(
                self._pipeline,
                system_prompt=seed["system_prompt"],
                user_content=user_content,
                temperature=seed["temperature"],
                top_p=seed["top_p"],
                max_tokens=seed["max_tokens"],
            )
        except Exception as e:
            return {"ok": False, "error": f"Generation failed: {e}"}

        # Record turn
        with self._lock:
            seed_display = {k: v for k, v in seed.items()
                           if k != "system_prompt"}
            session.turns.append(InteractiveTurn(
                turn_type="persona",
                name=seed["name"],
                role=seed["role"],
                content=response,
                timestamp=time.time(),
                seed=seed_display,
                stage_label=session.current_stage_label,
            ))

        return {
            "ok": True,
            "name": seed["name"],
            "role": seed["role"],
            "response": response,
            "n_turns": len(session.turns),
        }

    def apply_method(self, method_name: str,
                     system_prompt: str = None) -> dict:
        """Run an analysis method on the current stage's transcript."""
        if self._pipeline is None and method_name not in ("passthrough",
                                                          "aggregate"):
            return {"ok": False, "error": "No model loaded."}

        with self._lock:
            if self._session is None:
                return {"ok": False, "error": "No active session."}
            session = self._session

        method_info = RoundtableLMAModule._analysis_methods.get(method_name)
        if method_info is None:
            return {"ok": False,
                    "error": f"Unknown method '{method_name}'. "
                    f"Available: {list(RoundtableLMAModule._analysis_methods.keys())}"}

        handler = method_info["handler"]

        # Build seed
        seed = {
            "temperature": session.gen_config.get("temperature", 0.7),
            "max_tokens": session.gen_config.get("max_tokens", 256),
            "top_p": session.gen_config.get("top_p", 0.9),
            "system_prompt": (system_prompt
                              or method_info.get("default_prompt")
                              or "Process the following input."),
        }

        # Use current stage transcript as input
        transcript = session.transcript_text(current_stage_only=True)
        if not transcript.strip():
            return {"ok": False, "error": "No transcript content to process."}

        # Create a minimal module-like object for the handler
        class _Stub:
            pass
        stub = _Stub()
        stub._pipeline = self._pipeline

        try:
            result = handler(stub, transcript, seed, {}, lambda m: None)
        except Exception as e:
            return {"ok": False, "error": f"Method failed: {e}"}

        # Record turn
        with self._lock:
            session.turns.append(InteractiveTurn(
                turn_type="method",
                name=method_name,
                role="analysis",
                content=result,
                timestamp=time.time(),
                stage_label=session.current_stage_label,
            ))

        return {
            "ok": True,
            "method": method_name,
            "output": result,
            "n_turns": len(session.turns),
        }

    def apply_tool(self, tool_name: str, params: dict = None) -> dict:
        """Run a TOOL-stage function on the current transcript."""
        with self._lock:
            if self._session is None:
                return {"ok": False, "error": "No active session."}
            session = self._session

        handler = _TOOL_REGISTRY.get(tool_name)
        if handler is None:
            return {"ok": False,
                    "error": f"Unknown tool '{tool_name}'. "
                    f"Available: {list(_TOOL_REGISTRY.keys())}"}

        transcript = session.transcript_text(current_stage_only=True)
        if not transcript.strip():
            return {"ok": False, "error": "No transcript content."}

        seed = params or {}
        context = {
            "run_id": session.session_id,
            "topic": session.topic,
            "stage_index": len(session.stages),
        }

        try:
            result = handler(transcript, seed, context, lambda m: None)
        except Exception as e:
            return {"ok": False, "error": f"Tool failed: {e}"}

        # Record turn
        output_text = result if isinstance(result, str) else json.dumps(result)
        with self._lock:
            session.turns.append(InteractiveTurn(
                turn_type="method",
                name=f"tool:{tool_name}",
                role="tool",
                content=output_text,
                timestamp=time.time(),
                stage_label=session.current_stage_label,
            ))

        return {
            "ok": True,
            "tool": tool_name,
            "result": result,
            "n_turns": len(session.turns),
        }

    def new_stage(self, stage_type: str = "PANEL",
                  label: str = "") -> dict:
        """Close current stage, start a new one (blank canvas)."""
        with self._lock:
            if self._session is None:
                return {"ok": False, "error": "No active session."}
            info = self._session.mark_new_stage(stage_type, label)
            return {"ok": True, **info}

    def update_config(self, config: dict) -> dict:
        """Update generation config for the session."""
        with self._lock:
            if self._session is None:
                return {"ok": False, "error": "No active session."}
            self._session.gen_config.update(config)
            return {"ok": True, "gen_config": self._session.gen_config}

    def export(self) -> Optional[dict]:
        with self._lock:
            if self._session is None:
                return None
            return self._session.export()

    def reset(self) -> dict:
        with self._lock:
            had = self._session is not None
            self._session = None
            return {"ok": True, "had_session": had}


# Module-level singleton
_interactive_manager = InteractiveSessionManager()
