"""Roundtable LMA — configurable Language Model Array pipeline.

Two execution paths through the same infrastructure:

1. **Interactive** — click Run (no template) → chat window opens at
   /roundtable.  User types inquiry, selects personas, runs methods,
   manages stages step by step.

2. **Batch** — upload a CSV template via the file picker in the module
   panel → pipeline executes automatically.  Columns are stages
   (PANEL / ANALYSIS / TOOL), rows are agent seeds, cells are JSON
   dicts.  The module marches through columns left to right.

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

from .base import TASMModule, ModuleParameter

try:
    from src.engine import config as engine_config
except ImportError:
    engine_config = None

logger = logging.getLogger("src")

_CONFIG_DIR = Path.home() / ".tagm"
_CONFIG_PATH = _CONFIG_DIR / "roundtable_config.json"
_TRANSCRIPT_DIR = _CONFIG_DIR / "roundtable_transcripts"
_gen_lock = threading.Lock()


# ================================================================
#  1. Participant registry
# ================================================================

@dataclass
class Participant:
    id: str; name: str; role: str; system_prompt: str; active: bool = True
    def to_dict(self): return asdict(self)
    @classmethod
    def from_dict(cls, d):
        return cls(d.get("id",str(uuid4())[:8]), d.get("name","Unnamed"),
                   d.get("role",""), d.get("system_prompt","You are a helpful roundtable participant."),
                   d.get("active",True))

@dataclass
class RoundtableConfig:
    participants: list[Participant] = field(default_factory=list)
    default_topic: str = ""
    def to_dict(self):
        return {"participants":[p.to_dict() for p in self.participants], "default_topic":self.default_topic}
    @classmethod
    def from_dict(cls, d):
        return cls([Participant.from_dict(p) for p in d.get("participants",[])], d.get("default_topic",""))

def load_config():
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH) as f: return RoundtableConfig.from_dict(json.load(f))
        except Exception: pass
    return RoundtableConfig()

def save_config(cfg):
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_PATH,"w") as f: json.dump(cfg.to_dict(), f, indent=2)

def list_participants(): return [p.to_dict() for p in load_config().participants]

def upsert_participant(data):
    cfg = load_config(); pid = data.get("id")
    if pid:
        for i,p in enumerate(cfg.participants):
            if p.id == pid:
                cfg.participants[i] = Participant.from_dict({**p.to_dict(), **data})
                save_config(cfg); return cfg.participants[i].to_dict()
    if not pid: data["id"] = str(uuid4())[:8]
    cfg.participants.append(Participant.from_dict(data))
    save_config(cfg); return cfg.participants[-1].to_dict()

def remove_participant(pid):
    cfg = load_config(); n = len(cfg.participants)
    cfg.participants = [p for p in cfg.participants if p.id != pid]
    if len(cfg.participants) < n: save_config(cfg); return True
    return False

def update_default_topic(topic):
    cfg = load_config(); cfg.default_topic = topic; save_config(cfg); return topic


# ================================================================
#  2. Workflow dataclasses
# ================================================================

@dataclass
class WorkflowStage:
    stage_type: str   # PANEL, ANALYSIS, TOOL
    label: str
    seeds: list[dict] = field(default_factory=list)

@dataclass
class Workflow:
    stages: list[WorkflowStage] = field(default_factory=list)
    topic: str = ""
    def to_dict(self):
        return {"topic":self.topic, "stages":[
            {"stage_type":s.stage_type,"label":s.label,"n_agents":len(s.seeds),"seeds":s.seeds}
            for s in self.stages]}


# ================================================================
#  3. CSV template parser
# ================================================================

def parse_template(csv_text, registry=None):
    """Parse CSV → Workflow.  Headers: TYPE or TYPE:Label.  Cells: JSON dicts."""
    registry = registry or []
    by_id = {p.id: p for p in registry}
    by_name = {p.name.lower(): p for p in registry}

    rows = list(csv.reader(io.StringIO(csv_text)))
    if len(rows) < 2:
        raise ValueError("Template needs a header row and at least one data row.")

    stages = []
    for h in rows[0]:
        h = h.strip()
        stype, label = (h.split(":",1) if ":" in h else (h, h))
        stages.append(WorkflowStage(stype.strip().upper(), label.strip()))

    for row in rows[1:]:
        for col, cell in enumerate(row):
            if col >= len(stages): break
            cell = cell.strip()
            if not cell: continue
            try: seed = json.loads(cell)
            except json.JSONDecodeError: continue
            if not isinstance(seed, dict): continue
            ref = seed.pop("ref", None)
            if ref:
                p = by_id.get(ref) or by_name.get(ref.lower())
                if p: base = p.to_dict(); base.update(seed); seed = base
            stages[col].seeds.append(seed)

    return Workflow(stages=stages)


# ================================================================
#  4. Generation
# ================================================================

def _generate(pipeline, system_prompt, user_content,
              temperature=0.7, top_p=0.9, max_tokens=256):
    import torch
    model, tok, dev = pipeline.instruct_model, pipeline.tokenizer, pipeline.device
    msgs = [{"role":"system","content":system_prompt},{"role":"user","content":user_content}]
    with _gen_lock:
        try:
            inp = tok.apply_chat_template(msgs, return_tensors="pt",
                      add_generation_prompt=True, return_dict=True)
        except Exception:
            inp = tok(f"system: {system_prompt}\nuser: {user_content}\nassistant:",
                      return_tensors="pt",
                      add_special_tokens=engine_config.get("add_special_tokens") if engine_config else False)
        inp = {k:v.to(dev) for k,v in inp.items()}
        pl = inp["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=True,
                      temperature=max(temperature,0.01), top_p=top_p,
                      pad_token_id=tok.eos_token_id)
        return tok.decode(out[0,pl:], skip_special_tokens=True).strip()


# ================================================================
#  5. Default prompts
# ================================================================

_PROMPTS = {
    "synthesize": "You are a skilled analyst. Synthesize the discussion: key themes, agreement, disagreement, open questions.",
    "analyze": "You are a qualitative researcher. Code the discussion: emergent themes, frequency, co-occurrence.",
    "evaluate": "You are a critical evaluator. Assess each contribution for clarity, evidence, originality.",
    "extract": "Extract: (1) key claims, (2) evidence, (3) consensus, (4) open questions, (5) recommendations.",
    "report": "Produce a structured report: executive summary, findings, analysis, conclusions.",
}


# ================================================================
#  6. Module class
# ================================================================

class RoundtableLMAModule(TASMModule):
    name = "roundtable_lma"
    display_name = "Roundtable LMA"
    description = (
        "Configurable Language Model Array pipeline. Define workflows "
        "via CSV templates: columns are stages (PANEL, ANALYSIS, or TOOL), "
        "rows are agents, cells contain JSON seed parameters. Each panel "
        "is a fresh roundtable; intermediate stages transform and route "
        "data between panels. Click Run to open the interactive roundtable, "
        "or upload a CSV template for batch execution."
    )
    version = "2.0.0"
    min_results = 0
    requires_sfd = False
    requires_ltp = False
    requires_rd = False

    parameters = [
        ModuleParameter("template_csv","Workflow Template (CSV)",
            "Upload a CSV template for batch execution. Leave empty to open "
            "the interactive roundtable window instead.", "file", ""),
        ModuleParameter("topic","Discussion Topic",
            "The inquiry for the roundtable. Required for batch mode; "
            "in interactive mode you type it into the chat.", "textarea", ""),
        ModuleParameter("temperature","Temperature",
            "Sampling temperature.", "float", 0.7),
        ModuleParameter("max_tokens","Max Tokens",
            "Max tokens per generation.", "int", 256),
    ]

    _stage_handlers: dict[str, Callable] = {}
    _analysis_methods: dict[str, dict] = {}

    def __init__(self):
        self._pipeline = None

    def set_pipeline(self, pipeline):
        self._pipeline = pipeline
        _interactive_manager.set_pipeline(pipeline)

    def validate(self, session_results, params):
        return True, "OK"

    def run(self, session_results, params, progress=None):
        """Template uploaded → batch.  No template → open chat window."""
        template = params.get("template_csv", "")
        if template:
            return self._run_batch(template, params, progress)
        return self._run_interactive(params, progress)

    # ── Interactive: save config, return chat_url ───────────────

    def _run_interactive(self, params, progress=None):
        if progress: progress("Configuring roundtable")
        cfg = load_config()
        topic = params.get("topic","")
        if topic and topic.strip():
            cfg.default_topic = topic.strip(); save_config(cfg)
        config = {"temperature":float(params.get("temperature",0.7)),
                  "max_tokens":int(params.get("max_tokens",256))}
        config_path = Path(__file__).parent.parent.parent.parent / "roundtable_chat_config.json"
        try:
            with open(config_path,"w") as f: json.dump(config, f, indent=2)
        except Exception: pass
        active = [p for p in cfg.participants if p.active]
        if progress: progress(f"{len(active)} participants, ready")
        return {"config":config, "chat_url":"/roundtable",
                "participants":[p.to_dict() for p in active],
                "n_participants":len(active), "topic":cfg.default_topic,
                "message":"Roundtable configured. Click 'Open Roundtable' to start."}

    # ── Batch: resolve template, execute pipeline ──────────────

    def _run_batch(self, template_ref, params, progress=None):
        def prog(m):
            if progress: progress(m)

        # Resolve: filename from upload → read file.  Or raw CSV text.
        if "\n" not in template_ref and "," not in template_ref:
            path = Path(__file__).parent.parent.parent.parent / "templates" / template_ref
            if path.exists():
                prog(f"Loading template: {template_ref}")
                csv_text = path.read_text(encoding="utf-8")
            else:
                return {"error": f"Template not found: {template_ref}"}
        else:
            csv_text = template_ref

        cfg = load_config()
        topic = (params.get("topic","") or cfg.default_topic).strip()
        if not topic:
            return {"error": "No topic set. Enter a topic before running batch."}

        prog("Parsing template")
        try:
            workflow = parse_template(csv_text, cfg.participants)
        except Exception as e:
            return {"error": f"Template parse error: {e}"}

        workflow.topic = topic
        if not workflow.stages:
            return {"error": "Template has no stages."}

        prog(f"Pipeline: {len(workflow.stages)} stages — " +
             " → ".join(f"{s.stage_type}:{s.label}" for s in workflow.stages))

        # ── Execute column by column ────────────────────────────
        t0 = time.time()
        overrides = {"temperature":float(params.get("temperature",0.7)),
                     "max_tokens":int(params.get("max_tokens",256))}
        context = {"topic":topic, "prior_output":None, "stage_index":0, "overrides":overrides}
        stage_results = []

        for idx, stage in enumerate(workflow.stages):
            context["stage_index"] = idx
            prog(f"Stage {idx+1}/{len(workflow.stages)}: {stage.stage_type}:{stage.label}")

            handler = self._stage_handlers.get(stage.stage_type)
            if not handler:
                stage_results.append({"stage_type":stage.stage_type,"label":stage.label,
                                      "error":f"Unknown stage type '{stage.stage_type}'",
                                      "output":"","n_generations":0})
                continue

            result = handler(self, stage, context, prog)
            stage_results.append(result)
            context["prior_output"] = result.get("output","")

            # Save transcript
            _TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
            run_id = time.strftime("%Y%m%d_%H%M%S")
            try:
                fname = f"rt_{run_id}_{idx:02d}_{stage.stage_type.lower()}.json"
                with open(_TRANSCRIPT_DIR / fname,"w") as f: json.dump(result, f, indent=2)
            except Exception: pass

        elapsed = round(time.time()-t0, 2)
        n_gen = sum(r.get("n_generations",0) for r in stage_results)
        prog(f"Complete: {n_gen} generations in {elapsed}s")

        return {"topic":topic, "workflow":workflow.to_dict(), "stages":stage_results,
                "final_output":context.get("prior_output",""),
                "n_stages":len(workflow.stages), "n_total_generations":n_gen,
                "elapsed_seconds":elapsed}

    # ── Seed resolution ────────────────────────────────────────

    def _resolve_seed(self, seed, overrides):
        r = dict(seed)
        if "temperature" not in r: r["temperature"] = overrides.get("temperature",0.7)
        if "max_tokens" not in r: r["max_tokens"] = overrides.get("max_tokens",256)
        r.setdefault("top_p", 0.9)
        r.setdefault("name","Agent"); r.setdefault("role","")
        r.setdefault("system_prompt","You are a helpful roundtable participant.")
        return r

    @classmethod
    def register_stage(cls, stage_type):
        def dec(fn): cls._stage_handlers[stage_type.upper()] = fn; return fn
        return dec

    @classmethod
    def register_method(cls, name, description="", default_prompt=""):
        def dec(fn): cls._analysis_methods[name] = {"handler":fn,"description":description,"default_prompt":default_prompt}; return fn
        return dec

    @staticmethod
    def list_methods():
        return [{"name":n,"description":i.get("description","")} for n,i in RoundtableLMAModule._analysis_methods.items()]

    @staticmethod
    def list_tools():
        return [{"name":n,"description":getattr(fn,"_meta",{}).get("description","")} for n,fn in _TOOL_REGISTRY.items()]


# ================================================================
#  7. Stage handlers
# ================================================================

@RoundtableLMAModule.register_stage("PANEL")
def _stage_panel(module, stage, context, progress):
    """Multi-agent roundtable.  Each agent sees accumulating transcript.
    Blank-canvas isolation: no prior panel transcripts exposed."""
    topic = context["topic"]
    prior = context.get("prior_output")
    idx = context["stage_index"]
    panel_input = prior if (prior and idx > 0) else topic

    transcript_parts = []; responses = []; n_gen = 0

    for a_idx, raw in enumerate(stage.seeds):
        seed = module._resolve_seed(raw, context["overrides"])
        progress(f"  {stage.label} — {seed['name']} ({a_idx+1}/{len(stage.seeds)})")

        if transcript_parts:
            uc = f"Discussion topic:\n{panel_input}\n\nTranscript so far:\n" + "\n".join(transcript_parts) + "\n\nPlease contribute your perspective."
        else:
            uc = panel_input

        try:
            resp = _generate(module._pipeline, seed["system_prompt"], uc,
                             seed["temperature"], seed["top_p"], seed["max_tokens"])
            n_gen += 1
        except Exception as e:
            resp = f"[Generation error: {e}]"

        tag = f"{seed['name']} — {seed['role']}" if seed["role"] else seed["name"]
        responses.append({"name":seed["name"],"role":seed["role"],"response":resp})
        transcript_parts.append(f"[{tag}]\n{resp}\n")

    transcript = "\n".join(transcript_parts)
    return {"stage_type":"PANEL","label":stage.label,"responses":responses,
            "transcript":transcript,"output":transcript,"n_generations":n_gen}


@RoundtableLMAModule.register_stage("ANALYSIS")
def _stage_analysis(module, stage, context, progress):
    """Method-based processing.  Dispatches by 'method' field in seed."""
    prior = context.get("prior_output","")
    if not stage.seeds:
        return {"stage_type":"ANALYSIS","label":stage.label,"output":prior,"n_generations":0}

    current = prior; n_gen = 0; chain = []

    for raw in stage.seeds:
        seed = module._resolve_seed(raw, context["overrides"])
        method_name = raw.get("method","custom")
        progress(f"  {stage.label} — {method_name}")

        mi = module._analysis_methods.get(method_name)
        if not mi: mi = module._analysis_methods.get("custom")

        if "system_prompt" not in raw and mi.get("default_prompt"):
            seed["system_prompt"] = mi["default_prompt"]

        current = mi["handler"](module, current, seed, context, progress)
        chain.append(method_name)
        if method_name not in ("passthrough","aggregate"): n_gen += 1

    return {"stage_type":"ANALYSIS","label":stage.label,"method_chain":chain,
            "output":current,"n_generations":n_gen}


@RoundtableLMAModule.register_stage("TOOL")
def _stage_tool(module, stage, context, progress):
    """Programmatic functions — no model generation."""
    prior = context.get("prior_output","")
    if not stage.seeds:
        return {"stage_type":"TOOL","label":stage.label,"output":prior,"n_generations":0}

    current = prior; chain = []

    for raw in stage.seeds:
        tool_name = raw.get("tool","export_json")
        progress(f"  {stage.label} — tool:{tool_name}")
        handler = _TOOL_REGISTRY.get(tool_name)
        if not handler: chain.append(f"{tool_name}(unknown)"); continue
        try:
            result = handler(current, raw, context, progress)
            if isinstance(result, dict) and "output" in result: current = result["output"]
            elif isinstance(result, str): current = result
        except Exception as e:
            logger.error(f"[RT] Tool {tool_name}: {e}")
        chain.append(tool_name)

    return {"stage_type":"TOOL","label":stage.label,"tool_chain":chain,
            "output":current,"n_generations":0}


# ================================================================
#  8. Analysis methods
# ================================================================

def _gen_method(mod, prior, seed, ctx, prog):
    return _generate(mod._pipeline, seed["system_prompt"], prior,
                     seed["temperature"], seed.get("top_p",0.9), seed["max_tokens"])

for _n, _d, _p in [
    ("synthesize","Model-based synthesis",_PROMPTS["synthesize"]),
    ("analyze","Qualitative coding analysis",_PROMPTS["analyze"]),
    ("evaluate","Critical evaluation",_PROMPTS["evaluate"]),
    ("extract","Extract themes and claims",_PROMPTS["extract"]),
    ("report","Structured report",_PROMPTS["report"]),
    ("custom","Custom system prompt","Process the following input."),
]:
    RoundtableLMAModule._analysis_methods[_n] = {
        "handler": _gen_method, "description": _d, "default_prompt": _p}

RoundtableLMAModule._analysis_methods["passthrough"] = {
    "handler": lambda mod,prior,seed,ctx,prog: prior,
    "description": "Forward without modification", "default_prompt": ""}

RoundtableLMAModule._analysis_methods["aggregate"] = {
    "handler": lambda mod,prior,seed,ctx,prog: (
        f"Words: {len(prior.lower().split())} | "
        f"Unique: {len(set(prior.lower().split()))}\n"
        f"Top 20: {', '.join(f'{w}({c})' for w,c in Counter(prior.lower().split()).most_common(20))}"
    ), "description": "Word counts and stats (no model)", "default_prompt": ""}


# ================================================================
#  9. Tool registry
# ================================================================

_TOOL_REGISTRY: dict[str, Callable] = {}

def register_tool(name, description=""):
    def dec(fn): _TOOL_REGISTRY[name] = fn; fn._meta = {"description":description}; return fn
    return dec

@register_tool("export_json","Export context as JSON file")
def _t_export(prior, seed, ctx, prog):
    path = _TRANSCRIPT_DIR / f"roundtable_{ctx.get('run_id','export')}.json"
    _TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path,"w") as f: json.dump({"topic":ctx.get("topic",""),"output":prior},f,indent=2)
    return {"output":prior,"file":str(path)}

@register_tool("word_count","Word frequency analysis")
def _t_wc(prior, seed, ctx, prog):
    words = prior.lower().split()
    freq = Counter(words).most_common(30)
    return {"output":f"Words: {len(words)}, Unique: {len(set(words))}\n{', '.join(f'{w}({c})' for w,c in freq)}"}

@register_tool("snapshot","Save checkpoint of current output")
def _t_snap(prior, seed, ctx, prog):
    ctx.setdefault("_snapshots",{})[seed.get("label","snap")] = prior
    return {"output":prior}

@register_tool("truncate","Trim output to max_chars")
def _t_trunc(prior, seed, ctx, prog):
    mc = int(seed.get("max_chars",2000))
    return prior[:mc] + "\n[truncated]" if len(prior) > mc else prior


# ================================================================
#  10. Interactive session manager
# ================================================================

@dataclass
class _Turn:
    turn_type: str; name: str; role: str; content: str
    timestamp: float = 0.0; stage_label: str = ""
    def to_dict(self):
        return {"turn_type":self.turn_type,"name":self.name,"role":self.role,
                "content":self.content,"timestamp":self.timestamp,"stage_label":self.stage_label}

@dataclass
class _Session:
    session_id: str; topic: str
    turns: list = field(default_factory=list)
    stages: list = field(default_factory=list)
    current_stage_label: str = "Panel 1"
    current_stage_type: str = "PANEL"
    stage_counter: int = 1
    gen_config: dict = field(default_factory=lambda: {"temperature":0.7,"max_tokens":256,"top_p":0.9})

    def transcript(self, current_only=True):
        turns = [t for t in self.turns if t.stage_label == self.current_stage_label] if current_only else self.turns
        return "\n".join(f"[{t.name}]\n{t.content}\n" for t in turns)

    def mark_new_stage(self, stype, label=""):
        cur = [t for t in self.turns if t.stage_label == self.current_stage_label]
        self.stages.append({"type":self.current_stage_type,"label":self.current_stage_label,"n_turns":len(cur)})
        self.stage_counter += 1
        self.current_stage_label = label or f"{stype.title()} {self.stage_counter}"
        self.current_stage_type = stype.upper()
        return {"stage_type":self.current_stage_type,"label":self.current_stage_label}

    def to_dict(self):
        return {"session_id":self.session_id,"topic":self.topic,
                "current_stage":{"type":self.current_stage_type,"label":self.current_stage_label},
                "n_turns":len(self.turns),"turns":[t.to_dict() for t in self.turns],"gen_config":self.gen_config}

    def export(self):
        cur = [t for t in self.turns if t.stage_label == self.current_stage_label]
        all_s = self.stages + [{"type":self.current_stage_type,"label":self.current_stage_label,"n_turns":len(cur)}]
        full = "\n".join(f"[{t.name}]\n{t.content}\n" for t in self.turns)
        return {"session_id":self.session_id,"topic":self.topic,"stages":all_s,
                "full_transcript":full,"turns":[t.to_dict() for t in self.turns],"n_total_turns":len(self.turns)}


class InteractiveSessionManager:
    def __init__(self):
        self._session = None; self._lock = threading.Lock(); self._pipeline = None
    def set_pipeline(self, p): self._pipeline = p

    def start(self, topic, gen_config=None):
        with self._lock:
            self._session = _Session(session_id=time.strftime("%Y%m%d_%H%M%S"), topic=topic)
            if gen_config: self._session.gen_config.update(gen_config)
            self._session.turns.append(_Turn("user","User","",topic,time.time(),self._session.current_stage_label))
            return self._session.to_dict()

    def get_session(self):
        with self._lock: return self._session.to_dict() if self._session else None

    def send_user_message(self, msg):
        with self._lock:
            if not self._session: return {"ok":False,"error":"No session"}
            self._session.turns.append(_Turn("user","User","",msg,time.time(),self._session.current_stage_label))
            return {"ok":True,"n_turns":len(self._session.turns)}

    def apply_persona(self, participant_id=None, inline_seed=None):
        if not self._pipeline: return {"ok":False,"error":"No model loaded"}
        with self._lock:
            if not self._session: return {"ok":False,"error":"No session"}
            s = self._session
        seed = dict(inline_seed) if inline_seed else {}
        if participant_id and not inline_seed:
            p = next((x for x in load_config().participants if x.id == participant_id), None)
            if not p: return {"ok":False,"error":"Participant not found"}
            seed = p.to_dict()
        seed.setdefault("temperature", s.gen_config.get("temperature",0.7))
        seed.setdefault("max_tokens", s.gen_config.get("max_tokens",256))
        seed.setdefault("system_prompt","You are a helpful roundtable participant.")
        seed.setdefault("name","Agent"); seed.setdefault("role","")
        tr = s.transcript(current_only=True)
        uc = f"Discussion topic:\n{s.topic}\n\nTranscript so far:\n{tr}\n\nPlease contribute." if tr.strip() else s.topic
        try: resp = _generate(self._pipeline, seed["system_prompt"], uc, seed["temperature"], seed.get("top_p",0.9), seed["max_tokens"])
        except Exception as e: return {"ok":False,"error":str(e)}
        with self._lock:
            s.turns.append(_Turn("persona",seed["name"],seed["role"],resp,time.time(),s.current_stage_label))
        return {"ok":True,"name":seed["name"],"role":seed["role"],"response":resp,"n_turns":len(s.turns)}

    def apply_method(self, method_name="synthesize", system_prompt=None):
        if not self._pipeline and method_name not in ("passthrough","aggregate"):
            return {"ok":False,"error":"No model loaded"}
        with self._lock:
            if not self._session: return {"ok":False,"error":"No session"}
            s = self._session
        mi = RoundtableLMAModule._analysis_methods.get(method_name)
        if not mi: return {"ok":False,"error":f"Unknown method '{method_name}'"}
        tr = s.transcript(current_only=True)
        if not tr.strip(): return {"ok":False,"error":"No transcript content"}
        seed = {"temperature":s.gen_config.get("temperature",0.7),"max_tokens":s.gen_config.get("max_tokens",256),
                "top_p":0.9,"system_prompt":system_prompt or mi.get("default_prompt","Process this input.")}
        class _S: pass
        stub = _S(); stub._pipeline = self._pipeline
        try: result = mi["handler"](stub, tr, seed, {}, lambda m: None)
        except Exception as e: return {"ok":False,"error":str(e)}
        with self._lock:
            s.turns.append(_Turn("method",method_name,"analysis",result,time.time(),s.current_stage_label))
        return {"ok":True,"method":method_name,"output":result,"n_turns":len(s.turns)}

    def new_stage(self, stype="PANEL", label=""):
        with self._lock:
            if not self._session: return {"ok":False,"error":"No session"}
            return {"ok":True, **self._session.mark_new_stage(stype, label)}

    def export(self):
        with self._lock: return self._session.export() if self._session else None

    def reset(self):
        with self._lock: had = self._session is not None; self._session = None
        return {"ok":True,"had_session":had}

_interactive_manager = InteractiveSessionManager()
