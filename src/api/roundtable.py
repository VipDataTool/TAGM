"""Roundtable LMA endpoints.

Each route is a thin delegation to ``roundtable_lma``'s module-level
helpers. Previously every one of these re-imported the module inside the
function body; there is one module-level import now.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool

from src.engine.app_core import state
from src.engine.modules.roundtable_lma import (
    _interactive_manager,
    list_participants,
    remove_participant,
    reset_to_defaults,
    update_default_topic,
    upsert_participant,
)

from src.api._state import module_runner

router = APIRouter(prefix="/api/roundtable", tags=["roundtable"])


@router.get("/participants")
async def rt_list_participants():
    return {"ok": True, "participants": list_participants()}


@router.post("/participants")
async def rt_upsert_participant(request: Request):
    return {"ok": True, "participant": upsert_participant(await request.json())}


@router.delete("/participants/{pid}")
async def rt_remove_participant(pid: str):
    return {"ok": remove_participant(pid)}


@router.post("/participants/reset")
async def rt_reset_participants():
    return {"ok": True, "participants": reset_to_defaults()}


@router.post("/topic")
async def rt_set_topic(request: Request):
    d = await request.json()
    return {"ok": True, "topic": update_default_topic(d.get("topic", ""))}


@router.get("/methods")
async def rt_list_methods():
    mod = module_runner.get_module("roundtable_lma")
    if not mod:
        return {"ok": False}
    return {"ok": True, "methods": mod.list_methods(), "tools": mod.list_tools()}


@router.post("/interactive/start")
async def rt_start(request: Request):
    d = await request.json()
    return {"ok": True, "session": _interactive_manager.start(
        d.get("topic", ""), d.get("gen_config", {}))}


@router.get("/interactive/session")
async def rt_session():
    s = _interactive_manager.get_session()
    return {"ok": s is not None, "session": s}


@router.post("/interactive/send")
async def rt_send(request: Request):
    d = await request.json()
    return _interactive_manager.send_user_message(d.get("message", ""))


@router.post("/interactive/apply_persona")
async def rt_apply_persona(request: Request):
    d = await request.json()
    return await run_in_threadpool(_interactive_manager.apply_persona,
        participant_id=d.get("participant_id"), inline_seed=d.get("seed"))


@router.post("/interactive/apply_method")
async def rt_apply_method(request: Request):
    d = await request.json()
    return await run_in_threadpool(_interactive_manager.apply_method,
        method_name=d.get("method", "synthesize"),
        system_prompt=d.get("system_prompt"))


@router.post("/interactive/new_stage")
async def rt_new_stage(request: Request):
    d = await request.json()
    return _interactive_manager.new_stage(d.get("stage_type", "PANEL"),
                                          d.get("label", ""))


@router.post("/interactive/config")
async def rt_config(request: Request):
    return _interactive_manager.update_config(await request.json())


@router.post("/interactive/apply_tool")
async def rt_apply_tool(request: Request):
    d = await request.json()
    return await run_in_threadpool(_interactive_manager.apply_tool,
        tool_name=d.get("tool", "export_json"), params=d.get("params", {}))


@router.get("/interactive/export")
async def rt_export():
    e = _interactive_manager.export()
    return {"ok": e is not None, "export": e}


@router.post("/interactive/reset")
async def rt_reset():
    return _interactive_manager.reset()


@router.post("/batch")
async def rt_batch(request: Request):
    """Run a batch pipeline from CSV template text."""
    mod = module_runner.get_module("roundtable_lma")
    if not mod:
        return {"ok": False, "error": "Module not found."}
    if state.pipeline is None or not state.pipeline.loaded:
        return {"ok": False, "error": "No model loaded."}
    data = await request.json()
    try:
        result = await run_in_threadpool(mod._run_batch,
                                         data.get("template_csv", ""),
                                         data, lambda m: None)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}
