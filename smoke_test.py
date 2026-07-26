#!/usr/bin/env python3
"""TAGM post-refactor smoke test.

Run from the repo root in the Codespace:

    python smoke_test.py

Targets exactly the surfaces that static analysis could not reach. Every
check here failed-open in the offline review: syntax was valid, imports
resolved, routes matched — none of which proves the thing renders or
serves. Requires no model download and makes no network calls.

Exit code 0 = all passed. Non-zero = count of failures.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

PASS, FAIL = [], []


def check(name):
    """Decorator: run a check, record pass/fail, never abort the suite."""
    def deco(fn):
        sys.stdout.write(f"  {name:.<58}")
        sys.stdout.flush()
        try:
            detail = fn() or ""
            PASS.append(name)
            print(f" PASS {detail}")
        except Exception as e:
            FAIL.append((name, traceback.format_exc()))
            print(f" FAIL  {type(e).__name__}: {e}")
        return fn
    return deco


# ══ 1. Import + boot ════════════════════════════════════════════
# A circular import in the new src/api/* package means the server never
# starts. compileall cannot see this.
print("\n[1] Import graph")


@check("all first-party modules import")
def _imports():
    import importlib
    mods = [
        "src.core.db", "src.core.pipeline", "src.core.deltas.store",
        "src.core.deltas.compute", "src.core.deltas.spectral",
        "src.engine.metrics", "src.engine.statistics", "src.engine.result",
        "src.engine.ltp", "src.engine.sfd", "src.engine.analyzer",
        "src.engine.hooks", "src.engine.qk_intervention",
        "src.engine.modules.base", "src.probes.io", "src.probes.diagnostic",
        "src.service.plots", "src.service.chat", "src.service.events",
        "src.service.export", "src.engine.app_core",
        "src.api._state", "src.api.roundtable", "src.api.probes",
        "src.api.modules", "src.api.hep", "src.api.ecm_config", "src.app",
    ]
    for m in mods:
        importlib.import_module(m)
    return f"({len(mods)} modules)"


@check("module auto-discovery finds every module")
def _discovery():
    from src.api._state import module_runner
    mods = module_runner.list_modules()
    n = len(mods)
    if n < 15:
        raise AssertionError(f"only {n} modules discovered, expected ~17")
    return f"({n} found)"


# ══ 2. Plot rendering ═══════════════════════════════════════════
# THE highest residual risk. service/plots.py was converted to the
# matplotlib object-oriented API across every handler and not one figure
# has been rendered. A broken handler is invisible until a user clicks.
print("\n[2] Plot rendering (matplotlib OO conversion)")


@check("every plot handler renders without raising")
def _plots():
    import numpy as np
    from src.service import plots as P

    registry = getattr(P, "_PLOT_HANDLERS", {})
    if not registry:
        raise AssertionError("no plot handlers found — registry renamed?")

    # Synthetic result shaped like a real PromptResult dict.
    n = 12
    fake = {
        "prompt": "smoke test", "category": "benign", "seq_len": n,
        "tokens": [f"tok{i}" for i in range(n)],
        "stress_score": 0.5, "net_correction": 0.1, "entropy": 2.0,
        "top2_share": 0.4, "middle_share": 0.3, "interior_cv": 0.2,
        "kl_divergence": 0.05, "delta_scale": 1.0,
        "per_token_stress": np.linspace(0, 1, n).tolist(),
        "signed_attr": np.linspace(-1, 1, n).tolist(),
        "heatmap": np.random.rand(8, n).tolist(),
        "amplitude_trajectory": np.random.rand(8).tolist(),
        "amplitude_normalized": np.random.rand(8).tolist(),
        "per_layer_amplitude": {i: float(i) for i in range(4)},
        "instruct_topk": [("a", 0.5), ("b", 0.3)],
        "base_topk": [("a", 0.4), ("b", 0.35)],
        "ltp": {
            "mean_M": 1.0, "mean_V": 0.2, "mean_L": 0.8, "max_prc": 0.3,
            "n_directional": 2,
            "profiles": [np.random.rand(8).tolist() for _ in range(n)],
            "base_profiles": [np.random.rand(8).tolist() for _ in range(n)],
            "tension_magnitudes": np.random.rand(n).tolist(),
            "prc_per_token": np.random.rand(n).tolist(),
            "profile_shapes": ["flat"] * n,
            "monitored_layers": [0, 1, 2, 3],
            "offset_magnitude": {i: 1.0 for i in range(4)},
            "offset_variance": {i: 0.1 for i in range(4)},
            "lateral_coverage": {i: 0.5 for i in range(4)},
        },
        "sfd": {
            "per_token_density": np.random.rand(n).tolist(),
            "density_mean": 0.5, "density_max": 0.9, "density_var": 0.1,
            "density_p90": 0.8, "global_erank": 10.0, "n_layers_used": 4,
            "per_token_directions": [np.random.rand(8).tolist() for _ in range(n)],
        },
        # mean_tau is None-able since the rank-displacement fix — a
        # renderer that assumes a float will crash here, which is the point.
        "rank_displacement": {
            "mean_tau": None, "mean_overlap": 0.5, "n_comparable": 0,
            "n_tau_undefined": n, "n_positions": n,
            "per_position_tau": [None] * n,
            "per_position_overlap": np.random.rand(n).tolist(),
            "per_position": [
                {"total_disp": 0.1, "replacement_ratio": 0.2, "n_matched": 3,
                 "n_promoted": 1, "n_demoted": 1, "matched_disp": 0.05,
                 "promoted_mass": 0.02, "demoted_mass": 0.03,
                 "concentration": 0.1} for _ in range(n)
            ],
            "instruct_disp_profiles": [np.random.rand(8).tolist() for _ in range(n)],
            "base_disp_profiles": [np.random.rand(8).tolist() for _ in range(n)],
        },
    }

    # Handlers return RAW PNG BYTES (not a base64 data URL). Check the magic
    # bytes and a minimum size — "did it raise?" is too weak, because an
    # exception-swallowing handler returns a valid-but-empty placeholder and
    # would pass. Placeholders are ~5 KB; real plots are 15 KB+.
    PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
    broke, placeholders = [], []
    outdir = os.environ.get("SMOKE_PLOT_DIR")
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    for key, handler in registry.items():
        try:
            out = handler(fake)
            if not isinstance(out, (bytes, bytearray)):
                broke.append(f"{key}: returned {type(out).__name__}, not bytes")
            elif out[:8] != PNG_MAGIC:
                broke.append(f"{key}: not a PNG ({len(out)} bytes)")
            else:
                if len(out) < 8000:
                    placeholders.append(key)
                if outdir:
                    with open(os.path.join(outdir, f"{key}.png"), "wb") as f:
                        f.write(out)
        except Exception as e:
            broke.append(f"{key}: {type(e).__name__}: {e}")
    if broke:
        raise AssertionError(f"{len(broke)}/{len(registry)} failed -> {broke[:4]}")
    note = f"({len(registry)} handlers"
    if placeholders:
        note += f", {len(placeholders)} empty-state: {placeholders}"
    return note + ")"


@check("no figures leaked into the pyplot registry")
def _leak():
    # The OO conversion exists so failed renders stop accumulating figures.
    import matplotlib.pyplot as plt
    open_figs = plt.get_fignums()
    if open_figs:
        raise AssertionError(f"{len(open_figs)} figure(s) left open: {open_figs}")
    return "(0 open)"


# ══ 3. Live server ══════════════════════════════════════════════
# Boots the real app and serves real requests. Catches startup crashes,
# route shadowing and 500s that no static check can reach.
print("\n[3] Live server")

PORT = int(os.environ.get("SMOKE_PORT", "8199"))
BASE = f"http://127.0.0.1:{PORT}"
proc = None


def _get(path, timeout=15):
    req = urllib.request.Request(BASE + path)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


@check("server boots and answers /api/status")
def _boot():
    global proc
    env = dict(os.environ, TAGM_DB_PATH="/tmp/tagm_smoke.db")
    proc = subprocess.Popen(
        [sys.executable, "-m", "src", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(60):
        if proc.poll() is not None:
            raise AssertionError(
                "server exited during startup:\n" + (proc.stdout.read() or "")[-2000:])
        try:
            code, _body = _get("/api/status", timeout=2)
            if code == 200:
                return
        except Exception:
            time.sleep(1)
    raise AssertionError("server did not become ready within 60s")


@check("every static page returns 200")
def _pages():
    pages = ["/", "/static/index.html", "/static/chat.html",
             "/static/roundtable.html", "/static/template_maker.html",
             "/static/correction_prism_viz.html",
             "/static/probe_diagnostic_viz.html",
             "/static/domain_surface_viz.html",
             "/static/correction_field_topology_viz.html",
             "/static/js/main.js", "/static/css/main.css",
             "/static/js/common/esc.js", "/static/js/common/api.js",
             "/static/js/common/download.js", "/static/css/viz-base.css"]
    bad = []
    for p in pages:
        try:
            code, _ = _get(p)
            if code != 200:
                bad.append(f"{p}->{code}")
        except Exception as e:
            bad.append(f"{p}->{type(e).__name__}")
    if bad:
        raise AssertionError(f"{len(bad)} failed: {bad}")
    return f"({len(pages)} assets)"


@check("read-only API endpoints return 200")
def _api():
    eps = ["/api/status", "/api/models", "/api/config", "/api/dashboard",
           "/api/modules", "/api/session/results", "/api/prompts",
           "/api/templates"]
    bad = []
    for e in eps:
        try:
            code, _ = _get(e)
            if code != 200:
                bad.append(f"{e}->{code}")
        except urllib.error.HTTPError as he:
            bad.append(f"{e}->{he.code}")
        except Exception as ex:
            bad.append(f"{e}->{type(ex).__name__}")
    if bad:
        raise AssertionError(f"{bad}")
    return f"({len(eps)} endpoints)"


@check("cancel endpoints exist and reject when idle")
def _cancel():
    import json
    req = urllib.request.Request(BASE + "/api/analyze/cancel", method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        body = json.loads(r.read())
    # Nothing is running, so this must be a clean refusal, not a 404/500.
    if body.get("ok") is not False:
        raise AssertionError(f"expected ok:false when idle, got {body}")
    return "(refuses cleanly)"


# ══ Teardown + report ═══════════════════════════════════════════
if proc is not None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()

print("\n" + "=" * 68)
print(f"PASSED {len(PASS)}   FAILED {len(FAIL)}")
if FAIL:
    print("=" * 68)
    for name, tb in FAIL:
        print(f"\n--- {name} ---\n{tb}")
print("=" * 68)
sys.exit(len(FAIL))
