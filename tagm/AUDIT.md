# Frontend Rewiring Audit

This is the working spec for the frontend rewiring sessions. It maps
every fetch site in the TASM-derived frontend to its TAGM destination,
flags shape mismatches, and identifies TAGM backend additions needed.

The principle: **see TASM implementation for details.** If TASM had a
behavior, TAGM matches it. Where TAGM's backend doesn't currently
support something TASM did, the TAGM backend is extended to match.

Counts:
- 49 fetch sites in `static/js/main.js`
- 6 fetch sites across `static/chat.html` and the four `_viz.html` pages
- TASM has 49 backend routes; TAGM currently has 28.

## Endpoint mapping

Three columns: **TASM endpoint** (what main.js currently calls) → **TAGM
endpoint** (what it should call) → **Action**.

| TASM endpoint | TAGM endpoint | Action |
|---|---|---|
| GET `/api/status` | GET `/api/status` | RESHAPE_RESPONSE — TAGM wraps fields differently (`pipeline.model_pair.instruct` vs flat `model_loaded`/`current_model`); JS reads `s.model_loaded`, `s.loading_error`, etc. Add response shape adapter at top of main.js OR add fields to TAGM's status response to match TASM's flat shape. **Decision: add TASM-compat fields to TAGM's `/api/status`** (cheaper than rewriting every read site). |
| POST `/api/user_info` | — | NEW_TAGM_ENDPOINT — TASM stored a per-session user-info blob. Add `POST /api/user_info` accepting form data, mirror TASM's behavior. |
| GET `/api/models` | GET `/api/models` | RESHAPE_RESPONSE — TASM returns a flat list with `current` field; TAGM returns `{pairs: [...]}`. **Decision: change TAGM to return TASM's shape.** |
| POST `/api/models` | — | NEW_TAGM_ENDPOINT — TASM allowed adding/updating a pair in models.json from the UI. Add to TAGM. |
| POST `/api/load_model` | POST `/api/load` | RENAME + RESHAPE_REQUEST — TASM took form data; TAGM takes JSON. **Decision: add `POST /api/load_model` accepting form data, internally calling the same loader.** Keeps main.js unchanged. |
| POST `/api/set_inference_model` | — | NEW_TAGM_ENDPOINT — TASM allowed swapping the chat-side inference model independently of the analysis pair. Add to TAGM, mirror TASM. |
| POST `/api/reset` | POST `/api/session/reset` | RENAME — change main.js to call `/api/session/reset`, OR add `/api/reset` as alias. **Decision: add alias.** |
| POST `/api/session/clear_plots` | — | NEW_TAGM_ENDPOINT — TASM cleared cached plot images. TAGM doesn't generate server-side plots, but the call should succeed (no-op + 200 OK) so the UI button continues to work. Add as no-op. |
| POST `/api/session/clear_all` | POST `/api/session/reset` | RENAME or alias. **Decision: add alias.** |
| POST `/api/session/restore` | — | NEW_TAGM_ENDPOINT — TASM restored a previously-cleared session. TAGM needs equivalent: persist last-cleared session to disk on reset, restore on this call. |
| GET `/api/progress` | GET `/api/logs` | RENAME + RESHAPE_RESPONSE — TASM returned `{log: [{stage, message, ts}, ...]}`. TAGM returns `{entries: [...]}`. **Decision: add `/api/progress` returning TASM's shape.** |
| GET `/api/prompts` | GET `/api/prompts` | VERIFY_SHAPE — likely matches; check during session 2. |
| POST `/api/prompts` | — | NEW_TAGM_ENDPOINT — TASM let user save a prompt to the library. Add to TAGM. |
| POST `/api/analyze` | POST `/api/analyze` | RESHAPE_REQUEST/RESPONSE — TASM took form data with prompt + category + flags; returned `{ok, results: [{...measurements...}], plots, ...}`. TAGM takes JSON `{prompt, category}`; returns `{ok, prompt: {prompt, category, tokens, seq_len, measurements: {...}}}`. **Decision: add a TASM-compat wrapper that accepts form data and returns TASM-shape response.** |
| POST `/api/analyze_batch` | POST `/api/batch` | RENAME + RESHAPE — same as above. **Decision: add `/api/analyze_batch` form-data wrapper.** |
| GET `/api/session/results` | GET `/api/session` | RENAME + RESHAPE — TASM returns paginated. TAGM returns whole session. **Decision: add `/api/session/results` with pagination, derived from session state.** |
| POST `/api/session/remove` | — | NEW_TAGM_ENDPOINT — remove specific prompt indices from session. Add to TAGM. |
| POST `/api/session/rerun` | — | NEW_TAGM_ENDPOINT — re-run measurements on existing session prompts. Add to TAGM. |
| GET `/api/dashboard` | GET `/api/status` | RESHAPE — TASM's dashboard returned aggregate stats. **Decision: add `/api/dashboard` returning the aggregates the dashboard JS expects.** |
| GET `/api/plots/{plot_key}` | — | NEW_TAGM_ENDPOINT — TASM rendered matplotlib plots server-side and served them as PNG. TAGM doesn't have this. **Decision: add `/api/plots/{plot_key}` that generates the same plots from TAGM session data using matplotlib.** Will need a `tagm/service/plots.py` module. Roughly mirrors TASM's plot generation. |
| GET `/api/plots/individual/{index}/{plot_key}` | — | NEW_TAGM_ENDPOINT — same as above but per-prompt. |
| GET `/api/results/detail` | GET `/api/session` | RESHAPE — TASM returns chunked detail `{start, count, results: [...]}`. **Decision: add `/api/results/detail` accepting `start` & `count`, slicing TAGM session.prompts.** |
| POST `/api/export` | GET `/api/session/export` | METHOD_CHANGE + RESHAPE — TASM took JSON options POST; returned `{ok, ready: true}` and the download was via `/api/export/download`. TAGM streams the gzipped file directly from GET. **Decision: add `POST /api/export` that prepares an export and returns ok-true, then `GET /api/export/download` streams it. Keeps two-step pattern intact.** |
| GET `/api/export/download` | GET `/api/session/export` | RENAME — see above. **Decision: add as alias.** |
| GET `/api/modules` | GET `/api/measurements` + GET `/api/analyses` | RESHAPE — TASM unified both under `/api/modules`. **Decision: add `/api/modules` returning combined list with `kind: "measurement"|"analysis"` field.** |
| POST `/api/modules/upload_template` | — | NEW_TAGM_ENDPOINT — TASM let user upload a probe template CSV. TAGM has templates on disk but no upload route. Add. |
| POST `/api/modules/{name}/run` | POST `/api/analysis/{name}` (for analyses) OR — (for measurements) | RESHAPE — In TASM modules combined the configure-and-run flow. In TAGM measurements run as part of `/api/analyze`; analyses run separately. **Decision: add `/api/modules/{name}/run` that dispatches based on whether `{name}` is a measurement (call configure_measurements + analyze) or an analysis (call /api/analysis/{name}).** |
| GET `/api/modules/{name}/status` | — | NEW_TAGM_ENDPOINT — TASM tracked async per-module run status. Add equivalent. |
| GET `/api/modules/{name}/results` | derive from session.measurements[name] or session.analyses[name] | NEW_TAGM_ENDPOINT — `/api/modules/{name}/results` returning the merged data the viz pages expect. The 5 viz pages all hit this. |
| POST `/api/modules/{name}/reset` | — | NEW_TAGM_ENDPOINT — clear a module's results from the session. Add. |
| GET `/api/modules/{name}/download_log` | — | NEW_TAGM_ENDPOINT — module-specific log download. Add. |
| GET `/api/log` | GET `/api/logs` | RENAME — add alias. |
| GET `/api/config` | derive from `/api/status` | NEW_TAGM_ENDPOINT — TASM's config endpoint returned the active analysis config. **Decision: add `/api/config` (GET) returning the union of capture_config + selected_measurements.** |
| POST `/api/config` | POST `/api/capture` + POST `/api/configure` | RESHAPE — TASM accepted a unified config blob and applied it. **Decision: add `/api/config` (POST) that accepts the unified shape and dispatches to /api/capture and /api/configure internally.** |
| GET `/api/engine_config` | — | NEW_TAGM_ENDPOINT — engine-wide parameters. **Decision: add `/api/engine_config` returning a flat dict of orchestrator-level settings (defaults match TASM).** |
| POST `/api/engine_config` | — | NEW_TAGM_ENDPOINT — set engine-wide parameters. Add. |
| POST `/api/engine_config/reset` | — | NEW_TAGM_ENDPOINT — reset to defaults. Add. |
| POST `/api/probe_set/apply` | (uses generate + select) | NEW_TAGM_ENDPOINT — TASM applied a probe set (made it the active set for probe-using measurements). TAGM doesn't have a global "active probe set" — it's per-measurement. **Decision: add `/api/probe_set/apply` that sets a default template_id+capture_signature in app state, used by measurements that don't override.** |
| GET `/api/probe_set/apply_status` | — | NEW_TAGM_ENDPOINT — async status of probe-set apply. Add. |
| GET `/api/probe_set/status` | GET `/api/probes` | RESHAPE — TASM returned `{active, available, ...}`. TAGM returns `{sets: [...]}`. **Decision: add `/api/probe_set/status` returning TASM's shape.** |
| POST `/api/probe_set/clear_caches` | — | NEW_TAGM_ENDPOINT — clears probe-derived caches. Map to clearing the SFD cache + any other per-probe-set state. |
| POST `/api/chat` | — | NEW_TAGM_ENDPOINT — chat endpoint. TAGM's pipeline holds the instruct model already; expose `pipeline.instruct_model.generate()` via a chat endpoint. Streaming response. |

## Routes serving HTML pages

TASM serves these under specific paths; TAGM needs to serve them too.

| Route | Action |
|---|---|
| GET `/` → `index.html` | EXISTS — already serves index.html. |
| GET `/chat` → `chat.html` | NEW_TAGM_ROUTE — add. |
| GET `/domain_surface_viz` → `domain_surface_viz.html` | NEW_TAGM_ROUTE — add. |
| GET `/correction_manifold_viz` → `correction_manifold_viz.html` | NEW_TAGM_ROUTE — add. |
| GET `/correction_heatmap_viz` → `correction_heatmap_viz.html` | NEW_TAGM_ROUTE — add. |
| GET `/correction_backscatter_viz` → `correction_backscatter_viz.html` | NEW_TAGM_ROUTE — add. |
| GET `/favicon.svg` | EXISTS via StaticFiles. |

## Data shape adapters needed

These are the response shapes that need to match what main.js expects.

### `/api/status`
TASM shape (what main.js reads):
```json
{
  "model_loaded": bool,
  "loading_error": string|null,
  "current_model": string,
  "session_id": string,
  "n_results": int,
  "cache_bytes": int,
  ...
}
```
TAGM current shape:
```json
{
  "service": "TAGM",
  "pipeline": {"loaded": bool, "model_pair": {"instruct": ..., "base": ...}, ...},
  "loading": {"active": bool, "error": string|null},
  "capture_config": {...},
  "session": {"session_id": ..., "n_prompts": int, ...},
  ...
}
```
**Action:** add the TASM-flat fields alongside TAGM's nested shape. Don't remove TAGM's; the session-1 redesign of mine used the nested shape and there's no harm in keeping both. The keys that need adding flat:
`model_loaded` (= `pipeline.loaded`), `loading_error` (= `loading.error`),
`current_model` (= `pipeline.model_pair.instruct`),
`session_id`, `n_results` (= `session.n_prompts`), `cache_bytes`.

### `/api/analyze`
TASM request: form data with `prompt`, `category`, `compute_kl`, `extract_topk`, `topk`, `compute_signed_attribution`.
TASM response:
```json
{
  "ok": bool,
  "results": [
    {
      "prompt": string,
      "category": string,
      "tokens": [string],
      "stress_score": {"per_token": [...], "mean": float, ...},
      "lateral_tension_profile": {...},
      "amplitude_trajectory": {...},
      ...
    }
  ],
  "plots": {...},
  "session_id": string,
  "n_results": int
}
```

TAGM current request: JSON `{prompt, category}`.
TAGM current response:
```json
{
  "ok": true,
  "prompt": {
    "prompt": string,
    "category": string,
    "tokens": [string],
    "seq_len": int,
    "measurements": {
      "stress_score": {"scalars": {...}, "per_token": {...}, ...},
      ...
    },
    "metadata": {}
  }
}
```

**Action:** add a TASM-compat wrapper `/api/analyze_form` that accepts form data, calls the underlying logic, and reshapes the response into TASM's flat-per-measurement shape. main.js continues calling `/api/analyze` with form data, getting TASM-shape response. Internally, TAGM continues operating on its native MeasurementResult shape.

### `/api/modules/{name}/results` (used by 5 viz pages + main.js)
TASM shape: per-module canonical result. Each viz page knows what to expect from its module.
**Action:** for each measurement/analysis, expose its session entry verbatim plus any extra fields the corresponding viz page expects. Concretely:
- `correction_heatmap` → return `session.analyses.correction_heatmap` data
- `correction_manifold` → same
- `correction_backscatter` → same
- `domain_surface` → same
- `correction_field_topology` → same

The five viz pages each parse the response shape they expect from TASM. We need to verify each one parses correctly against TAGM's analysis output, and where it doesn't, reshape on the way out.

## Backend additions inventory

By count, this audit identifies these new endpoints/aliases to add to TAGM's `app.py`:

**Aliases (single-line forwarders, no logic change):**
1. POST `/api/reset` → /api/session/reset
2. POST `/api/session/clear_all` → /api/session/reset
3. GET `/api/log` → /api/logs

**Wrappers (accept TASM request shape, dispatch to TAGM internals):**
4. POST `/api/load_model` (form data → JSON load)
5. POST `/api/analyze` (form data + reshape response)  ← need to keep TAGM's JSON form too, branch on Content-Type
6. POST `/api/analyze_batch` (form data + reshape response)
7. POST `/api/config` (combined → /api/capture + /api/configure)
8. GET `/api/config` (combined view of capture + measurement config)

**Genuinely new functionality:**
9. POST `/api/user_info`
10. POST `/api/models` (add/update model pair in registry)
11. POST `/api/set_inference_model`
12. POST `/api/session/clear_plots` (no-op stub)
13. POST `/api/session/restore`
14. POST `/api/session/remove`
15. POST `/api/session/rerun`
16. GET `/api/dashboard`
17. GET `/api/plots/{plot_key}` + `/api/plots/individual/{index}/{plot_key}`
18. POST `/api/export` + GET `/api/export/download` (two-step pattern)
19. GET `/api/results/detail` (paginated)
20. GET `/api/progress` (TASM shape)
21. POST `/api/prompts` (save to library)
22. GET `/api/modules` (combined list)
23. POST `/api/modules/upload_template`
24. POST `/api/modules/{name}/run`
25. GET `/api/modules/{name}/status`
26. GET `/api/modules/{name}/results`
27. POST `/api/modules/{name}/reset`
28. GET `/api/modules/{name}/download_log`
29. GET `/api/engine_config` + POST `/api/engine_config` + POST `/api/engine_config/reset`
30. POST `/api/probe_set/apply` + GET `/api/probe_set/apply_status`
31. GET `/api/probe_set/status`
32. POST `/api/probe_set/clear_caches`
33. POST `/api/chat` (with streaming generation)

**HTML routes:**
34. GET `/chat`, `/domain_surface_viz`, `/correction_manifold_viz`,
    `/correction_heatmap_viz`, `/correction_backscatter_viz`

## Sequencing

The audit makes this look like a lot. It is. But the work decomposes
cleanly into bands by what's load-bearing for each user-visible workflow:

**Session 2 (happy path) ✅ COMPLETE:** items 1, 2, 3, 4, 5, plus the `/api/status`
flat-fields fix, plus GET `/api/config`, plus `/api/analyze_batch`
form-data wrapper. 9 endpoints added/extended. Output: load model,
configure capture+measurements, analyze prompts, see results.
`tagm/service/tasm_compat.py` created.

**Session 3 (session management + dashboard) ✅ COMPLETE:** items 12,
13, 14, 15, 16, 19, 20, 21, plus user_info. 8 endpoints. Output: full
session UI works — dashboard, results detail, batch result load,
remove/rerun/restore.

**Session 4 (modules + viz pages + probes) ✅ COMPLETE:** items
22-32, plus all 5 HTML viz routes, plus engine_config, plus the probe
apply/status flow. 19 endpoints/routes. Output: every viz page can
render from session data, modules tab fully functional, probe sets
applied through the UI.
New module: `tagm/service/modules_runner.py` (singleton
`ModuleRunner`).

**Session 5 (chat + plots + cleanup) ✅ COMPLETE:** items 17, 33, 34,
plus export 2-step. 6 endpoints. Output: chat works (instruct or base),
all 12 per-prompt plot keys render server-side as PNG, export 2-step
flow works.
New modules: `tagm/service/chat.py`, `tagm/service/plots.py`.
matplotlib added to requirements.

**Total backend additions:** ~43 endpoints across all four rewiring
sessions, plus 4 new service-layer modules (tasm_compat, modules_runner,
chat, plots), plus extensions to existing endpoints (status flat
fields, models POST, analyze content-type sniffing).

Endpoint count progression:
  Pre-rewiring: 28 endpoints
  After session 2: 38
  After session 3: 46
  After session 4: 65
  After session 5: 71

**Session 3 (session management + dashboard):** items 12, 13, 14, 15,
16, 19, 20, 21, plus the dashboard data assembly. Roughly 10 endpoints.
Output: full session UI works — dashboard, results detail, batch run,
remove/rerun/restore.

**Session 4 (modules + viz pages + probes):** items 22-32, plus all
five HTML viz routes, plus the probe apply/status flow. Output: every
viz page renders correctly from the session data, probe sets can be
applied through the UI, modules tab functions.

**Session 5 (chat + plots + cleanup):** items 17, 33, 34. Output: chat
page works against TAGM's loaded instruct model, plot generation works,
final polish, README update, repackage.

If sessions 2-5 each handle their assigned bands cleanly, the full UI
is restored to TASM's behavior at the end of session 5, with TAGM's
backend underneath.
