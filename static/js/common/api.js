/* TAGM — shared JSON fetch helper. Plain <script src>-able; attaches to TAGM.
 *
 * Three distinct failure shapes exist server-side and call sites used to
 * handle at most one of them:
 *   1. non-2xx with a FastAPI `{"detail": ...}` body — e.g. the 404 from
 *      GET /api/modules/{name}/results (src/app.py). No caller handled this;
 *      `d.ok` was simply undefined and the error vanished.
 *   2. HTTP 200 with `{"ok": false, "error": "..."}` — the app's own shape.
 *   3. network failure / non-JSON body (HTML error page, empty response).
 *
 * getJSON() collapses all three into one shape: `{ok:false, error:<string>,
 * status:<number>}`, so callers only ever branch on `ok`.
 */
(function (global) {
  var TAGM = global.TAGM = global.TAGM || {};

  function _detailToString(d) {
    if (d === null || d === undefined) return '';
    if (typeof d === 'string') return d;
    try { return JSON.stringify(d); } catch (e) { return String(d); }
  }

  async function getJSON(url, opts) {
    var resp;
    try {
      resp = await fetch(url, opts);
    } catch (e) {
      return { ok: false, status: 0, error: 'Network error: ' + e.message };
    }

    var text;
    try {
      text = await resp.text();
    } catch (e) {
      return { ok: false, status: resp.status, error: 'Unreadable response: ' + e.message };
    }

    var body;
    try {
      body = text ? JSON.parse(text) : {};
    } catch (e) {
      // Non-JSON body: an HTML error page, a proxy banner, or a truncated
      // stream. Report the HTTP status rather than a JSON.parse SyntaxError.
      return {
        ok: false,
        status: resp.status,
        error: resp.ok ? 'Malformed JSON response from ' + url
                       : ('HTTP ' + resp.status + ' ' + (resp.statusText || ''))
      };
    }
    if (body === null || typeof body !== 'object') body = { value: body };

    var detail = _detailToString(body.detail);

    if (!resp.ok) {
      return {
        ok: false,
        status: resp.status,
        error: detail || body.error || ('HTTP ' + resp.status + ' ' + (resp.statusText || '')),
        data: body
      };
    }
    if (body.ok === false) {
      return Object.assign({}, body, {
        ok: false,
        status: resp.status,
        error: body.error || detail || 'Request failed'
      });
    }
    // 200 with a bare {"detail": ...} and no ok flag is still an error body.
    if (body.ok === undefined && detail) {
      return { ok: false, status: resp.status, error: detail, data: body };
    }
    return Object.assign({ ok: true, status: resp.status }, body);
  }

  TAGM.getJSON = getJSON;
})(typeof window !== 'undefined' ? window : globalThis);
