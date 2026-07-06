/* ECM Module — results UI.
 *
 * Drop-in: include AFTER main.js in static/index.html:
 *   <script src="js/ecm_module_ui.js"></script>
 *
 * No edits to main.js required. This file wraps the global
 * renderModuleResults dispatcher and claims the 'ecm' module name;
 * every other module falls through to the original renderer. The
 * parameter widgets themselves are auto-rendered by the existing
 * module framework from the module's parameter schema.
 *
 * Visual language matches the app's tokens (--bg-0, --text-2, --mono,
 * mod-results / metrics conventions). The one bespoke element is the
 * generation-path strip: per-token temperature drawn as an SVG line
 * with the first-actuation and divergence points marked, so "where
 * the ECM rearranged the path" is readable at a glance.
 */
(function () {
  'use strict';

  var _esc = (typeof window.escHtml === 'function')
    ? window.escHtml
    : function (s) {
        return String(s == null ? '' : s)
          .replace(/&/g, '&amp;').replace(/</g, '&lt;')
          .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
      };

  var _orig = window.renderModuleResults;
  window.renderModuleResults = function (name, results) {
    if (name === 'ecm') {
      var c = document.getElementById('mod-results-ecm');
      if (c) c.innerHTML = renderEcmResults(results);
      return;
    }
    if (typeof _orig === 'function') return _orig(name, results);
  };

  // ── helpers ────────────────────────────────────────────────────

  function fmt(v, nd) {
    if (v === null || v === undefined || v !== v) return '—';
    if (typeof v !== 'number') return _esc(v);
    return v.toFixed(nd === undefined ? 3 : nd);
  }

  function card(label, value, accent) {
    return '<div style="background:var(--bg-0);border:1px solid var(--border);' +
      'border-radius:4px;padding:8px 12px;min-width:96px">' +
      '<div style="font-size:10px;color:var(--text-3);text-transform:uppercase;' +
      'letter-spacing:.05em">' + _esc(label) + '</div>' +
      '<div style="font-size:16px;font-weight:600;font-family:var(--mono);' +
      (accent ? 'color:var(--' + accent + ')' : 'color:var(--text-1)') + '">' +
      value + '</div></div>';
  }

  // ── generation-path strip ──────────────────────────────────────
  // Temperature trace (line, left axis implied 0..base) + fused signal
  // (filled steps) + markers: ▲ first actuation, ● divergence.

  function pathStrip(rec, cfg) {
    var tr = rec.trace || {};
    var temps = tr.temperature || [];
    if (!temps.length) return '';
    var fused = tr.fused || [];
    var W = 560, H = 64, PAD = 4;
    var n = temps.length;
    var base = (cfg && cfg.base_temperature) || 0.7;
    var tMax = base, tMin = 0;
    var x = function (i) { return PAD + (W - 2 * PAD) * (n <= 1 ? 0 : i / (n - 1)); };
    var yT = function (t) {
      var v = Math.max(tMin, Math.min(tMax, t == null ? base : t));
      return PAD + (H - 2 * PAD) * (1 - (v - tMin) / (tMax - tMin || 1));
    };
    var fMax = 0;
    for (var i = 0; i < fused.length; i++) if (fused[i] > fMax) fMax = fused[i];

    var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none" ' +
      'style="width:100%;height:' + H + 'px;display:block;background:var(--bg-0);' +
      'border:1px solid var(--border);border-radius:4px">';

    // fused signal as translucent columns from the bottom
    if (fMax > 0) {
      var bw = Math.max(1, (W - 2 * PAD) / n);
      for (var j = 0; j < fused.length; j++) {
        if (!fused[j]) continue;
        var h = (H - 2 * PAD) * Math.min(1, fused[j] / fMax) * 0.85;
        s += '<rect x="' + (x(j) - bw / 2).toFixed(1) + '" y="' + (H - PAD - h).toFixed(1) +
          '" width="' + bw.toFixed(1) + '" height="' + h.toFixed(1) +
          '" fill="var(--orange)" opacity="0.30"/>';
      }
    }

    // base temperature reference line
    s += '<line x1="' + PAD + '" y1="' + yT(base).toFixed(1) + '" x2="' + (W - PAD) +
      '" y2="' + yT(base).toFixed(1) + '" stroke="var(--border)" stroke-dasharray="3,3"/>';

    // temperature polyline
    var pts = [];
    for (var k = 0; k < n; k++) pts.push(x(k).toFixed(1) + ',' + yT(temps[k]).toFixed(1));
    s += '<polyline points="' + pts.join(' ') + '" fill="none" ' +
      'stroke="var(--cyan)" stroke-width="1.5"/>';

    // markers
    if (rec.first_actuation_idx !== null && rec.first_actuation_idx !== undefined) {
      var xa = x(rec.first_actuation_idx);
      s += '<line x1="' + xa.toFixed(1) + '" y1="' + PAD + '" x2="' + xa.toFixed(1) +
        '" y2="' + (H - PAD) + '" stroke="var(--orange)" stroke-width="1"/>';
    }
    if (rec.divergence_idx !== null && rec.divergence_idx !== undefined) {
      var xd = x(rec.divergence_idx);
      s += '<circle cx="' + xd.toFixed(1) + '" cy="' + yT(temps[rec.divergence_idx] || base).toFixed(1) +
        '" r="3.5" fill="var(--red)"/>';
    }
    s += '</svg>';

    var legend = '<div style="display:flex;gap:14px;font-size:10px;color:var(--text-3);' +
      'margin-top:3px;font-family:var(--mono)">' +
      '<span><span style="color:var(--cyan)">━</span> effective temperature</span>' +
      '<span><span style="color:var(--orange)">▮</span> fused signal</span>' +
      '<span><span style="color:var(--orange)">│</span> first actuation' +
      (rec.first_actuation_idx != null ? ' @' + rec.first_actuation_idx : '') + '</span>' +
      '<span><span style="color:var(--red)">●</span> path divergence' +
      (rec.divergence_idx != null ? ' @' + rec.divergence_idx : ' (none)') + '</span>' +
      '</div>';
    return s + legend;
  }

  // ── side-by-side texts with divergence emphasis ────────────────

  function pairTexts(rec) {
    function box(title, text, accent) {
      return '<div style="flex:1;min-width:240px">' +
        '<div style="font-size:10px;text-transform:uppercase;letter-spacing:.05em;' +
        'color:var(--' + accent + ');margin-bottom:3px">' + title + '</div>' +
        '<div style="background:var(--bg-0);border:1px solid var(--border);border-radius:4px;' +
        'padding:8px;font-size:11px;line-height:1.5;max-height:180px;overflow:auto;' +
        'white-space:pre-wrap">' + _esc(text || '') + '</div></div>';
    }
    return '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:8px">' +
      box('Control — ECM off', rec.control_text, 'text-3') +
      box('Treatment — ECM on', rec.treatment_text, 'cyan') +
      '</div>';
  }

  function measuresRow(rec) {
    var c = rec.control_measures, t = rec.treatment_measures;
    if (!c && !t) return '';
    function row(label, key, nd) {
      var cv = c ? c[key] : null, tv = t ? t[key] : null;
      var d = (cv != null && tv != null) ? (tv - cv) : null;
      var dCol = d == null ? 'var(--text-3)' : (d < 0 ? 'var(--cyan)' : 'var(--orange)');
      return '<tr>' +
        '<td style="padding:2px 10px;color:var(--text-2)">' + label + '</td>' +
        '<td style="padding:2px 10px;font-family:var(--mono)">' + fmt(cv, nd) + '</td>' +
        '<td style="padding:2px 10px;font-family:var(--mono)">' + fmt(tv, nd) + '</td>' +
        '<td style="padding:2px 10px;font-family:var(--mono);color:' + dCol + '">' +
        (d == null ? '—' : (d >= 0 ? '+' : '') + d.toFixed(nd)) + '</td></tr>';
    }
    return '<table style="font-size:11px;border-collapse:collapse;margin-top:8px">' +
      '<tr style="color:var(--text-3);font-size:10px;text-transform:uppercase">' +
      '<td style="padding:2px 10px"></td><td style="padding:2px 10px">control</td>' +
      '<td style="padding:2px 10px">treatment</td><td style="padding:2px 10px">Δ</td></tr>' +
      row('Stress', 'stress_score', 4) +
      row('KL(inst‖base)', 'kl_divergence', 4) +
      row('SFD density', 'density_mean', 4) +
      '</table>';
  }

  // ── record + category renderers ────────────────────────────────

  function renderRecord(rec, idx, cfg) {
    var badge = rec.error
      ? '<span style="color:var(--red)">error</span>'
      : (rec.divergence_idx != null
        ? '<span style="color:var(--red)">diverged @' + rec.divergence_idx + '</span>'
        : '<span style="color:var(--text-3)">identical paths</span>');
    var parity = rec.parity_ok === false
      ? ' <span style="color:var(--orange)" title="Divergence precedes first actuation — parity leak">⚠ parity</span>'
      : '';
    var head = '<div onclick="this.nextElementSibling.style.display=' +
      "this.nextElementSibling.style.display==='none'?'':'none'" + '" ' +
      'style="cursor:pointer;display:flex;gap:10px;align-items:baseline;padding:6px 10px;' +
      'border-top:1px solid var(--border);font-size:11px">' +
      '<span style="color:var(--text-3);font-family:var(--mono)">#' + idx + '</span>' +
      '<span style="color:var(--cyan);font-family:var(--mono)">' + _esc(rec.category) + '</span>' +
      '<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;' +
      'color:var(--text-2)">' + _esc(rec.prompt) + '</span>' +
      '<span style="font-family:var(--mono)">' + (rec.n_interventions || 0) + ' int</span>' +
      badge + parity + '</div>';

    var body;
    if (rec.error) {
      body = '<div style="display:none;padding:8px 10px;color:var(--orange);font-size:11px">' +
        _esc(rec.error) + '</div>';
    } else {
      body = '<div style="display:none;padding:8px 10px">' +
        pathStrip(rec, cfg) + pairTexts(rec) + measuresRow(rec) + '</div>';
    }
    return head + body;
  }

  function categoryTable(summary) {
    var cats = summary.categories || {};
    var keys = Object.keys(cats);
    if (!keys.length) return '';
    var cols = [
      ['n', 'n', 0], ['int rate', 'intervention_rate', 3],
      ['diverged', 'diverged_frac', 2], ['div idx', 'mean_divergence_idx', 1],
      ['act idx', 'mean_first_actuation_idx', 1],
      ['min T', 'mean_min_temperature', 3],
      ['Δstress', 'delta_stress_score', 4],
      ['Δdensity', 'delta_density_mean', 4],
      ['ΔKL', 'delta_kl_divergence', 4]
    ];
    var h = '<table style="font-size:11px;border-collapse:collapse;width:100%;margin-top:8px">';
    h += '<tr style="color:var(--text-3);font-size:10px;text-transform:uppercase">' +
      '<td style="padding:3px 8px">category</td>';
    cols.forEach(function (c) { h += '<td style="padding:3px 8px;text-align:right">' + c[0] + '</td>'; });
    h += '</tr>';
    keys.forEach(function (k) {
      var s = cats[k];
      h += '<tr><td style="padding:3px 8px;color:var(--cyan);font-family:var(--mono)">' + _esc(k) + '</td>';
      cols.forEach(function (c) {
        h += '<td style="padding:3px 8px;text-align:right;font-family:var(--mono)">' +
          fmt(s[c[1]], c[2]) + '</td>';
      });
      h += '</tr>';
    });
    h += '</table>';
    return h;
  }

  // ── top-level ──────────────────────────────────────────────────

  function renderEcmResults(r) {
    if (!r) return '<div class="mod-results"><div class="mod-results-body" ' +
      'style="padding:16px;color:var(--text-3)">No results yet.</div></div>';
    if (r.error) return '<div class="mod-results"><div class="mod-results-body" ' +
      'style="padding:16px;color:var(--orange)">' + _esc(r.error) + '</div></div>';

    var o = (r.summary && r.summary.overall) || {};
    var h = '<div class="mod-results">';
    h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">' +
      'ECM A/B Results</div>';
    h += '<div class="mod-results-body">';

    h += '<div style="display:flex;gap:8px;flex-wrap:wrap;padding:10px 12px">';
    h += card('pairs', r.n_pairs || 0);
    h += card('errors', r.n_errors || 0, r.n_errors ? 'orange' : null);
    h += card('int rate', fmt(o.intervention_rate, 3), 'orange');
    h += card('diverged', fmt(o.diverged_frac, 2), 'red');
    h += card('mean div idx', fmt(o.mean_divergence_idx, 1));
    h += card('mean min T', fmt(o.mean_min_temperature, 3), 'cyan');
    h += card('parity ⚠', o.parity_violations || 0,
      o.parity_violations ? 'orange' : null);
    h += card('loop releases', o.loop_releases || 0);
    h += '</div>';

    h += '<div style="padding:0 12px 8px">' + categoryTable(r.summary || {}) + '</div>';

    if (r.jsonl_path) {
      h += '<div style="padding:0 12px 8px;font-size:10px;color:var(--text-3);' +
        'font-family:var(--mono)">full records → ' + _esc(r.jsonl_path) + '</div>';
    }

    var recs = r.records || [];
    if (recs.length) {
      h += '<div style="border-top:1px solid var(--border)">';
      h += '<div style="padding:6px 10px;font-size:10px;color:var(--text-3);' +
        'text-transform:uppercase;letter-spacing:.05em">Generation pairs — click to expand</div>';
      var cfg = r.config || {};
      var cap = Math.min(recs.length, 200);
      for (var i = 0; i < cap; i++) h += renderRecord(recs[i], i, cfg);
      if (recs.length > cap) {
        h += '<div style="padding:6px 10px;font-size:11px;color:var(--text-3)">' +
          (recs.length - cap) + ' more in the JSONL.</div>';
      }
      h += '</div>';
    }

    h += '</div></div>';
    return h;
  }

})();
