/* ECM Module — analytical results UI (v2).
 *
 * Drop-in: include AFTER main.js in static/index.html:
 *   <script src="js/ecm_module_ui.js"></script>
 *
 * Renders aggregate ECM replay analysis from session data.
 * The ECM checkbox collects cascade-detector replay data during
 * inference; this module analyzes and visualizes that data.
 *
 * Visual language matches the app's tokens (--bg-0, --text-2, --mono,
 * mod-results / metrics conventions). The per-record view shows
 * per-channel signal traces as SVG sparklines.
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

  // ── intervention strip ───────────────────────────────────────
  // One segment per token along a baseline; segments light up where
  // the detector fired, with an amplitude bar (σ-excess) above.
  // Warmup region is shaded: the detector cannot fire there.

  function interventionStrip(signals, tokens, accentVar, warmup) {
    if (!signals || !signals.length) return '';
    var n = signals.length;
    var W = 560, H = 46, BASE = 34;       // bars grow upward from BASE
    var mx = 0;
    for (var i = 0; i < n; i++) if (signals[i] > mx) mx = signals[i];
    var scale = mx < 1e-8 ? 1 : mx;
    var pitch = W / n;
    var gap = pitch > 2.5 ? 0.5 : 0;
    var segW = Math.max(0.4, pitch - 2 * gap);

    var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none" ' +
      'style="width:100%;height:' + H + 'px;display:block;background:var(--bg-0);' +
      'border:1px solid var(--border);border-radius:3px">';

    // Warmup shading — no signal is possible before this index
    var wu = Math.max(0, Math.min(warmup || 0, n));
    if (wu > 0) {
      s += '<rect x="0" y="0" width="' + (wu * pitch).toFixed(1) + '" height="' + H +
        '" fill="var(--text-3)" opacity="0.07"><title>warmup (first ' + wu +
        ' tokens): detector calibrating, cannot fire</title></rect>';
    }

    for (var j = 0; j < n; j++) {
      var x = j * pitch + gap;
      var v = signals[j] || 0;
      var fired = v > 0;
      var tok = (tokens && tokens[j] != null) ? String(tokens[j]) : '';
      var tip = '<title>#' + j + (tok ? ' \u201C' + _esc(tok) + '\u201D' : '') +
        (fired ? '  \u03C3=' + v.toFixed(3) : '') + '</title>';

      // Baseline segment — every token gets one; fired tokens light up
      s += '<rect x="' + x.toFixed(2) + '" y="' + BASE + '" width="' + segW.toFixed(2) +
        '" height="' + (H - BASE - 3) + '" fill="var(--' + (fired ? accentVar : 'text-3') +
        ')" opacity="' + (fired ? '0.95' : '0.18') + '">' + tip + '</rect>';

      // Amplitude bar
      if (fired) {
        var bh = (BASE - 4) * Math.min(1, v / scale);
        s += '<rect x="' + x.toFixed(2) + '" y="' + (BASE - 2 - bh).toFixed(2) +
          '" width="' + segW.toFixed(2) + '" height="' + bh.toFixed(2) +
          '" fill="var(--' + accentVar + ')" opacity="0.55">' + tip + '</rect>';
      }
    }
    s += '</svg>';
    return s;
  }

  // ── category table ─────────────────────────────────────────────

  function categoryTable(summary) {
    var cats = summary.categories || {};
    var keys = Object.keys(cats);
    if (!keys.length) return '';
    var cols = [
      ['n', 'n', 0],
      ['fired', 'n_fired', 0],
      ['fired %', 'fired_frac', 2],
      ['mean max σ', 'mean_max_signal', 3],
      ['peak σ', 'max_max_signal', 3],
      ['mean ints', 'mean_total_interventions', 1],
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

  // ── channel breakdown table ────────────────────────────────────

  function channelTable(channelAgg) {
    if (!channelAgg) return '';
    var keys = Object.keys(channelAgg);
    if (!keys.length) return '';

    var h = '<div style="margin-top:10px;font-size:10px;color:var(--text-3);' +
      'text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px">Per-channel breakdown</div>';
    h += '<table style="font-size:11px;border-collapse:collapse;width:100%">';
    h += '<tr style="color:var(--text-3);font-size:10px;text-transform:uppercase">' +
      '<td style="padding:3px 8px">channel</td>' +
      '<td style="padding:3px 8px;text-align:right">records</td>' +
      '<td style="padding:3px 8px;text-align:right">fired</td>' +
      '<td style="padding:3px 8px;text-align:right">fired %</td>' +
      '<td style="padding:3px 8px;text-align:right">mean int rate</td>' +
      '<td style="padding:3px 8px;text-align:right">mean max σ</td>' +
      '<td style="padding:3px 8px;text-align:right">peak σ</td>' +
      '</tr>';
    keys.forEach(function (ch) {
      var s = channelAgg[ch];
      var color = ch === 'stress' ? 'orange' : (ch === 'kl' ? 'cyan' : 'red');
      h += '<tr>' +
        '<td style="padding:3px 8px;color:var(--' + color + ');font-family:var(--mono)">' + _esc(ch) + '</td>' +
        '<td style="padding:3px 8px;text-align:right;font-family:var(--mono)">' + (s.n_with_data || 0) + '</td>' +
        '<td style="padding:3px 8px;text-align:right;font-family:var(--mono)">' + (s.n_fired || 0) + '</td>' +
        '<td style="padding:3px 8px;text-align:right;font-family:var(--mono)">' + fmt(s.fired_frac, 2) + '</td>' +
        '<td style="padding:3px 8px;text-align:right;font-family:var(--mono)">' + fmt(s.mean_intervention_rate, 3) + '</td>' +
        '<td style="padding:3px 8px;text-align:right;font-family:var(--mono)">' + fmt(s.mean_max_signal, 3) + '</td>' +
        '<td style="padding:3px 8px;text-align:right;font-family:var(--mono)">' + fmt(s.max_max_signal, 3) + '</td>' +
        '</tr>';
    });
    h += '</table>';
    return h;
  }

  // ── correlation table ──────────────────────────────────────────

  function correlationTable(correlations) {
    if (!correlations) return '';
    var keys = Object.keys(correlations);
    if (!keys.length) return '';

    var h = '<div style="margin-top:10px;font-size:10px;color:var(--text-3);' +
      'text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px">' +
      'Signal–metric correlations (Pearson r)</div>';
    h += '<table style="font-size:11px;border-collapse:collapse">';
    keys.forEach(function (k) {
      var c = correlations[k];
      var r = c.pearson_r;
      var rColor = r == null ? 'text-3' : (Math.abs(r) > 0.3 ? 'orange' : 'text-2');
      h += '<tr>' +
        '<td style="padding:2px 10px;color:var(--text-2)">' + _esc(k) + '</td>' +
        '<td style="padding:2px 10px;font-family:var(--mono);color:var(--' + rColor + ')">' +
        fmt(r, 3) + '</td>' +
        '<td style="padding:2px 10px;color:var(--text-3);font-size:10px">n=' + (c.n || 0) + '</td>' +
        '</tr>';
    });
    h += '</table>';
    return h;
  }

  // ── per-record rendering ───────────────────────────────────────

  function renderRecord(rec, idx, warmup) {
    var fired = rec.any_fired;
    var badge = fired
      ? '<span style="color:var(--orange)">' + rec.total_interventions + ' int, max σ=' + fmt(rec.max_signal, 3) + '</span>'
      : '<span style="color:var(--text-3)">quiet</span>';

    var head = '<div onclick="this.nextElementSibling.style.display=' +
      "this.nextElementSibling.style.display==='none'?'':'none'" + '" ' +
      'style="cursor:pointer;display:flex;gap:10px;align-items:baseline;padding:6px 10px;' +
      'border-top:1px solid var(--border);font-size:11px">' +
      '<span style="color:var(--text-3);font-family:var(--mono)">#' + rec.index + '</span>' +
      '<span style="color:var(--cyan);font-family:var(--mono)">' + _esc(rec.category) + '</span>' +
      '<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;' +
      'color:var(--text-2)">' + _esc(rec.prompt) + '</span>' +
      badge + '</div>';

    // Expanded body
    var body = '<div style="display:none;padding:8px 10px">';

    // Channel details
    var chs = rec.channels || {};
    var chNames = Object.keys(chs);
    if (chNames.length) {
      body += '<table style="font-size:11px;border-collapse:collapse;margin-bottom:6px">';
      body += '<tr style="color:var(--text-3);font-size:10px;text-transform:uppercase">' +
        '<td style="padding:2px 8px">channel</td>' +
        '<td style="padding:2px 8px;text-align:right">ints</td>' +
        '<td style="padding:2px 8px;text-align:right">rate</td>' +
        '<td style="padding:2px 8px;text-align:right">max σ</td>' +
        '<td style="padding:2px 8px;text-align:right">first @</td></tr>';
      chNames.forEach(function (ch) {
        var d = chs[ch];
        var color = ch === 'stress' ? 'orange' : (ch === 'kl' ? 'cyan' : 'red');
        body += '<tr>' +
          '<td style="padding:2px 8px;color:var(--' + color + ');font-family:var(--mono)">' + _esc(ch) + '</td>' +
          '<td style="padding:2px 8px;text-align:right;font-family:var(--mono)">' + (d.n_interventions || 0) + '</td>' +
          '<td style="padding:2px 8px;text-align:right;font-family:var(--mono)">' + fmt(d.intervention_rate, 3) + '</td>' +
          '<td style="padding:2px 8px;text-align:right;font-family:var(--mono)">' + fmt(d.max_signal, 3) + '</td>' +
          '<td style="padding:2px 8px;text-align:right;font-family:var(--mono)">' +
          (d.first_signal_idx != null ? d.first_signal_idx : '—') + '</td></tr>';
      });
      body += '</table>';
    }

    // Signal traces (segmented intervention strips)
    var traces = rec.traces;
    if (traces) {
      var trNames = Object.keys(traces);
      var toks = rec.tokens;
      trNames.forEach(function (ch) {
        var color = ch === 'stress' ? 'orange' : (ch === 'kl' ? 'cyan' : 'red');
        var sig = traces[ch] || [];
        var pk = 0, nInt = 0;
        for (var i = 0; i < sig.length; i++) {
          if (sig[i] > 0) { nInt++; if (sig[i] > pk) pk = sig[i]; }
        }
        body += '<div style="margin-top:6px">' +
          '<div style="display:flex;justify-content:space-between;align-items:baseline">' +
          '<span style="font-size:10px;color:var(--' + color + ');font-family:var(--mono);' +
          'text-transform:uppercase;letter-spacing:.05em">' + _esc(ch) + ' signal</span>' +
          '<span style="font-size:10px;color:var(--text-3);font-family:var(--mono)">' +
          sig.length + ' tokens' +
          (nInt ? ' · ' + nInt + ' int · peak σ=' + pk.toFixed(3) : ' · quiet') +
          '</span></div>' +
          interventionStrip(sig, toks, color, warmup) + '</div>';
      });
    }

    // Companion metrics
    if (rec.stress_score != null || rec.kl_divergence != null || rec.density_mean != null) {
      body += '<div style="display:flex;gap:12px;margin-top:6px;font-size:10px;' +
        'font-family:var(--mono);color:var(--text-2)">';
      if (rec.stress_score != null) body += '<span>stress=' + fmt(rec.stress_score, 3) + '</span>';
      if (rec.kl_divergence != null) body += '<span>kl=' + fmt(rec.kl_divergence, 3) + '</span>';
      if (rec.density_mean != null) body += '<span>density=' + fmt(rec.density_mean, 4) + '</span>';
      body += '</div>';
    }

    body += '</div>';
    return head + body;
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
      'ECM Analysis Results</div>';
    h += '<div class="mod-results-body">';

    // Data coverage bar
    h += '<div style="padding:8px 12px;font-size:11px;color:var(--text-2)">' +
      'Analyzed <span style="color:var(--cyan);font-family:var(--mono)">' +
      (r.n_ecm || 0) + '</span> of ' + (r.n_total || 0) + ' session results' +
      (r.n_without_ecm ? ' <span style="color:var(--text-3)">(' + r.n_without_ecm +
        ' without ECM data)</span>' : '') + '</div>';

    // Summary cards
    h += '<div style="display:flex;gap:8px;flex-wrap:wrap;padding:10px 12px">';
    h += card('records', r.n_ecm || 0);
    h += card('fired', o.n_fired || 0, o.n_fired ? 'orange' : null);
    h += card('fired %', fmt(o.fired_frac, 2), o.fired_frac > 0 ? 'orange' : null);
    h += card('mean max σ', fmt(o.mean_max_signal, 3));
    h += card('peak σ', fmt(o.max_max_signal, 3), o.max_max_signal > 0.5 ? 'red' : null);
    h += card('mean ints', fmt(o.mean_total_interventions, 1));
    h += '</div>';

    // Detector config
    var det = r.detector || {};
    if (det.n_scales || det.deadband || det.agreement) {
      h += '<div style="padding:0 12px 4px;font-size:10px;color:var(--text-3);font-family:var(--mono)">' +
        'detector: scales=' + (det.n_scales || '?') +
        ' deadband=' + (det.deadband || '?') +
        'σ agreement=' + (det.agreement || '?') + '</div>';
    }

    // Category table
    h += '<div style="padding:0 12px 8px">' + categoryTable(r.summary || {}) + '</div>';

    // Per-channel breakdown (from overall)
    if (o.channels) {
      h += '<div style="padding:0 12px 8px">' + channelTable(o.channels) + '</div>';
    }

    // Correlations (from overall)
    if (o.correlations && Object.keys(o.correlations).length) {
      h += '<div style="padding:0 12px 8px">' + correlationTable(o.correlations) + '</div>';
    }

    // JSONL path
    if (r.jsonl_path) {
      h += '<div style="padding:0 12px 8px;font-size:10px;color:var(--text-3);' +
        'font-family:var(--mono)">full records → ' + _esc(r.jsonl_path) + '</div>';
    }

    // Per-record details
    var recs = r.records || [];
    if (recs.length) {
      h += '<div style="border-top:1px solid var(--border)">';
      h += '<div style="padding:6px 10px;font-size:10px;color:var(--text-3);' +
        'text-transform:uppercase;letter-spacing:.05em">Per-record ECM analysis — click to expand</div>';
      var cap = Math.min(recs.length, 200);
      var warmup = (r.detector && r.detector.warmup) || 8;
      for (var i = 0; i < cap; i++) h += renderRecord(recs[i], i, warmup);
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
