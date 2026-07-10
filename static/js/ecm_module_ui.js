/* ECM Module — analytical results UI (v3).
 *
 * Drop-in: include AFTER main.js in static/index.html:
 *   <script src="js/ecm_module_ui.js"></script>
 *
 * v3: rebuilt in the MI/CFT visual language — sectioned collapsible
 * headers, .mod-summary stat grids, .mod-tbl tables. Fully responsive
 * (grids and 100%-width tables; no fixed left-hugging blocks). Every
 * metric carries a title tooltip explaining what it measures. Channel
 * coverage is explicit: missing channels are listed with the reason,
 * never dropped silently.
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

  // ── shared metric tooltips ─────────────────────────────────────

  var TIP = {
    sigma: 'σ-excess: how far the detector signal rose above the deadband, in units of the trace\u2019s own typical volatility. 0 = quiet.',
    fired: 'A record \u201Cfires\u201D when any channel\u2019s signal exceeds 0 (or the Fired threshold parameter, if set).',
    firedPct: 'Fraction of records in this group that fired at least once.',
    intRate: 'Interventions per token: firing steps divided by trace length.',
    ints: 'Number of token steps where the detector signal was above zero.',
    firstAt: 'Token index of the first firing step (after warmup).',
    meanMax: 'Mean over records of each record\u2019s peak σ-excess.',
    peak: 'Largest single σ-excess seen in this group.',
    pearson: 'Pearson r between each record\u2019s peak replay signal and this result-level metric. |r|>0.3 highlighted.',
    channel: 'stress: correction-field stress trace (always collected). kl: instruct-vs-base KL divergence (KL checkbox). density: SFD spectral density, negated so collapse reads as a rise (SFD collection). entropy: the live harvest entropy trace (calibration channel).',
    live: 'What the runtime controller actually did during harvest generation (temperature was really modulated), as opposed to the replay audit, which asks whether the detector WOULD fire on the analytical traces.',
    loopRel: 'Times the live controller released cooling because the token stream became periodic.',
    coverage: 'Which results contributed each channel. Missing channels are listed with the reason \u2014 usually a collection checkbox that was off when the session ran.'
  };

  // ── helpers ────────────────────────────────────────────────────

  function fmt(v, nd) {
    if (v === null || v === undefined || v !== v) return '\u2014';
    if (typeof v !== 'number') return _esc(v);
    return v.toFixed(nd === undefined ? 3 : nd);
  }

  function section(title, bodyHtml, opts) {
    opts = opts || {};
    var collapsed = opts.collapsed ? ' collapsed' : '';
    var tip = opts.tip ? ' title="' + _esc(opts.tip) + '"' : '';
    return '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')"' + tip + '>' +
      _esc(title) + (opts.badge || '') + '</div>' +
      '<div class="mod-results-body' + collapsed + '"' +
      (opts.maxH ? ' style="max-height:' + opts.maxH + 'px"' : '') + '>' +
      bodyHtml + '</div>';
  }

  function stat(label, value, detail, tip, color) {
    return '<div class="mod-stat"' + (tip ? ' title="' + _esc(tip) + '"' : '') + '>' +
      '<div class="stat-label">' + _esc(label) + '</div>' +
      '<div class="stat-value"' + (color ? ' style="color:' + color + '"' : '') + '>' + value + '</div>' +
      (detail ? '<div class="stat-detail">' + detail + '</div>' : '') +
      '</div>';
  }

  function chColor(ch) {
    return ch === 'stress' ? 'var(--orange)'
      : ch === 'kl' ? 'var(--cyan)'
      : ch === 'entropy' ? 'var(--green)'
      : 'var(--red)';
  }

  // ── intervention strip (unchanged renderer, responsive by design:
  //    fixed viewBox, width:100%) ──────────────────────────────────

  function interventionStrip(signals, tokens, colorVar, warmup) {
    if (!signals || !signals.length) return '';
    var n = signals.length;
    var W = 560, H = 46, BASE = 34;
    var mx = 0;
    for (var i = 0; i < n; i++) if (signals[i] > mx) mx = signals[i];
    var scale = mx < 1e-8 ? 1 : mx;
    var pitch = W / n;
    var gap = pitch > 2.5 ? 0.5 : 0;
    var segW = Math.max(0.4, pitch - 2 * gap);

    var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" ' +
      'style="width:100%;height:' + H + 'px;display:block;background:var(--bg-0);' +
      'border:1px solid var(--border);border-radius:3px">';

    var wu = Math.max(0, Math.min(warmup || 0, n));
    if (wu > 0) {
      var wx = wu * pitch;
      s += '<rect x="0" y="0" width="' + wx.toFixed(1) + '" height="' + H +
        '" fill="var(--text-3)" opacity="0.12"><title>warmup (first ' + wu +
        ' tokens): detector calibrating, cannot fire</title></rect>';
      if (wu < n) {
        s += '<line x1="' + wx.toFixed(1) + '" y1="0" x2="' + wx.toFixed(1) + '" y2="' + H +
          '" stroke="var(--text-3)" stroke-width="1" stroke-dasharray="3,3" opacity="0.5"/>';
      }
      if (wx > 52) {
        s += '<text x="4" y="10" fill="var(--text-3)" font-size="8" ' +
          'font-family="var(--mono)" opacity="0.8">WARMUP</text>';
      }
    }

    for (var j = 0; j < n; j++) {
      var x = j * pitch + gap;
      var v = signals[j] || 0;
      var fired = v > 0;
      var tok = (tokens && tokens[j] != null) ? String(tokens[j]) : '';
      var tip = '<title>#' + j + (tok ? ' \u201C' + _esc(tok) + '\u201D' : '') +
        (fired ? '  \u03C3=' + v.toFixed(3) : '') + '</title>';

      s += '<rect x="' + x.toFixed(2) + '" y="' + BASE + '" width="' + segW.toFixed(2) +
        '" height="' + (H - BASE - 3) + '" fill="' + (fired ? colorVar : 'var(--text-3)') +
        '" opacity="' + (fired ? '0.95' : '0.35') + '">' + tip + '</rect>';

      if (fired) {
        var bh = (BASE - 4) * Math.min(1, v / scale);
        s += '<rect x="' + x.toFixed(2) + '" y="' + (BASE - 2 - bh).toFixed(2) +
          '" width="' + segW.toFixed(2) + '" height="' + bh.toFixed(2) +
          '" fill="' + colorVar + '" opacity="0.55">' + tip + '</rect>';
      }

      if (tok && pitch >= 26) {
        var label = tok.trim() || '\u2423';
        if (label.length > 10) label = label.slice(0, 9) + '\u2026';
        s += '<text x="' + (j * pitch + pitch / 2).toFixed(1) + '" y="' + (BASE - 6) +
          '" text-anchor="middle" fill="var(--text-2)" font-size="9" ' +
          'font-family="var(--mono)">' + _esc(label) + tip + '</text>';
      }
    }
    s += '</svg>';
    return s;
  }

  // ── coverage section ───────────────────────────────────────────

  function coverageSection(r) {
    var cov = r.coverage;
    if (!cov) return '';
    var h = '<table class="mod-tbl"><thead><tr>' +
      '<th title="' + _esc(TIP.channel) + '">Channel</th>' +
      '<th class="num" title="Results that contributed this channel to the replay">Available</th>' +
      '<th class="num">Missing</th>' +
      '<th>Why missing</th></tr></thead><tbody>';
    Object.keys(cov).forEach(function (ch) {
      var c = cov[ch];
      var reasons = Object.keys(c.reasons || {}).map(function (k) {
        return _esc(k) + ' (' + c.reasons[k] + ')';
      }).join('; ') || '\u2014';
      var okCol = c.available > 0 ? 'var(--green)' : 'var(--red)';
      h += '<tr>' +
        '<td style="color:' + chColor(ch) + ';font-weight:600">' + _esc(ch) + '</td>' +
        '<td class="num" style="color:' + okCol + '">' + c.available + '</td>' +
        '<td class="num" style="color:var(--text-3)">' + c.missing + '</td>' +
        '<td style="white-space:normal;color:var(--text-2);font-size:10px">' + reasons + '</td>' +
        '</tr>';
    });
    h += '</tbody></table>';
    var skipped = r.skipped || [];
    if (skipped.length) {
      h += '<div style="padding:8px 12px;font-size:10px;color:var(--text-3)">' +
        skipped.length + ' result(s) had no replayable traces at all and were skipped.</div>';
    }
    return h;
  }

  // ── category + channel tables ──────────────────────────────────

  function categoryTable(summary) {
    var cats = (summary && summary.categories) || {};
    var keys = Object.keys(cats);
    if (!keys.length) return '';
    var h = '<table class="mod-tbl"><thead><tr>' +
      '<th>Category</th>' +
      '<th class="num">n</th>' +
      '<th class="num" title="' + _esc(TIP.fired) + '">Fired</th>' +
      '<th class="num" title="' + _esc(TIP.firedPct) + '">Fired %</th>' +
      '<th class="num" title="' + _esc(TIP.meanMax) + '">Mean max σ</th>' +
      '<th class="num" title="' + _esc(TIP.peak) + '">Peak σ</th>' +
      '<th class="num" title="Mean firing steps per record">Mean ints</th>' +
      '</tr></thead><tbody>';
    keys.forEach(function (k) {
      var s = cats[k];
      h += '<tr>' +
        '<td style="color:var(--cyan)">' + _esc(k) + '</td>' +
        '<td class="num">' + (s.n || 0) + '</td>' +
        '<td class="num">' + (s.n_fired || 0) + '</td>' +
        '<td class="num">' + fmt(s.fired_frac, 2) + '</td>' +
        '<td class="num">' + fmt(s.mean_max_signal, 3) + '</td>' +
        '<td class="num">' + fmt(s.max_max_signal, 3) + '</td>' +
        '<td class="num">' + fmt(s.mean_total_interventions, 1) + '</td>' +
        '</tr>';
    });
    h += '</tbody></table>';
    return h;
  }

  function channelTable(channelAgg) {
    if (!channelAgg) return '';
    var keys = Object.keys(channelAgg);
    if (!keys.length) return '';
    var h = '<table class="mod-tbl"><thead><tr>' +
      '<th title="' + _esc(TIP.channel) + '">Channel</th>' +
      '<th class="num">Records</th>' +
      '<th class="num" title="' + _esc(TIP.fired) + '">Fired</th>' +
      '<th class="num" title="' + _esc(TIP.firedPct) + '">Fired %</th>' +
      '<th class="num" title="' + _esc(TIP.intRate) + '">Mean int rate</th>' +
      '<th class="num" title="' + _esc(TIP.meanMax) + '">Mean max σ</th>' +
      '<th class="num" title="' + _esc(TIP.peak) + '">Peak σ</th>' +
      '</tr></thead><tbody>';
    keys.forEach(function (ch) {
      var s = channelAgg[ch];
      h += '<tr>' +
        '<td style="color:' + chColor(ch) + ';font-weight:600">' + _esc(ch) + '</td>' +
        '<td class="num">' + (s.n_with_data || 0) + '</td>' +
        '<td class="num">' + (s.n_fired || 0) + '</td>' +
        '<td class="num">' + fmt(s.fired_frac, 2) + '</td>' +
        '<td class="num">' + fmt(s.mean_intervention_rate, 3) + '</td>' +
        '<td class="num">' + fmt(s.mean_max_signal, 3) + '</td>' +
        '<td class="num">' + fmt(s.max_max_signal, 3) + '</td>' +
        '</tr>';
    });
    h += '</tbody></table>';
    return h;
  }

  function correlationTable(correlations) {
    if (!correlations) return '';
    var keys = Object.keys(correlations);
    if (!keys.length) return '';
    var h = '<table class="mod-tbl" style="max-width:520px"><thead><tr>' +
      '<th>Result metric</th>' +
      '<th class="num" title="' + _esc(TIP.pearson) + '">Pearson r</th>' +
      '<th class="num">n</th></tr></thead><tbody>';
    keys.forEach(function (k) {
      var c = correlations[k];
      var rv = c.pearson_r;
      var col = rv == null ? 'var(--text-3)'
        : Math.abs(rv) > 0.3 ? 'var(--orange)' : 'var(--text-1)';
      h += '<tr>' +
        '<td style="color:var(--text-1)">' + _esc(k) + '</td>' +
        '<td class="num" style="color:' + col + '">' + fmt(rv, 3) + '</td>' +
        '<td class="num" style="color:var(--text-3)">' + (c.n || 0) + '</td></tr>';
    });
    h += '</tbody></table>';
    return h;
  }

  // ── live actuation section ─────────────────────────────────────

  function liveSection(live) {
    if (!live || !live.overall) return '';
    var o = live.overall;
    var h = '<div class="mod-summary">';
    h += stat('harvests', o.n || 0, null, 'Harvest generations that ran with ECM live');
    h += stat('actuated', o.n_fired || 0,
      'fired % ' + fmt(o.fired_frac, 2), 'Generations where the live controller reduced temperature at least once',
      o.n_fired ? 'var(--orange)' : null);
    h += stat('mean ints', fmt(o.mean_interventions, 1), null, TIP.ints);
    h += stat('peak σ', fmt(o.max_signal, 3), null, TIP.sigma);
    h += stat('loop releases', o.n_loop_releases || 0, null, TIP.loopRel);
    h += '</div>';
    var cats = live.categories || {};
    var keys = Object.keys(cats);
    if (keys.length) {
      h += '<table class="mod-tbl"><thead><tr><th>Category</th>' +
        '<th class="num">n</th><th class="num">Actuated</th>' +
        '<th class="num" title="' + _esc(TIP.intRate) + '">Mean int rate</th>' +
        '<th class="num" title="' + _esc(TIP.sigma) + '">Peak σ</th>' +
        '</tr></thead><tbody>';
      keys.forEach(function (k) {
        var s = cats[k];
        h += '<tr><td style="color:var(--cyan)">' + _esc(k) + '</td>' +
          '<td class="num">' + (s.n || 0) + '</td>' +
          '<td class="num">' + (s.n_fired || 0) + '</td>' +
          '<td class="num">' + fmt(s.mean_intervention_rate, 3) + '</td>' +
          '<td class="num">' + fmt(s.max_signal, 3) + '</td></tr>';
      });
      h += '</tbody></table>';
    }
    return h;
  }

  // ── per-record rendering ───────────────────────────────────────

  function renderRecord(rec, warmup, stripLimit) {
    var fired = rec.any_fired;
    var badge = fired
      ? '<span style="color:var(--orange);white-space:nowrap" title="' + _esc(TIP.sigma) + '">' +
        rec.total_interventions + ' int \u00B7 max \u03C3=' + fmt(rec.max_signal, 3) + '</span>'
      : '<span style="color:var(--text-3)">quiet</span>';

    var head = '<div onclick="this.nextElementSibling.style.display=' +
      "this.nextElementSibling.style.display==='none'?'':'none'" + '" ' +
      'style="cursor:pointer;display:flex;gap:10px;align-items:baseline;padding:6px 12px;' +
      'border-top:1px solid var(--border);font-size:11px;min-width:0">' +
      '<span style="color:var(--text-3);font-family:var(--mono);flex-shrink:0">#' + rec.index + '</span>' +
      '<span style="color:var(--cyan);font-family:var(--mono);flex-shrink:0">' + _esc(rec.category) + '</span>' +
      '<span style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;' +
      'color:var(--text-2)">' + _esc(rec.prompt) + '</span>' +
      badge + '</div>';

    var body = '<div style="display:none;padding:8px 12px">';

    var chs = rec.channels || {};
    var chNames = Object.keys(chs);
    if (chNames.length) {
      body += '<table class="mod-tbl" style="margin-bottom:6px"><thead><tr>' +
        '<th title="' + _esc(TIP.channel) + '">Channel</th>' +
        '<th class="num" title="' + _esc(TIP.ints) + '">Ints</th>' +
        '<th class="num" title="' + _esc(TIP.intRate) + '">Rate</th>' +
        '<th class="num" title="' + _esc(TIP.sigma) + '">Max σ</th>' +
        '<th class="num" title="' + _esc(TIP.firstAt) + '">First @</th>' +
        '</tr></thead><tbody>';
      chNames.forEach(function (ch) {
        var d = chs[ch];
        body += '<tr>' +
          '<td style="color:' + chColor(ch) + ';font-weight:600">' + _esc(ch) + '</td>' +
          '<td class="num">' + (d.n_interventions || 0) + '</td>' +
          '<td class="num">' + fmt(d.intervention_rate, 3) + '</td>' +
          '<td class="num">' + fmt(d.max_signal, 3) + '</td>' +
          '<td class="num">' + (d.first_signal_idx != null ? d.first_signal_idx : '\u2014') + '</td>' +
          '</tr>';
      });
      body += '</tbody></table>';
    }

    var miss = rec.missing_channels || {};
    var missNames = Object.keys(miss);
    if (missNames.length) {
      body += '<div style="font-size:10px;color:var(--text-3);margin-bottom:6px" title="' +
        _esc(TIP.coverage) + '">missing: ' +
        missNames.map(function (ch) {
          return '<span style="color:' + chColor(ch) + '">' + _esc(ch) + '</span> (' + _esc(miss[ch]) + ')';
        }).join(' \u00B7 ') + '</div>';
    }

    var traces = rec.traces;
    if (traces) {
      var toks = rec.tokens;
      Object.keys(traces).forEach(function (ch) {
        var full = traces[ch] || [];
        var pk = 0, nInt = 0;
        for (var i = 0; i < full.length; i++) {
          if (full[i] > 0) { nInt++; if (full[i] > pk) pk = full[i]; }
        }
        var lim = stripLimit > 0 ? Math.min(stripLimit, full.length) : full.length;
        var sig = full.slice(0, lim);
        var stoks = toks ? toks.slice(0, lim) : toks;
        var count = (lim < full.length)
          ? lim + ' of ' + full.length + ' tokens'
          : full.length + (full.length === 1 ? ' token' : ' tokens');
        body += '<div style="margin-top:6px">' +
          '<div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:4px">' +
          '<span style="font-size:10px;color:' + chColor(ch) + ';font-family:var(--mono);' +
          'text-transform:uppercase;letter-spacing:.05em">' + _esc(ch) + ' signal</span>' +
          '<span style="font-size:10px;color:var(--text-3);font-family:var(--mono)">' +
          count + (nInt ? ' \u00B7 ' + nInt + ' int \u00B7 peak \u03C3=' + pk.toFixed(3) : ' \u00B7 quiet') +
          '</span></div>' +
          interventionStrip(sig, stoks, chColor(ch), warmup) + '</div>';
      });
    }

    if (rec.stress_score != null || rec.kl_divergence != null || rec.density_mean != null) {
      body += '<div style="display:flex;gap:12px;margin-top:6px;font-size:10px;flex-wrap:wrap;' +
        'font-family:var(--mono);color:var(--text-2)">';
      if (rec.stress_score != null) body += '<span title="Result-level stress score">stress=' + fmt(rec.stress_score, 3) + '</span>';
      if (rec.kl_divergence != null) body += '<span title="Result-level KL divergence (last token)">kl=' + fmt(rec.kl_divergence, 3) + '</span>';
      if (rec.density_mean != null) body += '<span title="Mean SFD density over the sequence">density=' + fmt(rec.density_mean, 4) + '</span>';
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
    var det = r.detector || {};
    var h = '<div class="mod-results">';

    // Detector banner — what THIS replay used (Run-time parameters)
    h += '<div style="padding:10px 16px;background:color-mix(in srgb,var(--blue) 10%,transparent);' +
      'border-left:3px solid var(--blue);margin:8px 0;font-family:var(--mono);font-size:11px;color:var(--text-1)" ' +
      'title="These are the module parameters used for this replay. Change them above and click Run \u2014 no re-analysis needed.">' +
      '<span style="color:var(--blue);font-weight:700">REPLAY</span> ' +
      'scales=' + (det.n_scales != null ? det.n_scales : '?') +
      ' \u00B7 deadband=' + (det.deadband != null ? det.deadband : '?') + '\u03C3' +
      ' \u00B7 agreement=' + (det.agreement != null ? det.agreement : '?') +
      ' \u00B7 warmup=' + (det.warmup != null ? det.warmup : '?') +
      '<span style="color:var(--text-2);margin-left:10px">' +
      (r.n_ecm || 0) + '/' + (r.n_total || 0) + ' results replayed</span></div>';

    // Overview stats
    var ov = '<div class="mod-summary">';
    ov += stat('records', r.n_ecm || 0, null, 'Results with at least one replayable trace');
    ov += stat('fired', o.n_fired || 0, 'fired % ' + fmt(o.fired_frac, 2), TIP.fired,
      o.n_fired ? 'var(--orange)' : null);
    ov += stat('mean max σ', fmt(o.mean_max_signal, 3), null, TIP.meanMax);
    ov += stat('peak σ', fmt(o.max_max_signal, 3), null, TIP.peak,
      o.max_max_signal > 0.5 ? 'var(--red)' : null);
    ov += stat('mean ints', fmt(o.mean_total_interventions, 1), null, TIP.ints);
    ov += '</div>';
    h += section('Replay Audit \u2014 Overview', ov,
      { tip: 'Would the detector fire on the analytical traces, with the settings above?' });

    // Coverage
    h += section('Channel Coverage', coverageSection(r), { tip: TIP.coverage });

    // Category separability
    h += section('By Category', '<div style="padding:8px 12px">' +
      categoryTable(r.summary || {}) + '</div>');

    // Channel breakdown
    if (o.channels) {
      h += section('By Channel', '<div style="padding:8px 12px">' +
        channelTable(o.channels) + '</div>', { tip: TIP.channel });
    }

    // Correlations
    if (o.correlations && Object.keys(o.correlations).length) {
      h += section('Signal\u2013Metric Correlations',
        '<div style="padding:8px 12px">' + correlationTable(o.correlations) + '</div>',
        { collapsed: true, tip: TIP.pearson });
    }

    // Live actuation
    if (r.live_summary) {
      h += section('Live Actuation (harvest generations)', liveSection(r.live_summary),
        { collapsed: true, tip: TIP.live });
    }

    // Per-record details
    var recs = r.records || [];
    if (recs.length) {
      var body = '';
      var cap = Math.min(recs.length, 200);
      var warmup = det.warmup != null ? det.warmup : 4;
      var stripLimit = r.strip_token_limit || 0;
      for (var i = 0; i < cap; i++) body += renderRecord(recs[i], warmup, stripLimit);
      if (recs.length > cap) {
        body += '<div style="padding:6px 12px;font-size:11px;color:var(--text-3)">' +
          (recs.length - cap) + ' more in the JSONL.</div>';
      }
      if (r.jsonl_path) {
        body += '<div style="padding:6px 12px;font-size:10px;color:var(--text-3);' +
          'font-family:var(--mono);overflow-wrap:anywhere">full records \u2192 ' +
          _esc(r.jsonl_path) + '</div>';
      }
      h += section('Per-Record Replays \u2014 click a row to expand', body,
        { maxH: 700 });
    }

    h += '</div>';
    return h;
  }

})();
