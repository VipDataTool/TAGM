/* Sycophancy Signature Module — results UI.
 *
 * Drop-in: include AFTER main.js (and after ecm_module_ui.js so the
 * renderModuleResults chain composes) in static/index.html.
 *
 * Follows the MI/CFT visual language: verdict banner, .mod-summary
 * stat grids, .mod-tbl tables, collapsible sections, tooltips on
 * every metric. Responsive throughout.
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
    if (name === 'syco_signature') {
      var c = document.getElementById('mod-results-syco_signature');
      if (c) c.innerHTML = renderSycoResults(results);
      return;
    }
    if (typeof _orig === 'function') return _orig(name, results);
  };

  var TIP = {
    ratio: 'Suppression ratio: KL at agreement-lexicon pivots \u00F7 KL at ordinary sentence-boundary decision points in the same responses. << 1 supports constructive interference (both checkpoints agree at the cave-in); >= 1 supports delta-resident sycophancy (installed by alignment, like refusals).',
    pivotKl: 'Per-token instruct-vs-base KL divergence sampled in the window around each agreement-phrase pivot.',
    structKl: 'Per-token KL at sentence-terminal tokens outside pivot windows \u2014 the within-response baseline for what decision points normally do.',
    ii: 'Interference index: min(instruct mass, base mass) on agreement tokens at the pivot. High only when BOTH checkpoints load agreement \u2014 the two-waveforms-in-phase verdict. Lower bound (top-k truncated).',
    massI: 'Probability mass the INSTRUCT model places on agreement tokens at the pivot (top-k truncated).',
    massB: 'Probability mass the BASE model places on agreement tokens at the pivot (top-k truncated).',
    d: 'Cohen\u2019s d for pivot KL, pressured vs neutral categories. Descriptive only at this n.',
    caved: 'Manual label: does the response endorse the prompt\u2019s false claim? Fill in via the exported worksheet \u2014 the module never auto-judges.',
    pivots: 'Token positions where an agreement-lexicon phrase begins.'
  };

  function fmt(v, nd) {
    if (v === null || v === undefined || v !== v) return '\u2014';
    if (typeof v !== 'number') return _esc(v);
    return v.toFixed(nd === undefined ? 3 : nd);
  }

  function section(title, bodyHtml, opts) {
    opts = opts || {};
    return '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')"' +
      (opts.tip ? ' title="' + _esc(opts.tip) + '"' : '') + '>' + _esc(title) + '</div>' +
      '<div class="mod-results-body' + (opts.collapsed ? ' collapsed' : '') + '"' +
      (opts.maxH ? ' style="max-height:' + opts.maxH + 'px"' : '') + '>' + bodyHtml + '</div>';
  }

  function stat(label, value, detail, tip, color) {
    return '<div class="mod-stat"' + (tip ? ' title="' + _esc(tip) + '"' : '') + '>' +
      '<div class="stat-label">' + _esc(label) + '</div>' +
      '<div class="stat-value"' + (color ? ' style="color:' + color + '"' : '') + '>' + value + '</div>' +
      (detail ? '<div class="stat-detail">' + detail + '</div>' : '') + '</div>';
  }

  function ratioColor(r) {
    if (r == null) return null;
    if (r < 0.5) return 'var(--green)';   // interference signature
    if (r > 1.5) return 'var(--red)';     // delta-resident signature
    return 'var(--orange)';
  }

  function renderSycoResults(r) {
    if (!r) return '<div class="mod-results"><div class="mod-results-body" style="padding:16px;color:var(--text-3)">No results yet.</div></div>';
    if (r.error) return '<div class="mod-results"><div class="mod-results-body" style="padding:16px;color:var(--orange)">' + _esc(r.error) + '</div></div>';

    var o = (r.summary && r.summary.overall) || {};
    var msr = o.median_suppression_ratio;
    var h = '<div class="mod-results">';

    // Verdict banner
    if (r.verdict_hint) {
      var col = ratioColor(msr) || 'var(--orange)';
      h += '<div style="padding:12px 16px;background:color-mix(in srgb,' + col + ' 12%,transparent);' +
        'border-left:3px solid ' + col + ';margin:8px 0;font-size:12px;color:var(--text-1)" title="' + _esc(TIP.ratio) + '">' +
        '<span style="font-family:var(--mono);font-weight:700;color:' + col + '">SUPPRESSION RATIO ' + fmt(msr, 3) + '</span>' +
        '<span style="margin-left:8px">' + _esc(r.verdict_hint) + '</span></div>';
    }

    // Overview
    var ov = '<div class="mod-summary">';
    ov += stat('responses', r.n_records || 0, (o.n_with_pivots || 0) + ' with pivots', 'Response-phase records analyzed');
    ov += stat('pivots found', o.n_pivots_total || 0, null, TIP.pivots);
    ov += stat('suppression ratio', fmt(msr, 3), 'mean ' + fmt(o.mean_suppression_ratio, 3), TIP.ratio, ratioColor(msr));
    ov += stat('pivot KL', fmt(o.median_pivot_kl, 4), null, TIP.pivotKl);
    ov += stat('structural KL', fmt(o.median_structural_kl, 4), null, TIP.structKl);
    ov += stat('interference idx', fmt(o.mean_interference_index, 4),
      'i=' + fmt(o.mean_mass_instruct, 3) + ' b=' + fmt(o.mean_mass_base, 3), TIP.ii);
    if (r.effect_size_pressured_vs_neutral_d != null) {
      ov += stat('d (pressured\u2212neutral)', fmt(r.effect_size_pressured_vs_neutral_d, 3), null, TIP.d);
    }
    ov += '</div>';
    h += section('Signature Overview', ov);

    // Category table
    var cats = (r.summary && r.summary.categories) || {};
    var keys = Object.keys(cats);
    if (keys.length) {
      var t = '<div style="padding:8px 12px"><table class="mod-tbl"><thead><tr>' +
        '<th>Category</th><th class="num">n</th>' +
        '<th class="num" title="' + _esc(TIP.pivots) + '">Pivots</th>' +
        '<th class="num" title="' + _esc(TIP.ratio) + '">Suppr. ratio</th>' +
        '<th class="num" title="' + _esc(TIP.pivotKl) + '">Pivot KL</th>' +
        '<th class="num" title="' + _esc(TIP.structKl) + '">Struct KL</th>' +
        '<th class="num" title="' + _esc(TIP.ii) + '">Interf. idx</th>' +
        '</tr></thead><tbody>';
      keys.forEach(function (k) {
        var s = cats[k];
        var rc = ratioColor(s.median_suppression_ratio);
        t += '<tr><td style="color:var(--cyan)">' + _esc(k) + '</td>' +
          '<td class="num">' + (s.n || 0) + '</td>' +
          '<td class="num">' + (s.n_pivots_total || 0) + '</td>' +
          '<td class="num"' + (rc ? ' style="color:' + rc + ';font-weight:600"' : '') + '>' + fmt(s.median_suppression_ratio, 3) + '</td>' +
          '<td class="num">' + fmt(s.median_pivot_kl, 4) + '</td>' +
          '<td class="num">' + fmt(s.median_structural_kl, 4) + '</td>' +
          '<td class="num">' + fmt(s.mean_interference_index, 4) + '</td></tr>';
      });
      t += '</tbody></table></div>';
      h += section('By Category', t);
    }

    // Matched pairs
    var pairs = r.pairs || [];
    if (pairs.length) {
      var pb = '<div style="padding:8px 12px">';
      pairs.forEach(function (p) {
        pb += '<div style="padding:6px 0;border-bottom:1px solid var(--border)">' +
          '<div style="font-size:11px;color:var(--text-2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + _esc(p.source_prompt) + '</div>' +
          '<div style="display:flex;gap:14px;flex-wrap:wrap;font-family:var(--mono);font-size:11px;margin-top:2px">';
        (p.records || []).forEach(function (rec) {
          pb += '<span title="' + _esc(TIP.ratio) + '">' + _esc(rec.category) +
            ': <b style="color:' + (ratioColor(rec.suppression_ratio) || 'var(--text-1)') + '">' +
            fmt(rec.suppression_ratio, 3) + '</b> (' + rec.n_pivots + ' piv)</span>';
        });
        pb += '</div></div>';
      });
      pb += '</div>';
      h += section('Matched Pairs (same source prompt)', pb, { collapsed: true });
    }

    // Per-record
    var recs = r.records || [];
    if (recs.length) {
      var body = '';
      recs.forEach(function (rec) {
        var rc = ratioColor(rec.suppression_ratio);
        var badge = rec.n_pivots
          ? '<span style="white-space:nowrap;color:' + (rc || 'var(--text-1)') + '" title="' + _esc(TIP.ratio) + '">ratio ' + fmt(rec.suppression_ratio, 3) + ' \u00B7 ' + rec.n_pivots + ' piv</span>'
          : '<span style="color:var(--text-3)">no pivots</span>';
        body += '<div onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display===\'none\'?\'\':\'none\'" ' +
          'style="cursor:pointer;display:flex;gap:10px;align-items:baseline;padding:6px 12px;border-top:1px solid var(--border);font-size:11px;min-width:0">' +
          '<span style="color:var(--text-3);font-family:var(--mono);flex-shrink:0">#' + rec.index + '</span>' +
          '<span style="color:var(--cyan);font-family:var(--mono);flex-shrink:0">' + _esc(rec.category) + '</span>' +
          '<span style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--text-2)">' + _esc(rec.response_text) + '</span>' +
          badge + '</div>';
        var d = '<div style="display:none;padding:8px 12px">';
        if ((rec.pivots || []).length) {
          d += '<table class="mod-tbl"><thead><tr>' +
            '<th class="num">Pos</th><th>Phrase</th>' +
            '<th class="num" title="' + _esc(TIP.pivotKl) + '">KL @ pivot</th>' +
            '<th class="num" title="KL aggregated over the pivot window">Window KL</th>' +
            '<th class="num" title="' + _esc(TIP.massI) + '">Mass (inst)</th>' +
            '<th class="num" title="' + _esc(TIP.massB) + '">Mass (base)</th>' +
            '<th class="num" title="' + _esc(TIP.ii) + '">Interf.</th>' +
            '</tr></thead><tbody>';
          rec.pivots.forEach(function (pv) {
            d += '<tr><td class="num">' + pv.pos + '</td>' +
              '<td style="color:var(--cyan)">' + _esc(pv.ngram) + '</td>' +
              '<td class="num">' + fmt(pv.kl_at_pos, 4) + '</td>' +
              '<td class="num">' + fmt(pv.pivot_kl, 4) + '</td>' +
              '<td class="num">' + fmt(pv.mass_instruct, 4) + '</td>' +
              '<td class="num">' + fmt(pv.mass_base, 4) + '</td>' +
              '<td class="num" style="font-weight:600">' + fmt(pv.interference_index, 4) + '</td></tr>';
          });
          d += '</tbody></table>';
        } else {
          d += '<div style="font-size:11px;color:var(--text-3)">No agreement-lexicon phrases found in this response.</div>';
        }
        d += '<div style="font-size:10px;color:var(--text-3);margin-top:4px" title="' + _esc(TIP.caved) + '">caved: ' + (rec.caved == null ? 'unlabeled (see worksheet)' : _esc(rec.caved)) + '</div>';
        d += '</div>';
        body += d;
      });
      if (r.worksheet_path) {
        body += '<div style="padding:6px 12px;font-size:10px;color:var(--text-3);font-family:var(--mono);overflow-wrap:anywhere" title="' + _esc(TIP.caved) + '">labeling worksheet \u2192 ' + _esc(r.worksheet_path) + '</div>';
      }
      h += section('Per-Record Pivots \u2014 click a row to expand', body, { maxH: 700 });
    }

    // Caveats
    if ((r.caveats || []).length) {
      var cb = '<ul style="margin:8px 12px;padding-left:18px;font-size:11px;color:var(--text-2)">';
      r.caveats.forEach(function (c) { cb += '<li style="margin-bottom:4px">' + _esc(c) + '</li>'; });
      cb += '</ul>';
      h += section('Caveats', cb, { collapsed: true });
    }

    h += '</div>';
    return h;
  }

})();
