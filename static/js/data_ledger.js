/* TAGM — Data-tab ledger: field glossary + expandable per-record detail.
 *
 * Everything here is DERIVED from TAGM.FIELDS (static/js/common/fields.js).
 * Nothing in this file hardcodes what a field means; it only decides how a
 * given `format` code is drawn.  Adding a field to the registry makes it
 * appear in the glossary, in the tooltips and in the detail panel at once.
 *
 * Plain <script src>-able, no modules, no build step, no dependencies beyond
 * TAGM.esc (quote-safe escaper) and TAGM.FIELDS.
 *
 * SECURITY NOTE: every record value reaching innerHTML goes through E()
 * below, which is TAGM.esc.  Records contain user prompts and raw model
 * output, so this is not optional.
 */
(function (global) {
  var TAGM = global.TAGM = global.TAGM || {};

  function E(s) {
    return (TAGM.esc ? TAGM.esc(s)
      : String(s === null || s === undefined ? '' : s)
        .replace(/[&<>"']/g, function (c) {
          return { '&': '&amp;', '<': '&lt;', '>': '&gt;',
                   '"': '&quot;', "'": '&#39;' }[c];
        }));
  }

  /* ── Value classification ───────────────────────────────────────── */

  var ABSENT = '<span class="ldg-na" title="Key not present in this record">absent</span>';
  var NULLV  = '<span class="ldg-na" title="Present but null — measured as undefined, not zero">null</span>';

  function isNum(v) { return typeof v === 'number' && isFinite(v); }

  /* Min/max/mean over finite entries. Written as a loop rather than
     Math.min(...arr) because heatmap rows and embedding matrices routinely
     exceed the argument-count limit of spread. */
  function stats(arr) {
    var mn = Infinity, mx = -Infinity, sum = 0, n = 0, gaps = 0;
    for (var i = 0; i < arr.length; i++) {
      var v = arr[i];
      if (isNum(v)) { if (v < mn) mn = v; if (v > mx) mx = v; sum += v; n++; }
      else gaps++;
    }
    return n ? { min: mn, max: mx, mean: sum / n, n: n, gaps: gaps }
             : { min: null, max: null, mean: null, n: 0, gaps: gaps };
  }

  function num(v, digits) {
    if (!isNum(v)) return String(v);
    var a = Math.abs(v);
    // Very large / very small magnitudes are unreadable in fixed notation.
    if (a !== 0 && (a < 1e-4 || a >= 1e7)) return v.toExponential(3);
    return v.toFixed(digits === undefined ? 4 : digits);
  }

  /* ── Sparkline ──────────────────────────────────────────────────── */

  var SPARK_W = 120, SPARK_H = 28, SPARK_PAD = 2, SPARK_MAX_PTS = 300;

  /* Inline SVG polyline. No library, no canvas.
     WHY gaps matter: rank_displacement.per_position_tau stores null at every
     position where Kendall tau was undefined, and the list is deliberately
     kept position-indexed rather than compacted. Plotting those as 0.0 would
     draw a hard dip toward "the two models disagree completely" where the
     truth is "no comparison was possible". Non-finite entries therefore END
     the current polyline and start a new one, and each gap gets a faint
     baseline tick. */
  function sparkline(arr) {
    if (!arr || !arr.length) return '<span class="ldg-na">empty</span>';
    var st = stats(arr);
    if (!st.n) return '<span class="ldg-na">' + arr.length + ' entries, none finite</span>';

    var n = arr.length;
    var stride = Math.max(1, Math.ceil(n / SPARK_MAX_PTS));
    var span = (st.max - st.min) || 1;   // flat series draws a mid-height line
    var innerW = SPARK_W - 2 * SPARK_PAD;
    var innerH = SPARK_H - 2 * SPARK_PAD;
    var lastX = n > 1 ? n - 1 : 1;

    var segs = [], cur = [], gapTicks = '';
    for (var i = 0; i < n; i += stride) {
      var v = arr[i];
      var x = SPARK_PAD + (i / lastX) * innerW;
      if (!isNum(v)) {
        if (cur.length) { segs.push(cur); cur = []; }
        gapTicks += '<rect x="' + x.toFixed(2) + '" y="' + SPARK_PAD +
                    '" width="1" height="' + innerH +
                    '" class="ldg-spark-gap"></rect>';
        continue;
      }
      var y = SPARK_PAD + innerH - ((v - st.min) / span) * innerH;
      cur.push(x.toFixed(2) + ',' + y.toFixed(2));
    }
    if (cur.length) segs.push(cur);

    var paths = '';
    for (var s = 0; s < segs.length; s++) {
      if (segs[s].length === 1) {
        var p = segs[s][0].split(',');
        paths += '<circle cx="' + p[0] + '" cy="' + p[1] +
                 '" r="1.2" class="ldg-spark-dot"></circle>';
      } else {
        paths += '<polyline points="' + segs[s].join(' ') + '" class="ldg-spark-line"></polyline>';
      }
    }

    var caption = 'n=' + n +
      ' · min ' + num(st.min) + ' · max ' + num(st.max) +
      ' · mean ' + num(st.mean);
    if (st.gaps) caption += ' · ' + st.gaps + ' null/non-finite (drawn as gaps)';
    if (stride > 1) caption += ' · plotted every ' + stride + 'th point';

    return '<div class="ldg-spark-wrap">' +
      '<svg class="ldg-spark" width="' + SPARK_W + '" height="' + SPARK_H +
      '" viewBox="0 0 ' + SPARK_W + ' ' + SPARK_H + '" preserveAspectRatio="none" ' +
      'role="img" aria-label="' + E(caption) + '">' +
      gapTicks + paths + '</svg>' +
      '<span class="ldg-spark-cap">' + E(caption) + '</span></div>';
  }

  /* ── Composite renderers ────────────────────────────────────────── */

  function kvGrid(obj) {
    var keys = Object.keys(obj || {});
    if (!keys.length) return '<span class="ldg-na">empty</span>';
    var h = '<div class="ldg-kv">';
    for (var i = 0; i < keys.length; i++) {
      var v = obj[keys[i]];
      var shown;
      if (isNum(v)) shown = num(v);
      else if (v === null) shown = 'null';
      else if (typeof v === 'object') shown = Array.isArray(v)
        ? '[' + v.length + ' items]' : '{' + Object.keys(v).length + ' keys}';
      else shown = String(v);
      h += '<div class="ldg-kv-cell"><span class="ldg-kv-k">' + E(keys[i]) +
           '</span><span class="ldg-kv-v">' + E(shown) + '</span></div>';
    }
    return h + '</div>';
  }

  /* key -> numeric array (per_layer_signed_attr, and friends). */
  function dictOfArrays(obj) {
    var keys = Object.keys(obj || {});
    if (!keys.length) return '<span class="ldg-na">empty</span>';
    var cap = 24;
    var h = '<div class="ldg-dictarr">';
    for (var i = 0; i < Math.min(keys.length, cap); i++) {
      var v = obj[keys[i]];
      h += '<div class="ldg-dictarr-row"><span class="ldg-kv-k">' + E(keys[i]) +
           '</span>' + (Array.isArray(v) ? sparkline(v) : E(String(v))) + '</div>';
    }
    if (keys.length > cap) {
      h += '<div class="ldg-more">… ' + (keys.length - cap) + ' more keys (see Copy JSON)</div>';
    }
    return h + '</div>';
  }

  function matrix(rows) {
    if (!Array.isArray(rows) || !rows.length) return '<span class="ldg-na">empty</span>';
    var cols = Array.isArray(rows[0]) ? rows[0].length : null;
    if (cols === null) return renderAuto(rows);   // not actually 2-D
    var flatStats = { min: Infinity, max: -Infinity, sum: 0, n: 0 };
    var ragged = false;
    for (var r = 0; r < rows.length; r++) {
      if (!Array.isArray(rows[r])) { ragged = true; continue; }
      if (rows[r].length !== cols) ragged = true;
      for (var c = 0; c < rows[r].length; c++) {
        var v = rows[r][c];
        if (isNum(v)) {
          if (v < flatStats.min) flatStats.min = v;
          if (v > flatStats.max) flatStats.max = v;
          flatStats.sum += v; flatStats.n++;
        }
      }
    }
    var head = rows.length + ' × ' + (ragged ? 'ragged' : cols);
    if (flatStats.n) {
      head += ' · min ' + num(flatStats.min) + ' · max ' + num(flatStats.max) +
              ' · mean ' + num(flatStats.sum / flatStats.n);
    } else {
      head += ' · no finite entries';
    }
    // Row means give the shape of the matrix along its first axis without
    // pretending to show 10k numbers.
    var means = [];
    for (var r2 = 0; r2 < rows.length; r2++) {
      var st = Array.isArray(rows[r2]) ? stats(rows[r2]) : { mean: null };
      means.push(st.mean === null ? null : st.mean);
    }
    var body = '<div class="ldg-matrix-head">' + E(head) + '</div>' +
               '<div class="ldg-matrix-sub">row means:</div>' + sparkline(means);
    var preview = '';
    var pcap = Math.min(rows.length, 8);
    for (var r3 = 0; r3 < pcap; r3++) {
      var row = rows[r3];
      var cells = Array.isArray(row) ? row.slice(0, 12).map(function (v) {
        return isNum(v) ? num(v, 3) : String(v);
      }).join(', ') : String(row);
      if (Array.isArray(row) && row.length > 12) cells += ', … (' + row.length + ')';
      preview += '<div class="ldg-mono-row">[' + r3 + '] ' + E(cells) + '</div>';
    }
    if (rows.length > pcap) preview += '<div class="ldg-more">… ' + (rows.length - pcap) + ' more rows</div>';
    return body + '<details class="ldg-details"><summary>preview first rows</summary>' +
           preview + '</details>';
  }

  function topkTable(list, title) {
    if (!Array.isArray(list) || !list.length) {
      return '<div class="ldg-topk"><div class="ldg-topk-title">' + E(title) +
             '</div><span class="ldg-na">empty</span></div>';
    }
    var h = '<div class="ldg-topk"><div class="ldg-topk-title">' + E(title) +
            '</div><table class="ldg-topk-tbl"><thead><tr><th>#</th><th>token</th><th>p</th></tr></thead><tbody>';
    for (var i = 0; i < Math.min(list.length, 20); i++) {
      var e = list[i] || [];
      var tok = Array.isArray(e) ? e[0] : e;
      var p = Array.isArray(e) ? e[1] : null;
      h += '<tr><td>' + (i + 1) + '</td><td class="ldg-tok">' + E(tok) +
           '</td><td>' + (isNum(p) ? (p * 100).toFixed(2) + '%' : E(String(p))) + '</td></tr>';
    }
    if (list.length > 20) h += '<tr><td colspan="3" class="ldg-more">… ' + (list.length - 20) + ' more</td></tr>';
    return h + '</tbody></table></div>';
  }

  function topkPerPos(list, tokens) {
    if (!Array.isArray(list) || !list.length) return '<span class="ldg-na">empty</span>';
    var cap = Math.min(list.length, 24);
    var h = '<details class="ldg-details"><summary>' + list.length +
            ' positions</summary><div class="ldg-perpos">';
    for (var i = 0; i < cap; i++) {
      var alts = list[i] || [];
      var label = (tokens && tokens[i] !== undefined) ? String(tokens[i]) : '';
      var parts = [];
      for (var j = 0; j < Math.min(alts.length, 8); j++) {
        var a = alts[j] || [];
        var t = Array.isArray(a) ? a[0] : a;
        var p = Array.isArray(a) ? a[1] : null;
        parts.push(String(t) + (isNum(p) ? ' ' + (p * 100).toFixed(1) + '%' : ''));
      }
      if (alts.length > 8) parts.push('… +' + (alts.length - 8));
      h += '<div class="ldg-mono-row"><span class="ldg-pos">[' + i + ']' +
           (label ? ' ' + E(label) : '') + '</span> ' + E(parts.join(' · ')) + '</div>';
    }
    if (list.length > cap) h += '<div class="ldg-more">… ' + (list.length - cap) + ' more positions</div>';
    return h + '</div></details>';
  }

  /* Array of flat objects -> table over the union of keys. */
  function rowsTable(list) {
    if (!Array.isArray(list) || !list.length) return '<span class="ldg-na">empty</span>';
    var keys = [], seen = {};
    for (var i = 0; i < list.length; i++) {
      var o = list[i];
      if (!o || typeof o !== 'object') continue;
      for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k) && !seen[k]) { seen[k] = 1; keys.push(k); }
    }
    var cap = Math.min(list.length, 60);
    var h = '<details class="ldg-details"><summary>' + list.length +
            ' rows × ' + keys.length + ' keys</summary>' +
            '<div class="ldg-scrollbox"><table class="ldg-tbl"><thead><tr><th>#</th>';
    for (var c = 0; c < keys.length; c++) h += '<th>' + E(keys[c]) + '</th>';
    h += '</tr></thead><tbody>';
    for (var r = 0; r < cap; r++) {
      h += '<tr><td>' + r + '</td>';
      for (var c2 = 0; c2 < keys.length; c2++) {
        var v = (list[r] || {})[keys[c2]];
        var shown = v === undefined ? '' : (isNum(v) ? num(v, 4) :
          (v === null ? 'null' : (typeof v === 'object' ? JSON.stringify(v) : String(v))));
        h += '<td>' + E(shown) + '</td>';
      }
      h += '</tr>';
    }
    if (list.length > cap) {
      h += '<tr><td colspan="' + (keys.length + 1) + '" class="ldg-more">… ' +
           (list.length - cap) + ' more rows (see Copy JSON)</td></tr>';
    }
    return h + '</tbody></table></div></details>';
  }

  /* proof1_checks — collapsed by default, caveat printed ABOVE the toggle so
     it is read whether or not the table is opened. */
  function proof1(list, caveat) {
    var head = '<div class="ldg-warn ldg-warn-inline">⚠ ' + E(caveat || '') + '</div>';
    if (!Array.isArray(list) || !list.length) {
      return head + '<span class="ldg-na">empty</span>';
    }
    var nExact = 0, worst = 0;
    for (var i = 0; i < list.length; i++) {
      if (list[i] && list[i].exact) nExact++;
      var e = list[i] && Number(list[i].error);
      if (isNum(e) && e > worst) worst = e;
    }
    var summary = list.length + ' checks · ' + nExact + ' flagged exact · max |error| ' + num(worst);
    var h = head + '<details class="ldg-details"><summary>' + E(summary) +
            '</summary><div class="ldg-scrollbox"><table class="ldg-tbl"><thead><tr>' +
            '<th>layer</th><th>head</th><th>attr_sum</th><th>delta_norm</th>' +
            '<th>error</th><th>exact</th></tr></thead><tbody>';
    for (var j = 0; j < Math.min(list.length, 200); j++) {
      var c = list[j] || {};
      h += '<tr><td>' + E(c.layer) + '</td><td>' + E(c.head) + '</td><td>' +
           E(isNum(c.attr_sum) ? num(c.attr_sum, 8) : String(c.attr_sum)) + '</td><td>' +
           E(isNum(c.delta_norm) ? num(c.delta_norm, 8) : String(c.delta_norm)) + '</td><td>' +
           E(String(c.error)) + '</td><td class="' + (c.exact ? 'ldg-ok' : 'ldg-bad') + '">' +
           (c.exact ? 'yes' : 'no') + '</td></tr>';
    }
    if (list.length > 200) h += '<tr><td colspan="6" class="ldg-more">… ' + (list.length - 200) + ' more</td></tr>';
    return h + '</tbody></table></div></details>';
  }

  function ecmChannels(obj) {
    var names = Object.keys(obj || {});
    if (!names.length) return '<span class="ldg-na">no channels (no trace was extracted)</span>';
    var h = '';
    for (var i = 0; i < names.length; i++) {
      var ch = obj[names[i]] || {};
      var scal = {};
      for (var k in ch) {
        if (!Object.prototype.hasOwnProperty.call(ch, k)) continue;
        if (k === 'per_token_signal') continue;
        scal[k] = ch[k];
      }
      h += '<div class="ldg-subblock"><div class="ldg-subblock-title">' + E(names[i]) +
           '</div>' + kvGrid(scal);
      if (Array.isArray(ch.per_token_signal)) {
        h += '<div class="ldg-matrix-sub">per_token_signal (σ-excess):</div>' +
             sparkline(ch.per_token_signal);
      }
      h += '</div>';
    }
    return h;
  }

  /* ── Fallback for values with no registry format ─────────────────── */
  function renderAuto(v, depth) {
    depth = depth || 0;
    if (v === undefined) return ABSENT;
    if (v === null) return NULLV;
    if (typeof v === 'boolean') return '<span class="ldg-bool">' + (v ? 'true' : 'false') + '</span>';
    if (typeof v === 'number') return E(Number.isInteger(v) ? String(v) : num(v));
    if (typeof v === 'string') return '<span class="ldg-str">' + E(v) + '</span>';
    if (Array.isArray(v)) {
      if (!v.length) return '<span class="ldg-na">empty list</span>';
      var allNum = true, allStr = true, allArr = true, allObj = true;
      for (var i = 0; i < v.length; i++) {
        var e = v[i];
        if (!(typeof e === 'number' || e === null)) allNum = false;
        if (typeof e !== 'string') allStr = false;
        if (!Array.isArray(e)) allArr = false;
        if (!(e && typeof e === 'object' && !Array.isArray(e))) allObj = false;
      }
      if (allNum) return sparkline(v);
      if (allStr) return strChips(v);
      if (allArr) return matrix(v);
      if (allObj) return rowsTable(v);
      return jsonBlock(v);
    }
    if (typeof v === 'object') {
      // Flat scalar dict -> grid; anything nested -> JSON so nothing is lost.
      var flat = true;
      for (var k in v) {
        if (!Object.prototype.hasOwnProperty.call(v, k)) continue;
        if (v[k] !== null && typeof v[k] === 'object') { flat = false; break; }
      }
      if (flat) return kvGrid(v);
      if (depth > 1) return jsonBlock(v);
      return kvGrid(v) + jsonBlock(v);
    }
    return E(String(v));
  }

  function strChips(list) {
    var cap = Math.min(list.length, 200);
    var h = '<div class="ldg-chips">';
    for (var i = 0; i < cap; i++) h += '<span class="ldg-chip">' + E(list[i]) + '</span>';
    if (list.length > cap) h += '<span class="ldg-more">… ' + (list.length - cap) + ' more</span>';
    return h + '</div>';
  }

  function jsonBlock(v) {
    var s;
    try { s = JSON.stringify(v, null, 2); } catch (e) { s = String(v); }
    if (s === undefined) s = String(v);
    var truncated = false;
    if (s.length > 4000) { s = s.slice(0, 4000); truncated = true; }
    return '<details class="ldg-details"><summary>raw JSON</summary><pre class="ldg-pre">' +
           E(s) + (truncated ? '\n… truncated — use Copy JSON for the whole value' : '') +
           '</pre></details>';
  }

  /* ── format dispatch ────────────────────────────────────────────── */

  var WIDE = { spark: 1, matrix: 1, dict: 1, dict_arr: 1, topk: 1,
               topk_per_pos: 1, proof1: 1, rows: 1, ecm_channels: 1,
               strs: 1, nums: 1, ints: 1, json: 1 };

  function renderValue(v, f, rec) {
    var fmt = f && f.format;
    if (v === undefined) return ABSENT;
    if (v === null) {
      // null is meaningful for these: "measured, and undefined".
      return NULLV;
    }
    switch (fmt) {
      case 'text':   return v === '' ? '<span class="ldg-na">empty string</span>'
                                     : '<span class="ldg-str">' + E(v) + '</span>';
      case 'cat':    return '<span class="pill ' + E(pillCls(v)) + '">' + E(v) + '</span>';
      case 'int':    return E(isNum(v) ? String(Math.round(v)) : String(v));
      case 'f2':     return E(num(v, 2));
      case 'f3':     return E(num(v, 3));
      case 'f4':     return E(num(v, 4));
      case 'f5':     return E(num(v, 5));
      case 'f6':     return E(num(v, 6));
      case 'pct':    return isNum(v) ? E((v * 100).toFixed(2) + '%') : E(String(v));
      case 'pct_pp': return isNum(v) ? E((v * 100).toFixed(2) + ' pp') : E(String(v));
      case 'bool':   return '<span class="ldg-bool">' + (v ? 'true' : 'false') + '</span>';
      case 'spark':  return Array.isArray(v) ? sparkline(v) : renderAuto(v);
      case 'nums':
      case 'ints':   return Array.isArray(v) ? (v.length > 32 ? sparkline(v) : inlineNums(v)) : renderAuto(v);
      case 'strs':   return Array.isArray(v) ? strChips(v) : renderAuto(v);
      case 'dict':   return (v && typeof v === 'object') ? renderAuto(v, 0) : renderAuto(v);
      case 'dict_arr': return dictOfArrays(v);
      case 'matrix': return matrix(v);
      case 'topk':   return topkTable(v, f.label);
      case 'topk_per_pos': return topkPerPos(v, rec && rec.tokens);
      case 'rows':   return rowsTable(v);
      case 'proof1': return proof1(v, f.caveat);
      case 'ecm_channels': return ecmChannels(v);
      default:       return renderAuto(v);
    }
  }

  function inlineNums(list) {
    var parts = [];
    for (var i = 0; i < list.length; i++) {
      parts.push(isNum(list[i]) ? (Number.isInteger(list[i]) ? String(list[i]) : num(list[i], 4))
                                : String(list[i]));
    }
    return '<span class="ldg-mono">[' + E(parts.join(', ')) + ']</span>';
  }

  function pillCls(c) {
    var m = { benign: 'pill-benign', baseline: 'pill-benign', mild: 'pill-mild',
              harmful: 'pill-harmful', jailbreak: 'pill-jailbreak',
              adversarial: 'pill-adversarial', 'dual-use': 'pill-dual-use' };
    return m[c] || '';
  }

  /* ── Detail panel ───────────────────────────────────────────────── */

  /* Registry keys, grouped, but only the ones this record actually carries
     (or that are registered as core). Plus an "Unregistered" group holding
     every key present in the record with no registry entry — WHY: the ledger
     must stay complete as the backend evolves, so a new field surfaces
     loudly rather than being silently dropped. */
  function collectGroups(rec) {
    var out = [], covered = {};
    var groups = TAGM.fieldsByGroup();
    for (var g = 0; g < groups.length; g++) {
      var items = [];
      for (var i = 0; i < groups[g].fields.length; i++) {
        var f = groups[g].fields[i];
        var v = TAGM.fieldPath(rec, f.key);
        // Mark every registry key as covered even when absent, so a nested
        // block's own keys are not re-reported as "unregistered".
        covered[f.key] = 1;
        if (v === undefined) continue;   // absent fields are not listed
        items.push({ f: f, v: v });
      }
      if (items.length) out.push({ group: groups[g].group, label: groups[g].label,
                                   color: groups[g].color, items: items });
    }

    // Unregistered sweep: top-level keys, plus one level into the nested
    // blocks the registry describes with dotted keys.
    var NESTED = ['ltp', 'sfd', 'rank_displacement', 'ecm'];
    var extras = [];
    function sweep(obj, prefix) {
      if (!obj || typeof obj !== 'object') return;
      for (var k in obj) {
        if (!Object.prototype.hasOwnProperty.call(obj, k)) continue;
        var full = prefix ? prefix + '.' + k : k;
        if (covered[full]) continue;
        extras.push({
          f: { key: full, label: full, group: 'Unregistered', format: null,
               desc: 'Present in the record but absent from TAGM.FIELDS. ' +
                     'Displayed with an inferred renderer; add it to ' +
                     'static/js/common/fields.js to give it a description.' },
          v: obj[k]
        });
      }
    }
    sweep(rec, '');
    for (var n = 0; n < NESTED.length; n++) {
      if (rec && rec[NESTED[n]] && typeof rec[NESTED[n]] === 'object' && !Array.isArray(rec[NESTED[n]])) {
        sweep(rec[NESTED[n]], NESTED[n]);
      }
    }
    if (extras.length) {
      out.push({ group: 'Unregistered',
                 label: TAGM.FIELD_GROUP_LABELS.Unregistered,
                 color: TAGM.FIELD_GROUP_COLORS.Unregistered, items: extras });
    }
    return out;
  }

  function fieldBlock(entry, muted, rec) {
    var f = entry.f;
    var tip = TAGM.fieldTooltip ? TAGM.fieldTooltip(f.key) : f.key;
    if (f.group === 'Unregistered') tip = f.key + '\n\n' + f.desc;
    var wide = WIDE[f.format] || f.group === 'Unregistered';
    var cls = 'ldg-field' + (wide ? ' ldg-wide' : '') + (muted ? ' ldg-muted' : '');
    var marker = f.caveat ? '<sup class="ldg-caveat-mark" title="This field has a caveat — hover the label">°</sup>' : '';
    var h = '<div class="' + cls + '" title="' + E(tip) + '">' +
      '<div class="ldg-field-head"><span class="ldg-label">' + E(f.label) + marker +
      '</span><span class="ldg-path">' + E(f.key) + '</span>' +
      (f.units ? '<span class="ldg-units">' + E(f.units) + '</span>' : '') +
      (f.extensive ? '<span class="ldg-tag ldg-tag-ext" title="Extensive: grows with sequence length by construction">EXT</span>' : '') +
      (f.lengthSensitive ? '<span class="ldg-tag ldg-tag-len" title="Length-sensitive: expectation drifts with sequence length">LEN</span>' : '') +
      (muted ? '<span class="ldg-tag ldg-tag-mute" title="attribution_unavailable is set — this is a dataclass default, not a measurement">NOT MEASURED</span>' : '') +
      '</div>' +
      '<div class="ldg-value">' + renderValue(entry.v, f, rec) + '</div>';
    if (f.caveat) h += '<div class="ldg-caveat">⚠ ' + E(f.caveat) + '</div>';
    h += '</div>';
    return h;
  }

  /* Full detail panel body for one record. Returned as an HTML string so the
     caller can drop it into the colspan cell of the inserted <tr>. */
  function renderDetail(rec, srcIdx) {
    if (!rec) return '<div class="ldg-warn">Record unavailable.</div>';
    var unavailable = rec.attribution_unavailable;
    var mutedSet = {};
    if (unavailable) {
      var list = TAGM.ATTRIBUTION_DEPENDENT || [];
      for (var i = 0; i < list.length; i++) mutedSet[list[i]] = 1;
    }

    var h = '<div class="ldg-detail">';
    h += '<div class="ldg-bar">' +
      '<span class="ldg-bar-title">Record #' +
      E(rec._index !== undefined && rec._index !== null ? rec._index : srcIdx) + '</span>' +
      '<span class="ldg-bar-prompt" title="' + E(rec.prompt) + '">' +
      E(String(rec.prompt === undefined || rec.prompt === null ? '' : rec.prompt).slice(0, 160)) + '</span>' +
      '<button class="btn btn-secondary btn-sm" onclick="ldgCopyJSON(' + Number(srcIdx) + ',this)">Copy JSON</button>' +
      '<button class="btn btn-secondary btn-sm" onclick="dtToggleDetail(' + Number(srcIdx) + ')">Close</button>' +
      '</div>';

    if (unavailable) {
      h += '<div class="ldg-banner">' +
        '<div class="ldg-banner-title">ATTRIBUTION UNAVAILABLE</div>' +
        '<div class="ldg-banner-msg">' + E(unavailable) + '</div>' +
        '<div class="ldg-banner-note">entropy, top2_share, middle_share, interior_cv, ' +
        'net_correction and n_negative_tokens below are DATACLASS DEFAULTS, not ' +
        'measurements. They are muted. The statistics layer treats them as missing.</div>' +
        '</div>';
    }

    var groups = collectGroups(rec);
    for (var g = 0; g < groups.length; g++) {
      var grp = groups[g];
      h += '<div class="ldg-group"><div class="ldg-group-title" style="color:' +
           grp.color + ';border-color:' + grp.color + '">' + E(grp.label) +
           ' <span class="ldg-group-n">' + grp.items.length + '</span></div>' +
           '<div class="ldg-fields">';
      for (var i2 = 0; i2 < grp.items.length; i2++) {
        h += fieldBlock(grp.items[i2], !!mutedSet[grp.items[i2].f.key], rec);
      }
      h += '</div></div>';
    }
    return h + '</div>';
  }

  /* ── Glossary ───────────────────────────────────────────────────── */

  function glossaryEntriesHTML(filterText) {
    var q = (filterText || '').trim().toLowerCase();
    var groups = TAGM.fieldsByGroup();
    var h = '', nShown = 0;
    for (var g = 0; g < groups.length; g++) {
      var items = groups[g].fields.filter(function (f) {
        if (!q) return true;
        return (f.key + ' ' + f.label + ' ' + (f.desc || '') + ' ' +
                (f.caveat || '') + ' ' + (f.units || '') + ' ' + f.group)
          .toLowerCase().indexOf(q) >= 0;
      });
      if (!items.length) continue;
      nShown += items.length;
      h += '<div class="gls-group"><div class="gls-group-title" style="color:' +
           groups[g].color + ';border-color:' + groups[g].color + '">' +
           E(groups[g].label) + '</div>';
      for (var i = 0; i < items.length; i++) {
        var f = items[i];
        h += '<div class="gls-entry' + (f.caveat ? ' gls-has-caveat' : '') + '">' +
          '<div class="gls-head"><span class="gls-label">' + E(f.label) +
          (f.caveat ? '<sup class="ldg-caveat-mark">°</sup>' : '') + '</span>' +
          '<span class="gls-key">' + E(f.key) + '</span>' +
          (f.units ? '<span class="ldg-units">' + E(f.units) + '</span>' : '') +
          (f.extensive ? '<span class="ldg-tag ldg-tag-ext">EXT</span>' : '') +
          (f.lengthSensitive ? '<span class="ldg-tag ldg-tag-len">LEN</span>' : '') +
          (f.uiOnly ? '<span class="ldg-tag">UI</span>' : '') +
          '</div>' +
          '<div class="gls-desc">' + E(f.desc || '') + '</div>' +
          (f.notComparable ? '<div class="gls-desc">Not comparable ' + E(f.notComparable) + '.</div>' : '') +
          (f.caveat ? '<div class="gls-caveat">⚠ ' + E(f.caveat) + '</div>' : '') +
          '</div>';
      }
      h += '</div>';
    }
    if (!nShown) h = '<div class="ldg-na" style="padding:8px">No fields match “' + E(filterText) + '”.</div>';
    return h;
  }

  function renderGlossary(filterText) {
    var box = document.getElementById('glossaryEntries');
    if (!box) return;
    box.innerHTML = glossaryEntriesHTML(filterText);
    var count = document.getElementById('glossaryCount');
    if (count) {
      var shown = box.querySelectorAll('.gls-entry').length;
      count.textContent = shown + ' / ' + TAGM.FIELDS.length + ' fields';
    }
  }

  TAGM.ledger = {
    sparkline: sparkline,
    renderValue: renderValue,
    renderDetail: renderDetail,
    renderGlossary: renderGlossary,
    renderAuto: renderAuto
  };

  /* ── Globals used from inline handlers / main.js ─────────────────── */

  global.glossaryFilter = function (v) { renderGlossary(v); };

  global.ldgCopyJSON = function (srcIdx, btn) {
    var rec = global._dtRecordFor ? global._dtRecordFor(srcIdx) : null;
    if (!rec) { if (btn) btn.textContent = 'No record'; return; }
    var text;
    try { text = JSON.stringify(rec, null, 2); }
    catch (e) { text = '/* record could not be serialized: ' + e.message + ' */'; }
    var done = function (ok) {
      if (!btn) return;
      var orig = 'Copy JSON';
      btn.textContent = ok ? 'Copied' : 'Copy failed';
      setTimeout(function () { btn.textContent = orig; }, 1400);
    };
    if (global.navigator && navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { done(true); },
                                              function () { done(fallbackCopy(text)); });
    } else {
      done(fallbackCopy(text));
    }
  };

  /* No-network, no-permission fallback for file:// and older browsers. */
  function fallbackCopy(text) {
    try {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      var ok = document.execCommand('copy');
      document.body.removeChild(ta);
      return ok;
    } catch (e) { return false; }
  }

  // Populate the glossary as soon as the DOM is ready. It lives inside a
  // collapsed .card-body, so this costs one innerHTML and nothing visual.
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { renderGlossary(''); });
  } else {
    renderGlossary('');
  }
})(typeof window !== 'undefined' ? window : globalThis);
