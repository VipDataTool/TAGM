/* TAGM — shared HTML escaper.
 *
 * Plain <script src>-able (no ES module syntax) because the standalone viz
 * pages are opened directly by the browser. Everything hangs off window.TAGM.
 *
 * Why this exists: five hand-rolled escapers were in circulation across six
 * files, in two INCOMPATIBLE classes:
 *   - quote-safe regex escapers (probe_diagnostic_viz, template_maker)
 *   - quote-UNSAFE `textContent -> innerHTML` escapers (main.js, roundtable,
 *     correction_prism). That trick escapes & < > but NOT " or ', so every
 *     attribute-context use (title="...", value="...", data-key="...") could
 *     be broken out of with a single quote character.
 * This module is the quote-safe one; the unsafe copies now delegate to it.
 */
(function (global) {
  var TAGM = global.TAGM = global.TAGM || {};

  var MAP = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

  /* Null/undefined-safe, stringifies non-strings, escapes the five characters
     that matter in both element and attribute context. */
  function esc(s) {
    if (s === null || s === undefined) return '';
    return String(s).replace(/[&<>"']/g, function (c) { return MAP[c]; });
  }

  /* Truncate-then-escape, for fixed-width labels (domain_surface_viz's ladder).
     Truncation happens on the raw string so the ellipsis can never land in the
     middle of an entity. */
  function escTrunc(s, n) {
    s = (s === null || s === undefined) ? '' : String(s);
    if (n && s.length > n) s = s.slice(0, n - 1) + '…';
    return esc(s);
  }

  TAGM.esc = esc;
  TAGM.escTrunc = escTrunc;
})(typeof window !== 'undefined' ? window : globalThis);
