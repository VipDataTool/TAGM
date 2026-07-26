/* TAGM — shared blob download + CSV cell quoting.
 * Plain <script src>-able; attaches to TAGM.
 *
 * Three near-identical download helpers existed (correction_prism_viz,
 * template_maker, roundtable); roundtable's never called revokeObjectURL, so
 * every export pinned its blob in memory for the life of the tab.
 */
(function (global) {
  var TAGM = global.TAGM = global.TAGM || {};

  function downloadBlob(blob, filename) {
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = filename || 'download';
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    // Revoke on a later tick: revoking synchronously can cancel the download
    // in some browsers, but never revoking leaks the whole blob.
    setTimeout(function () {
      try { a.remove(); } catch (e) {}
      URL.revokeObjectURL(url);
    }, 200);
  }

  function downloadText(text, filename, mime) {
    downloadBlob(new Blob([text], { type: mime || 'text/plain' }), filename);
  }

  /* RFC 4180 cell quoting. */
  function csvCell(v) {
    v = String(v === null || v === undefined ? '' : v);
    if (/[",\n\r]/.test(v)) v = '"' + v.replace(/"/g, '""') + '"';
    return v;
  }

  TAGM.downloadBlob = downloadBlob;
  TAGM.downloadText = downloadText;
  TAGM.csvCell = csvCell;
})(typeof window !== 'undefined' ? window : globalThis);
