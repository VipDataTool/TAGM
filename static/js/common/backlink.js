/* TAGM — dashboard back button for standalone pages.
 *
 * Plain <script src>-able (no ES module syntax), same as esc.js.
 *
 * The popout pages (viz pages, chat, roundtable, template maker,
 * popout.html) are designed to open as separate windows via
 * window.open(). Embedded browsers (VS Code simple browser, Claude Code
 * preview pane) and popup-blocking configurations refuse that, so
 * main.js falls back to same-tab navigation — and this button is the
 * way back. A real popup window has window.opener set and keeps its
 * own close button, so the button only renders for same-tab visits.
 *
 * history.back() is preferred over location='/' because bfcache can
 * restore the dashboard instantly; init re-fetches the grid either way.
 */
(function () {
  if (window.opener) return;               // real popup — nothing to do
  if (history.length <= 1 && !document.referrer) return; // direct visit, nowhere back

  function go() {
    if (history.length > 1) history.back();
    else location.href = '/';
  }

  function inject() {
    var btn = document.createElement('button');
    btn.textContent = '← Dashboard';
    btn.title = 'Back to the TAGM dashboard';
    btn.onclick = go;
    btn.style.cssText =
      'position:fixed;top:10px;left:10px;z-index:1000;' +
      'padding:5px 12px;cursor:pointer;' +
      'font-family:var(--mono,monospace);font-size:11px;' +
      'color:var(--text-1,#ccc);background:var(--bg-1,#1a1a1a);' +
      'border:1px solid var(--border,#333);border-radius:4px;opacity:.85';
    btn.onmouseenter = function () { btn.style.opacity = '1'; };
    btn.onmouseleave = function () { btn.style.opacity = '.85'; };
    document.body.appendChild(btn);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', inject);
  } else {
    inject();
  }
})();
