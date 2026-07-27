/* TAGM — in-page confirm/prompt.
 *
 * Plain <script src>-able (no ES module syntax), same as esc.js.
 *
 * Why this exists: window.confirm()/prompt() are blocking native dialogs,
 * and embedded browsers (VS Code simple browser, the Claude Code preview
 * pane) suppress them — they return false/null immediately, so every
 * confirm-guarded button (Reset, Remove Selected, Apply & Reset, ...)
 * silently did nothing. These render a DOM overlay instead and return a
 * Promise, so callers await:
 *
 *     if (!(await TAGM.confirm('Clear session?'))) return;
 *     const id = await TAGM.prompt('Model ID:', 'Qwen/...');  // null = cancel
 *
 * Styling uses the app's CSS variables with hard fallbacks so the overlay
 * is legible even on a page that hasn't loaded main.css.
 */
(function (global) {
  var TAGM = global.TAGM = global.TAGM || {};

  function dialog(message, opts) {
    return new Promise(function (resolve) {
      var ov = document.createElement('div');
      ov.style.cssText =
        'position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:2000;' +
        'display:flex;align-items:center;justify-content:center';
      var box = document.createElement('div');
      box.style.cssText =
        'background:var(--bg-1,#1a1a1a);color:var(--text-1,#ccc);' +
        'border:1px solid var(--border,#333);border-radius:6px;' +
        'max-width:460px;width:90%;padding:18px;' +
        'font-family:var(--sans,sans-serif);font-size:13px;line-height:1.5';
      var msg = document.createElement('div');
      msg.style.whiteSpace = 'pre-wrap';
      msg.textContent = message;
      box.appendChild(msg);

      var input = null;
      if (opts.input) {
        input = document.createElement('input');
        input.type = 'text';
        input.value = opts.value || '';
        input.style.cssText =
          'width:100%;margin-top:12px;padding:6px 8px;box-sizing:border-box;' +
          'background:var(--bg-0,#111);color:var(--text-1,#ccc);' +
          'border:1px solid var(--border,#333);border-radius:4px;font:inherit';
        box.appendChild(input);
      }

      function finish(val) {
        document.removeEventListener('keydown', onKey, true);
        ov.remove();
        resolve(val);
      }
      function ok()     { finish(opts.input ? input.value : true); }
      function cancel() { finish(opts.input ? null : false); }

      var row = document.createElement('div');
      row.style.cssText = 'display:flex;gap:8px;justify-content:flex-end;margin-top:16px';
      var btnBase = 'padding:6px 14px;border-radius:4px;cursor:pointer;font:inherit;';
      var cbtn = document.createElement('button');
      cbtn.textContent = 'Cancel';
      cbtn.style.cssText = btnBase +
        'background:transparent;color:var(--text-2,#999);border:1px solid var(--border,#333)';
      cbtn.onclick = cancel;
      var obtn = document.createElement('button');
      obtn.textContent = 'OK';
      obtn.style.cssText = btnBase +
        'background:var(--blue,#4a7dbd);color:#fff;border:1px solid transparent';
      obtn.onclick = ok;
      row.appendChild(cbtn);
      row.appendChild(obtn);
      box.appendChild(row);

      // Escape cancels, Enter confirms (from the input, or anywhere for
      // confirm-style dialogs). Capture phase so page-level Esc handlers
      // (popout closers etc.) don't also fire.
      function onKey(e) {
        if (e.key === 'Escape') { e.stopPropagation(); cancel(); }
        else if (e.key === 'Enter' && (!opts.input || e.target === input)) {
          e.stopPropagation(); ok();
        }
      }
      document.addEventListener('keydown', onKey, true);
      ov.addEventListener('click', function (e) { if (e.target === ov) cancel(); });

      ov.appendChild(box);
      document.body.appendChild(ov);
      (input || obtn).focus();
      if (input) input.select();
    });
  }

  /* Drop-in async replacements. confirm resolves true/false;
     prompt resolves the entered string, or null on cancel. */
  TAGM.confirm = function (message)        { return dialog(message, {}); };
  TAGM.prompt  = function (message, value) { return dialog(message, { input: true, value: value }); };
})(typeof window !== 'undefined' ? window : globalThis);
