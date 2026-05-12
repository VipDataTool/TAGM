const $=id=>document.getElementById(id);
let modelLoaded=false, sessionResults=[], dashResults=[], promptLibraryData=[], _busy=false;
var _promptTotal=0;

function setStatus(s,t){const p=$('statusPill');p.className='status-pill '+s;p.textContent=t||s.toUpperCase()}
function log(m,t=''){const el=$('progressLog'),d=document.createElement('div');d.className='entry '+t;d.textContent=m;el.appendChild(d);el.scrollTop=el.scrollHeight}
function clearLog(){$('progressLog').innerHTML=''}
async function downloadLog(){
  // Server returns the file with Content-Disposition, so navigating to the
  // URL triggers a save dialog rather than rendering. Using location.href
  // keeps the user on the page; the browser intercepts the response.
  try {
    const r = await fetch('/api/log/download', {method:'HEAD'});
    if (!r.ok) {
      log('Log file not available on disk yet.', 'error');
      return;
    }
    window.location.href = '/api/log/download';
  } catch(e) {
    log('Download failed: ' + e.message, 'error');
  }
}
function setLoading(b,l){b.disabled=l;if(l)b.dataset.orig=b.textContent;b.textContent=l?'Processing...':(b.dataset.orig||b.textContent)}
function pillClass(c){return({'benign':'pill-benign','baseline':'pill-benign','mild':'pill-mild','harmful':'pill-harmful','jailbreak':'pill-jailbreak','adversarial':'pill-adversarial','dual-use':'pill-dual-use'})[c]||''}
function toggleFeature(el){
  const f=el.closest('.feature');if(!f)return;
  const b=f.querySelector('.feature-body');if(!b)return;
  b.classList.toggle('collapsed');
  const ch=el.querySelector('.chevron');
  if(ch)ch.textContent=b.classList.contains('collapsed')?'\u25B6':'\u25BC';
}
function escHtml(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
function fmtLN(v){if(v==null)return'';const s=v>0?'+':'',c=v>0.5?'positive':(v<-0.5?'negative':'neutral');return`<div class="metric-ln ${c}">${s}${v.toFixed(2)}sd</div>`}
function mc(l,v,ln,fmt,cls){const fv=fmt?fmt(v):(typeof v==='number'?v.toFixed(4):v);const extra=cls?` ${cls}`:'';return`<div class="metric-cell${extra}"><div class="metric-label">${l}</div><div class="metric-value">${fv}</div>${fmtLN(ln)}</div>`}
function switchMainTab(el,id){el.parentElement.querySelectorAll('.main-tab').forEach(t=>t.classList.remove('active'));el.classList.add('active');document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));$(id).classList.add('active')}
function showError(elId,msg){const el=$(elId);if(msg){el.innerHTML=`<div class="error-msg">${escHtml(msg)}</div>`;setTimeout(()=>el.innerHTML='',5000)}else el.innerHTML=''}

// ═══════════════════════════════════════════════════════════════
// SSE Event Stream — replaces all setInterval-based polling
// ═══════════════════════════════════════════════════════════════

let _eventSource = null;
let _sseConnected = false;

// One-shot waiter arrays: polling consumers register a callback here,
// and the SSE handler resolves it when the corresponding event arrives.
const _sseWaiters = {
  model_loaded: [],
  model_error: [],
  batch_done: [],
  export_ready: [],
  export_error: [],
  probe_status: [],
  pg_embed_status: [],
  module_status: {},  // keyed by module name
};

function initEventSource() {
  if (_eventSource) { try { _eventSource.close(); } catch(e) {} }
  _eventSource = new EventSource('/api/events');
  _eventSource.onmessage = function(e) {
    try { _handleSSEEvent(JSON.parse(e.data)); } catch(err) {}
  };
  _eventSource.onerror = function() {
    _sseConnected = false;
    // EventSource auto-reconnects with backoff.
    // On reconnect, server replays snapshot.
  };
}

function _handleSSEEvent(evt) {
  switch (evt.type) {
    case 'connected':
      _sseConnected = true;
      break;
    case 'progress':
      _handleProgressEvent(evt);
      break;
    case 'model_loaded':
      _handleModelLoaded(evt);
      break;
    case 'model_error':
      _handleModelError(evt);
      break;
    case 'module_status':
      _handleModuleStatusEvent(evt);
      break;
    case 'batch_done':
      _handleBatchDone(evt);
      break;
    case 'export_ready':
      _handleExportReady(evt);
      break;
    case 'export_error':
      _handleExportError(evt);
      break;
    case 'probe_status':
      _handleProbeStatus(evt);
      break;
    case 'pg_embed_status':
      _handlePgEmbedStatus(evt);
      break;
  }
}

function _handleProgressEvent(evt) {
  // Only log progress events after the initial snapshot replay.
  // The 'connected' event signals the snapshot is done.
  if (!_sseConnected) return;
  var cls = evt.stage === 'error' ? 'error' : (evt.stage === 'ready' || evt.stage === 'done' ? 'done' : '');
  log('[' + evt.stage + '] ' + evt.message, cls);
}

function _handleModelLoaded(evt) {
  modelLoaded = true;
  setStatus('ready', 'READY');
  $('analyzeBtn').disabled = false;
  $('batchBtn').disabled = false;
  var waiters = _sseWaiters.model_loaded.splice(0);
  for (var i = 0; i < waiters.length; i++) waiters[i](evt);
}

function _handleModelError(evt) {
  setStatus('idle', 'ERROR');
  log('Load failed: ' + (evt.error || 'unknown'), 'error');
  var waiters = _sseWaiters.model_error.splice(0);
  for (var i = 0; i < waiters.length; i++) waiters[i](evt);
}

function _handleModuleStatusEvent(evt) {
  var name = evt.name;
  if (!name) return;

  if (evt.status === 'running') {
    var prog = $('mod-progress-' + name);
    if (prog && evt.progress) prog.textContent = evt.progress;
    return;
  }

  if (evt.status === 'completed') {
    var prog2 = $('mod-progress-' + name);
    var btn = $('mod-run-' + name);
    updateModuleStatus(name, 'completed');
    playChime();
    if (btn) { btn.disabled = false; btn.textContent = 'Re-run'; }
    var elapsedStr = evt.elapsed ? evt.elapsed + 's' : '?';
    if (prog2) prog2.textContent = 'Completed in ' + elapsedStr;
    log('Module ' + name + ': completed in ' + elapsedStr, 'done');
    if (evt.has_log) {
      var existing = $('mod-log-' + name);
      if (!existing) {
        var logBtn = document.createElement('button');
        logBtn.id = 'mod-log-' + name;
        logBtn.className = 'btn btn-sm';
        logBtn.style.cssText = 'border:1px solid var(--border);color:var(--text-2);background:transparent;cursor:pointer';
        logBtn.textContent = 'Download Log';
        logBtn.onclick = function(e2) { e2.stopPropagation(); window.open('/api/modules/' + name + '/download_log', '_blank'); };
        if (btn && btn.parentNode) btn.parentNode.insertBefore(logBtn, btn.nextSibling);
      }
    }
    fetchModuleResults(name);
    var body = $('mod-body-' + name);
    var chev = $('mod-chev-' + name);
    if (body && body.style.display === 'none') {
      body.style.display = '';
      if (chev) chev.textContent = '\u25BC';
    }
    var mw = (_sseWaiters.module_status[name] || []).splice(0);
    for (var j = 0; j < mw.length; j++) mw[j](evt);

  } else if (evt.status === 'error') {
    updateModuleStatus(name, 'error');
    var btn2 = $('mod-run-' + name);
    if (btn2) { btn2.disabled = false; btn2.textContent = 'Retry'; }
    var errMsg = evt.error || 'unknown';
    var prog3 = $('mod-progress-' + name);
    if (prog3) prog3.textContent = 'Error: ' + errMsg;
    log('Module ' + name + ': error \u2014 ' + errMsg, 'error');
    var mw2 = (_sseWaiters.module_status[name] || []).splice(0);
    for (var k = 0; k < mw2.length; k++) mw2[k](evt);
  }
}

function _handleBatchDone(evt) {
  var waiters = _sseWaiters.batch_done.splice(0);
  for (var i = 0; i < waiters.length; i++) waiters[i](evt);
}

function _handleExportReady(evt) {
  var waiters = _sseWaiters.export_ready.splice(0);
  for (var i = 0; i < waiters.length; i++) waiters[i](evt);
}

function _handleExportError(evt) {
  var waiters = _sseWaiters.export_error.splice(0);
  for (var i = 0; i < waiters.length; i++) waiters[i](evt);
}

function _handleProbeStatus(evt) {
  var waiters = _sseWaiters.probe_status.splice(0);
  for (var i = 0; i < waiters.length; i++) waiters[i](evt);
}

function _handlePgEmbedStatus(evt) {
  var waiters = _sseWaiters.pg_embed_status.splice(0);
  for (var i = 0; i < waiters.length; i++) waiters[i](evt);
}

// ─── Card Collapse Defaults ─────────────────────────────────────
const CARD_COLLAPSE_DEFAULTS={prompt:'expanded',modules:'expanded',config:'expanded'};
const CARD_COLLAPSE_LABELS={prompt:'Prompt',modules:'Modules',config:'Configuration'};
function cardDefault(tab){return CARD_COLLAPSE_DEFAULTS[tab]==='collapsed'?' collapsed':''}
function applyCardDefaults(panelId,tab){
  const panel=$(panelId);if(!panel)return;
  const bodies=panel.querySelectorAll('.card-body');
  bodies.forEach(b=>{
    if(CARD_COLLAPSE_DEFAULTS[tab]==='collapsed')b.classList.add('collapsed');
    else b.classList.remove('collapsed');
  });
}

// ─── Export Defaults ────────────────────────────────────────────
const EXPORT_DEFAULTS={csv:true,pdf:false,json:false,charts:false,includeArrays:true,exportPath:'',embeddingPrecision:12};
const DISPLAY_DEFAULTS={terrainCharLimit:50,terrainRecordLimit:100,terrainTokenLimit:20,terrainCategory:'all',terrainAutoRotate:false,terrainRotateSpeed:0.3};

// Global delegated click handler for all card headers
document.addEventListener('click',function(e){
  // Don't toggle if clicking a button inside the header (e.g. popout)
  if(e.target.closest('button'))return;
  const header=e.target.closest('.card-header');
  if(!header)return;
  const body=header.nextElementSibling;
  if(body&&body.classList.contains('card-body'))body.classList.toggle('collapsed');
});

// ─── Visualization Registry ──────────────────────────────────────
const VIZ_REGISTRY = {
  // ── Per-Prompt: ASM Core ──
  signed_attribution:  {cat:'ASM Core',   title:'Signed Attribution',         on:true,  scope:'prompt', order:20, type:'plot', desc:'Per-token contribution to alignment correction'},
  stress_per_token:    {cat:'ASM Core',   title:'Per-Token Stress',           on:true,  scope:'prompt', order:30, type:'plot', desc:'Correction pressure at discriminative middle layers'},
  heatmap:             {cat:'ASM Core',   title:'Token × Layer Heatmap',      on:true,  scope:'prompt', order:40, type:'plot', desc:'2D correction intensity by token and sublayer'},
  amplitude_trajectory:{cat:'ASM Detail', title:'Layer Trajectory (single)',   on:true, scope:'prompt', order:10, type:'plot', desc:'Single-prompt correction through model depth'},

  // ── Per-Prompt: ASM Detail ──
  distribution_metrics:{cat:'ASM Detail', title:'Distribution Metrics',       on:true, scope:'prompt', order:50, type:'plot', desc:'Entropy, boundary/interior bars'},
  token_table:         {cat:'ASM Detail', title:'Attribution Table',          on:true,  scope:'prompt', order:60, type:'js',   desc:'Raw per-token values with inline bars'},
  model_predictions:   {cat:'ASM Detail', title:'Model Predictions',          on:true,  scope:'prompt', order:70, type:'js',   desc:'Instruct vs base top-k tokens'},

  // ── Per-Prompt: LTP ──
  ltp_tension_magnitudes:{cat:'LTP',      title:'Tension Magnitudes',         on:true,  scope:'prompt', order:80, type:'plot', needs:'ltp', desc:'Per-token lateral tension colored by profile shape'},
  ltp_profiles:        {cat:'LTP',        title:'Tension Profiles (Stacked)', on:true, scope:'prompt', order:90, type:'plot', needs:'ltp', desc:'Stacked bar of tension per counterfactual rank'},
  ltp_profile_heatmap: {cat:'LTP',        title:'Token × Rank Heatmap',       on:true, scope:'prompt', order:100, type:'plot', needs:'ltp', desc:'Dense heatmap: token vs rank'},
  ltp_summary_stats:   {cat:'LTP',        title:'LTP Summary Stats',          on:true, scope:'prompt', order:110, type:'plot', needs:'ltp', desc:'M, C, V scalar bars'},
  counterfactual_table:{cat:'LTP',        title:'Counterfactual Table',       on:true,  scope:'prompt', order:120, type:'js',   needs:'ltp', desc:'Alternative tokens at each position'},

  // ── Per-Prompt: SFD ──
  sfd_density:         {cat:'SFD',        title:'Per-Token QK Density',        on:true,  scope:'prompt', order:130, type:'plot',  needs:'sfd', desc:'How many dimensions of the QK routing subspace each token engages'},
  rank_displacement:   {cat:'SFD',        title:'Rank Displacement',           on:true,  scope:'prompt', order:160, type:'plot',  needs:'sfd', desc:'Kendall tau between base and instruct alternative orderings per position'},
};

// Per-prompt scope: drives the buttons under the Configuration tab's
// "Prompt Visualizations" card. (Batch-scoped viz was never gated by
// these toggles in the first place — the batch popout list reads
// directly from the module's plots output.)
function vizResetDefaults(){
  for(const[k,v]of Object.entries(VIZ_REGISTRY)){
    if(v.scope==='prompt') v.on=v._default;
  }
  vizRenderConfig(); saveConfig();
}
function vizEnableAll(){
  for(const v of Object.values(VIZ_REGISTRY)){
    if(v.scope==='prompt') v.on=true;
  }
  vizRenderConfig(); saveConfig();
}
function vizDisableAll(){
  for(const v of Object.values(VIZ_REGISTRY)){
    if(v.scope==='prompt') v.on=false;
  }
  vizRenderConfig(); saveConfig();
}

function vizRenderConfig(){
  const panel=$('vizConfigPanel'); if(!panel) return;
  const groups={};
  for(const[k,v]of Object.entries(VIZ_REGISTRY)){
    if(!groups[v.cat]) groups[v.cat]=[];
    groups[v.cat].push({key:k,...v});
  }
  let h='';
  // Font size config
  h+=`<div style="margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--border)">
    <div style="font-size:var(--font-desc);font-family:var(--mono);color:var(--text-1);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px">Font Sizes</div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;max-width:540px">
      <div><label style="font-size:var(--font-desc);color:var(--text-1)">Card titles</label><select id="cfgFontCardTitle" onchange="applyFontConfig();saveConfig()"><option value="12">12px</option><option value="14">14px</option><option value="16">16px</option><option value="18">18px</option><option value="20">20px</option><option value="24" selected>24px</option><option value="28">28px</option><option value="32">32px</option><option value="36">36px</option></select></div>
      <div><label style="font-size:var(--font-desc);color:var(--text-1)">Descriptions</label><select id="cfgFontDesc" onchange="applyFontConfig();saveConfig()"><option value="10">10px</option><option value="11">11px</option><option value="12">12px</option><option value="13" selected>13px</option><option value="14">14px</option><option value="15">15px</option><option value="16">16px</option><option value="18">18px</option><option value="20">20px</option><option value="24">24px</option><option value="28">28px</option><option value="32">32px</option></select></div>
      <div><label style="font-size:var(--font-desc);color:var(--text-1)">Legends</label><select id="cfgFontLegend" onchange="applyFontConfig();saveConfig()"><option value="8">8px</option><option value="9">9px</option><option value="10">10px</option><option value="11" selected>11px</option><option value="12">12px</option><option value="13">13px</option><option value="14">14px</option><option value="16">16px</option><option value="18">18px</option><option value="20">20px</option><option value="24">24px</option></select></div>
      <div><label style="font-size:var(--font-desc);color:var(--text-1)">Data tables</label><select id="cfgFontTable" onchange="applyFontConfig();saveConfig()"><option value="8">8px</option><option value="9">9px</option><option value="10">10px</option><option value="11" selected>11px</option><option value="12">12px</option><option value="13">13px</option><option value="14">14px</option><option value="16">16px</option><option value="18">18px</option><option value="20">20px</option><option value="24">24px</option></select></div>
    </div>
  </div>`;
  // Card collapse defaults per tab
  h+=`<div style="margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--border)">
    <div style="font-size:var(--font-desc);font-family:var(--mono);color:var(--text-1);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px">Card Default State</div>
    <div style="font-size:var(--font-legend);color:var(--text-0);margin-bottom:8px;line-height:1.5">Whether cards on each tab start expanded or collapsed.</div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;max-width:540px">`;
  for(const[tab,label]of Object.entries(CARD_COLLAPSE_LABELS)){
    const cur=CARD_COLLAPSE_DEFAULTS[tab];
    h+=`<div><label style="font-size:var(--font-desc);color:var(--text-1)">${label}</label><select id="cfgCardDefault_${tab}" onchange="CARD_COLLAPSE_DEFAULTS['${tab}']=this.value;saveConfig()"><option value="expanded"${cur==='expanded'?' selected':''}>Expanded</option><option value="collapsed"${cur==='collapsed'?' selected':''}>Collapsed</option></select></div>`;
  }
  h+=`</div></div>`;
  // Export options
  h+=`<div style="margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--border)">
    <div style="font-size:var(--font-desc);font-family:var(--mono);color:var(--text-1);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px">Export Options</div>
    <div style="font-size:var(--font-legend);color:var(--text-0);margin-bottom:8px;line-height:1.5">Select which formats to include when exporting. CSV is always generated.</div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;max-width:540px;margin-bottom:10px">
      <div class="checkbox-row"><input type="checkbox" id="cfgExportCsv" checked disabled><label for="cfgExportCsv" style="color:var(--text-2)">CSV (always)</label></div>
      <div class="checkbox-row"><input type="checkbox" id="cfgExportPdf" ${EXPORT_DEFAULTS.pdf?'checked':''} onchange="EXPORT_DEFAULTS.pdf=this.checked;saveConfig()"><label for="cfgExportPdf">PDF report</label></div>
      <div class="checkbox-row"><input type="checkbox" id="cfgExportJson" ${EXPORT_DEFAULTS.json?'checked':''} onchange="EXPORT_DEFAULTS.json=this.checked;saveConfig()"><label for="cfgExportJson">JSON results</label></div>
      <div class="checkbox-row"><input type="checkbox" id="cfgExportCharts" ${EXPORT_DEFAULTS.charts?'checked':''} onchange="EXPORT_DEFAULTS.charts=this.checked;saveConfig()"><label for="cfgExportCharts">Charts &amp; plots</label></div>
    </div>
    <div style="max-width:540px;margin-bottom:10px">
      <div class="checkbox-row"><input type="checkbox" id="cfgExportArrays" ${EXPORT_DEFAULTS.includeArrays?'checked':''} onchange="EXPORT_DEFAULTS.includeArrays=this.checked;saveConfig()"><label for="cfgExportArrays">Include per-token arrays in JSON <span style="color:var(--text-3)">(signed attribution, per-position tau, LTP profiles, SFD per-token — larger file)</span></label></div>
    </div>
    <div style="max-width:540px;margin-bottom:10px;display:flex;align-items:center;gap:10px">
      <label style="font-size:var(--font-desc);color:var(--text-1);white-space:nowrap">Embedding precision</label>
      <select id="cfgEmbeddingPrecision" onchange="EXPORT_DEFAULTS.embeddingPrecision=parseInt(this.value);saveConfig()" style="width:auto;min-width:80px">
        <option value="6" ${EXPORT_DEFAULTS.embeddingPrecision===6?'selected':''}>6 digits</option>
        <option value="8" ${EXPORT_DEFAULTS.embeddingPrecision===8?'selected':''}>8 digits</option>
        <option value="10" ${EXPORT_DEFAULTS.embeddingPrecision===10?'selected':''}>10 digits</option>
        <option value="12" ${EXPORT_DEFAULTS.embeddingPrecision===12?'selected':''}>12 digits</option>
        <option value="15" ${EXPORT_DEFAULTS.embeddingPrecision===15?'selected':''}>15 digits (full)</option>
      </select>
      <span style="font-size:var(--font-legend);color:var(--text-3)">Significant digits for embedding CSVs. Source precision is ~3 digits (bfloat16). Higher = larger files.</span>
    </div>
    <div style="max-width:540px">
      <label style="font-size:var(--font-desc);color:var(--text-1)">Export path <span style="color:var(--text-3)">(optional — leave blank to download via browser)</span></label>
      <input type="text" id="cfgExportPath" value="${EXPORT_DEFAULTS.exportPath}" placeholder="/path/to/export/directory" onchange="EXPORT_DEFAULTS.exportPath=this.value.trim();saveConfig()" style="margin-bottom:0">
    </div>
  </div>`;
  // Session management
  var sCount=_promptTotal||dashResults.length||sessionResults.length;
  var sessionInfo=sCount>0?(sCount+' results'):'No active session';
  h+=`<div style="margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--border)">
    <div style="font-size:var(--font-desc);font-family:var(--mono);color:var(--text-1);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px">Session</div>
    <div style="font-size:var(--font-legend);color:var(--text-0);margin-bottom:8px;line-height:1.5">Current session: <strong>${sessionInfo}</strong>. Session data persists on disk across browser refreshes and crashes. Use Restore to recover a session after a page reload.</div>
    <div style="display:flex;gap:8px;max-width:540px">
      <button class="btn btn-sm btn-secondary" onclick="restoreSessionFromDisk()">Restore Session</button>
    </div>
  </div>`;
  // Cache management
  h+=`<div style="margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--border)">
    <div style="font-size:var(--font-desc);font-family:var(--mono);color:var(--text-1);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px">Cache Management</div>
    <div style="font-size:var(--font-legend);color:var(--text-0);margin-bottom:8px;line-height:1.5">Session data cached on disk. Plots are the largest component. Cache is automatically cleared on server restart.</div>
    <div style="font-size:var(--font-desc);font-family:var(--mono);color:var(--text-0);margin-bottom:10px" id="cfgCacheSize">${_cacheBytes>0?fmtSize(_cacheBytes)+' total':'No data cached'}</div>
    <div style="display:flex;gap:8px;max-width:540px">
      <button class="btn btn-sm btn-secondary" onclick="clearPlotCache()">Clear Plot Cache</button>
      <button class="btn btn-sm btn-danger" onclick="clearAllSessionData()">Clear All Session Data</button>
    </div>
  </div>`;
  // Viz toggles by category. All entries are per-prompt; the batch
  // viz popout list lives in the Comparative Analysis module body and
  // reads directly from its plots output (no toggles needed —
  // available plots auto-render).
  for(const[cat,items]of Object.entries(groups)){
    const color=cat.includes('LTP')?'var(--cyan)':cat.includes('SFD')?'var(--orange)':cat.includes('Fusion')?'var(--green)':'var(--text-2)';
    const scope=items[0].scope;
    const scopeLabel=scope==='prompt'?'per-prompt':'analysis tab';
    h+=`<div style="margin-bottom:14px"><div style="font-size:var(--font-desc);font-family:var(--mono);color:${color};text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px">${cat} <span style="color:var(--text-3);text-transform:none;letter-spacing:0">(${scopeLabel})</span></div>`;
    for(const item of items){
      h+=`<div class="viz-config-row">
        <input type="checkbox" id="viz_${item.key}" ${item.on?'checked':''} onchange="VIZ_REGISTRY['${item.key}'].on=this.checked;saveConfig()">
        <label for="viz_${item.key}" style="flex:1"><span style="color:var(--text-0)">${item.title}</span> <span style="color:var(--text-2);font-size:var(--font-legend)">— ${item.desc}</span></label>
        <span style="font-family:var(--mono);font-size:var(--font-legend);color:var(--text-3);padding:3px 6px;border:1px solid var(--border);border-radius:3px;min-width:48px;text-align:center">${item.scope}</span>
        <input type="number" value="${item.order}" min="1" max="999" style="width:42px;font-size:var(--font-table);padding:3px;text-align:center" onchange="VIZ_REGISTRY['${item.key}'].order=parseInt(this.value)||0;saveConfig()">
      </div>`;
    }
    h+=`</div>`;
  }
  panel.innerHTML=h;
}

function applyFontConfig(){
  const cardTitle=$('cfgFontCardTitle').value+'px';
  const desc=$('cfgFontDesc').value+'px';
  const legend=$('cfgFontLegend').value+'px';
  const table=$('cfgFontTable').value+'px';
  document.documentElement.style.setProperty('--font-card-title',cardTitle);
  document.documentElement.style.setProperty('--font-desc',desc);
  document.documentElement.style.setProperty('--font-legend',legend);
  document.documentElement.style.setProperty('--font-table',table);
}

// Store defaults for reset
(function(){ for(const[k,v]of Object.entries(VIZ_REGISTRY)){v._default=v.on} })();

// ─── Config Persistence ─────────────────────────────────────────
async function saveConfig(){
  try{
    const vizState={};
    for(const[k,v]of Object.entries(VIZ_REGISTRY)) vizState[k]={on:v.on,order:v.order};
    const config={
      viz: vizState,
      fonts:{
        cardTitle: $('cfgFontCardTitle')?$('cfgFontCardTitle').value:'24',
        desc: $('cfgFontDesc')?$('cfgFontDesc').value:'13',
        legend: $('cfgFontLegend')?$('cfgFontLegend').value:'11',
        table: $('cfgFontTable')?$('cfgFontTable').value:'11',
      },
      ltp:{
        layerStrategy: $('cfgLtpLayerStrategy')?$('cfgLtpLayerStrategy').value:'late',
        k: $('cfgLtpK')?$('cfgLtpK').value:'8',
      },
      analysis:{
        computeTraj: $('cfgComputeTraj')?$('cfgComputeTraj').checked:true,
        captureResponses: $('cfgCaptureResponses')?$('cfgCaptureResponses').checked:false,
        fullCapture: $('cfgFullCapture')?$('cfgFullCapture').checked:false,
      },
      cardCollapse:{...CARD_COLLAPSE_DEFAULTS},
      exportOpts:{...EXPORT_DEFAULTS},
      display:{...DISPLAY_DEFAULTS}
    };
    await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(config)});
  }catch(e){}
}

async function loadConfig(){
  try{
    const resp=await fetch('/api/config');
    const data=await resp.json();
    if(!data.ok||!data.config||!Object.keys(data.config).length) return;
    const c=data.config;
    // Viz toggles
    if(c.viz){for(const[k,cfg]of Object.entries(c.viz)){if(VIZ_REGISTRY[k]){
      if(typeof cfg==='object'){VIZ_REGISTRY[k].on=!!cfg.on;if(cfg.order!=null)VIZ_REGISTRY[k].order=cfg.order}
      else{VIZ_REGISTRY[k].on=!!cfg} // backward compat: old format was just bool
    }}}
    // Font sizes
    if(c.fonts){
      if(c.fonts.cardTitle&&$('cfgFontCardTitle')){$('cfgFontCardTitle').value=c.fonts.cardTitle;document.documentElement.style.setProperty('--font-card-title',c.fonts.cardTitle+'px')}
      if(c.fonts.desc&&$('cfgFontDesc')){$('cfgFontDesc').value=c.fonts.desc;document.documentElement.style.setProperty('--font-desc',c.fonts.desc+'px')}
      if(c.fonts.legend&&$('cfgFontLegend')){$('cfgFontLegend').value=c.fonts.legend;document.documentElement.style.setProperty('--font-legend',c.fonts.legend+'px')}
      if(c.fonts.table&&$('cfgFontTable')){$('cfgFontTable').value=c.fonts.table;document.documentElement.style.setProperty('--font-table',c.fonts.table+'px')}
    }
    // LTP settings
    if(c.ltp){
      if(c.ltp.layerStrategy&&$('cfgLtpLayerStrategy'))$('cfgLtpLayerStrategy').value=c.ltp.layerStrategy;
      if(c.ltp.k&&$('cfgLtpK'))$('cfgLtpK').value=c.ltp.k;
    }
    // Analysis options
    if(c.analysis){
      if($('cfgComputeTraj'))$('cfgComputeTraj').checked=c.analysis.computeTraj!==false;
      if($('cfgCaptureResponses'))$('cfgCaptureResponses').checked=!!c.analysis.captureResponses;
      if($('cfgFullCapture'))$('cfgFullCapture').checked=!!c.analysis.fullCapture;
    }
    // Card collapse defaults
    if(c.cardCollapse){
      for(const[tab,val]of Object.entries(c.cardCollapse)){
        if(CARD_COLLAPSE_DEFAULTS.hasOwnProperty(tab))CARD_COLLAPSE_DEFAULTS[tab]=val;
      }
    }
    // Export options
    if(c.exportOpts){
      if(typeof c.exportOpts.pdf==='boolean')EXPORT_DEFAULTS.pdf=c.exportOpts.pdf;
      if(typeof c.exportOpts.json==='boolean')EXPORT_DEFAULTS.json=c.exportOpts.json;
      if(typeof c.exportOpts.charts==='boolean')EXPORT_DEFAULTS.charts=c.exportOpts.charts;
      if(typeof c.exportOpts.includeArrays==='boolean')EXPORT_DEFAULTS.includeArrays=c.exportOpts.includeArrays;
      if(typeof c.exportOpts.exportPath==='string')EXPORT_DEFAULTS.exportPath=c.exportOpts.exportPath;
      if(typeof c.exportOpts.embeddingPrecision==='number')EXPORT_DEFAULTS.embeddingPrecision=c.exportOpts.embeddingPrecision;
    }
    // Display settings
    if(c.display){
      if(c.display.terrainCharLimit)DISPLAY_DEFAULTS.terrainCharLimit=parseInt(c.display.terrainCharLimit)||50;
      if(c.display.terrainRecordLimit)DISPLAY_DEFAULTS.terrainRecordLimit=parseInt(c.display.terrainRecordLimit)||100;
      if(c.display.terrainTokenLimit)DISPLAY_DEFAULTS.terrainTokenLimit=parseInt(c.display.terrainTokenLimit)||20;
      if(c.display.terrainCategory)DISPLAY_DEFAULTS.terrainCategory=c.display.terrainCategory;
      if(c.display.terrainAutoRotate!=null)DISPLAY_DEFAULTS.terrainAutoRotate=!!c.display.terrainAutoRotate;
      if(c.display.terrainRotateSpeed)DISPLAY_DEFAULTS.terrainRotateSpeed=parseFloat(c.display.terrainRotateSpeed)||0.3;
    }
    vizRenderConfig();
    // Apply config tab card defaults to the 3 static config cards
    applyCardDefaults('panel-config','config');
  }catch(e){}
}

function resetLtpConfig(){$('cfgLtpLayerStrategy').value='late';$('cfgLtpK').value='8';saveConfig();log('LTP config reset to defaults','done')}

async function restoreSessionFromDisk(){
  try{
    log('Restoring session from disk...','');
    const r=await(await fetch('/api/session/restore',{method:'POST'})).json();
    if(r.ok){
      _promptTotal=r.n_results;
      if(r.cache_size_bytes)_cacheBytes=r.cache_size_bytes;
      updateSessionBadge();
      renderDataTable();
      $('dashBtn').disabled=false;
      $('exportBtn').disabled=false;
      log('Session restored: '+r.n_results+' results'+(r.model?' ('+r.model+')':''),'done');
      vizRenderConfig(); // refresh config panel to update session info
    }else{
      log('Restore failed: '+(r.error||'unknown error'),'error');
    }
  }catch(e){log('Restore error: '+e.message,'error')}
}

async function clearPlotCache(){
  if(!confirm('Delete all cached plot images? Data (CSV, JSON, metrics) will be kept.')) return;
  try{
    const r=await(await fetch('/api/session/clear_plots',{method:'POST'})).json();
    if(r.ok){_cacheBytes=r.cache_size_bytes||0;updateSessionBadge();if($('cfgCacheSize'))$('cfgCacheSize').textContent=_cacheBytes>0?fmtSize(_cacheBytes)+' total':'No data cached';log(`Plot cache cleared: ${r.freed_mb}MB freed`,'done')}
    else{log('Clear failed: '+(r.error||''),'error')}
  }catch(e){log('Clear error: '+e.message,'error')}
}

async function clearAllSessionData(){
  if(!confirm('Delete ALL session data? This includes CSV, JSON, plots, and aggregate stats. Export first if needed.')) return;
  try{
    const r=await(await fetch('/api/session/clear_all',{method:'POST'})).json();
    if(r.ok){sessionResults=[];dashResults=[];_promptTotal=0;_cacheBytes=0;updateSessionBadge();$('dataTableContainer').innerHTML='<p style="color:var(--text-3);font-size:11px">No data yet.</p>';if($('cfgCacheSize'))$('cfgCacheSize').textContent='No data cached';log(`Session data cleared: ${r.freed_mb}MB freed`,'done')}
    else{log('Clear failed: '+(r.error||''),'error')}
  }catch(e){log('Clear error: '+e.message,'error')}
}

var _cacheBytes=0;
function fmtSize(b){if(b<1024)return b+'B';if(b<1048576)return(b/1024).toFixed(0)+'KB';return(b/1048576).toFixed(1)+'MB'}
function updateSessionBadge(){
  var n=_promptTotal||dashResults.length||sessionResults.length;
  var label=n?`Session: ${n} prompts`:'No session';
  if(n&&_cacheBytes>0) label+=` · ${fmtSize(_cacheBytes)} cached`;
  $('sessionBadge').textContent=label;
  $('dashBtn').disabled=n<1;
  $('exportBtn').disabled=n<1;
  var cs=$('cfgCacheSize');if(cs) cs.textContent=_cacheBytes>0?fmtSize(_cacheBytes)+' total':'No data cached';
}

// ─── User Info ──────────────────────────────────────────────────
async function saveUserInfo(){
  const fd=new FormData();fd.append('name',$('userName').value);fd.append('organization',$('userOrg').value);fd.append('project',$('userProject').value);
  try{await fetch('/api/user_info',{method:'POST',body:fd})}catch(e){}
}

// ─── Models ─────────────────────────────────────────────────────
async function loadModelList(){
  try{
    const r=await(await fetch('/api/models')).json();
    const sel=$('modelSelect');sel.innerHTML='<option value="">-- Select --</option>';
    (r.models||[]).forEach(m=>{const o=document.createElement('option');o.value=m.id;o.textContent=`${m.name}${m.notes?' - '+m.notes:''}`;sel.appendChild(o)});
    const custom=document.createElement('option');custom.value='custom';custom.textContent='Custom pair...';sel.appendChild(custom);
  }catch(e){}
}
function toggleAddModel(){$('addModelForm').classList.toggle('visible')}
async function addModel(){
  const fd=new FormData();
  fd.append('id',$('newModelId').value);fd.append('name',$('newModelName').value);
  fd.append('base',$('newModelBase').value);fd.append('instruct',$('newModelInstruct').value);
  try{
    const r=await(await fetch('/api/models',{method:'POST',body:fd})).json();
    if(r.ok){log('Model pair added','done');$('addModelForm').classList.remove('visible');loadModelList();
      ['newModelId','newModelName','newModelBase','newModelInstruct'].forEach(i=>$(i).value='');}
    else log('Error: '+(r.error||''),'error');
  }catch(e){log('Error: '+e.message,'error')}
}

// ─── Model Loading ──────────────────────────────────────────────
async function loadModel(){
  const sel=$('modelSelect');
  if(!sel.value){showError('promptError','Select a model pair first.');return}
  const btn=$('loadModelBtn');setLoading(btn,true);clearLog();log('Starting model load...');setStatus('loading','LOADING');
  const fd=new FormData();
  if(sel.value==='custom'){
    const b=prompt('Base model HF ID:');const i=prompt('Instruct model HF ID:');
    if(!b||!i){setLoading(btn,false);return}
    fd.append('base_id',b);fd.append('instruct_id',i);
  } else fd.append('pair_id',sel.value);
  try{await fetch('/api/load_model',{method:'POST',body:fd})}catch(e){}

  // Wait for model_loaded or model_error via SSE.
  // Progress events are logged by _handleProgressEvent automatically.
  var result = await new Promise(function(resolve) {
    _sseWaiters.model_loaded.push(function(evt) { resolve({ok: true, evt: evt}); });
    _sseWaiters.model_error.push(function(evt) { resolve({ok: false, evt: evt}); });
  });

  setLoading(btn, false);

  if (result.ok) {
    // _handleModelLoaded already set modelLoaded, status, buttons.
    // Do the remaining one-time setup here.
    sessionResults = [];
    dashResults = [];
    _promptTotal = 0;
    updateSessionBadge();
    loadPromptLibrary();
    loadProbeFiles();
    try {
      var st = await(await fetch('/api/status')).json();
      if (st.user_info) {
        $('userName').value = st.user_info.name || '';
        $('userOrg').value = st.user_info.organization || '';
        $('userProject').value = st.user_info.project || '';
      }
    } catch(e) {}
    playChime();
    setInferenceModel($('inferenceModelSelect').value);
  }
}

async function setInferenceModel(cls){
  if(!modelLoaded)return;
  const sel=$('inferenceModelSelect');sel.disabled=true;
  log('Switching inference model to '+cls+'…','');
  const fd=new FormData();fd.append('model_class',cls);
  try{const r=await(await fetch('/api/set_inference_model',{method:'POST',body:fd})).json();
    if(!r.ok){showError('promptError',r.message||'Failed to switch inference model');sel.value=(cls==='base'?'instruct':'base')}
    else{log('Inference model: '+cls,'done')}
  }catch(e){log('Error switching model: '+e.message,'error');sel.value=(cls==='base'?'instruct':'base')}
  sel.disabled=false;
}

async function resetAll(){
  if(!confirm('Clear session and unload model?'))return;clearLog();setStatus('loading','RESETTING');
  try{const r=await(await fetch('/api/reset',{method:'POST'})).json();if(r.ok){modelLoaded=false;sessionResults=[];dashResults=[];_promptTotal=0;_cacheBytes=0;setStatus('idle','NO MODEL');$('analyzeBtn').disabled=true;$('batchBtn').disabled=true;$('dataTableContainer').innerHTML='<p style="color:var(--text-3);font-size:11px">No data yet.</p>';updateSessionBadge();log(r.message,'done')}}catch(e){log('Reset error: '+e.message,'error')}
}

// ─── Analysis ───────────────────────────────────────────────────
function _buildAnalysisFormData(){
  const fd=new FormData();
  fd.append('compute_kl',$('computeKL').checked);
  fd.append('compute_trajectory',$('cfgComputeTraj').checked);
  fd.append('full_capture',$('cfgFullCapture')?$('cfgFullCapture').checked:false);
  fd.append('capture_responses',$('cfgCaptureResponses').checked);
  fd.append('compute_ltp',$('computeLTP').checked);
  fd.append('compute_sfd',$('computeSFD').checked);
  fd.append('ltp_k',$('cfgLtpK').value);
  fd.append('ltp_layer_strategy',$('cfgLtpLayerStrategy').value);
  fd.append('ltp_svd_rank',0);
  return fd;
}

async function toggleBaselines(enabled){}

async function analyzePrompt(){
  const prompt=$('promptInput').value.trim();
  if(!prompt){showError('promptError','Enter a prompt to analyze.');return}
  if(prompt.length>5000){showError('promptError','Prompt too long (max 5000 chars).');return}
  showError('promptError','');
  const btn=$('analyzeBtn');setLoading(btn,true);log('Analyzing...');
  const fd=_buildAnalysisFormData();
  fd.append('prompt',prompt);fd.append('category',$('categorySelect').value);
  const prevN=_promptTotal;
  try{
    let resp,data;
    try{
      resp=await fetch('/api/analyze',{method:'POST',body:fd});
      data=await resp.json();
    }catch(fetchErr){
      // Browser timeout (Safari "Load failed") — poll for backend completion
      log('Connection interrupted, waiting for backend...','');
      let recovered=false;
      for(let attempt=0;attempt<15;attempt++){
        await new Promise(r=>setTimeout(r,4000));
        try{
          const chk=await(await fetch('/api/status')).json();
          const sn=chk.session?chk.session.n_results:0;
          if(sn>prevN){
            const allResp=await fetch(`/api/session/results?page=1&per_page=9999`);
            const allData=await allResp.json();
            if(allData.ok&&allData.results){
              sessionResults=allData.results;
              _promptTotal=allData.total;
              if(allData.cache_size_bytes!=null)_cacheBytes=allData.cache_size_bytes;
              updateSessionBadge();renderDataTable();
              dtGoPage(Math.ceil(allData.total/_dtPageSize)||1);
              log('Done (recovered after timeout)','done');
              playChime();
              recovered=true;break;
            }
          }
          log('Still waiting... ('+((attempt+1)*4)+'s)','');
        }catch(chkErr){}
      }
      if(recovered){setLoading(btn,false);return}
      throw fetchErr;
    }
    if(data.ok){log('Done','done');sessionResults.push(data.result);_promptTotal=data.session_n;if(data.cache_size_bytes!=null)_cacheBytes=data.cache_size_bytes;updateSessionBadge();renderDataTable();dtGoPage(Math.ceil(data.session_n/_dtPageSize)||1)}
    else{log('Error: '+(data.error||'Unknown'),'error');showError('promptError',data.error||'Analysis failed.')}
  }catch(e){log('Error: '+e.message,'error')}
  setLoading(btn,false);
}

async function analyzeBatch(){
  const file=$('csvFile').files[0];if(!file){log('No CSV selected','error');return}
  const btn=$('batchBtn');setLoading(btn,true);log('Starting batch...');
  const fd=_buildAnalysisFormData();
  fd.append('file',file);fd.append('compute_trajectory',false);
  try{
    const resp=await fetch('/api/analyze_batch',{method:'POST',body:fd});const data=await resp.json();
    if(!data.ok){log('Error: '+(data.error||''),'error');setLoading(btn,false);return}
    log(`Batch started: ${data.n_prompts} prompts...`);

    // Wait for batch_done via SSE. Progress logged by _handleProgressEvent.
    await new Promise(function(resolve) {
      _sseWaiters.batch_done.push(function() { resolve(); });
    });

    playChime();
    setLoading(btn,false);
    log('Loading results...');
    try{
      const allResp=await fetch(`/api/session/results?page=1&per_page=9999`);
      const allData=await allResp.json();
      if(allData.ok&&allData.results){
        sessionResults=allData.results;
        _promptTotal=allData.total;
        if(allData.cache_size_bytes!=null)_cacheBytes=allData.cache_size_bytes;
        updateSessionBadge();
        renderDataTable();
      }
    }catch(re){log('Results load: '+re.message,'error')}
    try{await refreshSession()}catch(de){log('Session refresh: '+de.message,'error')}
  }catch(e){log('Error: '+e.message,'error');setLoading(btn,false)}
}

// ─── Session Refresh ────────────────────────────────────────────
async function refreshSession(){
  if(_busy){log('Another operation in progress, please wait.','error');return}
  _busy=true;
  log('Refreshing session data...');
  try{
    const resp=await fetch('/api/dashboard');
    if(!resp.ok){log('Session refresh HTTP error: '+resp.status,'error');return}
    const data=await resp.json();
    if(!data.ok){log('Session refresh: '+(data.error||'unknown error'),'error');return}

    dashResults=data.results||[];
    if(data.cache_size_bytes!=null)_cacheBytes=data.cache_size_bytes;
    if(data.session_info){
      _promptTotal=data.session_info.n_results||dashResults.length;
    }
    updateSessionBadge();
    renderDataTable();

    log('Session refreshed: '+dashResults.length+' prompts','done');
  }catch(e){
    log('Session refresh error: '+e.message,'error');
    console.error('Session refresh error:',e);
  }finally{
    _busy=false;
  }
}

function plotCard(key,title,desc,idx,collapsed){
  const chevron=collapsed?'▶':'▼';
  return `<div class="feature"><div class="feature-header" onclick="toggleFeature(this)"><div class="feature-title"><span class="chevron" style="color:var(--text-2);font-size:12px;margin-right:6px">${chevron}</span>${title}</div><p class="feature-desc">${desc}</p></div><div class="feature-body${collapsed?' collapsed':''}"><div class="plot-container"><img src="/api/plots/individual/${idx}/${key}" loading="lazy" onerror="this.parentElement.innerHTML='<p style=\\'color:var(--text-3);font-size:11px;padding:12px\\'>Plot not available</p>'"></div></div></div>`;
}
function plotCardUrl(key,title,desc,collapsed){
  const chevron=collapsed?'▶':'▼';
  return `<div class="feature"><div class="feature-header" onclick="toggleFeature(this)"><div class="feature-title"><span class="chevron" style="color:var(--text-2);font-size:12px;margin-right:6px">${chevron}</span>${title}</div><p class="feature-desc">${desc}</p></div><div class="feature-body${collapsed?' collapsed':''}"><div class="plot-container"><img src="/api/plots/${key}" loading="lazy" onerror="this.parentElement.innerHTML='<p style=\\'color:var(--text-3);font-size:11px;padding:12px\\'>Plot not available</p>'"></div></div></div>`;
}

async function exportSession(){
  if(_busy){log('Another operation in progress, please wait.','error');return}
  _busy=true;
  const opts={csv:true,pdf:EXPORT_DEFAULTS.pdf,json:EXPORT_DEFAULTS.json,charts:EXPORT_DEFAULTS.charts,includeArrays:EXPORT_DEFAULTS.includeArrays,exportPath:EXPORT_DEFAULTS.exportPath,embeddingPrecision:EXPORT_DEFAULTS.embeddingPrecision};
  const enabledFmts=['CSV',opts.pdf?'PDF':null,opts.json?'JSON':null,opts.charts?'Charts':null].filter(Boolean).join(', ');
  log(`Exporting (${enabledFmts})...`);
  try{
    // Register the SSE waiter BEFORE the POST, because the server's
    // export handler uses await run_in_threadpool(_do_export) which
    // completes the export (and emits the SSE event) before returning
    // the HTTP response.  If we wait until after the fetch resolves
    // to register, the event has already fired and been discarded.
    var exportPromise = new Promise(function(resolve) {
      _sseWaiters.export_ready.push(function(evt) { resolve({ok: true, evt: evt}); });
      _sseWaiters.export_error.push(function(evt) { resolve({ok: false, evt: evt}); });
    });

    const r=await fetch('/api/export',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(opts)});const data=await r.json();
    if(!data.ok){
      // Clean up the dangling waiter
      _sseWaiters.export_ready.length = 0;
      _sseWaiters.export_error.length = 0;
      log('Export error: '+(data.error||''),'error');_busy=false;return;
    }
    log('Export started...');

    var result = await exportPromise;

    if (result.ok) {
      playChime();
      var pathSucceeded = opts.exportPath && result.evt.filename && !result.evt.filename.includes('path failed');
      if (pathSucceeded) {
        log('Export saved to: ' + opts.exportPath, 'done');
      } else {
        log('Downloading ZIP...');
        try {
          var a = document.createElement('a'); a.href = '/api/export/download'; a.download = 'tagm_session.zip'; document.body.appendChild(a); a.click(); document.body.removeChild(a);
          log('Download started', 'done');
        } catch(de) { log('Download error: ' + de.message, 'error'); }
      }
    }
    _busy=false;
  }catch(e){log('Export error: '+e.message,'error');_busy=false}
}

// ─── Visualization Pop-out Store ─────────────────────────────────
const _vizStore={};
function _plotHtml(plotKey,title,desc){
  return`<div class="feature"><div class="feature-header"><div class="feature-title">${title}</div><p class="feature-desc">${desc||''}</p></div><div class="feature-body"><div class="plot-container" id="pc_${plotKey}"><div style="font-family:var(--mono);font-size:12px;color:var(--text-2);padding:20px;text-align:center" id="pl_${plotKey}">Loading ${plotKey}.png…</div><img src="/api/plots/${plotKey}" style="max-width:100%;height:auto;display:none" onload="this.style.display='';var pl=document.getElementById('pl_${plotKey}');if(pl)pl.style.display='none'" onerror="var pl=document.getElementById('pl_${plotKey}');if(pl){pl.textContent='Failed to generate ${plotKey}';pl.style.color='var(--red)'}"></div></div></div>`;
}
function storeViz(key,title,html,scripts){_vizStore[key]={title:title,html:html,scripts:scripts||null}}
function popoutViz(key){
  const entry=_vizStore[key];
  if(!entry){console.warn('No viz stored for',key);return}
  const pw=1060,ph=820;
  const left=Math.round((screen.width-pw)/2);
  const top=Math.round((screen.height-ph)/2);
  const w=window.open('','_blank',`width=${pw},height=${ph},left=${left},top=${top},scrollbars=yes`);
  if(!w) return;
  const styles=Array.from(document.querySelectorAll('style,link[rel="stylesheet"]'))
    .map(s=>s.outerHTML).join('\n');
  const scriptBlock=entry.scripts?`<script>${entry.scripts}<\/script>`:'';
  w.document.write(`<!DOCTYPE html><html><head><title>${entry.title} — TAGM</title>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
    ${styles}
    <style>body{background:var(--bg-0);color:var(--text-1);font-family:var(--sans);padding:20px;margin:0}
    .feature{max-width:960px;margin:0 auto 16px auto}
    .plot-container img{max-width:100%;height:auto}
    </style>
    </head><body>${entry.html}${scriptBlock}</body></html>`);
  w.document.close();
}
function vizLabel(storeKey,title,desc,borderColor){
  const bc=borderColor?`border-left:3px solid ${borderColor};`:'';
  return`<div class="viz-label" onclick="popoutViz('${storeKey}')" style="${bc}"><span class="vl-title">${title}</span><span class="vl-desc">${desc||''}</span><span class="vl-open">↗ Open</span></div>`;
}

// ─── Pop-out Record Window ──────────────────────────────────────
function popoutRecord(rid){
  const el=document.getElementById(rid);
  if(!el) return;
  const pw=1000,ph=800;
  const left=Math.round((screen.width-pw)/2);
  const top=Math.round((screen.height-ph)/2);
  const w=window.open('','_blank',`width=${pw},height=${ph},left=${left},top=${top},scrollbars=yes`);
  if(!w) return;
  // Grab stylesheets from parent
  const styles=Array.from(document.querySelectorAll('style,link[rel="stylesheet"]'))
    .map(s=>s.outerHTML).join('\n');
  w.document.write(`<!DOCTYPE html><html><head><title>TAGM Record</title>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
    ${styles}
    <style>body{background:var(--bg-0);color:var(--text-1);font-family:var(--sans);padding:20px;margin:0}
    .card{max-width:960px;margin:0 auto}</style>
    </head><body>${el.outerHTML}</body></html>`);
  w.document.close();
}

function renderTopK(r){
  if(!r.instruct_topk||!r.instruct_topk.length)return'';
  const n=Math.max(r.instruct_topk.length,(r.base_topk||[]).length);let rows='';
  for(let i=0;i<n;i++){const it=r.instruct_topk[i],bt=(r.base_topk||[])[i];rows+=`<tr><td>${i+1}</td><td class="tok-inst">${it?escHtml(it[0]):'--'}</td><td>${it?(it[1]*100).toFixed(1)+'%':'--'}</td><td class="tok-base">${bt?escHtml(bt[0]):'--'}</td><td>${bt?(bt[1]*100).toFixed(1)+'%':'--'}</td></tr>`}
  return`<div class="feature"><div class="feature-header"><div class="feature-title">Model Predictions</div><p class="feature-desc">What the instruct-tuned model and the base model each predict as the most likely next tokens.</p></div><div class="feature-body" style="padding:10px"><table class="topk-table" style="width:100%"><tr><th>#</th><th>Instruct</th><th>P</th><th>Base</th><th>P</th></tr>${rows}</table></div></div>`
}
function renderTokenTable(r){
  if(!r.tokens||!r.signed_attr||!r.signed_attr.length)return'';
  const mx=Math.max(...r.signed_attr.map(Math.abs)),ms=r.per_token_stress?Math.max(...r.per_token_stress):1;let rows='';
  for(let i=0;i<r.tokens.length;i++){const a=r.signed_attr[i],s=r.per_token_stress?r.per_token_stress[i]:0,ap=mx>0?(Math.abs(a)/mx*100):0,ac=a<0?'var(--red)':'var(--green)',sp=ms>0?(s/ms*100):0;rows+=`<tr><td>${escHtml(r.tokens[i])}</td><td style="color:${ac}">${a>=0?'+':''}${a.toFixed(4)}</td><td><span class="token-bar" style="width:${ap}%;background:${ac}"></span></td><td>${s.toFixed(4)}</td><td><span class="token-bar" style="width:${sp}%;background:var(--blue)"></span></td></tr>`}
  return`<div class="feature"><div class="feature-header" onclick="toggleFeature(this)"><div class="feature-title">Per-Token Attribution Table</div><p class="feature-desc">Raw numeric values for each token's signed attribution and focused stress score. The colored bars show relative magnitude within this prompt — useful for identifying exactly which tokens drive the model's alignment response. Sort mentally by bar length to find the dominant tokens.</p><div class="feature-legend">🔑 Attr = signed contribution (green = reinforces, red = opposes) · Stress = correction pressure at signal layers · Bar width = relative magnitude within this prompt</div></div><div class="feature-body" style="padding:10px"><table class="data-table"><tr><th>Token</th><th>Attr</th><th></th><th>Stress</th><th></th></tr>${rows}</table></div></div>`
}
var _dtPage=1,_dtPageSize=25,_dtSorted=[];
function renderDataTable(){
  var source=dashResults.length?dashResults:sessionResults;
  if(!source.length){$('dataTableContainer').innerHTML='<p style="color:var(--text-3);font-size:var(--font-table)">No data yet.</p>';return}

  _dtSorted=[...source].map((r,i)=>({...r,_srcIdx:r._index!=null?r._index:i})).sort((a,b)=>a._srcIdx-b._srcIdx);

  _dtPage=Math.min(_dtPage,Math.ceil(_dtSorted.length/_dtPageSize)||1);
  _renderDataTablePage();
}

function _renderDataTablePage(){
  const sorted=_dtSorted;
  const totalPages=Math.ceil(sorted.length/_dtPageSize)||1;
  const start=(_dtPage-1)*_dtPageSize;
  const pageData=sorted.slice(start,start+_dtPageSize);

  // Column definitions: [key, label, format, heatHue, defaultWidth]
  const cols=[
    ['_cb','','cb',null,32],
    ['_index','#',null,null,32],
    ['prompt','Prompt','s',null,400],
    ['category','Cat','cat',null,52],
    ['role','Role','s',null,42],
    ['seq_len','Tok','i',null,36],
    ['stress_score','Stress','f3',200,60],
    ['net_correction','Net','f5',160,72],
    ['entropy','Ent','f4',40,58],
    ['middle_share','Int%','pct',280,52],
    ['interior_cv','IntCV','f4',null,54],
    ['top2_share','Bnd%','pct',null,52],
    ['kl_divergence','KL','f4',null,52],
    ['n_negative_tokens','Neg','i',null,34],
    ['_inst_top1','InstTop1','s',null,70],
    ['_inst_prob','InstP','pct3',null,50],
    ['_base_top1','BaseTop1','s',null,70],
    ['_base_prob','BaseP','pct3',null,50],
    ['_ltp_max_prc','MaxPRC','pct_pp',null,56],
    ['_ltp_n_dir','N_dir','i',null,40],
    ['_ltp_M','M','f6',null,68],
    ['_sfd_density','Dens','f4',120,56],
    ['_rank_tau','Tau','f3',60,48],
    ['_rank_overlap','Ovlp','pct',null,48],
  ];

  // Compute ranges for heatmap columns (across ALL data, not just page)
  const ranges={};
  ['stress_score','net_correction','entropy','middle_share','_sfd_density','_rank_tau'].forEach(k=>{
    const vals=sorted.map(r=>getVal(r,k)).filter(v=>v!=null&&!isNaN(v));
    if(vals.length) ranges[k]={min:Math.min(...vals),max:Math.max(...vals)};
  });
  function heatBg(val,key,hue){
    if(!hue)return'';
    const r=ranges[key];if(!r||r.max===r.min)return'';
    const t=(val-r.min)/(r.max-r.min);
    return`background:hsla(${hue},60%,50%,${(t*0.25).toFixed(2)})`;
  }

  function fmt(val,format){
    if(val==null||val===''||val===undefined) return'<span style="color:var(--text-3)">--</span>';
    if(format==='s') return escHtml(String(val).substring(0,120));
    if(format==='i') return Math.round(val);
    if(format==='f2') return Number(val).toFixed(2);
    if(format==='f3') return Number(val).toFixed(3);
    if(format==='f4') return Number(val).toFixed(4);
    if(format==='f5') return Number(val).toFixed(5);
    if(format==='f6') return Number(val).toFixed(6);
    if(format==='pct') return (Number(val)*100).toFixed(1)+'%';
    if(format==='pct3') return (Number(val)*100).toFixed(1)+'%';
    if(format==='pct_pp') return (Number(val)*100).toFixed(1)+'pp';
    if(format==='cat') return val?`<span class="pill ${pillClass(val)}">${val.substring(0,4)}</span>`:'--';
    return val;
  }

  function getVal(r,key){
    if(key==='_cb') return r._srcIdx;
    if(key==='_index') return r._srcIdx+1;
    if(key==='_inst_top1') return r.instruct_topk&&r.instruct_topk[0]?r.instruct_topk[0][0]:null;
    if(key==='_inst_prob') return r.instruct_topk&&r.instruct_topk[0]?r.instruct_topk[0][1]:null;
    if(key==='_base_top1') return r.base_topk&&r.base_topk[0]?r.base_topk[0][0]:null;
    if(key==='_base_prob') return r.base_topk&&r.base_topk[0]?r.base_topk[0][1]:null;
    if(key==='_ltp_M') return r.ltp?r.ltp.mean_M:null;
    if(key==='_ltp_max_prc') return r.ltp?r.ltp.max_prc:null;
    if(key==='_ltp_n_dir') return r.ltp?r.ltp.n_directional:null;
    if(key==='_sfd_density') return r.sfd?r.sfd.density_mean:null;
    if(key==='_rank_tau') return r.rank_displacement?r.rank_displacement.mean_tau:null;
    if(key==='_rank_overlap') return r.rank_displacement?r.rank_displacement.mean_overlap:null;
    return r[key];
  }

  // ─── Toolbar ──────────────────────────────────────────────────
  let h=`<div class="dt-toolbar">
    <label style="margin:0;display:flex;align-items:center;gap:4px;cursor:pointer">
      <input type="checkbox" id="dtSelectAll" onchange="dtToggleAll(this.checked)"> <span style="font-size:11px">All</span>
    </label>
    <span class="dt-sel-count" id="dtSelCount">0 selected</span>
    <button class="btn btn-secondary btn-sm" id="dtViewBtn" onclick="dtViewSelected()" disabled>View Selected</button>
    <button class="btn btn-secondary btn-sm" id="dtRerunBtn" onclick="dtRerunSelected()" disabled>Rerun Selected</button>
    <button class="btn btn-danger btn-sm" id="dtRemoveBtn" onclick="dtRemoveSelected()" disabled>Remove Selected</button>
    <span style="margin-left:auto;color:var(--text-1);font-size:11px">${sorted.length} records · page ${_dtPage}/${totalPages}${_cacheBytes>0?' · '+fmtSize(_cacheBytes)+' cached':''}</span>
  </div>`;

  // ─── Table ────────────────────────────────────────────────────
  h+=`<div class="dt-wrap"><div class="dt-scroll" id="dtScroll"><table class="dt-grid" id="dtGridTable"><thead><tr>`;

  for(let ci=0;ci<cols.length;ci++){
    const[key,label,format,,w]=cols[ci];
    const isCheckbox = key==='_cb';
    const cls = isCheckbox ? ' class="dt-cb"' : '';
    const content = isCheckbox
      ? ''
      : `${label}<div class="dt-resize" data-col="${ci}" onmousedown="dtStartResize(event,${ci})"></div>`;
    h+=`<th${cls} style="min-width:${w}px;width:${w}px">${content}</th>`;
  }
  h+=`</tr></thead><tbody>`;

  for(const r of pageData){
    const srcIdx = r._srcIdx;
    h+=`<tr data-idx="${srcIdx}" id="dtRow_${srcIdx}">`;
    for(const[key,label,format,hue,w]of cols){
      const val=getVal(r,key);
      if(key==='_cb'){
        h+=`<td class="dt-cb"><input type="checkbox" data-idx="${srcIdx}" onchange="dtUpdateSelection()"></td>`;
      } else if(key==='prompt'){
        h+=`<td class="dt-cell-prompt" title="${escHtml(r.prompt)}">${fmt(val,format)}</td>`;
      } else {
        const style=heatBg(val||0,key,hue);
        h+=`<td style="${style}">${fmt(val,format)}</td>`;
      }
    }
    h+=`</tr>`;
  }
  h+=`</tbody></table></div></div>`;

  // ─── Pagination controls ──────────────────────────────────────
  if(totalPages>1){
    h+=`<div style="display:flex;align-items:center;justify-content:center;gap:8px;padding:8px 0;font-family:var(--mono);font-size:11px">`;
    h+=`<button class="btn btn-secondary btn-sm" onclick="dtGoPage(1)" ${_dtPage<=1?'disabled':''} style="width:auto">⟨⟨</button>`;
    h+=`<button class="btn btn-secondary btn-sm" onclick="dtGoPage(${_dtPage-1})" ${_dtPage<=1?'disabled':''} style="width:auto">⟨ Prev</button>`;
    // Page numbers
    var startP=Math.max(1,_dtPage-2),endP=Math.min(totalPages,_dtPage+2);
    for(var p=startP;p<=endP;p++){
      if(p===_dtPage) h+=`<span style="color:var(--blue);font-weight:600;padding:0 4px">${p}</span>`;
      else h+=`<button class="btn btn-secondary btn-sm" onclick="dtGoPage(${p})" style="width:auto;padding:2px 6px">${p}</button>`;
    }
    h+=`<button class="btn btn-secondary btn-sm" onclick="dtGoPage(${_dtPage+1})" ${_dtPage>=totalPages?'disabled':''} style="width:auto">Next ⟩</button>`;
    h+=`<button class="btn btn-secondary btn-sm" onclick="dtGoPage(${totalPages})" ${_dtPage>=totalPages?'disabled':''} style="width:auto">⟩⟩</button>`;
    h+=`<span style="color:var(--text-3);margin-left:8px">${sorted.length} rows · ${_dtPageSize}/page</span>`;
    h+=`</div>`;
  }

  $('dataTableContainer').innerHTML=h;
}
function dtGoPage(p){
  var totalPages=Math.ceil(_dtSorted.length/_dtPageSize)||1;
  _dtPage=Math.max(1,Math.min(totalPages,p));
  _renderDataTablePage();
}

// ─── Data Table: Selection ──────────────────────────────────────
function dtGetSelectedIndices(){
  const cbs=document.querySelectorAll('#dtGridTable tbody input[type="checkbox"]:checked');
  return Array.from(cbs).map(cb=>parseInt(cb.dataset.idx));
}
function dtUpdateSelection(){
  const sel=dtGetSelectedIndices();
  const count=sel.length;
  const el=$('dtSelCount'); if(el) el.textContent=count+' selected';
  const reBtn=$('dtRerunBtn'); if(reBtn) reBtn.disabled=count===0;
  const rmBtn=$('dtRemoveBtn'); if(rmBtn) rmBtn.disabled=count===0;
  const vwBtn=$('dtViewBtn'); if(vwBtn) vwBtn.disabled=count===0;
  // Update "select all" checkbox state
  const allCbs=document.querySelectorAll('#dtGridTable tbody input[type="checkbox"]');
  const allCheck=$('dtSelectAll');
  if(allCheck){
    allCheck.checked=allCbs.length>0&&count===allCbs.length;
    allCheck.indeterminate=count>0&&count<allCbs.length;
  }
  // Highlight selected rows
  document.querySelectorAll('#dtGridTable tbody tr').forEach(tr=>{
    const cb=tr.querySelector('input[type="checkbox"]');
    tr.classList.toggle('dt-selected',cb&&cb.checked);
  });
}
function dtToggleAll(checked){
  document.querySelectorAll('#dtGridTable tbody input[type="checkbox"]').forEach(cb=>{cb.checked=checked});
  dtUpdateSelection();
}

// ─── Data Table: Remove ─────────────────────────────────────────
async function dtRemoveSelected(){
  const indices=dtGetSelectedIndices();
  if(!indices.length) return;
  if(!confirm(`Remove ${indices.length} result(s) from the session? This cannot be undone.`)) return;
  try{
    const r=await(await fetch('/api/session/remove',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({indices})})).json();
    if(r.ok){
      log(`Removed ${r.removed} results (${r.remaining} remaining)`,'done');
      _promptTotal=r.remaining;
      dashResults=[];
      updateSessionBadge();renderDataTable();
    } else { log('Remove failed: '+(r.error||'unknown'),'error') }
  }catch(e){log('Remove error: '+e.message,'error')}
}

// ─── Data Table: Rerun ──────────────────────────────────────────
async function dtRerunSelected(){
  const indices=dtGetSelectedIndices();
  if(!indices.length) return;
  // Gather current analysis options from the sidebar checkboxes (same IDs as analyze_single)
  const options={
    compute_kl:$('computeKL')&&$('computeKL').checked,
    compute_trajectory:$('cfgComputeTraj')&&$('cfgComputeTraj').checked,
    capture_responses:$('cfgCaptureResponses')&&$('cfgCaptureResponses').checked,
    full_capture:$('cfgFullCapture')&&$('cfgFullCapture').checked,
    compute_ltp:$('computeLTP')&&$('computeLTP').checked,
    compute_sfd:$('computeSFD')&&$('computeSFD').checked,
    ltp_k:parseInt(($('cfgLtpK')&&$('cfgLtpK').value)||8),
    ltp_layer_strategy:($('cfgLtpLayerStrategy')&&$('cfgLtpLayerStrategy').value)||'signal',
    ltp_svd_rank:0,
  };
  log(`Rerunning ${indices.length} prompt(s)...`,'analyzing');
  try{
    const r=await(await fetch('/api/session/rerun',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({indices,options})})).json();
    if(r.ok){
      log(`Reran ${r.rerun} prompts (${r.total} total in session)`,'done');
      _promptTotal=r.total;
      dashResults=[];
      updateSessionBadge();dashResults=[];renderDataTable();
    } else { log('Rerun failed: '+(r.error||'unknown'),'error') }
  }catch(e){log('Rerun error: '+e.message,'error')}
}

// ─── Data Table: View Selected ─────────────────────────────────
async function dtViewSelected(){
  const indices=dtGetSelectedIndices();
  if(!indices.length) return;
  try{
    // Fetch full result data for each selected index (one page at a time)
    const results=[];
    for(const idx of indices){
      // Use page=1,per_page large enough, then find by _index
      // More efficient: fetch page containing this index
      const page=Math.floor(idx/50)+1;
      const resp=await fetch(`/api/session/results?page=${page}&per_page=50`);
      const data=await resp.json();
      if(data.ok&&data.results){
        const match=data.results.find(r=>r._index===idx);
        if(match) results.push(match);
      }
    }
    if(!results.length){log('No results found for selected indices','error');return}

    // Build card HTML for each selected result
    const styles=Array.from(document.querySelectorAll('style,link[rel="stylesheet"]'))
      .map(s=>s.outerHTML).join('\n');
    let cardsHtml='';
    for(const r of results){
      const plotKeys=r._plot_keys||[];
      const idx=r._index;
      cardsHtml+=_buildRecordCardHtml(r,plotKeys,idx);
    }

    // Open popout
    const pw=1060,ph=850;
    const left=Math.round((screen.width-pw)/2);
    const top=Math.round((screen.height-ph)/2);
    const w=window.open('','_blank',`width=${pw},height=${ph},left=${left},top=${top},scrollbars=yes`);
    if(!w) return;
    w.document.write(`<!DOCTYPE html><html><head><title>TAGM — ${results.length} Record${results.length>1?'s':''}</title>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
      ${styles}
      <style>body{background:var(--bg-0);color:var(--text-1);font-family:var(--sans);padding:20px;margin:0}
      .card{max-width:960px;margin:0 auto 16px auto}
      .plot-container img{max-width:100%;height:auto}
      </style></head><body>${cardsHtml}</body></html>`);
    w.document.close();
  }catch(e){log('View error: '+e.message,'error');console.error(e)}
}

function _buildRecordCardHtml(r,plotKeys,idx){
  // Build a self-contained record card for popout display
  let h=`<div class="card">
    <div class="card-header record-header">
      <h3>${escHtml(r.prompt)}</h3>
      <div class="record-meta">
        ${r.category?`<span class="pill ${pillClass(r.category)}">${r.category}</span>`:''}
        <span style="font-family:var(--mono);font-size:12px;color:var(--text-1)">#${idx} · ${r.seq_len} tokens</span>
        ${r.ltp?'<span class="ltp-badge">LTP</span>':''}
      </div>
    </div>
    <div class="card-body">`;

  // Metrics grid
  h+=`<div class="metrics-grid">${mc('Stress',r.stress_score,null)}${mc('Net Corr',r.net_correction,null)}${mc('Entropy',r.entropy,null)}${mc('Int%',r.middle_share,null,v=>(v*100).toFixed(1)+'%')}${mc('Bnd%',r.top2_share,null,v=>(v*100).toFixed(1)+'%')}${mc('IntCV',r.interior_cv,null)}${r.kl_divergence!=null?mc('KL',r.kl_divergence,null):''}`;
  if(r.ltp){h+=mc('MaxPRC',r.ltp.max_prc,null,v=>(v*100).toFixed(1)+'pp','ltp-metric');h+=mc('N_dir',r.ltp.n_directional,null,null,'ltp-metric');h+=mc('M',r.ltp.mean_M,null,null,'ltp-metric')}
  if(r.sfd){h+=mc('Density',r.sfd.density_mean,null,null,'sfd-metric')}
  if(r.rank_displacement&&r.rank_displacement.mean_tau!=null){h+=mc('Tau',r.rank_displacement.mean_tau,null,v=>v.toFixed(3),'sfd-metric');h+=mc('Overlap',r.rank_displacement.mean_overlap,null,v=>(v*100).toFixed(1)+'%','sfd-metric')}
  if(r.candidate_graph){h+=mc('Contest',r.candidate_graph.contested_frac,null,v=>(v*100).toFixed(0)+'%','ltp-metric');h+=mc('Sw/tok',r.candidate_graph.switch_rate,null,v=>v.toFixed(2),'ltp-metric')}
  h+=`</div>`;
  if(r.has_negative_tokens)h+=`<div style="margin-bottom:8px;font-size:11px;color:var(--purple)">⚠ ${r.n_negative_tokens} of ${r.seq_len} tokens push against the alignment correction</div>`;

  // Inline plots (these render in the popout since images load from server)
  const featureMeta={signed_attribution:'Signed Attribution',stress_per_token:'Per-Token Stress',distribution_metrics:'Correction Distribution',amplitude_trajectory:'Layer Trajectory',heatmap:'Token × Layer Heatmap',sfd_density:'QK Density',rank_displacement:'Rank Displacement'};
  const ltpPlotMeta={ltp_profiles:'Tension Profiles',ltp_tension_magnitudes:'Tension Magnitudes',ltp_summary_stats:'LTP Summary',ltp_profile_heatmap:'Token × Rank Heatmap'};

  for(const k of plotKeys){
    if(k.startsWith('ltp_'))continue;
    if(VIZ_REGISTRY[k]&&VIZ_REGISTRY[k].scope!=='prompt')continue;
    const title=featureMeta[k]||k;
    h+=`<div class="feature"><div class="feature-header"><div class="feature-title">${title}</div></div><div class="feature-body"><div class="plot-container"><img src="/api/plots/individual/${idx}/${k}" style="max-width:100%;height:auto"></div></div></div>`;
  }

  // LTP plots
  if(r.ltp){
    const ltpKeys=plotKeys.filter(k=>k.startsWith('ltp_')&&k!=='ltp_dual_trajectory');
    if(ltpKeys.length){
      h+=`<div style="margin-top:14px;padding-top:12px;border-top:2px solid var(--cyan)"><div style="font-family:var(--mono);font-size:13px;font-weight:600;color:var(--cyan);margin-bottom:8px">LTP</div>`;
      for(const k of ltpKeys){
        const title=ltpPlotMeta[k]||k;
        h+=`<div class="feature" style="border-color:var(--cyan)"><div class="feature-header"><div class="feature-title" style="color:var(--cyan)">${title}</div></div><div class="feature-body"><div class="plot-container"><img src="/api/plots/individual/${idx}/${k}" style="max-width:100%;height:auto"></div></div></div>`;
      }
      h+=`</div>`;
    }
  }

  // Model predictions
  if(r.instruct_topk&&r.instruct_topk.length) h+=renderTopK(r);

  // Token table
  if(r.tokens&&r.signed_attr&&r.signed_attr.length) h+=renderTokenTable(r);

  h+=`</div></div>`;
  return h;
}

// ─── Data Table: Column Resize ──────────────────────────────────
let _dtResizeState=null;
function dtStartResize(e,colIndex){
  e.preventDefault();
  const table=document.getElementById('dtGridTable');
  if(!table) return;
  const th=table.querySelectorAll('thead th')[colIndex];
  if(!th) return;
  const startX=e.clientX;
  const startW=th.offsetWidth;
  const handle=e.target;
  handle.classList.add('active');

  function onMove(ev){
    const delta=ev.clientX-startX;
    const newW=Math.max(28,startW+delta);
    th.style.width=newW+'px';
    th.style.minWidth=newW+'px';
    // Also resize corresponding body cells for consistency
    const rows=table.querySelectorAll('tbody tr');
    rows.forEach(tr=>{
      const td=tr.children[colIndex];
      if(td){td.style.width=newW+'px';td.style.minWidth=newW+'px'}
    });
  }
  function onUp(){
    handle.classList.remove('active');
    document.removeEventListener('mousemove',onMove);
    document.removeEventListener('mouseup',onUp);
  }
  document.addEventListener('mousemove',onMove);
  document.addEventListener('mouseup',onUp);
}

// ─── Prompt Library ─────────────────────────────────────────────
async function loadPromptLibrary(){
  try{const r=await(await fetch('/api/prompts')).json();promptLibraryData=r.prompts||[];renderPromptLibrary()}catch(e){}
}
function renderPromptLibrary(){
  const s=$('promptLibrary');s.innerHTML='<option value="">-- Presets --</option>';
  const g={};promptLibraryData.forEach((p,i)=>{if(p.baseline)return;const c=p.category||'other';if(!g[c])g[c]=[];g[c].push({...p,index:i})});
  for(const[c,ps]of Object.entries(g)){const og=document.createElement('optgroup');og.label=c[0].toUpperCase()+c.slice(1);ps.forEach(p=>{const o=document.createElement('option');o.value=p.index;o.textContent=p.prompt.length>55?p.prompt.substring(0,52)+'...':p.prompt;og.appendChild(o)});s.appendChild(og)}
}
function loadFromLibrary(){const s=$('promptLibrary');if(s.value==='')return;const p=promptLibraryData[parseInt(s.value)];if(p){$('promptInput').value=p.prompt;$('categorySelect').value=p.category||''}}
async function addCurrentToLibrary(){
  const p=$('promptInput').value.trim();if(!p){showError('promptError','No prompt to save.');return}
  const fd=new FormData();fd.append('prompt',p);fd.append('category',$('categorySelect').value||'benign');fd.append('baseline',false);
  try{const r=await(await fetch('/api/prompts',{method:'POST',body:fd})).json();if(r.ok){promptLibraryData=r.prompts;renderPromptLibrary();log('Saved to library','done')}else showError('promptError',r.error||'Failed')}catch(e){}
}

// ─── Keyboard Navigation ────────────────────────────────────────
// ─── Feature card collapse (delegated) ──────────────────────────
document.addEventListener('click',e=>{
  const header=e.target.closest('.feature-header');
  if(!header) return;
  const feature=header.closest('.feature');
  if(!feature) return;
  const body=feature.querySelector('.feature-body');
  if(!body) return;
  body.classList.toggle('collapsed');
  const chevron=header.querySelector('.chevron');
  if(chevron) chevron.textContent=body.classList.contains('collapsed')?'▶':'▼';
});

// ─── Alignment Terrain Viewer ──────────────────────────────────
let _terrainState = null;

// Transforms the uniform module payload into the renderer's internal D shape.
// Accepts the full result object from the correction_field_topology module.
// Each row is [bm, im, kl, primary, ctoks]. Primary replaces the old
// replacement_ratio slot and carries the per-token primary-signal value
// that drives the spine ridge vertex.
function _terrainFromPayload(payload){
  const cats={};
  const prompts=payload.prompts||[];
  const targetMax=payload.target_max||0.008;
  const accentAvailable=!!payload.accent_available;

  for(const p of prompts){
    const cat=p.category||'unknown';
    const toks=p.tokens||[];
    const nT=toks.length;
    const primary=p.primary||[];
    const kls=p.per_token_kl||[];
    const iBank=p.instruct_bank||[];
    const bBank=p.base_bank||[];
    const iStatus=p.instruct_status||[];
    const bStatus=p.base_status||[];
    const cf=p.counterfactual_tokens||[];
    const baseCf=p.base_counterfactual_tokens||[];

    // Per-prompt normalization with separate scale factors for the two
    // axes of signal. The banks carry per-rank values whose variation
    // across ranks is the source of the terrain's texture; the primary
    // (spine) carries the per-token total, which is naturally an order
    // of magnitude larger because it aggregates across ranks.
    // Normalizing them jointly would crush the bank variation against
    // the floor, so we compute two independent scale factors so each
    // signal maps its own maximum onto target_max.
    let bankMax=0, primaryMax=0;
    for(let i=0;i<nT;i++){
      if(primary[i]>primaryMax) primaryMax=primary[i];
      const ip=iBank[i]||[],bp=bBank[i]||[];
      for(let j=0;j<8;j++){
        if(ip[j]>bankMax) bankMax=ip[j];
        if(bp[j]>bankMax) bankMax=bp[j];
      }
    }
    const bankScale=bankMax>1e-10?(targetMax/bankMax):1;
    const primaryScale=primaryMax>1e-10?(targetMax/primaryMax):1;

    if(!cats[cat]) cats[cat]=[];
    const rows=[];

    for(let i=0;i<nT;i++){
      const kl=(kls[i]!=null)?kls[i]:0;
      const primaryScaled=(primary[i]||0)*primaryScale;

      // Per-rank magnitudes used by the label underline layer.
      const iDisp=iBank[i]||[];
      const im=[];
      const imMag=[];
      for(let j=0;j<8;j++){
        const v=(iDisp[j]!=null?iDisp[j]:0);
        im.push(v*bankScale);
        imMag.push(v);
      }
      const bDisp=bBank[i]||[];
      const bm=[];
      const bmMag=new Array(8);
      // Base bank is reversed so columns 9..16 in the lattice render
      // highest-rank-first moving outward from the spine.
      for(let j=7;j>=0;j--){
        const v=(bDisp[j]!=null?bDisp[j]:0);
        bm.push(v*bankScale);
        bmMag[7-j]=v;
      }

      // Counterfactual labels
      const cands=cf[i]||[];
      const baseCands=baseCf[i]||[];
      const ctoks=[];
      for(let j=0;j<8;j++){
        if(j<cands.length&&cands[j]){
          ctoks.push(Array.isArray(cands[j])?cands[j][0]||'':String(cands[j]));
        }else ctoks.push('');
      }
      if(baseCands.length>0){
        for(let j=7;j>=0;j--){
          if(j<baseCands.length&&baseCands[j]){
            const nm=Array.isArray(baseCands[j])?baseCands[j][0]||'':String(baseCands[j]);
            ctoks.push('\u00B7'+nm);
          }else ctoks.push('\u00B7');
        }
      }else{
        for(let j=0;j<8;j++) ctoks.push('');
      }

      // Per-rank status strings, reversed on the base side to match the
      // lattice column ordering (column 9 = base rank 0).
      const iSt=iStatus[i]||[];
      const bSt=bStatus[i]||[];
      const statusRow=[];
      for(let j=0;j<8;j++) statusRow.push(iSt[j]||'');
      for(let j=7;j>=0;j--) statusRow.push(bSt[j]||'');

      // Row layout: [bm, im, kl, primary_scaled, ctoks, status_row, per_rank_magnitudes]
      // per_rank_magnitudes runs in the same column order as ctoks and statusRow,
      // i.e. imMag[0..7] for the instruct bank, bmMag[0..7] for the base bank.
      const perRankMag=[...imMag];
      for(let j=7;j>=0;j--) perRankMag.push(bmMag[j]);

      rows.push([bm,im,Math.round(kl*1e4)/1e4,primaryScaled,ctoks,statusRow,perRankMag]);
    }
    cats[cat].push([p.prompt||'',toks,rows]);
  }

  console.log('[Terrain] Payload transform: measure='+payload.measure+
              ', target_max='+targetMax+
              ', prompts='+prompts.length);
  return cats;
}

function _terrainRendererCore(container, D, opts){
  // Self-contained terrain renderer — used by both the embedded viewer
  // and the popout window (via .toString() serialization).
  // Only external dependency: THREE.js must be loaded.
  var charLimit=(opts&&opts.charLimit)||50;

  // Color ramps — dual palette
  function stops(t,s){for(let i=0;i<s.length-1;i++){if(t<=s[i+1][0]){const f=(t-s[i][0])/(s[i+1][0]-s[i][0]);return s[i][1].map((v,j)=>v+(s[i+1][1][j]-v)*f)}}return s[s.length-1][1]}
  const BASES={
    dual:{label:'Dual (Base \u2194 Instruct)',
      base:function(t){return stops(t,[[0,[.01,.02,.12]],[.15,[.02,.08,.32]],[.30,[.04,.20,.52]],[.45,[.06,.38,.68]],[.58,[.10,.56,.78]],[.70,[.22,.72,.84]],[.82,[.48,.85,.88]],[.92,[.72,.94,.96]],[1,[.90,.98,.99]]])},
      inst:function(t){return stops(t,[[0,[.12,.02,.01]],[.15,[.32,.06,.02]],[.30,[.52,.12,.02]],[.45,[.72,.28,.04]],[.58,[.86,.44,.06]],[.70,[.94,.60,.08]],[.82,[.98,.76,.18]],[.92,[.99,.88,.42]],[1,[1,.96,.72]]])},
      spine:function(t){return stops(t,[[0,[.06,.05,.06]],[.30,[.20,.18,.16]],[.60,[.50,.46,.38]],[.80,[.78,.72,.60]],[1,[.96,.92,.80]]])}},
    vivid:{label:'Vivid (shared)',
      base:function(t){return stops(t,[[0,[.04,.02,.28]],[.08,[.10,.04,.48]],[.16,[.18,.06,.60]],[.24,[.06,.22,.72]],[.32,[.02,.42,.78]],[.40,[.02,.62,.68]],[.48,[.10,.75,.45]],[.56,[.38,.82,.22]],[.64,[.65,.86,.12]],[.72,[.88,.84,.08]],[.80,[.96,.68,.06]],[.88,[.98,.42,.06]],[.94,[.92,.18,.08]],[1,[.72,.04,.14]]])},
      inst:null,spine:null},
    ember:{label:'Heat (shared)',
      base:function(t){return stops(t,[[0,[.01,0,0]],[.28,[.20,.02,.01]],[.52,[.62,.10,0]],[.72,[.92,.44,0]],[.88,[.98,.78,.10]],[1,[1,.98,.80]]])},
      inst:null,spine:null}
  };

  const isPopout=!!(opts&&opts.isPopout);
  let curCat=opts.cat||(Object.keys(D).includes('benign')?'benign':Object.keys(D)[0]);
  let curIdx=opts.idx||0, curBase=opts.base||opts.palette||'dual', labelScale=opts.labelScale||2.0;
  let renderMode='surface';
  let dataFilter=opts.dataFilter||'all';
  let slideshowPlaying=false, slideshowSpeed=4, slideshowTimer=null;
  const BG=[.047,.047,.07];
  const FY=-2.2, SCX=0.55, COLS=17, X_MAX=8*SCX;
  const HSCALE=(opts&&opts.heightScale)||180;
  const KL_BOOST=(opts&&opts.klBoost!=null)?opts.klBoost:0.3;
  const ACCENT_MODE=(opts&&opts.accentMode)||'kl_brightness';
  const SHOW_GRID=!opts||opts.showGrid!==false;
  const SHOW_LABELS=!opts||opts.showLabels!==false;
  const SHOW_LEGEND=!opts||opts.showLegend!==false;
  const SHOW_SPINE=!opts||opts.showSpine!==false;

  // Accent palette for promoted/demoted/matched label styling. Desaturated
  // greens and vermillions to avoid collision with the warm-cool bank
  // ramps. Okabe-Ito adjacent: bluish-green for promoted, vermillion for
  // demoted, muted cream for matched so promoted/demoted come forward.
  const ACCENT_COLORS={
    promoted: {underline:'rgba(0,158,115,0.85)',  text:'rgba(120,220,175,1.0)'},
    demoted:  {underline:'rgba(213,94,0,0.85)',   text:'rgba(245,165,115,1.0)'},
    matched:  {underline:'rgba(160,160,170,0.45)',text:'rgba(220,218,208,1.0)'},
  };

  function _rampFor(colIdx){
    const profile=BASES[curBase];
    if(colIdx===8 && profile.spine) return profile.spine;
    if(colIdx>8 && profile.inst) return profile.inst;
    return profile.base;
  }

  // Build DOM
  container.innerHTML='<div class="terrain-wrap">'
    +'<canvas id="tCv"></canvas>'
    +'<div class="terrain-ui" id="tUI"></div>'
    +'<div class="terrain-inf" id="tInf"></div>'
    +'<div class="terrain-leg" id="tLeg">'
    +'<div style="display:flex;gap:4px;align-items:center"><span id="tLegMin">0</span><canvas id="tLegBase"></canvas><span id="tLegMax">0</span></div>'
    +'</div></div>';

  const cv=document.getElementById('tCv');
  const wrap=cv.parentElement;
  const sc=new THREE.Scene();
  const cam=new THREE.PerspectiveCamera(44,1,.1,200);
  const ren=new THREE.WebGLRenderer({canvas:cv,antialias:true});
  ren.setPixelRatio(Math.min(devicePixelRatio,2));
  ren.setClearColor(0x0c0c12,1);
  function onResize(){
    const w=wrap.clientWidth,h=wrap.clientHeight;
    if(w<10||h<10)return;
    ren.setSize(w,h);cam.aspect=w/h;cam.updateProjectionMatrix();
  }
  onResize();
  new ResizeObserver(onResize).observe(wrap);

  sc.add(new THREE.AmbientLight(0xaabbcc,1.0));
  const dl=new THREE.DirectionalLight(0xccddff,1.0);dl.position.set(3,10,6);sc.add(dl);
  const dl2=new THREE.DirectionalLight(0x667788,.5);dl2.position.set(-4,5,-3);sc.add(dl2);

  let msh=null,wir=null,flr=null,lbs=[];

  function mixColor(mag,mn,mx,klT,xAbs,colIdx){
    const t=Math.max(0,Math.min(1,(mag-mn)/(mx-mn||1)));
    const rampFn=_rampFor(colIdx);
    var rgb=rampFn(t);
    var r=rgb[0],g=rgb[1],b=rgb[2];
    const klBoost=1.0+klT*KL_BOOST;
    r=Math.min(1,r*klBoost);g=Math.min(1,g*klBoost);b=Math.min(1,b*klBoost);
    const fade=Math.pow(xAbs/X_MAX,1.6);
    r=r*(1-fade)+BG[0]*fade;g=g*(1-fade)+BG[1]*fade;b=b*(1-fade)+BG[2]*fade;
    return new THREE.Color(r,g,b);
  }

  function build(){
    [msh,wir,flr].forEach(function(m){if(!m)return;sc.remove(m);m.geometry.dispose();m.material.dispose()});
    lbs.forEach(function(s){sc.remove(s);if(s.geometry)s.geometry.dispose();if(s.material){if(s.material.map)s.material.map.dispose();s.material.dispose()}});
    msh=wir=flr=null;lbs=[];
    var entries=D[curCat];
    if(!entries||!entries[curIdx])return;
    var parts=entries[curIdx],prompt=parts[0],toks=parts[1],rows=parts[2];
    var nT=toks.length,scZ=SCX;
    var zDepth=(nT-1)*scZ;
    var zOffset=zDepth/2;
    // Magnitude pool is built from the bank values only. Pooling the
    // spine primary here would dominate the mn/mx window (the primary
    // is naturally an order of magnitude larger than the per-rank bank
    // values) and flatten the bank variation that gives the terrain its
    // texture. The transform already applies separate scale factors so
    // the primary lands within target_max without needing to be pooled.
    var allMags=[];
    for(var ri=0;ri<rows.length;ri++){
      var rr=rows[ri];
      rr[0].forEach(function(v){allMags.push(v)});
      rr[1].forEach(function(v){allMags.push(v)});
    }
    var mn=Math.min.apply(null,allMags),mx=Math.max.apply(null,allMags);
    var klVals=rows.map(function(r){return r[2]});
    var klMin=Math.min.apply(null,klVals),klMax=Math.max.apply(null,klVals);

    // Surface
    var vsArr=[],csArr=[],ixArr=[],vtx=[];
    for(var t=0;t<nT;t++){
      vtx.push([]);
      var bm=rows[t][0],im=rows[t][1],kl=rows[t][2];
      var z=-t*scZ+zOffset,klT=(kl-klMin)/(klMax-klMin||1);
      // Spine vertex reads the per-token primary scalar from slot 3 of
      // the row, which the client transform derives from the selected
      // measure's natural per-token signal (rank-displacement total_disp,
      // LTP tension magnitude, ASM stress, etc.). Previous versions
      // synthesized this from the mean of the two banks, which made the
      // spine a summary of its own context rather than a focal signal.
      var spineMag=rows[t][3]!=null?rows[t][3]:0;
      for(var c=0;c<COLS;c++){
        var x,mag;
        if(c<8){x=(c-8)*SCX;mag=bm[c]}
        else if(c===8){x=0;mag=spineMag}
        else{x=(c-8)*SCX;mag=im[c-9]}
        var y=FY+(mag-mn)*HSCALE;
        vtx[t].push({x:x,y:y,z:z});
        vsArr.push(x,y,z);
        var col=mixColor(mag,mn,mx,klT,Math.abs(x),c);
        csArr.push(col.r,col.g,col.b);
      }
    }
    // Filter mask
    var klMean=klVals.reduce(function(a,v){return a+v},0)/klVals.length;
    var allMagsSorted=allMags.slice().sort(function(a,b){return a-b});
    var medianMag=allMagsSorted[Math.floor(allMagsSorted.length/2)];
    var tokenVisible=[];
    for(var t=0;t<nT;t++){
      if(dataFilter==='highkl') tokenVisible.push(rows[t][2]>klMean);
      else if(dataFilter==='hotspots'){
        var tmB=rows[t][0],tmI=rows[t][1];
        var tMean=(tmB.reduce(function(a,v){return a+v},0)+tmI.reduce(function(a,v){return a+v},0))/(tmB.length+tmI.length);
        tokenVisible.push(tMean>medianMag);
      }
      else tokenVisible.push(true);
    }
    var dimFactor=0.15;
    for(var t=0;t<nT;t++){
      if(!tokenVisible[t]){
        for(var c=0;c<COLS;c++){var i=(t*COLS+c)*3;csArr[i]*=dimFactor;csArr[i+1]*=dimFactor;csArr[i+2]*=dimFactor}
      }
    }

    // Render
    var geo=new THREE.BufferGeometry();
    geo.setAttribute('position',new THREE.Float32BufferAttribute(vsArr,3));
    geo.setAttribute('color',new THREE.Float32BufferAttribute(csArr,3));

    if(renderMode==='points'){
      var sizes=new Float32Array(nT*COLS);
      for(var t=0;t<nT;t++){var vis=tokenVisible[t];for(var c=0;c<COLS;c++){var ef=1-Math.pow(Math.abs(c-8)/8,0.8)*0.6;sizes[t*COLS+c]=vis?(0.06*ef):0.01}}
      geo.setAttribute('size',new THREE.Float32BufferAttribute(sizes,1));
      msh=new THREE.Points(geo,new THREE.PointsMaterial({vertexColors:true,size:0.08,sizeAttenuation:true,transparent:true,opacity:0.9}));
      sc.add(msh);
    }else{
      for(var t=0;t<nT-1;t++)for(var c=0;c<COLS-1;c++){var a=t*COLS+c,b=a+1,cc=a+COLS,dd=cc+1;ixArr.push(a,b,cc,b,dd,cc)}
      geo.setIndex(ixArr);geo.computeVertexNormals();
      msh=new THREE.Mesh(geo,new THREE.MeshPhongMaterial({vertexColors:true,side:THREE.DoubleSide,shininess:30,emissive:0x444444,emissiveIntensity:0.35}));
      sc.add(msh);
    }

    // Floor
    var fG=new THREE.PlaneGeometry(COLS*SCX+1.2,nT*scZ+2);fG.rotateX(-Math.PI/2);
    flr=new THREE.Mesh(fG,new THREE.MeshBasicMaterial({color:0x0c0c12,transparent:true,opacity:.25,side:THREE.DoubleSide}));
    flr.position.set(0,FY-.01,0);sc.add(flr);

    // Grid lines
    var OFF=0.032;
    if(renderMode==='surface' && SHOW_GRID){
      var gridP=[],gridC=[];
      var pushSeg=function(ax,ay,az,bx,by,bz,r,g,b){gridP.push(ax,ay+OFF,az,bx,by+OFF,bz);gridC.push(r,g,b,r,g,b)};
      for(var t=0;t<nT;t++){for(var c=0;c<COLS-1;c++){var v0=vtx[t][c],v1=vtx[t][c+1];var near=(c===7||c===8)?.30:.14;pushSeg(v0.x,v0.y,v0.z,v1.x,v1.y,v1.z,near*.55,near*.65,near)}}
      for(var c=0;c<COLS;c++){var isSpine=(c===8);for(var t=0;t<nT-1;t++){var v0=vtx[t][c],v1=vtx[t+1][c];if(isSpine){pushSeg(v0.x,v0.y,v0.z,v1.x,v1.y,v1.z,.75,.72,.62)}else{var fade=1-Math.pow(Math.abs(c-8)/8,.7);pushSeg(v0.x,v0.y,v0.z,v1.x,v1.y,v1.z,fade*.22,fade*.26,fade*.38)}}}
      var gridGeo=new THREE.BufferGeometry();
      gridGeo.setAttribute('position',new THREE.Float32BufferAttribute(gridP,3));
      gridGeo.setAttribute('color',new THREE.Float32BufferAttribute(gridC,3));
      wir=new THREE.LineSegments(gridGeo,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.92}));
      sc.add(wir);
    }

    // Spine line + dots
    if(SHOW_SPINE){
      var spinePts=[];
      for(var t=0;t<nT;t++){var sv=vtx[t][8];spinePts.push(new THREE.Vector3(sv.x,sv.y+OFF+0.04,sv.z))}
      var spineGeo=new THREE.BufferGeometry().setFromPoints(spinePts);
      var spineLine=new THREE.Line(spineGeo,new THREE.LineBasicMaterial({color:0xf0ede4,linewidth:2,transparent:true,opacity:0.85}));
      sc.add(spineLine);lbs.push(spineLine);
      var dotGeo=new THREE.SphereGeometry(0.035,6,6);
      var dotMat=new THREE.MeshBasicMaterial({color:0xf0ede4,transparent:true,opacity:0.7});
      for(var t=0;t<nT;t++){var sv=vtx[t][8];var dot=new THREE.Mesh(dotGeo,dotMat);dot.position.set(sv.x,sv.y+OFF+0.04,sv.z);sc.add(dot);lbs.push(dot)}
    }

    // Labels at every vertex
    if(SHOW_LABELS){
      var LW=Math.round(96*labelScale),LH=Math.round(24*labelScale);
      var LSX=SCX*0.90*labelScale,LSY=LSX*(LH/LW);
      var SPINE_BOOST=1.5;
      var SLW=Math.round(LW*SPINE_BOOST),SLH=Math.round(LH*SPINE_BOOST);
      var SLSX=LSX*SPINE_BOOST,SLSY=SLSX*(SLH/SLW);
      var labelFontPx=Math.round(12*labelScale);
      var spineFontPx=Math.round(12*labelScale*SPINE_BOOST);

      // Find max per-rank magnitude across the prompt so underline length
      // in promoted_demoted mode is normalized to the prompt's range
      // rather than to whatever KL range happens to exist.
      var magMaxThis=1e-10;
      for(var tt=0;tt<nT;tt++){
        var mr=rows[tt][6]||[];
        for(var jj=0;jj<mr.length;jj++) if(mr[jj]>magMaxThis) magMaxThis=mr[jj];
      }

      var texCache={};
      // Label texture generator. In promoted_demoted mode the underline
      // color encodes the set-membership status of the candidate at that
      // rank (promoted / demoted / matched) and its length encodes the
      // candidate's per-rank magnitude. In kl_brightness mode the previous
      // behavior is preserved: underline color is sampled from the active
      // palette and length tracks the token's normalized KL value. In
      // off mode no underline is drawn at all.
      var getTexture=function(str,klT,isSpine,colIdx,status,rankMagT){
        var accentKey=(ACCENT_MODE==='promoted_demoted'&&status)?status:(ACCENT_MODE==='off'?'none':'kl');
        var bucket=Math.round(klT*8);
        var magBucket=Math.round((rankMagT||0)*8);
        var key=str+'|'+bucket+'|'+magBucket+'|'+labelScale+'|'+(isSpine?'S':'C')+'|'+colIdx+'|'+accentKey;
        if(texCache[key]) return texCache[key];
        var w=isSpine?SLW:LW, h=isSpine?SLH:LH, fpx=isSpine?spineFontPx:labelFontPx;
        var S=2,lc=document.createElement('canvas');lc.width=w*S;lc.height=h*S;
        var lx=lc.getContext('2d');lx.scale(S,S);
        lx.font='700 '+fpx+'px system-ui,sans-serif';lx.textAlign='center';lx.textBaseline='middle';

        // Text stroke (always) plus text fill. Fill color depends on mode:
        // promoted/demoted tints for bank tokens; matched and legacy modes
        // stay with the existing cream; spine tokens always render blue.
        lx.lineWidth=Math.max(isSpine?3:2,labelScale*(isSpine?2.2:1.5));
        lx.strokeStyle=isSpine?'rgba(0,0,0,0.98)':'rgba(0,0,0,0.95)';
        lx.strokeText(str,w/2,h/2);
        if(isSpine){
          lx.fillStyle='rgba(86,180,233,1.0)';
        }else if(ACCENT_MODE==='promoted_demoted' && status && ACCENT_COLORS[status]){
          lx.fillStyle=ACCENT_COLORS[status].text;
        }else{
          lx.fillStyle='rgba(240,237,228,1.0)';
        }
        lx.fillText(str,w/2,h/2);

        // Underline layer. Off suppresses it entirely; kl_brightness uses
        // the legacy behavior; promoted_demoted uses the status color and
        // the per-rank magnitude for length.
        if(ACCENT_MODE!=='off'){
          var ulLen, ulColor, ulAlpha, ulOffset;
          if(isSpine){
            ulLen=Math.max(4,Math.round((w-10)*klT));
            ulColor='rgba(86,180,233,0.8)';
            ulAlpha=0.6;
            ulOffset=Math.round(10*labelScale*SPINE_BOOST);
          }else if(ACCENT_MODE==='promoted_demoted' && status && ACCENT_COLORS[status]){
            var lenT=Math.max(0,Math.min(1,rankMagT||0));
            ulLen=Math.max(4,Math.round((w-10)*lenT));
            ulColor=ACCENT_COLORS[status].underline;
            ulAlpha=0.85;
            ulOffset=Math.round(8*labelScale);
          }else{
            // kl_brightness legacy path
            ulLen=Math.max(4,Math.round((w-10)*klT));
            var rgb2=_rampFor(colIdx)(Math.max(.15,klT*.75+.18));
            ulColor='rgb('+Math.round(rgb2[0]*255)+','+Math.round(rgb2[1]*255)+','+Math.round(rgb2[2]*255)+')';
            ulAlpha=0.7;
            ulOffset=Math.round(8*labelScale);
          }
          lx.globalAlpha=ulAlpha;
          lx.fillStyle=ulColor;
          lx.fillRect((w-ulLen)/2,h/2+ulOffset,ulLen,Math.max(2,Math.round(2*labelScale)));
        }

        texCache[key]=new THREE.CanvasTexture(lc);return texCache[key];
      };

      for(var t=0;t<nT;t++){
        var ctoks=rows[t][4]||[];
        var statusRow=rows[t][5]||[];
        var rankMags=rows[t][6]||[];
        var klT2=(rows[t][2]-klMin)/(klMax-klMin||1);
        for(var c=0;c<COLS;c++){
          var vv=vtx[t][c];
          var isSpine=(c===8);
          var str;
          if(c<8) str=(ctoks[8+c]||'').replace(/^[\s\u0120\u00c2]/,'').trim().slice(0,12);
          else if(isSpine) str=toks[t].replace(/^[\s\u0120\u00c2]/,'').trim().slice(0,12);
          else str=(ctoks[16-c]||'').replace(/^[\s\u0120\u00c2]/,'').trim().slice(0,12);
          if(!str) str='\u00B7';

          // Map lattice column to status-row index. Columns 0..7 are the
          // base bank (reversed): column 0 = farthest from spine = base
          // rank 7; column 7 = adjacent to spine = base rank 0. The
          // status row from the transform lists instruct 0..7 first then
          // base 7..0 (reversed), so column index maps directly for the
          // instruct side (c=9..16 -> statusRow[c-9..c-16]... ).
          var status=null, rankMag=0;
          if(!isSpine){
            if(c<8){
              // Base bank. statusRow indices 8..15 hold base 7..0.
              var baseStatusIdx=8+(7-c);
              status=statusRow[baseStatusIdx]||null;
              rankMag=rankMags[baseStatusIdx]||0;
            }else{
              // Instruct bank. statusRow indices 0..7 hold instruct 0..7.
              var instIdx=c-9;
              status=statusRow[instIdx]||null;
              rankMag=rankMags[instIdx]||0;
            }
          }
          var rankMagT=magMaxThis>1e-10?(rankMag/magMaxThis):0;

          var tex=getTexture(str,klT2,isSpine,c,status,rankMagT);
          var fd=isSpine?1.0:Math.max(.28,1-Math.pow(Math.abs(c-8)/8,1.2)*.68);
          var sx=isSpine?SLSX:LSX, sy=isSpine?SLSY:LSY;
          var sp=new THREE.Sprite(new THREE.SpriteMaterial({map:tex,transparent:true,depthTest:false,opacity:fd}));
          sp.position.set(vv.x,vv.y+OFF+(isSpine?.10:.06),vv.z);sp.scale.set(sx,sy,1);
          sc.add(sp);lbs.push(sp);
        }
      }
    }

    // Axis labels
    if(SHOW_LEGEND){
      var frontZ=vtx[0][8].z+scZ*.6;
      var makeAx=function(txt,xp,align){
        var ac=document.createElement('canvas');ac.width=160;ac.height=22;
        var ax=ac.getContext('2d');ax.font='500 10px system-ui,sans-serif';
        ax.fillStyle='rgba(110,110,130,0.55)';ax.textAlign=align||'center';
        ax.fillText(txt,align==='right'?155:align==='left'?5:80,14);
        var sp=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(ac),transparent:true,depthTest:false}));
        sp.position.set(xp,FY+.1,frontZ);sp.scale.set(1.3,.19,1);sc.add(sp);lbs.push(sp);
      };
      makeAx('\u2190 base',-2.2,'right');
      makeAx('instruct \u2192',2.2,'left');
    }

    // Info
    var peakKl=Math.max.apply(null,klVals);
    var peakTok=toks[klVals.indexOf(peakKl)].replace(/^[\s\u0120\u00c2]/,'').trim().slice(0,14);
    var baseHasData=rows.some(function(r){return r[0].some(function(v){return v>0})});
    var infEl=document.getElementById('tInf');
    if(infEl) infEl.textContent=nT+' tokens \u00B7 tension '+(mn*1e3).toFixed(2)+'\u2013'+(mx*1e3).toFixed(2)+' \u00D710\u207B\u00B3 \u00B7 peak KL '+peakKl.toFixed(3)+' "'+peakTok+'" \u00B7 base: '+(baseHasData?'\u2705 data':'\u274C NO DATA');

    // Legend
    var legC=document.getElementById('tLegBase');
    if(legC){
      var profile=BASES[curBase];var hasDual=!!profile.inst;
      legC.width=hasDual?200:100;legC.height=hasDual?18:6;
      var ctx=legC.getContext('2d');ctx.clearRect(0,0,legC.width,legC.height);
      if(hasDual){
        var g1=ctx.createLinearGradient(0,0,legC.width,0);for(var i=0;i<=20;i++){var tt=i/20;var rgb=profile.base(tt);g1.addColorStop(tt,'rgb('+Math.round(rgb[0]*255)+','+Math.round(rgb[1]*255)+','+Math.round(rgb[2]*255)+')')}
        ctx.fillStyle=g1;ctx.fillRect(0,0,legC.width,7);ctx.fillStyle='#556';ctx.font='7px monospace';ctx.fillText('base',2,6);
        var g2=ctx.createLinearGradient(0,0,legC.width,0);for(var i=0;i<=20;i++){var tt=i/20;var rgb=profile.inst(tt);g2.addColorStop(tt,'rgb('+Math.round(rgb[0]*255)+','+Math.round(rgb[1]*255)+','+Math.round(rgb[2]*255)+')')}
        ctx.fillStyle=g2;ctx.fillRect(0,11,legC.width,7);ctx.fillStyle='#556';ctx.font='7px monospace';ctx.fillText('inst',2,17);
      }else{
        var g=ctx.createLinearGradient(0,0,100,0);for(var i=0;i<=20;i++){var tt=i/20;var rgb=profile.base(tt);g.addColorStop(tt,'rgb('+Math.round(rgb[0]*255)+','+Math.round(rgb[1]*255)+','+Math.round(rgb[2]*255)+')')}
        ctx.fillStyle=g;ctx.fillRect(0,0,100,6);
      }
    }
    var lMin=document.getElementById('tLegMin'),lMax=document.getElementById('tLegMax');
    if(lMin)lMin.textContent=(mn*1e3).toFixed(1);
    if(lMax)lMax.textContent=(mx*1e3).toFixed(1);

    // Center camera on terrain — frame based on larger dimension
    th=0.0;ph=0.08;
    var terrainW=COLS*SCX;
    rad=Math.max(10, Math.max(terrainW, zDepth)*0.8+4);
    target.set(0,FY+1.0,0);
    updateCam();
  }

  // UI controls
  function buildUI(){
    var ui=document.getElementById('tUI');
    if(!ui) return;
    var h='';
    h+='<select onchange="window._terrainSelectCat(this.value)" style="min-width:80px">';
    for(var c in D){if(D.hasOwnProperty(c))h+='<option value="'+c+'"'+(c===curCat?' selected':'')+'>'+c+'</option>'}
    h+='</select><span class="tsep">|</span>';
    h+='<select id="tPSel" onchange="window._terrainSelectPrompt(+this.value)">';
    var entries=D[curCat]||[];
    entries.forEach(function(e,i){h+='<option value="'+i+'"'+(i===curIdx?' selected':'')+'>'+e[0].slice(0,charLimit)+'</option>'});
    h+='</select>';
    h+='<button class="tb" onclick="window._terrainPrev()">Prev</button>';
    var playLabel=slideshowPlaying?'Stop':'Play';
    h+='<button class="tb'+(slideshowPlaying?' on':'')+'" id="tPlayBtn" onclick="window._terrainPlayPause()">'+playLabel+'</button>';
    h+='<button class="tb" onclick="window._terrainNext()">Next</button>';
    h+='<input type="range" min="1" max="10" value="'+slideshowSpeed+'" style="width:32px;vertical-align:middle" oninput="window._terrainSetSpeed(this.value)" title="Slideshow speed">';
    h+='<span class="tsep">|</span>';
    h+='<select onchange="window._terrainSetBase(this.value)" title="Color ramp">';
    for(var k in BASES){if(BASES.hasOwnProperty(k))h+='<option value="'+k+'"'+(k===curBase?' selected':'')+'>'+BASES[k].label+'</option>'}
    h+='</select>';
    if(!isPopout){
      h+='<select onchange="window._terrainSetMode(this.value)" title="Render mode">';
      h+='<option value="surface"'+(renderMode==='surface'?' selected':'')+'>Surface</option>';
      h+='<option value="points"'+(renderMode==='points'?' selected':'')+'>Points</option>';
      h+='</select>';
      h+='<select onchange="window._terrainSetFilter(this.value)" title="Data filter">';
      h+='<option value="all"'+(dataFilter==='all'?' selected':'')+'>All tokens</option>';
      h+='<option value="highkl"'+(dataFilter==='highkl'?' selected':'')+'>High KL</option>';
      h+='<option value="hotspots"'+(dataFilter==='hotspots'?' selected':'')+'>Hot spots</option>';
      h+='</select>';
    }
    h+='<span class="tsep">|</span>';
    h+='<span style="font-size:11px;color:#bbc">Aa</span><input type="range" min="5" max="40" value="'+Math.round(labelScale*10)+'" style="width:36px;vertical-align:middle" oninput="window._terrainSetLabelScale(this.value)" title="Label size">';
    h+='<span class="tsep">|</span>';
    h+='<button class="tb'+(autoRotate?' on':'')+'" onclick="window._terrainToggleRotate()">Rotate</button>';
    h+='<input type="range" min="1" max="20" value="'+Math.round(rotateSpeed*10)+'" style="width:32px;vertical-align:middle" oninput="window._terrainSetRotateSpeed(this.value)" title="Rotation speed (RPM)">';
    h+='<span class="tsep">|</span>';
    h+='<button class="tb" onclick="window._terrainResetView()">Reset</button>';
    if(!isPopout) h+='<button class="tb" onclick="window._terrainPopout()">Pop out</button>';
    ui.innerHTML=h;
  }

  function refreshPromptSelect(){
    var sel=document.getElementById('tPSel');if(!sel)return;sel.innerHTML='';
    (D[curCat]||[]).forEach(function(e,i){var o=document.createElement('option');o.value=i;o.textContent=e[0].slice(0,charLimit);if(i===curIdx)o.selected=true;sel.appendChild(o)});
  }

  window._terrainSelectCat=function(c){curCat=c;curIdx=0;if(slideshowPlaying){slideshowPlaying=false;if(slideshowTimer){clearTimeout(slideshowTimer);slideshowTimer=null}}buildUI();build()};
  window._terrainSelectPrompt=function(i){curIdx=parseInt(i)||0;if(slideshowPlaying){slideshowPlaying=false;if(slideshowTimer){clearTimeout(slideshowTimer);slideshowTimer=null}}buildUI();build()};
  window._terrainSetBase=function(k){curBase=k;build()};
  window._terrainSetMode=function(m){renderMode=m;build()};
  window._terrainSetFilter=function(f){dataFilter=f;build()};
  window._terrainSetLabelScale=function(v){labelScale=parseInt(v)/10;build()};
  window._terrainToggleRotate=function(){autoRotate=!autoRotate;buildUI()};
  window._terrainSetRotateSpeed=function(v){rotateSpeed=parseInt(v)/10};
  window._terrainNext=function(){var entries=D[curCat]||[];if(!entries.length)return;curIdx=(curIdx+1)%entries.length;buildUI();build()};
  window._terrainPrev=function(){var entries=D[curCat]||[];if(!entries.length)return;curIdx=(curIdx-1+entries.length)%entries.length;buildUI();build()};
  window._terrainPlayPause=function(){slideshowPlaying=!slideshowPlaying;buildUI();if(slideshowPlaying)_terrainAutoAdvance();else if(slideshowTimer){clearTimeout(slideshowTimer);slideshowTimer=null}};
  window._terrainSetSpeed=function(v){slideshowSpeed=parseInt(v)||4};
  window._terrainResetView=function(){th=0.0;ph=0.08;var entries=D[curCat]||[];var nT=entries[curIdx]?entries[curIdx][1].length:10;var terrainW=COLS*SCX;var zDepth=(nT-1)*SCX;rad=Math.max(10,Math.max(terrainW,zDepth)*0.8+4);target.set(0,FY+1.0,0);updateCam()};

  function _terrainAutoAdvance(){
    if(!slideshowPlaying)return;
    var entries=D[curCat]||[];
    if(!entries.length){slideshowPlaying=false;return}
    curIdx=(curIdx+1)%entries.length;
    if(curIdx===0){var cats=Object.keys(D);var ci=cats.indexOf(curCat);if(cats.length>1){curCat=cats[(ci+1)%cats.length];curIdx=0}}
    buildUI();build();
    var delay=Math.max(500,4000-slideshowSpeed*350);
    slideshowTimer=setTimeout(_terrainAutoAdvance,delay);
  }

  // Camera
  var th=0.0,ph=0.08,rad=14;
  var target=new THREE.Vector3(0,FY+1.0,0);
  var autoRotate=!!(opts&&opts.autoRotate);
  var rotateSpeed=(opts&&opts.rotateSpeed)||0.3; // RPM
  var dragging=false;
  function updateCam(){cam.position.set(target.x+rad*Math.sin(ph)*Math.cos(th),target.y+rad*Math.cos(ph),target.z+rad*Math.sin(ph)*Math.sin(th));cam.lookAt(target)}
  var ptrs={};
  cv.addEventListener('contextmenu',function(e){e.preventDefault()});
  cv.addEventListener('pointerdown',function(e){cv.setPointerCapture(e.pointerId);ptrs[e.pointerId]={btn:e.button,x:e.clientX,y:e.clientY};dragging=true});
  cv.addEventListener('pointerup',function(e){cv.releasePointerCapture(e.pointerId);delete ptrs[e.pointerId];if(Object.keys(ptrs).length===0)dragging=false});
  cv.addEventListener('pointermove',function(e){
    var p=ptrs[e.pointerId];if(!p)return;
    var dx=e.clientX-p.x,dy=e.clientY-p.y;p.x=e.clientX;p.y=e.clientY;
    if(p.btn===0){th+=dx*.008;ph=Math.max(.08,Math.min(1.55,ph-dy*.008))}
    else{
      var speed=rad*.0012;
      var right=new THREE.Vector3();
      right.crossVectors(new THREE.Vector3(0,1,0),new THREE.Vector3(Math.sin(ph)*Math.cos(th),Math.cos(ph),Math.sin(ph)*Math.sin(th))).normalize();
      var fwd=new THREE.Vector3(Math.sin(ph)*Math.cos(th),Math.cos(ph),Math.sin(ph)*Math.sin(th)).normalize();
      var up=new THREE.Vector3().crossVectors(right,fwd).normalize();
      target.addScaledVector(right,-dx*speed);target.addScaledVector(up,-dy*speed);
    }
    updateCam();
  });
  cv.addEventListener('wheel',function(e){e.preventDefault();rad=Math.max(4,Math.min(30,rad+e.deltaY*.012));updateCam()},{passive:false});

  // Start
  buildUI();
  updateCam();
  build();
  var running=true;
  var lastTime=performance.now();
  (function loop(){if(!running)return;requestAnimationFrame(loop);
    var now=performance.now();var dt=(now-lastTime)/1000;lastTime=now;
    if(autoRotate&&!dragging){th+=rotateSpeed*Math.PI*2/60*dt;updateCam()}
    ren.render(sc,cam)})();

  // Cleanup
  var observer=new MutationObserver(function(){
    if(!document.getElementById('tCv')){running=false;observer.disconnect();ren.dispose();if(slideshowTimer){clearTimeout(slideshowTimer);slideshowTimer=null;slideshowPlaying=false}}
  });
  observer.observe(document.body,{childList:true,subtree:true});
}


function _openTerrainWindow(D, opts){
    opts = opts || {};
    var charLimit = opts.charLimit || 50;
    var autoRotate = !!opts.autoRotate;
    var rotateSpeed = opts.rotateSpeed || 0.3;
    var pw=1200,ph=800;
    var left=Math.round((screen.width-pw)/2);
    var top=Math.round((screen.height-ph)/2);
    var w=window.open('','_blank','width='+pw+',height='+ph+',left='+left+',top='+top);
    if(!w)return;
    var dataJSON=JSON.stringify(D);
    var optsJSON=JSON.stringify({
      isPopout: true,
      charLimit: charLimit,
      autoRotate: autoRotate,
      rotateSpeed: rotateSpeed,
      palette: opts.palette || 'dual',
      heightScale: opts.heightScale || 180,
      klBoost: opts.klBoost != null ? opts.klBoost : 0.3,
      accentMode: opts.accentMode || 'kl_brightness',
      accentAvailable: !!opts.accentAvailable,
      dataFilter: opts.dataFilter || 'all',
      showGrid: opts.showGrid !== false,
      showLabels: opts.showLabels !== false,
      showLegend: opts.showLegend !== false,
      showSpine: opts.showSpine !== false,
    });
    w.document.write('<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Correction Field Topology</title>');
    w.document.write('<style>*{margin:0;padding:0;box-sizing:border-box}html,body{width:100%;height:100%;overflow:hidden;background:#0c0c12;color:#c2c0b6;font-family:system-ui,sans-serif}');
    w.document.write('.terrain-wrap{position:absolute;inset:0;background:#0c0c12}');
    w.document.write('.terrain-wrap canvas{position:absolute;inset:0;width:100%!important;height:100%!important;cursor:grab}.terrain-wrap canvas:active{cursor:grabbing}');
    w.document.write('.terrain-ui{position:absolute;bottom:8px;left:8px;display:flex;flex-wrap:wrap;gap:3px;align-items:center;background:rgba(8,8,18,.75);padding:3px 7px;border-radius:6px;max-width:calc(100% - 16px);z-index:5}');
    w.document.write('.terrain-ui .tb{font-size:11px;padding:2px 8px;background:transparent;border:.5px solid rgba(255,255,255,.16);border-radius:4px;color:#888;cursor:pointer;font-family:monospace}.terrain-ui .tb:hover{background:rgba(255,255,255,.05)}.terrain-ui .tb.on{border-color:#4A9EE0;color:#85B7EB}');
    w.document.write('.terrain-ui .tsep{color:rgba(255,255,255,.12);margin:0 2px;font-size:11px}');
    w.document.write('.terrain-ui select{background:rgba(255,255,255,.05);border:.5px solid rgba(255,255,255,.15);border-radius:3px;color:#ccc;font-size:11px;padding:2px 5px;cursor:pointer;max-width:200px;font-family:monospace}');
    w.document.write('.terrain-inf{position:absolute;top:6px;left:8px;font-size:11px;color:#bbc;background:rgba(8,8,18,.55);padding:2px 6px;border-radius:4px;z-index:5;font-family:monospace}');
    w.document.write('.terrain-leg{position:absolute;bottom:8px;right:8px;display:flex;flex-direction:column;gap:2px;background:rgba(8,8,18,.55);padding:3px 6px;border-radius:4px;z-index:5}.terrain-leg canvas{height:5px;width:100px;border-radius:2px}.terrain-leg span{font-size:11px;color:#bbc}');
    w.document.write('</style>');
    w.document.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"><\/script>');
    w.document.write('</head><body>');
    w.document.write('<div id="terrainContainer" style="position:absolute;inset:0"></div>');
    w.document.write('<script>var D='+dataJSON+';var OPTS='+optsJSON+';<\/script>');
    w.document.write('<script>var _terrainRendererCore='+_terrainRendererCore.toString()+';<\/script>');
    w.document.write('<script>requestAnimationFrame(function(){_terrainRendererCore(document.getElementById("terrainContainer"),D,OPTS)});<\/script>');
    w.document.write('</body></html>');
    w.document.close();
}

async function _lazyLoadTerrainPopout(totalCount, params){
  params = params || {};
  var charLimit = params.charLimit || DISPLAY_DEFAULTS.terrainCharLimit || 50;
  var autoRotate = params.autoRotate != null ? params.autoRotate : DISPLAY_DEFAULTS.terrainAutoRotate;
  var rotateSpeed = params.rotateSpeed || DISPLAY_DEFAULTS.terrainRotateSpeed || 0.3;

  try{
    log('Fetching terrain payload from module results...');
    var r = await fetch('/api/modules/correction_field_topology/results');
    var d = await r.json();
    if(!d.ok || !d.results){
      log('No module results available. Run Correction Field Topology first.','error');
      return;
    }
    var payload = d.results;
    if(payload.error){
      log('Module reported error: '+payload.error,'error');
      return;
    }
    if(!payload.prompts || payload.prompts.length === 0){
      log('Terrain payload is empty.','error');
      return;
    }

    var D = _terrainFromPayload(payload);
    if(!Object.keys(D).length){
      log('Terrain payload produced no renderable categories.','error');
      return;
    }

    // Visual parameters captured at run time flow from payload.launch_params
    // into the renderer via opts, keeping the renderer stateless.
    var lp = payload.launch_params || {};
    var viewerOpts = {
      charLimit: lp.char_limit || charLimit,
      autoRotate: lp.auto_rotate != null ? lp.auto_rotate : autoRotate,
      rotateSpeed: lp.rotate_speed || rotateSpeed,
      palette: lp.palette || 'dual',
      heightScale: lp.height_scale || 180,
      klBoost: lp.kl_boost != null ? lp.kl_boost : 0.3,
      accentMode: lp.accent_mode || 'kl_brightness',
      accentAvailable: !!payload.accent_available,
      dataFilter: lp.data_filter || 'all',
      showGrid: lp.show_grid !== false,
      showLabels: lp.show_labels !== false,
      showLegend: lp.show_legend !== false,
      showSpine: lp.show_spine !== false,
    };

    _openTerrainWindow(D, viewerOpts);
  }catch(e){
    console.error('Terrain lazy load error:',e);
    log('Terrain load failed: '+e.message,'error');
  }
}

// ─── Advanced Engine Parameters ────────────────────────────────

var _engineConfig = {};
var _engineDefaults = {};
var _enginePending = {};  // local edits not yet applied
var _advUnlocked = false;

// Parameter metadata for display
var ENGINE_PARAM_META = {
  signal_layer_fraction:  {group:'Signal Layers',   label:'Signal layer fraction',       desc:'Fraction of model depth for each third. 0.333 = middle third.', type:'float', step:0.01, min:0.1, max:0.5},
  sfd_use_signal_layers:  {group:'SFD',             label:'SFD uses signal layers',      desc:'If enabled, SFD uses the same layer range as ASM.', type:'bool'},
  sfd_layer_start:        {group:'SFD',             label:'SFD layer start',             desc:'First layer for SFD (when "uses signal layers" is off).', type:'int', min:0, max:30},
  sfd_layer_end:          {group:'SFD',             label:'SFD layer end (exclusive)',    desc:'Last layer (exclusive) for SFD.', type:'int', min:1, max:30},
  sfd_svd_k:              {group:'SFD',             label:'SFD SVD components',          desc:'Right singular vectors retained for per-token projection.', type:'int', min:4, max:64},
  sfd_svd_seed:           {group:'SFD',             label:'SFD SVD seed',                desc:'Random seed for deterministic SVD. Ensures reproducible SFD across sessions.', type:'int', min:0, max:99999},
  rd_min_shared:          {group:'Rank Displacement',label:'Min shared candidates',      desc:'Min shared candidates for Kendall tau. Below this, tau = 0.', type:'int', min:1, max:8},
  boundary_fraction:      {group:'ASM Attribution', label:'Boundary fraction',           desc:'Fraction of tokens at each end for interior/boundary split.', type:'float', step:0.01, min:0.01, max:0.4},
  response_topk:          {group:'ASM Attribution', label:'Response top-k',              desc:'Next-token predictions captured from each model.', type:'int', min:3, max:50},
  proof1_threshold:       {group:'ASM Attribution', label:'Proof-1 exactness',           desc:'Error threshold for exact attribution decomposition.', type:'float', step:0.0001, min:0.00001, max:0.01},
  delta_svd_k:            {group:'Weight Delta',    label:'Delta SVD rank',              desc:'Max singular values for weight delta spectral summary.', type:'int', min:16, max:128},
  serialization_precision:{group:'Serialization',   label:'Decimal precision',           desc:'Decimal places for measurement values in JSON. Higher = larger files, more precision.', type:'int', min:4, max:12},
  n_bootstrap:            {group:'Statistics',      label:'Bootstrap resamples',         desc:'Bootstrap iterations for CIs. Higher = slower, tighter.', type:'int', min:500, max:20000, step:500},
  ci_level:               {group:'Statistics',      label:'Confidence level',            desc:'CI width. 0.95 = 95% confidence intervals.', type:'float', step:0.01, min:0.80, max:0.99},
  threshold_steps:        {group:'Statistics',      label:'Threshold search steps',      desc:'Granularity of optimal classification threshold search.', type:'int', min:50, max:2000, step:50},
  min_valid_separability: {group:'Statistics',      label:'Min valid for separability',  desc:'Min non-null values per metric for separability analysis.', type:'int', min:2, max:20},
  min_samples_d:          {group:'Statistics',      label:'Min samples for Cohen\'s d',  desc:'Min per-group samples for effect size.', type:'int', min:2, max:10},
  ltp_overfetch_first:    {group:'LTP',             label:'LTP overfetch (first)',       desc:'Extra candidates on first pass for k non-chosen alternatives.', type:'int', min:1, max:10},
  ltp_overfetch_second:   {group:'LTP',             label:'LTP overfetch (fallback)',    desc:'Wider fetch if first pass insufficient.', type:'int', min:2, max:20},
  disc_sublayers_top_n:   {group:'Visualization',   label:'Discriminative sublayers N',  desc:'Sublayers shown in discriminative sublayers plot.', type:'int', min:5, max:30},
  persist_probe_caches:    {group:'Modules',         label:'Persist probe caches',        desc:'Keep probe embeddings across sessions. When off, caches clear on startup.', type:'bool'},
  domain_embedding_layer_frac:{group:'Modules',     label:'Angular probe depth',         desc:'Embedding depth for subject/topic identity (angular axis). Lower = better noun separation. 0.50=default.', type:'float', step:0.05, min:0.05, max:0.95},
  domain_escalation_layer_frac:{group:'Modules',    label:'Radial probe depth',          desc:'Embedding depth for escalation level (radial axis). Higher = better discourse framing. 0.75=default.', type:'float', step:0.05, min:0.05, max:0.95},
  include_first_token:     {group:'Token Processing', label:'Include first token',        desc:'Include position-0 token in per-token analysis. On by default. Disable only if your tokenizer prepends a BOS token that would dominate per-position metrics.', type:'bool'},
  probe_projection_space:  {group:'Modules',         label:'Delta-projected probes',      desc:'Project embeddings through the o_proj weight delta before probe matching. Matches in the correction field coordinate system instead of raw hidden-state space.', type:'bool'},
  attention_weighted_pool:  {group:'Modules',        label:'Attention-weighted pooling',   desc:'Weight each token by model attention at the domain layer instead of uniform mean-pool. De-emphasizes function words without external filtering.', type:'bool'},
  export_domain_embeddings: {group:'Serialization',  label:'Export domain embeddings',     desc:'Include per-token domain embeddings in session JSON export. Large but enables offline re-analysis with different probe sets.', type:'bool'},
  chat_temperature:        {group:'Chat Generation', label:'Temperature',                 desc:'Sampling temperature for chat responses. Lower = more deterministic.', type:'float', step:0.05, min:0.0, max:2.0},
  chat_top_p:              {group:'Chat Generation', label:'Top-p (nucleus)',              desc:'Nucleus sampling threshold. Lower = fewer candidate tokens.', type:'float', step:0.05, min:0.1, max:1.0},
  chat_max_tokens:         {group:'Chat Generation', label:'Max tokens',                  desc:'Maximum tokens generated per chat response.', type:'int', min:32, max:2048, step:32},
};

function _advToggleLock(unlocked){
  _advUnlocked = unlocked;
  var btns = $('advParamsButtons');
  if(unlocked){
    _enginePending = {};
    loadAdvancedParams();
    if(btns) btns.style.display = 'flex';
  } else {
    _enginePending = {};
    if(btns) btns.style.display = 'none';
    $('advancedParamsPanel').textContent = 'Locked.';
  }
}

async function loadAdvancedParams(){
  try{
    var r = await(await fetch('/api/engine_config')).json();
    if(!r.ok) return;
    _engineConfig = r.config;
    _engineDefaults = r.defaults;
    _enginePending = {};
    renderAdvancedParams();
    _advUpdateApplyBtn();
  }catch(e){
    $('advancedParamsPanel').textContent = 'Failed to load: '+e.message;
  }
}

function _advGetValue(key){
  return key in _enginePending ? _enginePending[key] : _engineConfig[key];
}

function renderAdvancedParams(){
  var panel = $('advancedParamsPanel');
  if(!panel || !_advUnlocked) return;
  var groups = {};
  for(var key in ENGINE_PARAM_META){
    var m = ENGINE_PARAM_META[key];
    if(!groups[m.group]) groups[m.group]=[];
    groups[m.group].push({key:key, meta:m});
  }
  var h = '';
  for(var group in groups){
    var items = groups[group];
    var color = group==='SFD'?'var(--orange)':group==='LTP'?'var(--cyan)':group==='Statistics'?'var(--green)':group==='Serialization'?'var(--purple)':'var(--text-2)';
    h += '<div style="margin-bottom:14px"><div style="font-size:var(--font-desc);font-family:var(--mono);color:'+color+';text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px">'+group+'</div>';
    h += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 16px">';
    items.forEach(function(item){
      var m = item.meta;
      var val = _advGetValue(item.key);
      var def = _engineDefaults[item.key];
      var isPending = item.key in _enginePending;
      var isDefault = String(val) === String(def);
      var id = 'ep_'+item.key;
      h += '<div style="margin-bottom:4px">';
      h += '<label style="font-size:var(--font-desc);color:var(--text-1);font-family:var(--mono)">'+escHtml(m.label);
      if(isPending) h += ' <span style="color:var(--orange);font-size:10px" title="Pending change">●</span>';
      else if(!isDefault) h += ' <span style="color:var(--yellow);font-size:10px" title="Changed from default">◆</span>';
      h += '</label>';
      if(m.type === 'bool'){
        h += '<div class="checkbox-row" style="margin-top:2px"><input type="checkbox" id="'+id+'"'+(val?' checked':'')+' onchange="_advEdit(\''+item.key+'\',this.checked)"><label for="'+id+'">Enable</label></div>';
      } else if(m.type === 'float'){
        h += '<input type="number" id="'+id+'" value="'+val+'" step="'+(m.step||0.01)+'"'+(m.min!=null?' min="'+m.min+'"':'')+(m.max!=null?' max="'+m.max+'"':'')+(isPending?' style="width:100px;font-size:var(--font-table);border-color:var(--orange)"':' style="width:100px;font-size:var(--font-table)"')+' onchange="_advEdit(\''+item.key+'\',parseFloat(this.value))">';
      } else {
        h += '<input type="number" id="'+id+'" value="'+val+'" step="'+(m.step||1)+'"'+(m.min!=null?' min="'+m.min+'"':'')+(m.max!=null?' max="'+m.max+'"':'')+(isPending?' style="width:100px;font-size:var(--font-table);border-color:var(--orange)"':' style="width:100px;font-size:var(--font-table)"')+' onchange="_advEdit(\''+item.key+'\',parseInt(this.value))">';
      }
      h += '<div style="font-size:var(--font-legend);color:var(--text-3);margin-top:1px">'+escHtml(m.desc)+' Default: '+def+'</div>';
      h += '</div>';
    });
    h += '</div></div>';
  }
  panel.innerHTML = h;
}

function _advEdit(key, value){
  // Check if value matches current server value — if so, remove from pending
  if(String(value) === String(_engineConfig[key])){
    delete _enginePending[key];
  } else {
    _enginePending[key] = value;
  }
  renderAdvancedParams();
  _advUpdateApplyBtn();
}

function _advUpdateApplyBtn(){
  var btn = $('advApplyBtn');
  if(!btn) return;
  var n = Object.keys(_enginePending).length;
  btn.disabled = (n === 0);
  btn.textContent = n > 0
    ? 'Apply '+n+' Change'+(n>1?'s':'')+' & Reset Application'
    : 'Apply Changes & Reset Application';
}

async function _advApplyChanges(){
  var n = Object.keys(_enginePending).length;
  if(n === 0) return;
  var msg = 'Apply '+n+' parameter change'+(n>1?'s':'')+'?\n\nThis will:\n• Update engine configuration\n• Unload the model\n• Clear the current session\n• Invalidate all caches\n\nYou will need to reload the model and re-collect data.\n\nChanged parameters:\n';
  for(var k in _enginePending){
    var m = ENGINE_PARAM_META[k];
    msg += '  '+m.label+': '+_engineConfig[k]+' → '+_enginePending[k]+'\n';
  }
  if(!confirm(msg)) return;

  try{
    // 1. Apply config changes
    var r = await(await fetch('/api/engine_config',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(_enginePending)
    })).json();
    if(!r.ok){ log('Config update failed','error'); return; }
    log('Engine config updated: '+Object.keys(_enginePending).join(', '),'done');

    // 2. Reset application
    var resetR = await(await fetch('/api/reset',{method:'POST'})).json();
    if(resetR.ok){
      log('Application reset. Reload model to continue.','done');
      modelLoaded = false;
      sessionResults = [];
      dashResults = [];
      _promptTotal = 0;
      setStatus('idle','IDLE');
      $('analyzeBtn').disabled = true;
      $('batchBtn').disabled = true;
      updateSessionBadge();
    }

    // 3. Refresh panel
    _enginePending = {};
    await loadAdvancedParams();
    _advUpdateApplyBtn();

  }catch(e){
    log('Apply failed: '+e.message,'error');
  }
}

async function resetAdvancedParams(){
  if(!confirm('Reset all advanced parameters to defaults?\n\nThis will reset the application (unload model, clear session).')) return;
  try{
    var r = await(await fetch('/api/engine_config/reset',{method:'POST'})).json();
    if(r.ok){
      _engineConfig = r.config;
      _enginePending = {};
      // Reset application
      await fetch('/api/reset',{method:'POST'});
      modelLoaded = false;
      sessionResults = [];
      dashResults = [];
      _promptTotal = 0;
      setStatus('idle','IDLE');
      $('analyzeBtn').disabled = true;
      $('batchBtn').disabled = true;
      updateSessionBadge();
      renderAdvancedParams();
      _advUpdateApplyBtn();
      log('Advanced parameters and application reset to defaults','done');
    }
  }catch(e){log('Reset failed: '+e.message,'error')}
}

// ─── High-Efficiency Pipeline ──────────────────────────────────

async function loadHepStatus() {
  try {
    var r = await(await fetch('/api/hep/status')).json();
    renderHepPanel(r);
  } catch(e) {
    var status = $('hepStatus');
    var controls = $('hepControls');
    if (status) status.innerHTML = '<span style="color:var(--text-2)">Status: <b>Inactive</b> (standard memory mode)</span>';
    if (controls) controls.innerHTML = '<button class="btn btn-sm" style="border:1px solid var(--yellow);color:var(--yellow);background:transparent;font-weight:600" onclick="showHepConfirm()">Initialize High-Efficiency Pipeline</button>';
  }
}

function renderHepPanel(st) {
  var status = $('hepStatus');
  var controls = $('hepControls');
  if (!status || !controls) return;

  var diskPct = ((st.disk_used / st.disk_total) * 100).toFixed(0);
  var diskFreeGB = (st.disk_free / 1e9).toFixed(1);
  var ramAvailGB = (st.ram_available / 1e9).toFixed(1);
  var hfCacheGB = (st.hf_cache_bytes / 1e9).toFixed(1);

  if (st.active) {
    var mmapInfo = st.mmap_file
      ? 'Mmap file: ' + st.mmap_file.split('/').pop() + ' (' + (st.mmap_size_bytes / 1e9).toFixed(1) + ' GB)'
      : 'No mmap file (load a model to create one)';
    status.innerHTML = '<div style="padding:8px 10px;border:1px solid var(--yellow);border-radius:4px;background:rgba(240,228,66,0.06);margin-bottom:8px">'
      + '<span style="color:var(--yellow);font-weight:600">HIGH-EFFICIENCY PIPELINE: ACTIVE</span><br>'
      + '<span style="color:var(--text-2)">'
      + 'Delta backend: memory-mapped (disk)<br>'
      + mmapInfo + '<br>'
      + 'Disk: ' + diskPct + '% used (' + diskFreeGB + ' GB free)<br>'
      + 'RAM available: ' + ramAvailGB + ' GB'
      + '</span></div>';
    controls.innerHTML = '<button class="btn btn-sm" style="border:1px solid var(--yellow);color:var(--yellow);background:transparent" onclick="deactivateHep()">Deactivate HEP</button>';
  } else {
    status.innerHTML = '<div style="color:var(--text-2)">'
      + 'Status: <b>Inactive</b> (standard memory mode)<br>'
      + 'Disk: ' + diskPct + '% used (' + diskFreeGB + ' GB free)<br>'
      + 'HF model cache: ' + hfCacheGB + ' GB<br>'
      + 'RAM available: ' + ramAvailGB + ' GB'
      + '</div>';
    controls.innerHTML = '<button class="btn btn-sm" style="border:1px solid var(--yellow);color:var(--yellow);background:transparent;font-weight:600" onclick="showHepConfirm()">Initialize High-Efficiency Pipeline</button>';
  }
}

function showHepConfirm() {
  // Fetch fresh status for the dialog
  fetch('/api/hep/status').then(function(r){ return r.json(); }).then(function(st) {
    var hfGB = (st.hf_cache_bytes / 1e9).toFixed(1);
    var freeAfter = ((st.disk_free + st.hf_cache_bytes) / 1e9).toFixed(1);

    var overlay = document.createElement('div');
    overlay.id = 'hepConfirmOverlay';
    overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.6);z-index:9999;display:flex;align-items:center;justify-content:center';

    overlay.innerHTML = '<div style="background:var(--bg-1);border:1px solid var(--yellow);border-radius:8px;padding:24px;max-width:480px;width:90%;color:var(--text-1)">'
      + '<h3 style="margin:0 0 12px;color:var(--yellow)">Initialize High-Efficiency Pipeline</h3>'
      + '<p style="margin:0 0 10px;line-height:1.5;color:var(--text-2)">This mode enables analysis of models up to 3B parameters by using disk-backed storage for weight deltas.</p>'
      + '<p style="margin:0 0 10px;line-height:1.5;color:var(--text-2)">To free disk space, this will:</p>'
      + '<div style="margin:0 0 12px;padding:8px 12px;background:var(--bg-2);border-radius:4px;line-height:1.6;color:var(--text-1)">'
      + '• Clear the HuggingFace model cache (' + hfGB + ' GB)<br>'
      + '• Remove any existing mmap delta files<br>'
      + '• Reset the current pipeline and session'
      + '</div>'
      + '<p style="margin:0 0 10px;line-height:1.5;color:var(--text-2)">All cached model weights will need to be re-downloaded. Probe-dependent modules will require re-embedding (~15-20 min at 3B).</p>'
      + '<div style="margin:0 0 16px;padding:8px 10px;background:var(--bg-2);border-radius:4px;color:var(--text-2)">'
      + 'After cleanup: ~' + freeAfter + ' GB free disk'
      + '</div>'
      + '<div style="display:flex;gap:10px;justify-content:flex-end">'
      + '<button class="btn btn-sm" style="border:1px solid var(--border);color:var(--text-2);background:transparent" onclick="document.getElementById(\'hepConfirmOverlay\').remove()">Cancel</button>'
      + '<button class="btn btn-sm" style="border:1px solid var(--yellow);color:var(--bg-1);background:var(--yellow);font-weight:600" onclick="confirmInitHep()">Confirm</button>'
      + '</div></div>';

    document.body.appendChild(overlay);
  });
}

async function confirmInitHep() {
  var overlay = $('hepConfirmOverlay');
  if (overlay) overlay.remove();

  log('Initializing High-Efficiency Pipeline...');
  try {
    var r = await(await fetch('/api/hep/initialize', {method: 'POST'})).json();
    if (r.ok) {
      modelLoaded = false;
      sessionResults = [];
      dashResults = [];
      _promptTotal = 0;
      setStatus('idle', 'IDLE');
      $('analyzeBtn').disabled = true;
      $('batchBtn').disabled = true;
      updateSessionBadge();
      var freedGB = ((r.hf_freed.bytes_freed + r.mmap_freed.bytes_freed) / 1e9).toFixed(1);
      log('HEP initialized. Freed ' + freedGB + ' GB. Load a model to begin.', 'done');
      loadHepStatus();
    } else {
      log('HEP init failed: ' + (r.error || 'unknown'), 'error');
    }
  } catch(e) {
    log('HEP init error: ' + e.message, 'error');
  }
}

async function deactivateHep() {
  if (!confirm('Deactivate High-Efficiency Pipeline?\n\nThis will reset the pipeline and return to standard memory mode.')) return;
  try {
    var r = await(await fetch('/api/hep/deactivate', {method: 'POST'})).json();
    if (r.ok) {
      modelLoaded = false;
      sessionResults = [];
      setStatus('idle', 'IDLE');
      $('analyzeBtn').disabled = true;
      $('batchBtn').disabled = true;
      log('HEP deactivated. Standard memory mode.', 'done');
      loadHepStatus();
    }
  } catch(e) { log('HEP deactivate error: ' + e.message, 'error'); }
}

var _probeFiles = [];

async function applyProbeSet(){
  var picker = $('probeFilePicker');
  var btn = $('probeApplyBtn');
  var status = $('probeStatus');
  if(!picker.files || picker.files.length === 0){
    status.textContent = 'No file selected.';
    status.style.color = 'var(--red)';
    return;
  }
  if(!modelLoaded){
    status.textContent = 'Load a model first.';
    status.style.color = 'var(--red)';
    return;
  }
  btn.disabled = true;
  btn.textContent = 'Applying...';
  status.textContent = 'Uploading...';
  status.style.color = 'var(--text-2)';
  try {
    var fd = new FormData();
    fd.append('file', picker.files[0]);
    var r = await(await fetch('/api/probe_set/apply', {method:'POST', body:fd})).json();
    if(!r.ok){
      status.textContent = 'Error: ' + (r.error||'Unknown');
      status.style.color = 'var(--red)';
      btn.disabled = false;
      btn.textContent = 'Apply';
      return;
    }
    status.textContent = 'Embedding probes...';
    // Wait for probe_status via SSE
    var probeResult = await new Promise(function(resolve) {
      _sseWaiters.probe_status.push(function(evt) { resolve(evt); });
    });
    btn.disabled = false;
    btn.textContent = 'Apply';
    if (probeResult.error) {
      status.textContent = 'Error: ' + probeResult.error;
      status.style.color = 'var(--red)';
    } else if (probeResult.result) {
      var res = probeResult.result;
      status.innerHTML = '<span style="color:var(--green)">✓ ' + escHtml(res.filename) + '</span>'
        + ' — ' + res.n_probes + ' probes, ' + res.n_subjects + ' subjects, ' + res.n_levels + ' subclasses'
        + ' — embedded at L' + res.layer_L50 + ' and L' + res.layer_L75;
      status.style.color = 'var(--text-1)';
      log('Probe set applied: ' + res.filename + ' (' + res.n_probes + ' probes)', 'done');
      playChime();
    }
  } catch(e) {
    status.textContent = 'Failed: ' + e.message;
    status.style.color = 'var(--red)';
    btn.disabled = false;
    btn.textContent = 'Apply';
  }
}

async function loadProbeFiles(){
  // Show current active probe if any
  var status = $('probeStatus');
  if(!status) return;
  try {
    var r = await(await fetch('/api/probe_set/status')).json();
    if(r.ok && r.active){
      // Top line: filename + probe count + cache presence.
      var h = '<span style="color:var(--green)">✓ ' + escHtml(r.filename) + '</span>'
            + ' — ' + r.n_probes + ' probes, ' + r.n_subjects + ' subjects'
            + (r.cached ? ' — <span style="color:var(--green)">cached</span>'
                        : ' — <span style="color:var(--text-3)">not cached</span>');

      // Second line: the binding — which model this set was applied for,
      // and whether it matches the currently loaded model. This is the
      // user-visible signal that "Apply" recorded a model-bound cache.
      if (r.legacy) {
        h += '<br><span style="color:var(--orange);font-size:11px">'
           + '⚠ Legacy probe_config.json record (no model recorded). '
           + 'Re-Apply to bind this probe set to the current model.</span>';
      } else if (r.model_id) {
        var depthsStr = (r.depths || [])
          .map(function(d){ return 'L' + Math.round(d * 100); }).join(', ');
        var bindLine = '<span style="color:var(--text-2);font-size:11px">'
          + 'Applied for <span style="color:var(--text-1)">'
          + escHtml(r.model_id) + '</span>'
          + (depthsStr ? ' at ' + depthsStr : '')
          + (r.projected ? ' (projected)' : '')
          + '</span>';

        if (r.stale_for_loaded_model && r.loaded_model_id) {
          bindLine = '<span style="color:var(--orange);font-size:11px">'
            + '⚠ Applied for <span style="color:var(--text-1)">'
            + escHtml(r.model_id) + '</span>, but '
            + '<span style="color:var(--text-1)">'
            + escHtml(r.loaded_model_id) + '</span> is currently loaded. '
            + 'Re-Apply to embed for the current model.</span>';
        }
        h += '<br>' + bindLine;
      }

      status.innerHTML = h;
    } else {
      status.innerHTML = '<span style="color:var(--text-3)">No probe set active.</span>';
    }
  } catch(e) {}
}

async function clearProbeCaches(){
  var status = $('probeStatus');
  try {
    var r = await(await fetch('/api/probe_set/clear_caches', {method:'POST'})).json();
    if(r.ok){
      status.textContent = 'Cleared ' + r.deleted + ' cache file(s).';
      status.style.color = 'var(--text-2)';
      log('Probe caches cleared: ' + r.deleted + ' files', 'done');
    } else {
      status.textContent = 'Error: ' + (r.error||'Unknown');
      status.style.color = 'var(--red)';
    }
  } catch(e) {
    status.textContent = 'Failed: ' + e.message;
    status.style.color = 'var(--red)';
  }
}

// ─── Notification Chime ─────────────────────────────────────────
function playChime() {
  if (!$('chimeToggle') || !$('chimeToggle').checked) return;
  try {
    var ctx = new (window.AudioContext || window.webkitAudioContext)();
    var now = ctx.currentTime;

    function tone(freq, start, dur, gain) {
      var osc = ctx.createOscillator();
      var g = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = freq;
      g.gain.setValueAtTime(0, now + start);
      g.gain.linearRampToValueAtTime(gain, now + start + 0.02);
      g.gain.exponentialRampToValueAtTime(0.001, now + start + dur);
      osc.connect(g);
      g.connect(ctx.destination);
      osc.start(now + start);
      osc.stop(now + start + dur);
    }

    // Two-note major third: E5 → G5, soft and brief
    tone(659, 0.0, 0.4, 0.12);
    tone(784, 0.15, 0.5, 0.10);

    setTimeout(function() { ctx.close(); }, 1000);
  } catch(e) {}
}

// ─── Init ───────────────────────────────────────────────────────
(async function(){
  try{
    // Initialize SSE event stream before anything else.
    initEventSource();

    loadModelList();loadPromptLibrary();await loadConfig();vizRenderConfig();loadAdvancedParams();loadProbeFiles();loadHepStatus();
    if($('chimeToggle'))$('chimeToggle').checked=localStorage.getItem('tagm_chime')==='1';
    const st=await(await fetch('/api/status')).json();
    if(st.user_info){$('userName').value=st.user_info.name||'';$('userOrg').value=st.user_info.organization||'';$('userProject').value=st.user_info.project||''}
    if(st.model_loaded){modelLoaded=true;setStatus('ready','READY');$('analyzeBtn').disabled=false;$('batchBtn').disabled=false}
    else if(st.loading){
      // Load in progress (page refreshed mid-load). SSE will deliver
      // progress events and eventually model_loaded or model_error.
      setStatus('loading','LOADING');
      $('loadModelBtn').disabled=true;
      $('loadModelBtn').textContent='Processing...';
      _sseWaiters.model_loaded.push(function() {
        $('loadModelBtn').disabled=false;
        $('loadModelBtn').textContent='Load Model';
        loadPromptLibrary();
        playChime();
      });
      _sseWaiters.model_error.push(function() {
        $('loadModelBtn').disabled=false;
        $('loadModelBtn').textContent='Load Model';
      });
    }
    // Restored session: populate data table and enable dashboard
    if(st.session&&st.session.n_results>0){
      _promptTotal=st.session.n_results;
      if(st.session.cache_size_bytes)_cacheBytes=st.session.cache_size_bytes;
      updateSessionBadge();
      renderDataTable();
      $('dashBtn').disabled=false;
      $('exportBtn').disabled=false;
      log('Session restored: '+st.session.n_results+' results'+(st.session.model?' ('+st.session.model+')':''),'done');
    }
  }catch(e){}
})();

// ─── Candidate Graph Topology ──────────────────────────────────
function computeCandidateGraph(r){
  const tokens=r.tokens||[];const instCf=(r.ltp&&r.ltp.counterfactual_tokens)||[];const baseCf=r.base_counterfactual_tokens||[];
  const nPos=Math.min(instCf.length,baseCf.length,tokens.length);if(nPos===0)return null;
  const candidates={};const posStats=[];
  function ec(t){if(!candidates[t])candidates[t]={inst:[],base:[],promoted:[],demoted:[],matched:[]};return candidates[t]}
  for(let pos=0;pos<nPos;pos++){
    const iAlts=instCf[pos]||[];const bAlts=baseCf[pos]||[];const instSet={};const baseSet={};
    for(let j=0;j<iAlts.length;j++){const t=Array.isArray(iAlts[j])?iAlts[j][0]:iAlts[j];const p=Array.isArray(iAlts[j])?iAlts[j][1]:0;instSet[t]={rank:j,prob:p}}
    for(let j=0;j<bAlts.length;j++){const t=Array.isArray(bAlts[j])?bAlts[j][0]:bAlts[j];const p=Array.isArray(bAlts[j])?bAlts[j][1]:0;baseSet[t]={rank:j,prob:p}}
    let nP=0,nD=0,nM=0;const prom=[],demo=[],match=[];
    for(const t in instSet){const c=ec(t);c.inst.push({pos,rank:instSet[t].rank,prob:instSet[t].prob});if(baseSet[t]!==undefined){c.matched.push(pos);match.push(t);nM++}else{c.promoted.push(pos);prom.push(t);nP++}}
    for(const t in baseSet){const c=ec(t);c.base.push({pos,rank:baseSet[t].rank,prob:baseSet[t].prob});if(instSet[t]===undefined){c.demoted.push(pos);demo.push(t);nD++}}
    posStats.push({pos,token:tokens[pos],nPromoted:nP,nDemoted:nD,nMatched:nM,contested:nP>0&&nD>0,intensity:nP+nD,promoted:prom,demoted:demo,matched:match})
  }
  const dualRole=[];for(const t in candidates){const c=candidates[t];if(c.promoted.length>0&&c.demoted.length>0)dualRole.push({token:t,promotedAt:c.promoted.map(p=>({pos:p,tok:tokens[p]})),demotedAt:c.demoted.map(p=>({pos:p,tok:tokens[p]})),matchedAt:c.matched.map(p=>({pos:p,tok:tokens[p]}))})}
  const roleSwitches=[];for(const t in candidates){const c=candidates[t];const ap=[...new Set([...c.inst.map(x=>x.pos),...c.base.map(x=>x.pos)])].sort((a,b)=>a-b);if(ap.length<2)continue;for(let i=0;i<ap.length-1;i++){if(ap[i+1]-ap[i]>2)continue;const p1=ap[i],p2=ap[i+1];const r1=c.promoted.includes(p1)?'promoted':(c.demoted.includes(p1)?'demoted':'matched');const r2=c.promoted.includes(p2)?'promoted':(c.demoted.includes(p2)?'demoted':'matched');if(r1!==r2)roleSwitches.push({candidate:t,fromPos:p1,fromTok:tokens[p1]||'?',fromRole:r1,toPos:p2,toTok:tokens[p2]||'?',toRole:r2})}}
  const trajectories=[];for(const t in candidates){const c=candidates[t];const aps=new Set([...c.inst.map(x=>x.pos),...c.base.map(x=>x.pos)]);if(aps.size<2)continue;const im={};c.inst.forEach(x=>im[x.pos]=x);const bm={};c.base.forEach(x=>bm[x.pos]=x);const positions=[...aps].sort((a,b)=>a-b).map(pos=>({pos,tok:tokens[pos]||'?',role:c.promoted.includes(pos)?'promoted':(c.demoted.includes(pos)?'demoted':'matched'),inst:im[pos]||null,base:bm[pos]||null}));trajectories.push({candidate:t,nAppearances:aps.size,positions,isDualRole:c.promoted.length>0&&c.demoted.length>0})}
  trajectories.sort((a,b)=>b.nAppearances-a.nAppearances);
  const nContested=posStats.filter(p=>p.contested).length;
  return{nPositions:nPos,tokens,posStats,candidates,dualRole,roleSwitches,trajectories:trajectories.slice(0,20),nContested,contestedFrac:nContested/nPos,nDualRole:dualRole.length,nRoleSwitches:roleSwitches.length,switchRate:roleSwitches.length/nPos,nUniqueCandidates:Object.keys(candidates).length,nMultiPosition:trajectories.length}
}
function renderCandidateGraphSection(r){
  const g=computeCandidateGraph(r);if(!g)return'';
  const uid='cg_'+Date.now()+'_'+Math.random().toString(36).substr(2,5);
  let mapH='<div style="font-family:var(--mono);font-size:12px;letter-spacing:1px;line-height:1.8;margin:8px 0">';
  for(let i=0;i<g.posStats.length;i++){const ps=g.posStats[i];let ch,col,title;if(!ps.contested){ch='·';col='var(--text-3)';title=ps.token+': stable ('+ps.nMatched+' matched)'}else if(ps.intensity>=6){ch='▓';col='var(--red)';title=ps.token+': heavy (+'+ps.nPromoted+' -'+ps.nDemoted+')'}else if(ps.intensity>=3){ch='▒';col='var(--orange)';title=ps.token+': moderate (+'+ps.nPromoted+' -'+ps.nDemoted+')'}else{ch='░';col='#e3b341';title=ps.token+': light (+'+ps.nPromoted+' -'+ps.nDemoted+')'}mapH+='<span style="color:'+col+';cursor:help" title="'+escHtml(title)+'">'+ch+'</span>'}
  mapH+='</div><div style="display:flex;flex-wrap:wrap;gap:2px;margin-bottom:12px">';
  for(let i=0;i<g.tokens.length;i++){const ps=g.posStats[i];const bg=ps.contested?(ps.intensity>=6?'rgba(248,81,73,0.15)':(ps.intensity>=3?'rgba(210,153,34,0.15)':'rgba(227,179,65,0.08)')):'transparent';mapH+='<span style="font-family:var(--mono);font-size:12px;padding:2px 5px;background:'+bg+';border-radius:3px;color:var(--text-0)">'+escHtml(g.tokens[i])+'</span>'}
  mapH+='</div>';
  const cp=(g.contestedFrac*100).toFixed(0),bw=Math.round(g.contestedFrac*100),bc=g.contestedFrac>0.85?'var(--red)':(g.contestedFrac>0.7?'var(--orange)':'var(--green)');
  let mH=`<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;margin-bottom:14px"><div><div style="font-size:11px;color:var(--text-2);margin-bottom:3px">CONTESTED</div><div style="font-family:var(--mono);font-size:16px;font-weight:600;color:${bc}">${cp}%</div><div style="width:100%;height:3px;background:var(--bg-3);border-radius:2px;margin-top:3px"><div style="width:${bw}%;height:100%;background:${bc};border-radius:2px"></div></div></div><div><div style="font-size:11px;color:var(--text-2);margin-bottom:3px">DUAL-ROLE</div><div style="font-family:var(--mono);font-size:16px;font-weight:600;color:var(--text-0)">${g.nDualRole}</div><div style="font-size:11px;color:var(--text-2)">candidates</div></div><div><div style="font-size:11px;color:var(--text-2);margin-bottom:3px">SWITCHES</div><div style="font-family:var(--mono);font-size:16px;font-weight:600;color:var(--text-0)">${g.nRoleSwitches}</div><div style="font-size:11px;color:var(--text-2)">${g.switchRate.toFixed(2)}/token</div></div><div><div style="font-size:11px;color:var(--text-2);margin-bottom:3px">TRAJECTORIES</div><div style="font-family:var(--mono);font-size:16px;font-weight:600;color:var(--text-0)">${g.nMultiPosition}</div><div style="font-size:11px;color:var(--text-2)">of ${g.nUniqueCandidates} unique</div></div></div>`;
  let ptH='<div style="max-height:200px;overflow-y:auto;margin-bottom:14px"><table class="data-table"><tr><th>Pos</th><th>Token</th><th>Match</th><th>+Prom</th><th>-Demo</th><th>Status</th></tr>';
  for(const ps of g.posStats){const st=ps.contested?'<span style="color:var(--red)">contested</span>':'<span style="color:var(--green)">stable</span>';ptH+=`<tr><td>${ps.pos}</td><td>${escHtml(ps.token)}</td><td>${ps.nMatched}</td><td>${ps.nPromoted}</td><td>${ps.nDemoted}</td><td>${st}</td></tr>`}
  ptH+='</table></div>';
  let dH='';if(g.dualRole.length>0){dH='<div style="margin-bottom:14px"><div style="font-size:12px;font-weight:600;color:var(--purple);margin-bottom:8px">DUAL-ROLE CANDIDATES</div>';for(const dr of g.dualRole){if(dr.token.trim()==='')continue;const pc=dr.promotedAt.map(x=>'<span style="color:var(--green)">pos '+x.pos+'</span> <span style="color:var(--text-3)">('+escHtml(x.tok)+')</span>').join(', ');const dc=dr.demotedAt.map(x=>'<span style="color:var(--red)">pos '+x.pos+'</span> <span style="color:var(--text-3)">('+escHtml(x.tok)+')</span>').join(', ');dH+=`<div style="margin-bottom:8px;padding:6px 8px;background:var(--bg-2);border-radius:4px;border-left:3px solid var(--purple)"><div style="font-family:var(--mono);font-size:11px;font-weight:600;color:var(--text-0)">"${escHtml(dr.token)}"</div><div style="font-size:11px;margin-top:4px">↑ promoted: ${pc}</div><div style="font-size:11px;margin-top:3px">↓ demoted: ${dc}</div>${dr.matchedAt.length?'<div style="font-size:11px;margin-top:3px;color:var(--text-3)">= matched: '+dr.matchedAt.length+' positions</div>':''}</div>`}dH+='</div>'}
  let sH='';if(g.roleSwitches.length>0){sH='<div style="margin-bottom:14px"><div style="font-size:12px;font-weight:600;color:var(--orange);margin-bottom:8px">ROLE SWITCHES</div><div style="max-height:180px;overflow-y:auto"><table class="data-table"><tr><th>Candidate</th><th>From</th><th>→</th><th>To</th></tr>';for(const sw of g.roleSwitches.slice(0,20)){const fc=sw.fromRole==='promoted'?'var(--green)':(sw.fromRole==='demoted'?'var(--red)':'var(--text-2)');const tc=sw.toRole==='promoted'?'var(--green)':(sw.toRole==='demoted'?'var(--red)':'var(--text-2)');sH+=`<tr><td style="font-family:var(--mono)">"${escHtml(sw.candidate)}"</td><td><span style="color:${fc}">${sw.fromRole}</span> <span style="color:var(--text-3)">at ${sw.fromPos} (${escHtml(sw.fromTok)})</span></td><td style="color:var(--text-3)">→</td><td><span style="color:${tc}">${sw.toRole}</span> <span style="color:var(--text-3)">at ${sw.toPos} (${escHtml(sw.toTok)})</span></td></tr>`}sH+='</table></div></div>'}
  let tH='';const topT=g.trajectories.filter(t=>t.nAppearances>=2).slice(0,8);if(topT.length>0){tH='<div style="margin-bottom:14px"><div style="font-size:12px;font-weight:600;color:var(--cyan);margin-bottom:8px">CANDIDATE TRAJECTORIES</div>';for(const tr of topT){const lb=tr.isDualRole?' <span style="color:var(--purple);font-size:11px;font-weight:600">DUAL</span>':'';tH+=`<div style="margin-bottom:8px;padding:6px 8px;background:var(--bg-2);border-radius:4px"><div style="font-family:var(--mono);font-size:12px;font-weight:600;color:var(--text-0)">"${escHtml(tr.candidate)}" <span style="color:var(--text-3);font-weight:400">${tr.nAppearances} pos</span>${lb}</div><div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:4px">`;for(const p of tr.positions){const rc=p.role==='promoted'?'var(--green)':(p.role==='demoted'?'var(--red)':'var(--text-2)');const rs=p.role==='promoted'?'↑':(p.role==='demoted'?'↓':'=');const ip=p.inst?(p.inst.prob*100).toFixed(1)+'%':'—';const bp=p.base?(p.base.prob*100).toFixed(1)+'%':'—';tH+=`<span style="font-size:11px;padding:3px 6px;background:var(--bg-1);border-radius:3px;border:1px solid var(--border)" title="inst:${ip} base:${bp}"><span style="color:${rc}">${rs}</span> <span style="color:var(--text-1)">${p.pos}:${escHtml(p.tok)}</span></span>`}tH+='</div></div>'}tH+='</div>'}
  return`<div style="margin-top:16px;padding-top:14px;border-top:2px solid var(--purple)"><div style="display:flex;align-items:center;gap:8px;margin-bottom:12px"><div style="font-family:var(--mono);font-size:14px;font-weight:600;color:var(--purple)">Candidate Graph</div><button class="btn-popout" onclick="popoutCandidateGraph('${uid}')" title="Full graph in new window">↗ Pop out</button></div><div id="${uid}">${mapH}${mH}<div class="feature" style="border-color:var(--purple)"><div class="feature-header" onclick="toggleFeature(this)"><div class="feature-title" style="color:var(--purple)">Position Detail</div><p class="feature-desc">Per-token contested status. Contested = both models swapping candidates at this position.</p></div><div class="feature-body collapsed" style="padding:10px">${ptH}</div></div>${dH}${sH}${tH}</div></div>`
}
function popoutCandidateGraph(uid){
  const el=document.getElementById(uid);if(!el)return;
  const pw=1000,ph=850,left=Math.round((screen.width-pw)/2),top=Math.round((screen.height-ph)/2);
  const w=window.open('','_blank',`width=${pw},height=${ph},left=${left},top=${top},scrollbars=yes`);if(!w)return;
  const styles=Array.from(document.querySelectorAll('style,link[rel="stylesheet"]')).map(s=>s.outerHTML).join('\n');
  w.document.write(`<!DOCTYPE html><html><head><title>TAGM — Candidate Graph</title><link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">${styles}<style>body{background:var(--bg-0);color:var(--text-1);font-family:var(--sans);padding:20px;margin:0}.feature-body.collapsed{display:block!important}</style></head><body><div style="max-width:960px;margin:0 auto">${el.outerHTML}</div></body></html>`);
  w.document.close();
}

function _computeGraphsChunked(results){
  // Build graph objects from server pre-computed candidate_graph summaries
  var graphs=[];
  for(var i=0;i<results.length;i++){
    var r=results[i];
    var cg=r.candidate_graph;
    if(!cg)continue;
    cg.category=r.category||'unknown';
    cg.prompt=r.prompt||'';
    cg.contestedFrac=cg.contested_frac||0;
    cg.nDualRole=cg.n_dual_role||0;
    cg.nRoleSwitches=cg.n_role_switches||0;
    cg.switchRate=cg.switch_rate||0;
    cg.nMultiPosition=cg.n_multi_position||0;
    cg.nUniqueCandidates=cg.n_unique_candidates||0;
    cg.nPositions=cg.n_positions||0;
    cg.posStats=[];
    var cmap=cg.contest_map||'';
    for(var j=0;j<cmap.length;j++){
      var ch=cmap[j];
      cg.posStats.push({contested:ch!=='.',intensity:ch==='H'?6:(ch==='M'?3:1)});
    }
    graphs.push(cg);
  }
  return _buildCandidateGraphAggregateFromGraphs(graphs);
}

function _buildCandidateGraphAggregateFromGraphs(graphs){
  if(graphs.length<2)return'';

  // Group by category
  var catOrder=['benign','dual-use','mild','harmful','adversarial','jailbreak'];
  var catGraphs={};catOrder.forEach(function(c){catGraphs[c]=[]});
  graphs.forEach(function(g){if(catGraphs[g.category])catGraphs[g.category].push(g);else{if(!catGraphs['unknown'])catGraphs['unknown']=[];catGraphs['unknown'].push(g)}});

  // Helper: mean and std
  function stats(arr){
    if(!arr.length)return{mean:0,std:0,n:0};
    var m=arr.reduce(function(a,v){return a+v},0)/arr.length;
    var v=arr.reduce(function(a,v){return a+(v-m)*(v-m)},0)/arr.length;
    return{mean:m,std:Math.sqrt(v),n:arr.length};
  }

  // Cohen's d
  function cohend(a,b){
    var sa=stats(a),sb=stats(b);
    if(sa.n<2||sb.n<2)return 0;
    var pooled=Math.sqrt((sa.std*sa.std+sb.std*sb.std)/2);
    return pooled>0?Math.abs(sa.mean-sb.mean)/pooled:0;
  }

  // ── Category summary table ──
  var tblS='';
  tblS+='<table style="width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11px">';
  tblS+='<tr style="border-bottom:1px solid rgba(255,255,255,0.12)">';
  tblS+='<th style="text-align:left;padding:8px 10px;color:#e6edf3">Category</th>';
  tblS+='<th style="text-align:right;padding:8px 6px;color:#e6edf3">N</th>';
  tblS+='<th style="text-align:right;padding:8px 6px;color:#e6edf3">Contested %</th>';
  tblS+='<th style="text-align:right;padding:8px 6px;color:#e6edf3">Dual-Role</th>';
  tblS+='<th style="text-align:right;padding:8px 6px;color:#e6edf3">Switches</th>';
  tblS+='<th style="text-align:right;padding:8px 6px;color:#e6edf3">Sw/token</th>';
  tblS+='<th style="text-align:right;padding:8px 6px;color:#e6edf3">Unique Cands</th>';
  tblS+='</tr>';

  var catColors={benign:'#3fb950','dual-use':'#009E73',mild:'#58a6ff',harmful:'#d29a22',adversarial:'#882255',jailbreak:'#f85149'};

  catOrder.forEach(function(cat){
    var gs=catGraphs[cat];if(!gs.length)return;
    var cf=stats(gs.map(function(g){return g.contestedFrac}));
    var dr=stats(gs.map(function(g){return g.nDualRole}));
    var sw=stats(gs.map(function(g){return g.nRoleSwitches}));
    var sr=stats(gs.map(function(g){return g.switchRate}));
    var uc=stats(gs.map(function(g){return g.nUniqueCandidates}));
    var cc=catColors[cat]||'#b1bac4';

    tblS+='<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">';
    tblS+='<td style="padding:6px 10px"><span style="color:'+cc+';font-weight:600">'+cat+'</span></td>';
    tblS+='<td style="text-align:right;padding:6px;color:#b1bac4">'+gs.length+'</td>';
    tblS+='<td style="text-align:right;padding:6px;color:#e6edf3;font-weight:600">'+(cf.mean*100).toFixed(1)+'%<span style="color:var(--text-1);font-weight:400"> ±'+(cf.std*100).toFixed(1)+'</span></td>';
    tblS+='<td style="text-align:right;padding:6px;color:#e6edf3">'+dr.mean.toFixed(1)+'<span style="color:var(--text-1)"> ±'+dr.std.toFixed(1)+'</span></td>';
    tblS+='<td style="text-align:right;padding:6px;color:#e6edf3">'+sw.mean.toFixed(1)+'<span style="color:var(--text-1)"> ±'+sw.std.toFixed(1)+'</span></td>';
    tblS+='<td style="text-align:right;padding:6px;color:#e6edf3">'+sr.mean.toFixed(3)+'</td>';
    tblS+='<td style="text-align:right;padding:6px;color:#b1bac4">'+uc.mean.toFixed(0)+'</td>';
    tblS+='</tr>';
  });
  tblS+='</table>';

  // ── Effect sizes ──
  var benignGraphs=catGraphs['benign']||[];
  var jailGraphs=catGraphs['jailbreak']||[];
  var effH='';
  if(benignGraphs.length>=2&&jailGraphs.length>=2){
    var metrics=[
      {name:'Contested %',fn:function(g){return g.contestedFrac}},
      {name:'Switch rate',fn:function(g){return g.switchRate}},
      {name:'Dual-role count',fn:function(g){return g.nDualRole}},
      {name:'Role switches',fn:function(g){return g.nRoleSwitches}},
      {name:'Multi-pos cands',fn:function(g){return g.nMultiPosition}},
    ];
    var effRows='';
    metrics.forEach(function(m){
      var bVals=benignGraphs.map(m.fn),jVals=jailGraphs.map(m.fn);
      var d=cohend(bVals,jVals);
      var bm=stats(bVals),jm=stats(jVals);
      var dir=jm.mean>bm.mean?'↑':'↓';
      var barW=Math.min(d/3*100,100);
      var barCol=d>1.5?'#3fb950':(d>0.8?'#58a6ff':(d>0.5?'#d29a22':'#8b949e'));

      effRows+='<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">';
      effRows+='<td style="padding:5px 10px;color:#e6edf3">'+m.name+'</td>';
      effRows+='<td style="padding:5px 6px;text-align:right;color:#b1bac4">'+bm.mean.toFixed(3)+'</td>';
      effRows+='<td style="padding:5px 6px;text-align:right;color:#b1bac4">'+jm.mean.toFixed(3)+'</td>';
      effRows+='<td style="padding:5px 6px;text-align:right;font-weight:600;color:'+barCol+'">'+d.toFixed(2)+' '+dir+'</td>';
      effRows+='<td style="padding:5px 6px;width:120px"><div style="width:100%;height:8px;background:rgba(255,255,255,0.06);border-radius:4px"><div style="width:'+barW+'%;height:100%;background:'+barCol+';border-radius:4px"></div></div></td>';
      effRows+='</tr>';
    });

    effH='<div style="margin-top:16px"><div style="font-size:11px;font-weight:600;color:#e6edf3;margin-bottom:8px">Separation Power (benign vs jailbreak)</div>';
    effH+='<table style="width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11px">';
    effH+='<tr style="border-bottom:1px solid rgba(255,255,255,0.12)"><th style="text-align:left;padding:5px 10px;color:var(--text-1)">Metric</th><th style="text-align:right;padding:5px 6px;color:#3fb950">Benign</th><th style="text-align:right;padding:5px 6px;color:#f85149">Jailbreak</th><th style="text-align:right;padding:5px 6px;color:var(--text-1)">Cohen\'s d</th><th style="padding:5px 6px;color:var(--text-1)"></th></tr>';
    effH+=effRows+'</table></div>';
  }

  // ── Cross-prompt candidate frequency (requires raw data — skip for server summaries) ──
  var hasRawData=graphs.some(function(g){return g.candidates&&typeof g.candidates==='object'&&Object.keys(g.candidates).length>0});
  var candH='';
  if(hasRawData){
    var candFreq={};
    graphs.forEach(function(g){
      if(!g.candidates)return;
      for(var t in g.candidates){
        var c=g.candidates[t];
        if(!candFreq[t])candFreq[t]={promoted:0,demoted:0,matched:0,dual:0,prompts:0,categories:{}};
        var cf=candFreq[t];
        cf.prompts++;
        cf.promoted+=c.promoted.length;
        cf.demoted+=c.demoted.length;
        cf.matched+=c.matched.length;
        if(c.promoted.length>0&&c.demoted.length>0)cf.dual++;
        cf.categories[g.category]=(cf.categories[g.category]||0)+1;
      }
    });

  // Most contested: highest (promoted+demoted) / total appearances
  var candList=[];
  for(var t in candFreq){
    var cf=candFreq[t];
    if(cf.prompts<2)continue;// skip single-prompt
    var contestRatio=(cf.promoted+cf.demoted)/(cf.promoted+cf.demoted+cf.matched+0.001);
    candList.push({token:t,label:t.trim()||'⎵',freq:cf,contestRatio:contestRatio});
  }
  candList.sort(function(a,b){return b.freq.dual-a.freq.dual||(b.contestRatio-a.contestRatio)});

  var candH='';
  if(candList.length>0){
    candH='<div style="margin-top:16px"><div style="font-size:11px;font-weight:600;color:#e6edf3;margin-bottom:8px">Most Contested Candidates (across all prompts)</div>';
    candH+='<table style="width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11px">';
    candH+='<tr style="border-bottom:1px solid rgba(255,255,255,0.12)"><th style="text-align:left;padding:5px 10px;color:var(--text-1)">Token</th><th style="text-align:right;padding:5px 6px;color:var(--text-1)">Prompts</th><th style="text-align:right;padding:5px 6px;color:#3fb950">+Prom</th><th style="text-align:right;padding:5px 6px;color:#f85149">-Demo</th><th style="text-align:right;padding:5px 6px;color:#58a6ff">=Match</th><th style="text-align:right;padding:5px 6px;color:#bc8cff">Dual</th><th style="text-align:left;padding:5px 6px;color:var(--text-1)">Categories</th></tr>';
    var shown=0;
    for(var ci=0;ci<candList.length&&shown<15;ci++){
      var cl=candList[ci];
      if(cl.label==='⎵')continue;// skip EOS
      var catStr=catOrder.filter(function(c){return cl.freq.categories[c]}).map(function(c){
        return '<span style="color:'+catColors[c]+'">'+c.substr(0,3)+':'+cl.freq.categories[c]+'</span>';
      }).join(' ');
      candH+='<tr style="border-bottom:1px solid rgba(255,255,255,0.04)">';
      candH+='<td style="padding:4px 10px;color:#e6edf3;font-weight:600">"'+escHtml(cl.label)+'"</td>';
      candH+='<td style="text-align:right;padding:4px 6px;color:#b1bac4">'+cl.freq.prompts+'</td>';
      candH+='<td style="text-align:right;padding:4px 6px;color:#3fb950">'+cl.freq.promoted+'</td>';
      candH+='<td style="text-align:right;padding:4px 6px;color:#f85149">'+cl.freq.demoted+'</td>';
      candH+='<td style="text-align:right;padding:4px 6px;color:#58a6ff">'+cl.freq.matched+'</td>';
      candH+='<td style="text-align:right;padding:4px 6px;color:'+(cl.freq.dual>0?'#bc8cff':'#8b949e')+'">'+cl.freq.dual+'</td>';
      candH+='<td style="padding:5px 8px;font-size:11px">'+catStr+'</td>';
      candH+='</tr>';
      shown++;
    }
    candH+='</table></div>';
  }
  } // end hasRawData

  // ── Per-prompt breakdown ──
  var promptH='<div style="margin-top:16px"><div style="font-size:11px;font-weight:600;color:#e6edf3;margin-bottom:8px">Per-Prompt Candidate Graph Metrics</div>';
  promptH+='<div style="max-height:300px;overflow-y:auto">';
  promptH+='<table style="width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11px">';
  promptH+='<tr style="border-bottom:1px solid rgba(255,255,255,0.12);position:sticky;top:0;background:var(--bg-1)"><th style="text-align:left;padding:4px 8px;color:var(--text-1)">Prompt</th><th style="text-align:left;padding:4px 4px;color:var(--text-1)">Cat</th><th style="text-align:right;padding:4px 4px;color:var(--text-1)">Contest</th><th style="text-align:right;padding:4px 4px;color:var(--text-1)">Dual</th><th style="text-align:right;padding:4px 4px;color:var(--text-1)">Sw/tok</th><th style="padding:4px 4px;color:var(--text-1);width:80px">Map</th></tr>';

  // Sort by contested fraction descending
  var sorted=graphs.slice().sort(function(a,b){return b.contestedFrac-a.contestedFrac});
  sorted.forEach(function(g){
    var cc=catColors[g.category]||'#b1bac4';
    var cfCol=g.contestedFrac>0.85?'#f85149':(g.contestedFrac>0.7?'#d29a22':'#3fb950');
    // Build mini contestation map
    var miniMap='';
    for(var i=0;i<g.posStats.length;i++){
      var ps=g.posStats[i];
      if(!ps.contested)miniMap+='<span style="color:#333">·</span>';
      else if(ps.intensity>=6)miniMap+='<span style="color:#f85149">▓</span>';
      else if(ps.intensity>=3)miniMap+='<span style="color:#d29a22">▒</span>';
      else miniMap+='<span style="color:var(--text-1)">░</span>';
    }
    promptH+='<tr style="border-bottom:1px solid rgba(255,255,255,0.03)">';
    promptH+='<td style="padding:3px 8px;color:#b1bac4;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="'+escHtml(g.prompt)+'">'+escHtml(g.prompt.substr(0,40))+'</td>';
    promptH+='<td style="padding:3px 4px;color:'+cc+'">'+g.category.substr(0,4)+'</td>';
    promptH+='<td style="text-align:right;padding:3px 4px;color:'+cfCol+';font-weight:600">'+(g.contestedFrac*100).toFixed(0)+'%</td>';
    promptH+='<td style="text-align:right;padding:3px 4px;color:#b1bac4">'+g.nDualRole+'</td>';
    promptH+='<td style="text-align:right;padding:3px 4px;color:#b1bac4">'+g.switchRate.toFixed(2)+'</td>';
    promptH+='<td style="padding:3px 4px;font-family:var(--mono);font-size:11px;letter-spacing:0.5px">'+miniMap+'</td>';
    promptH+='</tr>';
  });
  promptH+='</table></div></div>';

  // ── Assemble ──
  var h='<div class="feature" style="border-color:var(--purple);margin-top:16px">';
  h+='<div class="feature-header" onclick="toggleFeature(this)">';
  h+='<div class="feature-title" style="color:#bc8cff">Candidate Graph Topology</div>';
  h+='<p class="feature-desc" style="color:#b1bac4">Cross-prompt analysis of alignment decision stability. Contested = both models swapping candidates at that position. Dual-role = same token promoted at some positions, demoted at others. Switch rate = role changes per token.</p>';
  h+='</div>';
  h+='<div class="feature-body" style="padding:16px">';
  h+=tblS+effH+candH+promptH;
  h+='</div></div>';
  return h;
}

// ─── Module Framework ──────────────────────────────────────────

var _moduleData = {};  // cached module metadata
var _modulePollers = {};  // active polling intervals
var _moduleStartTimes = {};  // wall-clock start time per module (fallback for elapsed)

async function loadModules() {
  try {
    const r = await fetch('/api/modules');
    const d = await r.json();
    if (!d.ok) { $('modulesContainer').innerHTML = '<div class="error-msg">Failed to load modules</div>'; return; }
    $('modulesLoading').style.display = 'none';
    renderModules(d.modules);
  } catch(e) {
    $('modulesLoading').textContent = 'Failed to load modules: ' + e.message;
  }
}

function renderModules(modules) {
  if (!modules.length) {
    $('modulesContainer').innerHTML = '<div style="padding:20px;color:var(--text-3);font-family:var(--mono);font-size:12px">No modules discovered. Place module files in engine/modules/</div>';
    return;
  }
  var h = '';
  modules.forEach(function(m) {
    _moduleData[m.name] = m;
    h += renderModuleCard(m);
  });
  $('modulesContainer').innerHTML = h;

  // Resume polling for any running modules
  modules.forEach(function(m) {
    if (m.status === 'running') startModulePoll(m.name);
    if (m.has_results) fetchModuleResults(m.name);
  });
}

function renderModuleCard(m) {
  // Tier classes on the card give different border colors:
  //   showpiece      → cyan   (featured analysis)
  //   mi             → lime   (mechanistic-interpretability work)
  //   legacy         → purple (legacy modules)
  //   infrastructure → orange (utility: probe gen, dialogue)
  //   standard       → no border accent
  var tierClass = '';
  if      (m.tier === 'showpiece')      tierClass = ' mod-card--showpiece';
  else if (m.tier === 'mi')             tierClass = ' mod-card--mi';
  else if (m.tier === 'legacy')         tierClass = ' mod-card--legacy';
  else if (m.tier === 'infrastructure') tierClass = ' mod-card--infra';
  var h = '<div class="mod-card' + tierClass + '" id="mod-' + m.name + '">';

  var modCollapsed = CARD_COLLAPSE_DEFAULTS.modules === 'collapsed';
  var chevSymbol = modCollapsed ? '▶' : '▼';
  var bodyStyle = modCollapsed ? ' style="display:none"' : '';

  // Collapsible header - click to toggle body
  h += '<div class="mod-header" onclick="toggleModCard(\'' + m.name + '\')" style="cursor:pointer">';
  h += '<div class="mod-info">';
  h += '<div class="mod-title"><span class="mod-chevron" id="mod-chev-' + m.name + '" style="font-size:11px;color:var(--text-2);flex-shrink:0">' + chevSymbol + '</span> ' + escHtml(m.display_name) + ' <span class="mod-ver">v' + m.version + '</span></div>';
  h += '<div class="mod-desc">' + escHtml(m.description) + '</div>';
  h += '</div>';
  h += '<div class="mod-status ' + m.status + '" id="mod-status-' + m.name + '">' + m.status + '</div>';
  h += '</div>';

  // Collapsible body
  h += '<div class="mod-body" id="mod-body-' + m.name + '"' + bodyStyle + '>';

  // Requirements
  var reqs = [];
  if (m.requires_sfd) reqs.push('SFD');
  if (m.requires_ltp) reqs.push('LTP');
  if (m.requires_rd) reqs.push('RD');
  if (reqs.length || m.min_results > 1) {
    h += '<div style="padding:6px 16px 0;display:flex;gap:6px;flex-wrap:wrap">';
    if (m.min_results > 1) h += '<span class="mod-req">' + m.min_results + '+ results</span>';
    reqs.forEach(function(r) { h += '<span class="mod-req">' + r + '</span>'; });
    h += '</div>';
  }

  // Parameters
  if (m.parameters && m.parameters.length) {
    h += '<div class="mod-params">';
    m.parameters.forEach(function(p) {
      h += '<div class="mod-param">';
      h += '<label>' + escHtml(p.display_name) + '</label>';
      h += renderParamControl(m.name, p);
      if (p.description) h += '<div class="param-desc">' + escHtml(p.description) + '</div>';
      h += '</div>';
    });
    h += '</div>';
  }

  // Actions
  h += '<div class="mod-actions">';
  h += '<button class="btn btn-primary btn-sm" id="mod-run-' + m.name + '" onclick="event.stopPropagation();runModule(\'' + m.name + '\')">Run</button>';
  h += '<button class="btn btn-sm" style="border:1px solid var(--border);color:var(--text-2);background:transparent" title="Clear results and restore all parameters to defaults" onclick="event.stopPropagation();resetModule(\'' + m.name + '\')">Reset Defaults</button>';
  // Probe Generator: diagnostic popout, available without requiring a fresh run.
  // Operates on whatever probe set is on disk (active set by default).
  if (m.name === 'probe_generator') {
    h += '<button class="btn btn-sm btn-popout" style="margin-left:auto" onclick="event.stopPropagation();popoutProbeDiagnostic()" title="Inspect lattice properties of the active probe set">↗ Probe Diagnostics</button>';
  }
  if (m.name === 'token_pair_coupling') {
    h += '<button class="btn btn-sm" style="border:1px solid var(--border);color:var(--text-2);background:transparent" onclick="event.stopPropagation();window.open(\'/api/modules/token_pair_coupling/export_cache\',\'_blank\')">Export Cache</button>';
    h += '<button class="btn btn-sm" style="border:1px solid var(--red);color:var(--red);background:transparent" onclick="event.stopPropagation();confirmResetTokenPairCache()">Reset Cache</button>';
  }
  h += '<div class="mod-progress" id="mod-progress-' + m.name + '"></div>';
  h += '</div>';

  // Results placeholder
  h += '<div id="mod-results-' + m.name + '"></div>';

  h += '</div>'; // end mod-body
  h += '</div>'; // end mod-card
  return h;
}

function toggleModCard(name) {
  var body = $('mod-body-' + name);
  var chev = $('mod-chev-' + name);
  if (!body) return;
  if (body.style.display === 'none') {
    body.style.display = '';
    if (chev) chev.textContent = '▼';
  } else {
    body.style.display = 'none';
    if (chev) chev.textContent = '▶';
  }
}

function renderParamControl(modName, p) {
  var id = 'modp-' + modName + '-' + p.name;
  if (p.type === 'bool') {
    var checked = p.default ? ' checked' : '';
    return '<div class="checkbox-row" style="margin-top:3px"><input type="checkbox" id="' + id + '"' + checked + '><label for="' + id + '">Enable</label></div>';
  }
  if (p.type === 'select') {
    var s = '<select id="' + id + '">';
    p.options.forEach(function(o) {
      var sel = o === p.default ? ' selected' : '';
      s += '<option value="' + o + '"' + sel + '>' + o + '</option>';
    });
    s += '</select>';
    return s;
  }
  if (p.type === 'int' || p.type === 'float') {
    var step = p.type === 'float' ? '0.01' : '1';
    var minA = p.min_val != null ? ' min="' + p.min_val + '"' : '';
    var maxA = p.max_val != null ? ' max="' + p.max_val + '"' : '';
    return '<input type="number" id="' + id + '" value="' + p.default + '" step="' + step + '"' + minA + maxA + ' style="width:80px">';
  }
  if (p.type === 'file') {
    return '<div class="file-input-wrapper"><input type="file" id="' + id + '" accept=".csv" data-param-type="file"></div>';
  }
  if (p.type === 'textarea') {
    var val = (p.default || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    return '<textarea id="' + id + '" rows="4" style="width:100%;resize:vertical;font-size:11px;line-height:1.5">' + val + '</textarea>';
  }
  return '<input type="text" id="' + id + '" value="' + (p.default || '') + '">';
}

function getModuleParams(modName) {
  var m = _moduleData[modName];
  if (!m || !m.parameters) return {};
  var params = {};
  m.parameters.forEach(function(p) {
    var el = $('modp-' + modName + '-' + p.name);
    if (!el) { params[p.name] = p.default; return; }
    if (p.type === 'bool') params[p.name] = el.checked;
    else if (p.type === 'file') params[p.name] = '';  // resolved by upload in runModule
    else if (p.type === 'int') { var v = parseInt(el.value); params[p.name] = isNaN(v) ? p.default : v; }
    else if (p.type === 'float') { var v = parseFloat(el.value); params[p.name] = isNaN(v) ? p.default : v; }
    else params[p.name] = el.value;
  });
  return params;
}

async function resetModule(name) {
  try {
    var r = await fetch('/api/modules/' + name + '/reset', {method: 'POST'});
    var d = await r.json();
    if (d.ok) {
      var st = $('mod-status-' + name);
      if (st) { st.className = 'mod-status idle'; st.textContent = 'idle'; }
      var pr = $('mod-progress-' + name);
      if (pr) pr.textContent = '';
      var res = $('mod-results-' + name);
      if (res) res.innerHTML = '';
      var btn = $('mod-run-' + name);
      if (btn) btn.disabled = false;
      // Remove download log button
      var logBtn = $('mod-log-' + name);
      if (logBtn) logBtn.remove();
      // Reset params to defaults
      var m = _moduleData[name];
      if (m && m.parameters) {
        m.parameters.forEach(function(p) {
          var el = $('modp-' + name + '-' + p.name);
          if (!el) return;
          if (p.type === 'bool') el.checked = !!p.default;
          else if (p.type === 'file') el.value = '';
          else if (p.type === 'textarea') el.value = p.default || '';
          else el.value = p.default != null ? p.default : '';
        });
      }
    }
  } catch(e) {
    console.error('Reset failed:', e);
  }
}

async function runModule(name) {
  var btn = $('mod-run-' + name);
  var prog = $('mod-progress-' + name);
  btn.disabled = true;
  btn.textContent = 'Running...';
  prog.textContent = 'Starting...';
  // Remove stale download log button
  var oldLog = $('mod-log-' + name);
  if (oldLog) oldLog.remove();

  log('Module ' + name + ': starting');

  var params = getModuleParams(name);

  // Upload any file params first
  try {
    var fileInputs = document.querySelectorAll('[data-param-type="file"]');
    for (var i = 0; i < fileInputs.length; i++) {
      var fi = fileInputs[i];
      if (fi.id.startsWith('modp-' + name + '-') && fi.files && fi.files.length > 0) {
        var paramName = fi.id.replace('modp-' + name + '-', '');
        prog.textContent = 'Uploading ' + fi.files[0].name + '...';
        var fd = new FormData();
        fd.append('file', fi.files[0]);
        var ur = await fetch('/api/modules/upload_template', {method:'POST', body:fd});
        var ud = await ur.json();
        if (!ud.ok) { throw new Error(ud.error || 'Upload failed'); }
        params[paramName] = ud.filename;
      }
    }
  } catch(e) {
    prog.textContent = '';
    btn.disabled = false;
    btn.textContent = 'Run';
    log('Module ' + name + ': upload error — ' + e.message, 'error');
    return;
  }

  try {
    const r = await fetch('/api/modules/' + name + '/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({params: params})
    });
    const d = await r.json();
    if (!d.ok) {
      prog.textContent = '';
      btn.disabled = false;
      btn.textContent = 'Run';
      updateModuleStatus(name, 'error');
      log('Module ' + name + ': failed to start — ' + (d.error || 'unknown'), 'error');
      return;
    }
    updateModuleStatus(name, 'running');
    // Stash start time on poller dict for elapsed-time computation on completion.
    // (The backend also reports elapsed; this is a fallback for when
    // started_at/completed_at are missing.)
    _moduleStartTimes[name] = Date.now();
    startModulePoll(name);
  } catch(e) {
    prog.textContent = '';
    btn.disabled = false;
    btn.textContent = 'Run';
    log('Module ' + name + ': error — ' + e.message, 'error');
  }
}

function startModulePoll(name) {
  // Module status now delivered via SSE (module_status events).
  // Retained as no-op so callers don't need to change.
}

async function pollModule(name) {
  // No longer called. Module status arrives via SSE.
}

function updateModuleStatus(name, status) {
  var el = $('mod-status-' + name);
  if (!el) return;
  el.className = 'mod-status ' + status;
  el.textContent = status;
}

async function fetchModuleResults(name) {
  try {
    const r = await fetch('/api/modules/' + name + '/results');
    const d = await r.json();
    if (!d.ok) return;
    renderModuleResults(name, d.results);
  } catch(e) { /* silent */ }
}

function renderModuleResults(name, results) {
  var container = $('mod-results-' + name);
  if (!container) return;

  // Route to module-specific renderer
  if (name === 'domain_surface') {
    _dsSurfaceData = results;
    container.innerHTML = renderDomainSurfaceResults(results);
  } else if (name === 'probe_generator') {
    container.innerHTML = renderProbeGeneratorResults(results);
  } else if (name === 'correction_prism') {
    container.innerHTML = renderCorrectionPrismResults(results);
  } else if (name === 'mechanistic_interpretability') {
    container.innerHTML = renderMIResults(results);
  } else if (name === 'correction_field_topology') {
    container.innerHTML = renderCFTResults(results);
  } else if (name === 'comparative_analysis') {
    container.innerHTML = renderComparativeResults(results);
  } else if (name === 'mi_instrumentation') {
    container.innerHTML = renderMIInstrumentationResults(results);
  } else if (name === 'model_dialogue') {
    container.innerHTML = renderModelDialogueResults(results);
    // Auto-open chat window when Run completes
    if (results && results.chat_url) popoutChat();
  } else if (name === 'roundtable_lma') {
    container.innerHTML = renderRoundtableLMAResults(results);
    if (results && results.chat_url) popoutRoundtable();
  } else if (name === 'arditi_benchmarks') {
    container.innerHTML = renderArditiBenchmarksResults(results);
  } else if (name === 'token_pair_coupling') {
    container.innerHTML = renderTokenPairResults(results);
  } else {
    // Generic JSON dump fallback
    container.innerHTML = '<div class="mod-results"><div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Results</div><div class="mod-results-body"><pre style="padding:12px;font-size:11px;color:var(--text-1);overflow:auto;max-height:400px">' + escHtml(JSON.stringify(results, null, 2).substring(0, 5000)) + '</pre></div></div>';
  }
}

// ─── Model Dialogue Results Renderer ────────────────────────────

function renderModelDialogueResults(r) {
  var cfg = r.config || {};
  var h = '<div class="mod-results">';
  h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">';
  h += 'Chat Configuration';
  h += '<button class="btn-popout" style="margin-left:auto" onclick="event.stopPropagation();popoutChat()">↗ Open Chat</button>';
  h += '</div>';
  h += '<div class="mod-results-body">';
  h += '<table style="font-size:12px;border-collapse:collapse;width:100%">';
  h += '<tr><td style="padding:4px 12px;color:var(--text-2)">Temperature</td><td style="padding:4px 12px;font-weight:600">' + (cfg.temperature || 0.7) + '</td>';
  h += '<td style="padding:4px 12px;color:var(--text-2)">Top-P</td><td style="padding:4px 12px;font-weight:600">' + (cfg.top_p || 0.9) + '</td>';
  h += '<td style="padding:4px 12px;color:var(--text-2)">Max Tokens</td><td style="padding:4px 12px;font-weight:600">' + (cfg.max_tokens || 256) + '</td></tr>';
  h += '<tr><td style="padding:4px 12px;color:var(--text-2)">Analyze Prompts</td><td style="padding:4px 12px;font-weight:600">' + (cfg.analyze_prompts ? '✓' : '—') + '</td>';
  h += '<td style="padding:4px 12px;color:var(--text-2)">Analyze Responses</td><td style="padding:4px 12px;font-weight:600">' + (cfg.analyze_responses ? '✓' : '—') + '</td>';
  h += '<td style="padding:4px 12px;color:var(--text-2)">LTP / SFD</td><td style="padding:4px 12px;font-weight:600">' + (cfg.compute_ltp ? 'LTP' : '') + (cfg.compute_ltp && cfg.compute_sfd ? ' + ' : '') + (cfg.compute_sfd ? 'SFD' : '') + '</td></tr>';
  h += '</table>';
  h += '<div style="margin-top:10px;padding:8px;background:var(--bg-0);border-radius:4px;font-size:11px;color:var(--text-2)">';
  h += 'Click <strong>Open Chat</strong> to launch the dialogue window. ';
  h += 'Each conversation turn is analyzed under the current configuration and recorded into the session. ';
  h += 'Switch between instruct and base models using the toggle in the chat window.';
  h += '</div>';
  h += '</div></div>';
  return h;
}

function popoutChat() {
  var pw = 640, ph = 720;
  var left = Math.round((screen.width - pw) / 2);
  var top = Math.round((screen.height - ph) / 2);
  window.open('/chat', '_blank', 'width=' + pw + ',height=' + ph + ',left=' + left + ',top=' + top + ',scrollbars=yes');
}

// ─── Roundtable LMA Results Renderer ──────────────────────────

function renderRoundtableLMAResults(r) {
  var cfg = r.config || {};
  var parts = r.participants || [];
  var methods = r.methods || [];
  var tools = r.tools || [];
  var topic = r.topic || '(not set)';
  var h = '<div class="mod-results">';
  h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">';
  h += 'Roundtable Configuration';
  h += '<button class="btn-popout" style="margin-left:auto" onclick="event.stopPropagation();popoutRoundtable()">↗ Open Roundtable</button>';
  h += '</div>';
  h += '<div class="mod-results-body">';

  // Config table
  h += '<table style="font-size:12px;border-collapse:collapse;width:100%">';
  h += '<tr><td style="padding:4px 12px;color:var(--text-2)">Temperature</td><td style="padding:4px 12px;font-weight:600">' + (cfg.temperature || 0.7) + '</td>';
  h += '<td style="padding:4px 12px;color:var(--text-2)">Max Tokens</td><td style="padding:4px 12px;font-weight:600">' + (cfg.max_tokens || 256) + '</td>';
  h += '<td style="padding:4px 12px;color:var(--text-2)">Panels</td><td style="padding:4px 12px;font-weight:600">' + (cfg.n_roundtables || 2) + '</td></tr>';
  h += '</table>';

  // Topic
  h += '<div style="margin-top:10px;padding:8px 12px;background:var(--bg-0);border-radius:4px;font-size:12px">';
  h += '<span style="color:var(--text-2)">Topic:</span> <span style="color:var(--text-0)">' + escHtml(topic) + '</span>';
  h += '</div>';

  // Participants
  h += '<div style="margin-top:8px;padding:8px 12px;background:var(--bg-0);border-radius:4px;font-size:12px">';
  h += '<span style="color:var(--text-2)">Participants (' + parts.length + '):</span> ';
  if (parts.length === 0) {
    h += '<span style="color:var(--orange)">None registered — add participants in the Roundtable window</span>';
  } else {
    h += parts.map(function(p) { return '<span style="color:var(--cyan)">' + escHtml(p.name) + '</span>'; }).join(', ');
  }
  h += '</div>';

  // Capabilities
  h += '<div style="margin-top:8px;padding:8px 12px;background:var(--bg-0);border-radius:4px;font-size:11px;color:var(--text-2)">';
  h += '<span style="color:var(--text-2)">Stage types:</span> ' + (r.stage_types || ['PANEL','ANALYSIS','TOOL']).map(function(s) {
    var c = s === 'PANEL' ? '--cyan' : s === 'TOOL' ? '--orange' : '--purple';
    return '<span style="color:var(' + c + ');font-family:var(--mono);font-weight:600">' + s + '</span>';
  }).join(' · ') + ' &nbsp;│&nbsp; ';
  h += '<span style="color:var(--text-2)">Methods:</span> ' + methods.join(', ') + ' &nbsp;│&nbsp; ';
  h += '<span style="color:var(--text-2)">Tools:</span> ' + tools.join(', ');
  h += '</div>';

  // Help text
  h += '<div style="margin-top:10px;padding:8px;background:var(--bg-0);border-radius:4px;font-size:11px;color:var(--text-2)">';
  h += 'Click <strong>Open Roundtable</strong> to launch the interactive panel. ';
  h += 'Type your inquiry, select personas, and build the discussion step by step. ';
  h += 'Or upload a CSV template below for automated batch execution.';
  h += '</div>';

  // Template upload + batch run
  h += '<div style="margin-top:10px;padding:10px 12px;background:var(--bg-0);border:1px solid var(--border);border-radius:4px;display:flex;align-items:center;gap:10px;flex-wrap:wrap">';
  h += '<label style="display:inline-flex;align-items:center;gap:6px;padding:5px 12px;background:var(--bg-2);border:1px solid var(--border);border-radius:4px;cursor:pointer;font-size:12px;color:var(--text-1)">';
  h += '<input type="file" accept=".csv" onchange="rtSelectTemplate(this)" style="display:none">';
  h += '📄 Choose Template CSV';
  h += '</label>';
  h += '<span id="rtTemplateFile" style="font-size:11px;color:var(--text-2);font-family:var(--mono)">no file selected</span>';
  h += '<button onclick="runRoundtableBatch()" style="margin-left:auto;padding:5px 14px;background:var(--cyan);color:#000;border:none;border-radius:4px;font-size:12px;font-weight:600;cursor:pointer;font-family:var(--sans)">Run Batch</button>';
  h += '</div>';
  h += '<div id="rtBatchOutput" style="margin-top:8px"></div>';

  h += '</div></div>';
  return h;
}

var _rtTemplateText = null;

function rtSelectTemplate(input) {
  var file = input.files[0];
  if (!file) return;
  var el = document.getElementById('rtTemplateFile');
  if (el) el.textContent = file.name;
  var reader = new FileReader();
  reader.onload = function() { _rtTemplateText = reader.result; };
  reader.readAsText(file);
}

function runRoundtableBatch() {
  if (!_rtTemplateText) {
    var out = document.getElementById('rtBatchOutput');
    if (out) out.innerHTML = '<div style="padding:8px;font-size:12px;color:var(--orange)">Select a CSV template first.</div>';
    return;
  }
  var topic = '';
  // Try to read topic from the config
  try { topic = document.querySelector('#topicInput') ? document.querySelector('#topicInput').value : ''; } catch(e) {}
  if (!topic) topic = 'Roundtable discussion';

  var out = document.getElementById('rtBatchOutput');
  if (out) out.innerHTML = '<div style="padding:8px;font-size:12px;color:var(--text-2);font-family:var(--mono)">Running batch pipeline...</div>';

  fetch('/api/roundtable/batch', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({template_csv: _rtTemplateText, topic: topic})
  }).then(function(r) { return r.json(); }).then(function(data) {
    if (!out) return;
    if (data.ok && data.result) {
      var res = data.result;
      var h = '<div style="padding:10px;background:var(--bg-0);border:1px solid var(--border);border-radius:4px">';
      h += '<div style="font-size:12px;font-weight:600;color:var(--green);margin-bottom:8px">✓ Batch complete — '
           + res.n_stages + ' stages, ' + res.n_total_generations + ' generations, ' + res.elapsed_seconds + 's</div>';
      // Stage summary
      if (res.stages) {
        res.stages.forEach(function(s, i) {
          var c = s.stage_type === 'PANEL' ? '--cyan' : s.stage_type === 'TOOL' ? '--orange' : '--purple';
          h += '<div style="font-size:11px;padding:2px 0;color:var(--text-1)">';
          h += '<span style="color:var(' + c + ');font-family:var(--mono);font-weight:600">' + s.stage_type + '</span>';
          h += ' <span style="color:var(--text-2)">' + (s.label || '') + '</span>';
          if (s.n_generations) h += ' · ' + s.n_generations + ' gen';
          h += '</div>';
        });
      }
      // Final output preview
      if (res.final_output) {
        h += '<div style="margin-top:8px;padding:8px;background:var(--bg-1);border-radius:4px;font-size:11px;font-family:var(--mono);color:var(--text-1);max-height:200px;overflow:auto;white-space:pre-wrap">';
        h += escHtml(res.final_output.substring(0, 3000));
        h += '</div>';
      }
      h += '</div>';
      out.innerHTML = h;
    } else {
      out.innerHTML = '<div style="padding:8px;font-size:12px;color:var(--red)">' + escHtml(data.error || 'Batch failed') + '</div>';
    }
  }).catch(function(e) {
    if (out) out.innerHTML = '<div style="padding:8px;font-size:12px;color:var(--red)">Error: ' + escHtml(e.message) + '</div>';
  });
}

function popoutRoundtable() {
  var pw = 960, ph = 760;
  var left = Math.round((screen.width - pw) / 2);
  var top = Math.round((screen.height - ph) / 2);
  window.open('/roundtable', '_blank', 'width=' + pw + ',height=' + ph + ',left=' + left + ',top=' + top + ',scrollbars=yes');
}

// ─── Arditi Benchmarks Results Renderer ────────────────────────
//
// Renders the results of the arditi_benchmarks module: a summary banner
// plus collapsible panels for each sub-benchmark (causal/steering/scan).
// Follows the conventions of renderMIInstrumentationResults.

// ─── Token Pair Coupling Results Renderer ──────────────────────

function renderTokenPairResults(r) {
  if (!r) return '<div class="mod-results"><div class="mod-results-body" style="padding:16px;color:var(--text-3)">No results yet.</div></div>';
  if (r.error) return '<div class="mod-results"><div class="mod-results-body" style="padding:16px;color:var(--orange)">' + escHtml(r.error) + '</div></div>';

  var cs = r.cache_summary || {};
  var h = '<div class="mod-results">';

  // Session summary
  h += '<div style="padding:12px 16px;border-bottom:1px solid var(--border)">';
  h += '<div class="metrics-grid">';
  h += mc('Pairs found', r.session_pairs_found, null, function(v){ return v; });
  h += mc('New added', r.session_pairs_added, null, function(v){ return v; });
  h += mc('Duplicates', r.duplicates_skipped, null, function(v){ return v || '0'; });
  h += mc('Prompts', r.prompts_processed, null, function(v){ return v; });
  h += mc('Skipped', r.prompts_skipped, null, function(v){ return v || '0'; });
  h += mc('Cache total', cs.n_observations, null, function(v){ return v || '0'; });
  h += mc('Unique pairs', cs.n_unique_pairs, null, function(v){ return v || '0'; });
  h += mc('Sessions', cs.n_sessions, null, function(v){ return v || '0'; });
  h += '</div>';
  if (r.prompts_skipped > 0 && r.skip_reason) {
    h += '<div style="margin-top:6px;font-size:11px;color:var(--orange)">Skipped: ' + escHtml(r.skip_reason) + '</div>';
  }
  if (cs.models && cs.models.length > 0) {
    h += '<div style="margin-top:6px;font-size:11px;color:var(--text-2)">Models: ' + cs.models.map(function(m){ return escHtml(m.split('/').pop()); }).join(', ') + '</div>';
  }
  h += '</div>';

  // Cache management buttons
  h += '<div style="padding:10px 16px;border-bottom:1px solid var(--border);display:flex;gap:8px;align-items:center">';
  h += '<button class="btn btn-sm" style="border:1px solid var(--border);color:var(--text-2);background:transparent" onclick="window.open(\'/api/modules/token_pair_coupling/export_cache\',\'_blank\')">Export Cache</button>';
  h += '<button class="btn btn-sm" style="border:1px solid var(--red);color:var(--red);background:transparent" onclick="confirmResetTokenPairCache()">Reset Cache</button>';
  h += '<span style="flex:1"></span>';
  h += '<span style="font-size:11px;color:var(--text-3)">Cache persists across sessions and resets</span>';
  h += '</div>';

  // Top pairs table
  var pairs = r.top_pairs || [];
  if (pairs.length > 0) {
    h += '<div style="padding:12px 16px">';
    h += '<div style="font-size:12px;font-weight:600;color:var(--text-1);margin-bottom:8px">Strongly Coupled Pairs (' + (r.all_pairs_count || pairs.length) + ' total)</div>';
    h += '<div style="overflow-x:auto;max-height:500px;overflow-y:auto">';
    h += '<table style="width:100%;font-size:11px;border-collapse:collapse">';
    h += '<thead><tr style="border-bottom:1px solid var(--border);color:var(--text-2);text-align:left">';
    h += '<th style="padding:4px 8px">Prompt Token</th>';
    h += '<th style="padding:4px 8px">→</th>';
    h += '<th style="padding:4px 8px">Counterfactual</th>';
    h += '<th style="padding:4px 8px;text-align:right">Count</th>';
    h += '<th style="padding:4px 8px;text-align:right">Mean Score</th>';
    h += '<th style="padding:4px 8px;text-align:right">Max Score</th>';
    h += '<th style="padding:4px 8px">Variant</th>';
    h += '<th style="padding:4px 8px">Position</th>';
    h += '<th style="padding:4px 8px">Categories</th>';
    h += '</tr></thead><tbody>';

    for (var i = 0; i < pairs.length; i++) {
      var p = pairs[i];
      var rowBg = i % 2 === 0 ? '' : 'background:var(--bg-0);';
      h += '<tr style="border-bottom:1px solid var(--border-dim);' + rowBg + '">';
      h += '<td style="padding:4px 8px;color:var(--cat-harmful);font-family:var(--mono)">' + escHtml(p.prompt_token) + '</td>';
      h += '<td style="padding:4px 8px;color:var(--text-3)">→</td>';
      h += '<td style="padding:4px 8px;color:var(--cat-benign);font-family:var(--mono)">' + escHtml(p.counterfactual) + '</td>';
      h += '<td style="padding:4px 8px;text-align:right;font-weight:600">' + p.count + '</td>';
      h += '<td style="padding:4px 8px;text-align:right">' + (p.mean_score || 0).toFixed(4) + '</td>';
      h += '<td style="padding:4px 8px;text-align:right">' + (p.max_score || 0).toFixed(4) + '</td>';
      h += '<td style="padding:4px 8px;color:var(--text-2);font-size:10px">' + (p.variants || []).join(', ') + '</td>';
      h += '<td style="padding:4px 8px;color:var(--text-2)">' + (p.position_tendency || '') + '</td>';
      h += '<td style="padding:4px 8px">' + (p.categories || []).map(function(c){ return '<span class="pill ' + pillClass(c) + '" style="font-size:9px">' + c.substr(0,4) + '</span>'; }).join(' ') + '</td>';
      h += '</tr>';
    }

    h += '</tbody></table></div></div>';
  } else {
    h += '<div style="padding:16px;color:var(--text-3)">No pairs above threshold. Try lowering the minimum interaction score or running more prompts with LTP and KL divergence enabled.</div>';
  }

  h += '</div>';
  return h;
}

function confirmResetTokenPairCache() {
  var overlay = document.createElement('div');
  overlay.id = 'tpcResetOverlay';
  overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.6);z-index:9999;display:flex;align-items:center;justify-content:center';
  overlay.innerHTML = '<div style="background:var(--bg-1);border:1px solid var(--red);border-radius:8px;padding:24px;max-width:420px;width:90%;color:var(--text-1)">'
    + '<h3 style="margin:0 0 12px;color:var(--red)">Reset Token Pair Cache</h3>'
    + '<p style="margin:0 0 16px;line-height:1.5;color:var(--text-2)">This will permanently delete all accumulated token pair observations across all sessions. This action cannot be undone.</p>'
    + '<div style="display:flex;gap:10px;justify-content:flex-end">'
    + '<button class="btn btn-sm" style="border:1px solid var(--border);color:var(--text-2);background:transparent" onclick="document.getElementById(\'tpcResetOverlay\').remove()">Cancel</button>'
    + '<button class="btn btn-sm" style="border:1px solid var(--red);color:var(--bg-1);background:var(--red);font-weight:600" onclick="executeResetTokenPairCache()">Delete All Observations</button>'
    + '</div></div>';
  document.body.appendChild(overlay);
}

async function executeResetTokenPairCache() {
  var overlay = $('tpcResetOverlay');
  if (overlay) overlay.remove();
  try {
    var r = await(await fetch('/api/modules/token_pair_coupling/reset_cache', {method: 'POST'})).json();
    if (r.ok) {
      log('Token pair cache cleared: ' + r.observations_removed + ' observations removed', 'done');
      fetchModuleResults('token_pair_coupling');
    }
  } catch(e) { log('Cache reset error: ' + e.message, 'error'); }
}

function renderArditiBenchmarksResults(r) {
  if (r && r.error) {
    return '<div class="mod-results"><div class="mod-results-body" ' +
           'style="padding:16px;color:var(--orange)">' +
           escHtml(r.error) + '</div></div>';
  }

  var h = '<div class="mod-results">';
  var sm = r.summary || {};

  // ═══ TOP SUMMARY BANNER ═══
  var bidirColor = sm.bidirectional_confirmed ? 'var(--green)' : 'var(--text-2)';
  var bidirLabel = sm.bidirectional_confirmed ? 'Confirmed' : '—';
  h += '<div style="padding:12px 16px;border-bottom:1px solid var(--border)">';
  h += '<div class="metrics-grid">';
  h += mc('Train AUROC', sm.train_auroc, null,
          function(v){ return (v||0).toFixed(4); });
  h += mc('Hidden Dim', sm.hidden_dim, null, function(v){ return v||'—'; });
  h += mc('Causal Ran', sm.causal_ran ? 'Yes' : 'No', null, function(v){ return v; });
  h += mc('Steering Ran', sm.steering_ran ? 'Yes' : 'No', null, function(v){ return v; });
  h += mc('Scan Ran', sm.alpha_scan_ran ? 'Yes' : 'No', null, function(v){ return v; });
  h += mc('Bidirectional', bidirLabel, null,
          function(v){ return '<span style="color:' + bidirColor + '">' + v + '</span>'; });
  h += '</div>';
  if (sm.combined_verdict) {
    h += '<div style="margin-top:10px;padding:10px 12px;background:var(--bg-0);' +
         'border-radius:4px;font-size:12px;color:var(--text-1);line-height:1.5">' +
         escHtml(sm.combined_verdict) + '</div>';
  }
  h += '</div>';

  // ═══ CAUSAL (ABLATION) PANEL ═══
  if (r.causal) {
    h += _renderArditiCausalPanel(r.causal);
  }

  // ═══ STEERING (ADDITION) PANEL ═══
  if (r.steering) {
    h += _renderArditiSteeringPanel(r.steering);
  }

  // ═══ ALPHA SCAN PANEL ═══
  if (r.alpha_scan) {
    h += _renderArditiAlphaScanPanel(r.alpha_scan);
  }

  h += '</div>';
  return h;
}

// ─── Causal (ablation) sub-panel ───────────────────────────────────
function _renderArditiCausalPanel(c) {
  var h = '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Causal Test (Ablation)</div>';
  h += '<div class="mod-results-body">';
  h += '<div style="font-size:11px;color:var(--text-2);margin-bottom:8px;line-height:1.5">';
  h += 'Projects the refusal direction out of the residual stream on held-out harmful prompts. ';
  h += 'A large drop in refusal rate is Arditi\'s necessity signature.</div>';

  if (c.error) {
    h += '<div style="padding:10px;color:var(--orange)">' + escHtml(c.error) + '</div>';
    h += '</div>';
    return h;
  }

  var s = c.summary || {};
  var verdictCol = _arditiVerdictColor(s.verdict_tag);

  h += '<div class="metrics-grid">';
  h += mc('Baseline refusal', s.baseline_refusal_rate, null,
          function(v){ return ((v||0)*100).toFixed(1) + '%'; });
  h += mc('Ablated refusal', s.intervened_refusal_rate, null,
          function(v){ return ((v||0)*100).toFixed(1) + '%'; });
  h += mc('Δ (drop)', s.delta, null,
          function(v){ return '<span style="color:' + verdictCol + '">' +
                          (v>=0 ? '+' : '') + (v*100).toFixed(1) + '%</span>'; });
  h += mc('95% CI', s.delta_ci_95, null,
          function(v){ return v ? '['+(v[0]*100).toFixed(1)+'%, '+(v[1]*100).toFixed(1)+'%]' : '—'; });
  h += mc('Layers', s.n_layers_intervened, null, function(v){ return v||'—'; });
  h += mc('Alpha', s.alpha, null, function(v){ return (v||0).toFixed(2); });
  h += mc('N held-out', s.n_held_prompts, null, function(v){ return v||'—'; });
  h += mc('Significant', s.delta_excludes_zero ? '✓' : '—', null,
          function(v){ return '<span style="color:' + (v==='✓' ? 'var(--green)' : 'var(--text-3)') + '">' + v + '</span>'; });
  h += '</div>';

  if (s.verdict) {
    h += '<div style="margin-top:10px;padding:8px 10px;background:var(--bg-0);' +
         'border-radius:4px;font-size:12px;color:' + verdictCol + ';line-height:1.5">';
    h += '<strong>' + (s.verdict_tag||'').replace(/_/g,' ') + '</strong> &mdash; ';
    h += escHtml(s.verdict) + '</div>';
  }

  // Per-prompt table
  h += _renderArditiPromptTable(c.baseline, c.intervened,
                                  'Baseline reply', 'Ablated reply');
  h += '</div>';
  return h;
}

// ─── Steering (addition) sub-panel ─────────────────────────────────
function _renderArditiSteeringPanel(c) {
  var h = '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Steering Test (Addition)</div>';
  h += '<div class="mod-results-body">';
  h += '<div style="font-size:11px;color:var(--text-2);margin-bottom:8px;line-height:1.5">';
  h += 'Adds the refusal direction to held-out benign prompts. ';
  h += 'A large rise in refusal rate is the sufficiency signature.</div>';

  if (c.error) {
    h += '<div style="padding:10px;color:var(--orange)">' + escHtml(c.error) + '</div>';
    h += '</div>';
    return h;
  }

  var s = c.summary || {};
  var verdictCol = _arditiVerdictColor(s.verdict_tag);

  h += '<div class="metrics-grid">';
  h += mc('Baseline refusal', s.baseline_refusal_rate, null,
          function(v){ return ((v||0)*100).toFixed(1) + '%'; });
  h += mc('Steered refusal', s.intervened_refusal_rate, null,
          function(v){ return ((v||0)*100).toFixed(1) + '%'; });
  h += mc('Induction rate', s.induction_rate, null,
          function(v){ return '<span style="color:' + verdictCol + '">' +
                          (v>=0 ? '+' : '') + (v*100).toFixed(1) + '%</span>'; });
  h += mc('95% CI', s.induction_ci_95, null,
          function(v){ return v ? '['+(v[0]*100).toFixed(1)+'%, '+(v[1]*100).toFixed(1)+'%]' : '—'; });
  h += mc('Layers', s.n_layers_intervened, null, function(v){ return v||'—'; });
  h += mc('Alpha', s.alpha, null, function(v){ return (v||0).toFixed(2); });
  h += mc('N held-out', s.n_held_prompts, null, function(v){ return v||'—'; });
  h += mc('Significant', s.induction_excludes_zero ? '✓' : '—', null,
          function(v){ return '<span style="color:' + (v==='✓' ? 'var(--green)' : 'var(--text-3)') + '">' + v + '</span>'; });
  h += '</div>';

  if (s.verdict) {
    h += '<div style="margin-top:10px;padding:8px 10px;background:var(--bg-0);' +
         'border-radius:4px;font-size:12px;color:' + verdictCol + ';line-height:1.5">';
    h += '<strong>' + (s.verdict_tag||'').replace(/_/g,' ') + '</strong> &mdash; ';
    h += escHtml(s.verdict) + '</div>';
  }

  h += _renderArditiPromptTable(c.baseline, c.intervened,
                                  'Baseline reply', 'Steered reply');
  h += '</div>';
  return h;
}

// ─── Alpha scan sub-panel ──────────────────────────────────────────
function _renderArditiAlphaScanPanel(c) {
  var h = '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Alpha-Scan Dose Response</div>';
  h += '<div class="mod-results-body">';
  h += '<div style="font-size:11px;color:var(--text-2);margin-bottom:8px;line-height:1.5">';
  h += 'Runs the intervention at a grid of alpha values, sharing one baseline pass. ';
  h += 'Useful for finding the alpha at which an effect first appears and where coherence breaks.</div>';

  if (c.error) {
    h += '<div style="padding:10px;color:var(--orange)">' + escHtml(c.error) + '</div>';
    h += '</div>';
    return h;
  }

  var s = c.summary || {};
  var curve = s.curve || [];

  h += '<div class="metrics-grid">';
  h += mc('Mode', s.mode, null, function(v){ return v||'—'; });
  h += mc('Baseline refusal', s.baseline_refusal_rate, null,
          function(v){ return ((v||0)*100).toFixed(1) + '%'; });
  h += mc('Peak alpha', s.peak_alpha, null,
          function(v){ return v===null||v===undefined ? '—' : v; });
  h += mc('Peak effect', s.peak_effect, null,
          function(v){ return v===null||v===undefined ? '—' :
                          ((v>=0?'+':'') + (v*100).toFixed(1) + '%'); });
  h += mc('First signif α', s.first_significant_alpha, null,
          function(v){ return v===null||v===undefined ? '—' : v; });
  h += mc('Monotone', s.monotone_increasing ? 'Yes' : 'No', null, function(v){ return v; });
  h += mc('Breakdown?', s.likely_coherence_breakdown ? 'Likely' : '—', null,
          function(v){ return '<span style="color:' + (v==='Likely' ? 'var(--orange)' : 'var(--text-3)') + '">' + v + '</span>'; });
  h += mc('N layers', s.n_layers_intervened, null, function(v){ return v||'—'; });
  h += '</div>';

  if (s.verdict) {
    h += '<div style="margin-top:10px;padding:8px 10px;background:var(--bg-0);' +
         'border-radius:4px;font-size:12px;color:var(--text-1);line-height:1.5">';
    h += escHtml(s.verdict) + '</div>';
  }

  // Dose-response bar chart
  if (curve.length) {
    var maxAbs = Math.max.apply(null, curve.map(function(c){ return Math.abs(c.effect); }));
    maxAbs = Math.max(maxAbs, 0.1);  // never divide by zero
    h += '<div style="margin-top:12px;margin-bottom:8px">';
    curve.forEach(function(pt) {
      var pct = pt.effect * 100;
      var w = Math.min(100, Math.abs(pt.effect) / maxAbs * 100);
      var sig = pt.effect_excludes_zero;
      var col = sig
          ? (pt.effect > 0 ? 'var(--green)' : 'var(--red)')
          : 'var(--text-3)';
      h += '<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px">';
      h += '<span style="width:42px;text-align:right;font-size:11px;color:var(--text-2)">α=' + pt.alpha + '</span>';
      h += '<div style="flex:1;height:14px;background:var(--bg-3);border-radius:2px;overflow:hidden;position:relative">';
      h += '<div style="width:' + w + '%;height:100%;background:' + col + ';border-radius:2px"></div></div>';
      h += '<span style="width:60px;font-size:11px;color:' + col + ';text-align:right">' +
           (pct>=0?'+':'') + pct.toFixed(1) + '%' + (sig?' *':'') + '</span>';
      h += '</div>';
    });
    h += '</div>';
  }

  // Curve table
  if (curve.length) {
    h += '<table class="mod-tbl"><thead><tr>';
    h += '<th>α</th><th class="num">Refusal rate</th><th class="num">Effect</th>';
    h += '<th class="num">95% CI</th><th>Significant</th></tr></thead><tbody>';
    curve.forEach(function(pt) {
      var sig = pt.effect_excludes_zero;
      var col = sig
          ? (pt.effect > 0 ? 'var(--green)' : 'var(--red)')
          : 'var(--text-3)';
      h += '<tr><td>' + pt.alpha + '</td>';
      h += '<td class="num">' + (pt.intervened_refusal_rate*100).toFixed(1) + '%</td>';
      h += '<td class="num" style="color:' + col + '">' +
           (pt.effect>=0?'+':'') + (pt.effect*100).toFixed(1) + '%</td>';
      h += '<td class="num">[' + (pt.effect_ci_95[0]*100).toFixed(1) + '%, ' +
           (pt.effect_ci_95[1]*100).toFixed(1) + '%]</td>';
      h += '<td style="color:' + col + '">' + (sig ? '✓' : '—') + '</td></tr>';
    });
    h += '</tbody></table>';
  }

  h += '</div>';
  return h;
}

// ─── Per-prompt side-by-side table (causal + steering share this) ──
function _renderArditiPromptTable(baseline, intervened, baseLabel, intvLabel) {
  if (!baseline || !intervened || !baseline.length) return '';
  var n = Math.min(baseline.length, intervened.length);
  var h = '<table class="mod-tbl" style="margin-top:10px"><thead><tr>';
  h += '<th style="width:28%">Prompt</th>';
  h += '<th style="width:33%">' + baseLabel + '</th>';
  h += '<th style="width:33%">' + intvLabel + '</th>';
  h += '<th>B/I</th></tr></thead><tbody>';
  for (var i = 0; i < n; i++) {
    var b = baseline[i], v = intervened[i];
    var bFlag = b.refused ? '<span style="color:var(--orange)">R</span>' : '<span style="color:var(--green)">C</span>';
    var iFlag = v.refused ? '<span style="color:var(--orange)">R</span>' : '<span style="color:var(--green)">C</span>';
    h += '<tr>';
    h += '<td style="vertical-align:top;font-size:11px">' + escHtml(_truncate(b.prompt, 120)) + '</td>';
    h += '<td style="vertical-align:top;font-size:11px;color:var(--text-2)">' + escHtml(_truncate(b.reply, 180)) + '</td>';
    h += '<td style="vertical-align:top;font-size:11px;color:var(--text-2)">' + escHtml(_truncate(v.reply, 180)) + '</td>';
    h += '<td style="vertical-align:top;font-size:11px">' + bFlag + '/' + iFlag + '</td>';
    h += '</tr>';
  }
  h += '</tbody></table>';
  h += '<div style="font-size:10px;color:var(--text-3);margin-top:4px">R = refused, C = complied (phrase-match heuristic on first 200 chars).</div>';
  return h;
}

// ─── Helpers ───────────────────────────────────────────────────────
function _arditiVerdictColor(tag) {
  if (tag === 'large_effect') return 'var(--green)';
  if (tag === 'moderate_effect') return 'var(--cyan)';
  if (tag === 'small_or_absent') return 'var(--text-3)';
  return 'var(--text-1)';
}

function _truncate(s, n) {
  s = s || '';
  return s.length > n ? s.substring(0, n) + '…' : s;
}

function tvStat(label, value, detail) {
  return '<div class="mod-stat"><div class="stat-label">' + escHtml(label) + '</div><div class="stat-value">' + value + '</div>' + (detail ? '<div class="stat-detail">' + escHtml(String(detail)) + '</div>' : '') + '</div>';
}

function tvTokenTable(title, tokens) {
  var h = '<div class="mod-section-title">' + escHtml(title) + '</div>';
  h += '<div class="mod-results-body" style="max-height:350px">';
  h += '<table class="mod-tbl"><thead><tr>';
  h += '<th>Token</th><th class="num">n</th><th class="num">cats</th>';
  h += '<th class="num">den_cv</th><th class="num">str_cv</th>';
  h += '<th class="num">den_mean</th><th class="num">str_mean</th>';
  h += '<th class="num">η²_den</th>';
  h += '</tr></thead><tbody>';
  tokens.forEach(function(t) {
    var cv = t.density_cv || 0;
    var cvCol = cv > 0.05 ? 'var(--orange)' : (cv < 0.01 ? 'var(--green)' : 'var(--text-0)');
    h += '<tr>';
    h += '<td style="color:var(--cyan);font-weight:500">' + escHtml(t.token) + '</td>';
    h += '<td class="num">' + t.n + '</td>';
    h += '<td class="num">' + t.n_cats + '</td>';
    h += '<td class="num" style="color:' + cvCol + '">' + (t.density_cv || 0).toFixed(4) + '</td>';
    h += '<td class="num">' + (t.stress_cv || 0).toFixed(4) + '</td>';
    h += '<td class="num">' + (t.density_mean || 0).toFixed(4) + '</td>';
    h += '<td class="num">' + (t.stress_mean || 0).toFixed(3) + '</td>';
    h += '<td class="num">' + (t.eta_sq_density || 0).toFixed(4) + '</td>';
    h += '</tr>';
  });
  h += '</tbody></table></div>';
  return h;
}

function tvPairwiseTable(title, pairs) {
  var h = '<div class="mod-section-title">' + escHtml(title) + '</div>';
  h += '<div class="mod-results-body" style="max-height:300px">';
  h += '<table class="mod-tbl"><thead><tr>';
  h += '<th>Token</th><th class="num">Mean A</th><th class="num">n</th><th class="num">Mean B</th><th class="num">n</th><th class="num">Δ</th>';
  h += '</tr></thead><tbody>';
  pairs.forEach(function(p) {
    var dc = p.diff > 0 ? 'var(--red)' : (p.diff < 0 ? 'var(--green)' : 'var(--text-0)');
    h += '<tr>';
    h += '<td style="color:var(--cyan)">' + escHtml(p.token) + '</td>';
    h += '<td class="num">' + p.mean_a.toFixed(4) + '</td>';
    h += '<td class="num" style="color:var(--text-3)">' + p.n_a + '</td>';
    h += '<td class="num">' + p.mean_b.toFixed(4) + '</td>';
    h += '<td class="num" style="color:var(--text-3)">' + p.n_b + '</td>';
    h += '<td class="num" style="color:' + dc + ';font-weight:600">' + (p.diff > 0 ? '+' : '') + p.diff.toFixed(4) + '</td>';
    h += '</tr>';
  });
  h += '</tbody></table></div>';
  return h;
}

// ─── Domain Surface Results Renderer ──────────────────────────

var _dsSurfaceData = null;

function renderDomainSurfaceResults(r) {
  var h = '<div class="mod-results">';
  var ln = r.level_names || ['nouns','phrase','question','instruct','meta'];
  var cats = ['b','m','h','j'];
  var catNames = {b:'benign',m:'mild',h:'harmful',j:'jailbreak'};
  var catCols = {b:'var(--cat-benign)',m:'var(--cat-mild)',h:'var(--cat-harmful)',j:'var(--cat-jailbreak)'};

  // Summary
  h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Summary';
  if (r.ladder && r.ladder.right_individual && r.ladder.right_individual.items && r.ladder.right_individual.items.length > 0) {
    h += '<button class="btn-popout" style="margin-left:auto" onclick="event.stopPropagation();popoutDomainSurface()">↗ Probe Ladder</button>';
  }
  h += '</div>';
  h += '<div class="mod-results-body"><div class="mod-summary">';
  h += tvStat('Prompts', r.n_prompts_used, r.n_prompts_total > r.n_prompts_used ? (r.n_prompts_total - r.n_prompts_used) + ' without embeddings' : 'all with embeddings');
  h += tvStat('Observations', r.observations ? r.observations.length : 0, r.tokens ? r.tokens.length + ' tokens tracked' : '');
  h += tvStat('Subjects', r.subjects ? r.subjects.length : 0, r.anchors ? r.anchors.length + ' probe anchors' : '');
  h += tvStat('PCA Variance', r.pca ? r.pca[0] + '% + ' + r.pca[1] + '%' : '?', 'PC1 + PC2');
  h += tvStat('Probe File', r.probe_file || '?', '');
  h += '</div></div>';

  // Stratification by discourse level
  if (r.stratification && r.stratification.by_level) {
    h += '<div class="mod-section-title">Stratification by Discourse Level</div>';
    h += '<div class="mod-results-body">';
    h += '<table class="mod-tbl"><thead><tr>';
    h += '<th>Level</th>';
    cats.forEach(function(c) { h += '<th class="num" style="color:' + catCols[c] + '">' + catNames[c] + '</th>'; });
    h += '<th class="num">total</th><th class="num">% jailbreak</th>';
    h += '</tr></thead><tbody>';
    for (var li = 0; li < 5; li++) {
      var lk = String(li);
      var lv = r.stratification.by_level[lk] || {};
      var total = 0;
      cats.forEach(function(c) { total += (lv[c] || 0); });
      var jPct = total > 0 ? ((lv.j || 0) / total * 100) : 0;
      var isMeta = (li === 4);
      h += '<tr' + (isMeta ? ' style="background:rgba(204,121,167,0.06)"' : '') + '>';
      h += '<td style="color:var(--cyan);font-weight:500">' + (ln[li] || li) + '</td>';
      cats.forEach(function(c) {
        var n = lv[c] || 0;
        h += '<td class="num"' + (n > 0 ? ' style="color:' + catCols[c] + '"' : '') + '>' + n + '</td>';
      });
      h += '<td class="num" style="color:var(--text-2)">' + total + '</td>';
      h += '<td class="num" style="color:' + (jPct > 30 ? 'var(--cat-jailbreak)' : 'var(--text-2)') + ';font-weight:' + (jPct > 30 ? '600' : '400') + '">' + jPct.toFixed(1) + '%</td>';
      h += '</tr>';
    }
    h += '</tbody></table></div>';
  }

  // Stratification by subject
  if (r.stratification && r.stratification.by_subject) {
    h += '<div class="mod-section-title">Stratification by Subject</div>';
    h += '<div class="mod-results-body" style="max-height:350px">';
    h += '<table class="mod-tbl"><thead><tr>';
    h += '<th>Subject</th>';
    cats.forEach(function(c) { h += '<th class="num" style="color:' + catCols[c] + '">' + catNames[c] + '</th>'; });
    h += '<th class="num">total</th>';
    h += '</tr></thead><tbody>';
    var subjects = Object.keys(r.stratification.by_subject).sort();
    subjects.forEach(function(s) {
      var sv = r.stratification.by_subject[s];
      var total = 0;
      cats.forEach(function(c) { total += (sv[c] || 0); });
      h += '<tr><td style="color:var(--cyan)">' + escHtml(s.replace(/_/g, ' ')) + '</td>';
      cats.forEach(function(c) {
        var n = sv[c] || 0;
        h += '<td class="num"' + (n > 0 ? ' style="color:' + catCols[c] + '"' : '') + '>' + n + '</td>';
      });
      h += '<td class="num" style="color:var(--text-2)">' + total + '</td></tr>';
    });
    h += '</tbody></table></div>';
  }

  // Probe table — probe-centric view of how the session engaged each
  // probe in the active set. Replaces the prior token-CV table.
  // Click any column header (except cat mix) to sort. Click again to
  // reverse direction. Default: CV ascending.
  // Filter row above the table hides probes below the engagement
  // threshold (signal-clearing, not data-pruning — the underlying
  // probe set is preserved on the state object).
  if (r.probes && r.probes.length) {
    h += '<div class="mod-section-title">Probes by Engagement</div>';
    h += '<div class="mod-results-body" style="max-height:520px">';
    h += '<div id="dsProbeFilterRow" style="display:flex;align-items:center;gap:8px;padding:6px 0 10px 0;font-size:11px;color:var(--text-2);border-bottom:1px solid var(--border);margin-bottom:8px">';
    h +=   '<span>Show:</span>';
    h +=   '<button data-filter="all"       class="ds-pf-btn">all</button>';
    h +=   '<button data-filter="threshold" class="ds-pf-btn">n ≥</button>';
    h +=   '<input id="dsProbeFilterK" type="number" min="1" max="999" value="1" ';
    h +=     'style="width:48px;background:#22272E;border:1px solid var(--border);color:var(--text-0);';
    h +=     'border-radius:3px;padding:2px 4px;font-family:inherit;font-size:11px">';
    h +=   '<span id="dsProbeFilterCount" style="margin-left:auto;color:var(--text-3)"></span>';
    h += '</div>';
    h += '<div id="dsProbeTableMount"></div>';
    h += '</div>';

    // Cache the data and initial sort/filter state on a module-scoped
    // object so re-renders don't refetch.
    window._dsProbeState = {
      probes: r.probes,
      subjects: r.subjects || [],
      level_names: r.level_names || [],
      sortKey: 'cv',
      sortDir: 'asc',
      filterMode: 'threshold',  // 'all' | 'threshold'
      filterK: 1,
    };
    // Defer the first render to next tick so mount nodes exist in DOM.
    setTimeout(renderDsProbeTable, 0);
  }

  h += '</div>';
  return h;
}

// ─── Sortable Probes-by-Engagement table ──────────────────────
// State lives on window._dsProbeState, set by renderDomainSurfaceResults.
// Header clicks toggle direction when re-clicking the active column;
// otherwise switch to the new column with a sensible default direction.

function renderDsProbeTable() {
  var mount = document.getElementById('dsProbeTableMount');
  if (!mount || !window._dsProbeState) return;
  var st = window._dsProbeState;

  var SCOL = ["#D55E00","#56B4E9","#CC79A7","#009E73","#0072B2",
              "#E69F00","#F0E442","#882255","#44AA99","#AA4499"];
  var subjIdx = {};
  st.subjects.forEach(function(s, i){ subjIdx[s] = i; });

  // Column metadata: key, label, numeric flag, getter for sort value.
  // cat_mix has no scalar — it's omitted from the sortable set.
  var COLS = [
    { key:'text',      label:'Probe',    numeric:false, get:function(p){ return p.text || ''; } },
    { key:'subject',   label:'Subject',  numeric:false, get:function(p){ return p.subject || ''; } },
    { key:'level',     label:'Level',    numeric:true,  get:function(p){ return p.level; } },
    { key:'cv',        label:'CV',       numeric:true,  get:function(p){ return p.cv; }, classes:'num' },
    { key:'n',         label:'n',        numeric:true,  get:function(p){ return p.n; }, classes:'num' },
    { key:'mean_dist', label:'dist μ',   numeric:true,  get:function(p){ return p.mean_dist; }, classes:'num' },
  ];

  // Filter step: apply current filter mode before sorting. The full
  // probe set is preserved on st.probes — we only narrow the rendered
  // rows. Threshold uses st.filterK; "all" passes everything through.
  var filtered;
  if (st.filterMode === 'threshold') {
    var k = Math.max(1, parseInt(st.filterK, 10) || 1);
    filtered = st.probes.filter(function(p){ return (p.n || 0) >= k; });
  } else {
    filtered = st.probes.slice();
  }

  // Build sorted rows. Unengaged probes (n=0) participate in the sort
  // like everything else when the filter is "all"; only the dimming
  // carries the n=0 signal.
  var rows = filtered.sort(function(a, b){
    var col = COLS.find(function(c){ return c.key === st.sortKey; });
    if (!col) return 0;
    var av = col.get(a), bv = col.get(b);
    var cmp;
    if (col.numeric) cmp = (av || 0) - (bv || 0);
    else cmp = String(av).localeCompare(String(bv));
    return st.sortDir === 'asc' ? cmp : -cmp;
  });

  // Header
  var h = '<table class="mod-tbl"><thead><tr>';
  COLS.forEach(function(c){
    var active = c.key === st.sortKey;
    var arrow = active ? (st.sortDir === 'asc' ? ' ▲' : ' ▼') : '';
    var headerStyle = 'cursor:pointer;user-select:none' +
                      (active ? ';color:var(--text-0);font-weight:600' : '');
    h += '<th class="' + (c.classes || '') + '" ' +
         'data-sort-key="' + c.key + '" ' +
         'style="' + headerStyle + '">' +
         escHtml(c.label) + arrow + '</th>';
  });
  // cat mix header — non-sortable
  h += '<th>cat mix</th>';
  h += '</tr></thead><tbody>';

  rows.forEach(function(p){
    var dimmed = p.n === 0;
    var rowStyle = dimmed ? 'style="opacity:0.35"' : '';
    var swatch = SCOL[(subjIdx[p.subject] || 0) % SCOL.length];
    var lvName = (st.level_names && st.level_names[p.level]) || ('L' + p.level);
    var cvCol = dimmed ? 'var(--text-3)'
                : (p.cv < 0.35 ? 'var(--green)'
                   : (p.cv < 0.6 ? 'var(--orange)' : 'var(--red)'));

    var cm = p.cat_mix || {b:0,m:0,h:0,j:0};
    var total = cm.b + cm.m + cm.h + cm.j;
    var mix = '';
    if (total > 0) {
      mix += '<span style="display:inline-flex;height:10px;width:80px;border-radius:2px;overflow:hidden;background:#2D333B;vertical-align:middle">';
      ["b","m","h","j"].forEach(function(c){
        if (cm[c] > 0) {
          var pct = (cm[c] / total) * 100;
          mix += '<span title="' + c + ': ' + cm[c] + '" style="background:var(--cat-' +
                 {b:'benign',m:'mild',h:'harmful',j:'jailbreak'}[c] +
                 ');height:100%;width:' + pct.toFixed(2) + '%"></span>';
        }
      });
      mix += '</span>';
    } else {
      mix = '<span style="color:var(--text-3)">—</span>';
    }

    h += '<tr ' + rowStyle + '>';
    h += '<td style="color:var(--cyan);font-weight:500">' + escHtml(p.text) + '</td>';
    h += '<td><span style="display:inline-block;width:8px;height:8px;border-radius:1px;background:' +
         swatch + ';vertical-align:middle;margin-right:6px"></span>' +
         escHtml(p.subject.replace(/_/g, ' ')) + '</td>';
    h += '<td style="color:var(--text-2)">' + escHtml(lvName) + '</td>';
    h += '<td class="num" style="color:' + cvCol + '">' + p.cv.toFixed(3) + '</td>';
    h += '<td class="num">' + p.n + '</td>';
    h += '<td class="num">' + p.mean_dist.toFixed(3) + '</td>';
    h += '<td>' + mix + '</td>';
    h += '</tr>';
  });
  h += '</tbody></table>';
  mount.innerHTML = h;

  // Wire header clicks. Re-clicking the active column flips direction.
  // Switching to a new column picks a default direction: ascending for
  // text-y columns (Probe/Subject/Level) and CV (low-CV-first matches
  // the prior framing); descending for n and dist μ (largest first is
  // usually what you want).
  var defaultDirs = { text:'asc', subject:'asc', level:'asc',
                       cv:'asc', n:'desc', mean_dist:'desc' };
  mount.querySelectorAll('th[data-sort-key]').forEach(function(th){
    th.addEventListener('click', function(){
      var key = th.dataset.sortKey;
      if (key === st.sortKey) {
        st.sortDir = (st.sortDir === 'asc') ? 'desc' : 'asc';
      } else {
        st.sortKey = key;
        st.sortDir = defaultDirs[key] || 'asc';
      }
      renderDsProbeTable();
    });
  });

  // Filter UI: paint active button, update count, sync K input value,
  // wire handlers (idempotent — handlers are re-bound each render).
  var filterRow = document.getElementById('dsProbeFilterRow');
  var kInput = document.getElementById('dsProbeFilterK');
  var countEl = document.getElementById('dsProbeFilterCount');
  if (filterRow) {
    var btns = filterRow.querySelectorAll('.ds-pf-btn');
    btns.forEach(function(b){
      var active = (b.dataset.filter === st.filterMode);
      b.style.cssText =
        'border:1px solid ' + (active ? 'var(--cyan)' : 'var(--border)') + ';' +
        'background:' + (active ? 'rgba(86,180,233,0.12)' : 'transparent') + ';' +
        'color:' + (active ? 'var(--cyan)' : 'var(--text-2)') + ';' +
        'border-radius:3px;padding:2px 8px;cursor:pointer;font-family:inherit;font-size:11px';
      b.onclick = function(){
        st.filterMode = b.dataset.filter;
        renderDsProbeTable();
      };
    });
  }
  if (kInput) {
    kInput.value = st.filterK;
    kInput.style.opacity = (st.filterMode === 'threshold') ? '1' : '0.45';
    kInput.oninput = function(){
      var v = parseInt(kInput.value, 10);
      st.filterK = (isFinite(v) && v >= 1) ? v : 1;
      // Editing K implies the user wants the threshold filter active.
      if (st.filterMode !== 'threshold') st.filterMode = 'threshold';
      renderDsProbeTable();
    };
  }
  if (countEl) {
    var totalProbes = st.probes.length;
    var shown = rows.length;
    countEl.textContent = shown === totalProbes
      ? (totalProbes + ' probes')
      : (shown + ' of ' + totalProbes + ' probes');
  }
}

function popoutDomainSurface(){
  if(!_dsSurfaceData||!_dsSurfaceData.ladder)return;
  var pw=1100,ph=780,left=Math.round((screen.width-pw)/2),top=Math.round((screen.height-ph)/2);
  window.open('/domain_surface_viz','_blank','width='+pw+',height='+ph+',left='+left+',top='+top+',scrollbars=yes');
}

// ─── Probe Generator Results Renderer ──────────────────────────

function renderProbeGeneratorResults(r) {
  var h = '<div class="mod-results">';
  var outputFile = r.output_file || '';

  // Summary
  h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Summary</div>';
  h += '<div class="mod-results-body"><div class="mod-summary">';
  h += '<div class="mod-stat"><span class="mod-stat-val">' + r.total_raw_tokens + '</span><span class="mod-stat-label">Raw Tokens</span></div>';
  h += '<div class="mod-stat"><span class="mod-stat-val">' + r.total_shared_removed + '</span><span class="mod-stat-label">Shared (removed)</span></div>';
  h += '<div class="mod-stat"><span class="mod-stat-val">' + r.total_discriminative + '</span><span class="mod-stat-label">Discriminative</span></div>';
  h += '<div class="mod-stat"><span class="mod-stat-val">' + r.queries_per_subject + '</span><span class="mod-stat-label">Queries/Subject</span></div>';
  h += '<div class="mod-stat"><span class="mod-stat-val">' + r.sampling_seconds + 's</span><span class="mod-stat-label">Sampling Time</span></div>';
  h += '<div class="mod-stat"><span class="mod-stat-val">' + escHtml(r.output_file) + '</span><span class="mod-stat-label">Output File</span></div>';
  if (r.catalog_file) h += '<div class="mod-stat"><span class="mod-stat-val">' + escHtml(r.catalog_file) + '</span><span class="mod-stat-label">Catalog File</span></div>';
  h += '</div></div>';

  // Embed status row
  if (outputFile) {
    var ae = r.auto_embed;
    h += '<div style="display:flex;align-items:center;gap:10px;padding:10px 16px;border-bottom:1px solid var(--border);background:#1a2233">';
    if (ae && ae.applied) {
      // Auto-embed succeeded — show green status
      h += '<span style="color:var(--green);font-size:12px;font-weight:600">⬡ Embedded &amp; Activated</span>';
      h += '<span style="color:var(--text-2);font-size:11px;flex:1">'
        + ae.n_probes + ' probes, ' + ae.n_subjects + ' subjects, ' + ae.n_levels + ' levels'
        + ' — depths L' + (ae.depths || []).join(', L')
        + '</span>';
    } else {
      // Auto-embed didn't run or failed — show manual button
      h += '<button id="pgEmbedBtn" onclick="embedActiveProbes(' + JSON.stringify(outputFile) + ')" style="padding:8px 18px;border:none;color:#000;background:var(--cyan);border-radius:4px;cursor:pointer;font-family:inherit;font-size:12px;font-weight:600">⬡ Embed &amp; Activate Probe Set</button>';
      var statusMsg = 'Ready to embed <code>' + escHtml(outputFile) + '</code> into the loaded model';
      if (ae && ae.error) statusMsg = '<span style="color:var(--yellow)">Auto-embed skipped: ' + escHtml(ae.error) + '</span>';
      h += '<span id="pgEmbedStatus" style="color:var(--text-2);font-size:11px;flex:1">' + statusMsg + '</span>';
    }
    h += '</div>';
  }

  // Per-subject table
  h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Per Subject Breakdown</div>';
  h += '<div class="mod-results-body" style="max-height:500px">';
  h += '<table class="data-table" style="width:100%;font-size:11px">';
  h += '<tr><th style="text-align:left">Subject</th><th>Raw</th><th>Discriminative</th><th style="text-align:left">Top Words</th></tr>';

  var subjects = r.subjects || [];
  var ps = r.per_subject || {};
  subjects.forEach(function(subj) {
    var s = ps[subj] || {};
    var top = (s.top_20 || []).slice(0, 10).join(', ');
    var pct = s.raw_tokens > 0 ? Math.round(100 * s.discriminative_tokens / s.raw_tokens) : 0;
    h += '<tr>';
    h += '<td style="text-align:left;font-weight:600;color:var(--text-0)">' + escHtml(subj.replace('_', ' ')) + '</td>';
    h += '<td>' + (s.raw_tokens || 0) + '</td>';
    h += '<td><span style="color:var(--green)">' + (s.discriminative_tokens || 0) + '</span> <span style="color:var(--text-3);font-size:10px">(' + pct + '%)</span></td>';
    h += '<td style="text-align:left;color:var(--text-2);font-size:10px">' + escHtml(top) + '</td>';
    h += '</tr>';
  });

  h += '</table></div>';

  // Shared tokens list
  var shared = r.shared_tokens || [];
  if (shared.length > 0) {
    h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Shared Tokens Removed (' + shared.length + ')</div>';
    h += '<div class="mod-results-body collapsed" style="max-height:300px">';
    h += '<div style="padding:10px 16px;font-size:10px;color:var(--text-2);font-family:var(--mono);line-height:1.8;word-break:break-all">';
    h += escHtml(shared.join(', '));
    h += '</div></div>';
  }

  // Inference catalog
  var catalog = r.catalog || {};
  var catalogKeys = Object.keys(catalog);
  if (catalogKeys.length > 0) {
    h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Inference Catalog (' + (r.total_queries||0) + ' queries' + (r.catalog_file ? ' — ' + escHtml(r.catalog_file) : '') + ')</div>';
    h += '<div class="mod-results-body collapsed" style="max-height:600px">';
    catalogKeys.forEach(function(key) {
      var parts = key.split('|');
      var cls = parts[0] || '?';
      var sub = parts[1] || '?';
      var entries = catalog[key] || [];
      h += '<div style="border-bottom:1px solid var(--border);padding:8px 12px">';
      h += '<div style="cursor:pointer;font-weight:600;color:var(--text-0);font-size:11px" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">';
      h += escHtml(cls.replace('_',' ')) + ' × ' + escHtml(sub.replace('_',' '));
      h += ' <span style="color:var(--text-3);font-weight:400">(' + entries.length + ' queries)</span></div>';
      h += '<div class="collapsed" style="max-height:400px;overflow-y:auto">';
      entries.forEach(function(e) {
        var nTok = (e.tokens_extracted && e.tokens_extracted.length) || e.n_tokens || 0;
        var respFull = e.response || '';
        var respPrev = respFull.length > 300 ? respFull.slice(0,300) + '…' : respFull;
        h += '<div style="padding:6px 0;border-top:1px solid var(--border);font-size:10px">';
        h += '<span style="color:var(--cyan);font-weight:600">Q' + e.q + '</span> ';
        h += '<span style="color:var(--text-3)">(' + nTok + ' tokens)</span>';
        h += '<div style="color:var(--text-2);margin-top:3px;line-height:1.5;white-space:pre-wrap;word-break:break-word">' + escHtml(respPrev) + '</div>';
        h += '</div>';
      });
      h += '</div></div>';
    });
    h += '</div>';
  }

  h += '</div>';
  return h;
}

function popoutCorrectionPrism(){
  var pw=1200,ph=900,left=Math.round((screen.width-pw)/2),top=Math.round((screen.height-ph)/2);
  window.open('/correction_prism_viz','_blank','width='+pw+',height='+ph+',left='+left+',top='+top+',scrollbars=yes');
}

function popoutProbeDiagnostic(file){
  var pw=1100,ph=900,left=Math.round((screen.width-pw)/2),top=Math.round((screen.height-ph)/2);
  var url = '/probe_diagnostic_viz';
  if (file) url += '?file=' + encodeURIComponent(file);
  window.open(url,'_blank','width='+pw+',height='+ph+',left='+left+',top='+top+',scrollbars=yes');
}

async function embedActiveProbes(filename){
  if (!filename) return;
  var btn = document.getElementById('pgEmbedBtn');
  var status = document.getElementById('pgEmbedStatus');
  if (!btn || !status) return;
  btn.disabled = true;
  btn.textContent = '⬡ Embedding...';
  btn.style.opacity = '0.6';
  status.textContent = 'Submitting embed request...';
  status.style.color = 'var(--text-2)';
  try {
    var r = await(await fetch('/api/modules/probe_generator/embed_active', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({filename: filename}),
    })).json();
    if (!r.ok) {
      status.textContent = 'Error: ' + (r.error || 'Unknown');
      status.style.color = 'var(--red)';
      btn.disabled = false;
      btn.style.opacity = '1';
      btn.textContent = '⬡ Embed & Activate Probe Set';
      return;
    }
    // Wait for pg_embed_status via SSE
    var pgResult = await new Promise(function(resolve) {
      _sseWaiters.pg_embed_status.push(function(evt) { resolve(evt); });
    });
    btn.disabled = false;
    btn.style.opacity = '1';
    btn.textContent = '⬡ Embed & Activate Probe Set';
    if (pgResult.error) {
      status.innerHTML = '<span style="color:var(--red)">✗ ' + escHtml(pgResult.error) + '</span>';
    } else if (pgResult.result) {
      var res = pgResult.result;
      status.innerHTML = '<span style="color:var(--green)">✓ Embedded and activated</span>'
        + ' — ' + res.n_probes + ' probes, ' + res.n_subjects + ' subjects, ' + res.n_levels + ' levels'
        + ' — depths L' + (res.depths || []).join(', L');
      log('Probe set applied: ' + res.filename + ' (' + res.n_probes + ' probes)', 'done');
      if (typeof loadProbeFiles === 'function') loadProbeFiles();
      if (typeof playChime === 'function') playChime();
    }
  } catch(e) {
    status.textContent = 'Failed: ' + e.message;
    status.style.color = 'var(--red)';
    btn.disabled = false;
    btn.style.opacity = '1';
    btn.textContent = '⬡ Embed & Activate Probe Set';
  }
}

// ─── Correction Prism Results Renderer ──────────────────────────
//
// Renders the signed heatmap from correction_prism. Each cell carries
// a signed scalar (sum of per-probe responses by default); positive =
// the correction excites the probes in that cell, negative = it opposes
// them. Diverging blue↔white↔red colormap with autoscale ±vmax so
// white sits at exactly zero.
//
// The per-cell composition panel exposes the dispersion that the cell
// scalar hides — n_excited / n_opposed / n_neutral counts and the
// individual probe responses, sortable.

function renderCorrectionPrismResults(r) {
  var h = '<div class="mod-results">';
  var subjects = r.subjects || [];
  var levels = r.levels || [];
  var subjShort = r.subj_short || subjects;
  var agg = r.aggregate || [];
  var variance = r.variance || [];
  var cellDetails = r.cell_details || {};
  var cfg = r.config || {};
  var decomp = r.decomposition || {};
  var circOrder = r.circuit_order || [];
  var circLabels = r.circuit_labels || {};
  var diag = r.diagnostic || {ok: true};
  var primary = r.primary_circuit || cfg.circuit || 'full';

  // Stash cell details + subject/level lookups on window so prismShowCell
  // can find them. Setting them here (synchronously, before innerHTML
  // injection) is necessary because <script> tags concatenated into an
  // innerHTML string do NOT execute — that's a browser security behavior.
  window.__prismCells = cellDetails;
  window.__prismSubjects = subjects;
  window.__prismLevels = levels;

  // Failure banner.
  if (diag.ok === false) {
    h += '<div style="margin:8px 12px;padding:12px 14px;border:1px solid #D55E00;border-radius:6px;background:rgba(213,94,0,0.08);color:var(--text-0);font-size:12px;line-height:1.5">';
    h += '<div style="font-weight:600;color:#E69F00;margin-bottom:4px">⚠ Prism failed to compute</div>';
    h += '<div>' + escHtml(diag.message || '') + '</div>';
    h += '</div>';
  }

  // Symmetric scale around zero so white = exactly 0.
  function divergingScale(grid) {
    var vmax = 0;
    for (var si = 0; si < grid.length; si++) {
      var row = grid[si] || [];
      for (var li = 0; li < row.length; li++) {
        var a = Math.abs(row[li] || 0);
        if (a > vmax) vmax = a;
      }
    }
    if (vmax < 1e-10) vmax = 1e-10;
    return {vmax: vmax};
  }

  // Blue (negative) ↔ neutral (zero) ↔ red (positive).
  // RGB stops chosen to match the existing dark-themed UI palette.
  function divergingColor(val, sc) {
    var t = Math.max(-1, Math.min(1, val / sc.vmax));
    if (t > 0) {
      // 0 → near-bg, 1 → saturated red
      var r1 = Math.round(30 + t * 200);
      var g1 = Math.round(30 + t * 30);
      var b1 = Math.round(36 + t * 20);
      return 'rgb(' + r1 + ',' + g1 + ',' + b1 + ')';
    }
    if (t < 0) {
      // 0 → near-bg, -1 → saturated blue
      var s = -t;
      var r2 = Math.round(30 + s * 30);
      var g2 = Math.round(30 + s * 80);
      var b2 = Math.round(36 + s * 200);
      return 'rgb(' + r2 + ',' + g2 + ',' + b2 + ')';
    }
    return 'rgb(30,30,36)';
  }

  function divergingTextColor(val, sc) {
    var t = Math.abs(val) / sc.vmax;
    return t > 0.45 ? '#fff' : 'var(--text-2)';
  }

  // Render one signed heatmap. Cells are clickable and emit a JS call
  // to popDetail(si, li) — defined later — so the user can drill in.
  function renderHeatmap(grid, sc, digits, makeCellId) {
    var th = '<table style="border-collapse:collapse;font-size:10px;font-family:var(--mono)">';
    th += '<tr><th style="padding:3px 6px"></th>';
    for (var li = 0; li < levels.length; li++) {
      th += '<th style="padding:3px 5px;color:var(--text-2);font-weight:500;text-align:center;font-size:9px">'
          + escHtml(levels[li]) + '</th>';
    }
    th += '</tr>';
    for (var si = 0; si < subjects.length; si++) {
      th += '<tr>';
      th += '<td style="padding:3px 6px;color:var(--text-1);font-weight:600;text-align:left;font-size:9px;white-space:nowrap">'
          + escHtml(subjShort[si]) + '</td>';
      for (var li = 0; li < levels.length; li++) {
        var val = (grid[si] && grid[si][li] != null) ? grid[si][li] : 0;
        var bg = divergingColor(val, sc);
        var tc = divergingTextColor(val, sc);
        var cellId = makeCellId ? makeCellId(si, li) : '';
        var clickable = cellId
          ? ' onclick="prismShowCell(\'' + cellId + '\')" style="cursor:pointer;'
          : ' style="';
        th += '<td' + clickable
            + 'padding:3px 5px;text-align:center;background:' + bg
            + ';color:' + tc + ';font-size:9px">'
            + (val >= 0 ? '+' : '') + val.toFixed(digits) + '</td>';
      }
      th += '</tr>';
    }
    th += '</table>';
    return th;
  }

  // Color legend strip showing the diverging scale.
  function renderLegend(sc) {
    var stops = '';
    var n = 11;
    for (var i = 0; i < n; i++) {
      var v = -sc.vmax + (2 * sc.vmax * i / (n - 1));
      stops += '<div style="flex:1;height:8px;background:'
            + divergingColor(v, sc) + '"></div>';
    }
    var leg = '<div style="display:flex;flex-direction:column;gap:2px;font-size:9px;color:var(--text-2);font-family:var(--mono);margin-top:4px;max-width:240px">';
    leg += '<div style="display:flex">' + stops + '</div>';
    leg += '<div style="display:flex;justify-content:space-between">';
    leg += '<span>−' + sc.vmax.toFixed(3) + ' (opposed)</span>';
    leg += '<span>0</span>';
    leg += '<span>+' + sc.vmax.toFixed(3) + ' (excited)</span>';
    leg += '</div></div>';
    return leg;
  }

  // ── Top header: primary heatmap with view toggle ──
  // Three views, all sharing the same diverging colormap & cell click:
  //   primary   = the chosen circuit's heatmap (correction lens on)
  //   baseline  = the same prompt's response with ΔW = I (no correction)
  //   diff      = primary − baseline, isolating the correction-induced
  //               response
  // The baseline + diff views are only available if include_baseline ran.
  var hasBaseline = !!(decomp._baseline && decomp._baseline.aggregate);

  function gridSubtract(a, b) {
    var out = [];
    for (var si = 0; si < a.length; si++) {
      var row = [];
      for (var li = 0; li < (a[si]||[]).length; li++) {
        row.push((a[si][li] || 0) - ((b[si]||[])[li] || 0));
      }
      out.push(row);
    }
    return out;
  }
  function gridForView(view) {
    var p = (decomp[primary] && decomp[primary].aggregate) || agg;
    if (view === 'baseline' && hasBaseline) return decomp._baseline.aggregate;
    if (view === 'diff' && hasBaseline) {
      return gridSubtract(p, decomp._baseline.aggregate);
    }
    return p;
  }

  // Stash on window so the toggle handler can recompute without
  // re-running anything. Mirrors the cell-details stashing pattern.
  window.__prismDecomp = decomp;
  window.__prismPrimary = primary;
  window.__prismLevels = levels;
  window.__prismSubjects = subjects;
  window.__prismSubjShort = subjShort;
  window.__prismCircLabels = circLabels;
  window.__prismHasBaseline = hasBaseline;

  var primaryAgg = gridForView('primary');
  var primarySc = divergingScale(primaryAgg);

  h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">';
  h += 'Prism Heatmap — ' + escHtml(circLabels[primary] || primary);
  h += '<button class="btn-popout" style="margin-left:auto" onclick="event.stopPropagation();popoutCorrectionPrism()">↗ Open in Explorer</button>';
  h += '</div>';
  h += '<div class="mod-results-body">';
  h += '<div style="padding:10px;display:flex;flex-direction:column;gap:8px" id="prism-primary-wrap">';

  // View toggle (only meaningful if baseline was computed)
  if (hasBaseline) {
    h += '<div style="display:flex;gap:6px;font-family:var(--mono);font-size:10px">';
    h += '<button class="prism-view-btn active" data-view="primary" onclick="prismSwitchView(\'primary\',this)" '
       + 'style="padding:3px 10px;background:var(--cyan);color:#000;border:none;border-radius:3px;cursor:pointer">'
       + 'Correction'
       + '</button>';
    h += '<button class="prism-view-btn" data-view="baseline" onclick="prismSwitchView(\'baseline\',this)" '
       + 'style="padding:3px 10px;background:var(--bg-2);color:var(--text-1);border:1px solid var(--border);border-radius:3px;cursor:pointer">'
       + 'Baseline (ΔW = I)'
       + '</button>';
    h += '<button class="prism-view-btn" data-view="diff" onclick="prismSwitchView(\'diff\',this)" '
       + 'style="padding:3px 10px;background:var(--bg-2);color:var(--text-1);border:1px solid var(--border);border-radius:3px;cursor:pointer">'
       + 'Correction − Baseline'
       + '</button>';
    h += '</div>';
  }

  h += '<div style="font-size:11px;color:var(--text-2);line-height:1.5" id="prism-primary-caption">';
  h += 'Each cell is the <b>' + escHtml(cfg.cell_aggregation || 'sum') + '</b> of '
     + 'signed probe responses in that (subject, level). ';
  h += '<span style="color:#ff6060">Red</span> = correction excites the cell\'s probes; ';
  h += '<span style="color:#6090ff">blue</span> = correction opposes them. ';
  h += 'Click a cell to see its probe composition.';
  h += '</div>';
  h += '<div id="prism-primary-heatmap">';
  h += renderHeatmap(primaryAgg, primarySc, 3,
      function(si, li){ return si + '_' + li; });
  h += '</div>';
  h += '<div id="prism-primary-legend">' + renderLegend(primarySc) + '</div>';
  h += '</div></div>';

  // ── Decomposition: all circuits side by side ──
  if (circOrder.length > 1) {
    h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">';
    h += 'Circuit Decomposition (' + circOrder.length + ' circuits)';
    h += '</div>';
    h += '<div class="mod-results-body">';
    h += '<div style="padding:10px;display:flex;flex-wrap:wrap;gap:16px">';
    circOrder.forEach(function(ck) {
      var d = decomp[ck];
      if (!d) return;
      var dAgg = d.aggregate || [];
      var sc = divergingScale(dAgg);
      var label = circLabels[ck] || ck;
      var isBaseline = (ck === '_baseline');
      h += '<div style="min-width:200px">';
      h += '<div style="font-size:11px;font-weight:600;color:'
         + (isBaseline ? 'var(--text-2)' : 'var(--cyan)')
         + ';margin-bottom:4px">'
         + escHtml(isBaseline ? 'Baseline (ΔW = I)' : label) + '</div>';
      h += renderHeatmap(dAgg, sc, 3, null);
      h += '</div>';
    });
    h += '</div></div>';
  }

  // ── Cell detail panel (populated dynamically when a cell is clicked) ──
  h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Cell Composition</div>';
  h += '<div class="mod-results-body" id="prism-cell-panel">';
  h += '<div style="padding:10px;color:var(--text-3);font-size:11px;font-style:italic">'
     + 'Click any cell in the heatmap above to inspect its probes.</div>';
  h += '</div>';

  // ── Per-subject summary ──
  h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Per Subject Summary</div>';
  h += '<div class="mod-results-body">';
  h += '<table class="data-table" style="width:100%;font-size:11px">';
  h += '<tr><th style="text-align:left">Subject</th>'
     + '<th>Mean (signed)</th><th>Mean |val|</th><th>Max |val|</th>'
     + '<th style="color:#ff6060">N excited</th>'
     + '<th style="color:#6090ff">N opposed</th></tr>';
  var ps = r.per_subject || {};
  subjects.forEach(function(subj) {
    var s = ps[subj] || {};
    var meanS = s.mean_signed || 0;
    h += '<tr>';
    h += '<td style="text-align:left;font-weight:600;color:var(--text-0)">'
       + escHtml(subj.replace(/_/g, ' ')) + '</td>';
    h += '<td style="color:'
       + (meanS > 0.05 ? '#ff8060' : meanS < -0.05 ? '#6090ff' : 'var(--text-2)')
       + '">' + (meanS >= 0 ? '+' : '') + meanS.toFixed(4) + '</td>';
    h += '<td>' + (s.mean_abs || 0).toFixed(4) + '</td>';
    h += '<td>' + (s.max_abs || 0).toFixed(4) + '</td>';
    h += '<td style="color:#ff6060">' + (s.n_excited || 0) + '</td>';
    h += '<td style="color:#6090ff">' + (s.n_opposed || 0) + '</td>';
    h += '</tr>';
  });
  h += '</table></div>';

  // ── Configuration ──
  h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Configuration</div>';
  h += '<div class="mod-results-body collapsed">';
  h += '<div style="padding:10px;font-size:10px;color:var(--text-2);line-height:1.8;font-family:var(--mono)">';
  h += 'Circuit: <span style="color:var(--text-1)">' + escHtml(cfg.circuit || primary) + '</span> · ';
  h += 'Beam reduction: <span style="color:var(--text-1)">' + escHtml(cfg.beam_reduction || 'mean') + '</span> · ';
  h += 'Prism metric: <span style="color:var(--text-1)">' + escHtml(cfg.prism_metric || 'signed_cosine') + '</span> · ';
  h += 'Cell agg: <span style="color:var(--text-1)">' + escHtml(cfg.cell_aggregation || 'sum') + '</span> · ';
  h += 'Prompt agg: <span style="color:var(--text-1)">' + escHtml(cfg.prompt_aggregation || 'mean') + '</span><br>';
  h += 'L_low: <span style="color:var(--text-1)">' + ((cfg.L_low || 0.5) * 100).toFixed(0) + '%</span> · ';
  h += 'L_high: <span style="color:var(--text-1)">' + ((cfg.L_high || 0.75) * 100).toFixed(0) + '%</span> · ';
  h += 'Layers: <span style="color:var(--text-1)">' + (cfg.n_signal_layers || 0) + '</span> · ';
  h += 'd_model: <span style="color:var(--text-1)">' + (cfg.model_dim || 0) + '</span> · ';
  h += 'Degenerate probes: <span style="color:var(--text-1)">' + (cfg.n_probes_degenerate || 0) + '</span><br>';
  if (cfg.probe_cache_used) {
    h += 'Caches: <span style="color:var(--text-1)">'
       + escHtml(cfg.probe_cache_used.L_low || '?') + ', '
       + escHtml(cfg.probe_cache_used.L_high || '?') + '</span>';
  }
  if (cfg.active_probe_model_id) {
    h += '<br>Model: <span style="color:var(--text-1)">'
       + escHtml(cfg.active_probe_model_id) + '</span>';
  }
  h += '</div></div>';

  h += '</div>';
  return h;
}

// Click-handler for prism cells. Fills the Cell Composition panel with
// the probes of the clicked cell, sorted by |response|.
function prismShowCell(cellKey) {
  var panel = document.getElementById('prism-cell-panel');
  if (!panel) return;
  var cells = window.__prismCells || {};
  var cell = cells[cellKey];
  if (!cell) {
    panel.innerHTML = '<div style="padding:10px;color:var(--text-3)">No data for cell ' + cellKey + '.</div>';
    return;
  }

  var parts = cellKey.split('_');
  var si = parseInt(parts[0], 10);
  var li = parseInt(parts[1], 10);
  var subj = (window.__prismSubjects || [])[si] || ('subject ' + si);
  var lvl = (window.__prismLevels || [])[li] || ('level ' + li);

  var probes = (cell.probes || []).slice().sort(function(a, b) {
    return Math.abs(b.response || 0) - Math.abs(a.response || 0);
  });

  var v = cell.cell_value || 0;
  var html = '<div style="padding:10px;font-size:11px">';
  html += '<div style="margin-bottom:6px">';
  html += '<b>' + escHtml(subj.replace(/_/g, ' ')) + ' / ' + escHtml(lvl) + '</b> · ';
  html += 'cell value <span style="color:'
       + (v > 0 ? '#ff8060' : v < 0 ? '#6090ff' : 'var(--text-2)')
       + ';font-family:var(--mono)">' + (v >= 0 ? '+' : '') + v.toFixed(4) + '</span>';
  html += '</div>';

  html += '<div style="font-size:10px;color:var(--text-2);margin-bottom:6px;line-height:1.5">';
  html += 'Probes: ' + (cell.n_probes || 0) + ' active';
  if (cell.n_probes_degenerate) html += ' (' + cell.n_probes_degenerate + ' degenerate filtered)';
  html += ' · <span style="color:#ff6060">' + (cell.n_excited || 0) + ' excited</span>';
  html += ' · <span style="color:#6090ff">' + (cell.n_opposed || 0) + ' opposed</span>';
  html += ' · <span style="color:var(--text-3)">' + (cell.n_neutral || 0) + ' neutral</span>';
  html += '<br>response range [' + (cell.probe_response_min || 0).toFixed(4)
       +  ', ' + (cell.probe_response_max || 0).toFixed(4) + ']';
  html += ' · median ' + (cell.probe_response_median || 0).toFixed(4);
  html += ' · variance ' + (cell.cell_variance || 0).toFixed(6);
  html += '</div>';

  html += '<table class="data-table" style="width:100%;font-size:10px;font-family:var(--mono)">';
  html += '<tr><th style="text-align:left">Probe</th><th>Response</th><th>Direction</th></tr>';
  probes.forEach(function(p) {
    var resp = p.response || 0;
    var color = (p.direction === 'excited') ? '#ff6060'
              : (p.direction === 'opposed') ? '#6090ff'
              : 'var(--text-3)';
    html += '<tr>';
    html += '<td style="text-align:left;color:var(--text-1)">' + escHtml(p.text || ('#' + p.probe_idx)) + '</td>';
    html += '<td style="color:' + color + '">' + (resp >= 0 ? '+' : '') + resp.toFixed(4) + '</td>';
    html += '<td style="color:' + color + '">' + escHtml(p.direction || '—') + '</td>';
    html += '</tr>';
  });
  html += '</table></div>';

  panel.innerHTML = html;
  // Make sure the panel is open.
  panel.classList.remove('collapsed');
}

// View toggle for the primary heatmap. Re-renders the heatmap and legend
// (and rewrites the caption) without re-running the module. Reads from
// window.__prism* state stashed by renderCorrectionPrismResults.
function prismSwitchView(view, btn) {
  var decomp = window.__prismDecomp || {};
  var primary = window.__prismPrimary;
  var levels = window.__prismLevels || [];
  var subjects = window.__prismSubjects || [];
  var subjShort = window.__prismSubjShort || subjects;
  var hasBaseline = !!window.__prismHasBaseline;

  // Resolve the grid to display.
  var primaryAgg = (decomp[primary] && decomp[primary].aggregate) || [];
  var grid;
  var captionExtra = '';
  if (view === 'baseline' && hasBaseline) {
    grid = decomp._baseline.aggregate;
    captionExtra = ' (showing baseline: ΔW = I, no correction lens applied — '
                 + 'this is the prompt\'s field decomposed against the probe '
                 + 'lattice with the correction switched off)';
  } else if (view === 'diff' && hasBaseline) {
    var b = decomp._baseline.aggregate;
    grid = [];
    for (var si = 0; si < primaryAgg.length; si++) {
      var row = [];
      for (var li = 0; li < (primaryAgg[si]||[]).length; li++) {
        row.push((primaryAgg[si][li] || 0) - ((b[si]||[])[li] || 0));
      }
      grid.push(row);
    }
    captionExtra = ' (showing correction − baseline: this isolates the '
                 + 'cell-by-cell response that is uniquely attributable to '
                 + 'fine-tuning, with the prompt\'s pre-existing geometry '
                 + 'subtracted out)';
  } else {
    grid = primaryAgg;
  }

  // Self-contained color/scale helpers (mirror the closure-scoped ones in
  // renderCorrectionPrismResults so the toggle doesn't depend on its closure).
  function _scale(g) {
    var vmax = 0;
    for (var si = 0; si < g.length; si++) {
      var r = g[si] || [];
      for (var li = 0; li < r.length; li++) {
        var a = Math.abs(r[li] || 0);
        if (a > vmax) vmax = a;
      }
    }
    if (vmax < 1e-10) vmax = 1e-10;
    return {vmax: vmax};
  }
  function _color(val, sc) {
    var t = Math.max(-1, Math.min(1, val / sc.vmax));
    if (t > 0) return 'rgb(' + Math.round(30 + t * 200) + ','
                       + Math.round(30 + t * 30) + ','
                       + Math.round(36 + t * 20) + ')';
    if (t < 0) {
      var s = -t;
      return 'rgb(' + Math.round(30 + s * 30) + ','
                    + Math.round(30 + s * 80) + ','
                    + Math.round(36 + s * 200) + ')';
    }
    return 'rgb(30,30,36)';
  }
  function _textColor(val, sc) {
    return (Math.abs(val) / sc.vmax) > 0.45 ? '#fff' : 'var(--text-2)';
  }

  var sc = _scale(grid);

  // Heatmap HTML. Cells stay clickable so cell-composition still works
  // in any view (it's keyed off the underlying probe responses, not the
  // display grid).
  var th = '<table style="border-collapse:collapse;font-size:10px;font-family:var(--mono)">';
  th += '<tr><th style="padding:3px 6px"></th>';
  for (var li = 0; li < levels.length; li++) {
    th += '<th style="padding:3px 5px;color:var(--text-2);font-weight:500;text-align:center;font-size:9px">'
        + escHtml(levels[li]) + '</th>';
  }
  th += '</tr>';
  for (var si = 0; si < subjects.length; si++) {
    th += '<tr>';
    th += '<td style="padding:3px 6px;color:var(--text-1);font-weight:600;text-align:left;font-size:9px;white-space:nowrap">'
        + escHtml(subjShort[si]) + '</td>';
    for (var li = 0; li < levels.length; li++) {
      var val = (grid[si] && grid[si][li] != null) ? grid[si][li] : 0;
      th += '<td onclick="prismShowCell(\'' + si + '_' + li + '\')" '
          + 'style="cursor:pointer;padding:3px 5px;text-align:center;background:'
          + _color(val, sc) + ';color:' + _textColor(val, sc)
          + ';font-size:9px">'
          + (val >= 0 ? '+' : '') + val.toFixed(3) + '</td>';
    }
    th += '</tr>';
  }
  th += '</table>';
  document.getElementById('prism-primary-heatmap').innerHTML = th;

  // Legend
  var stops = '';
  for (var i = 0; i < 11; i++) {
    var v = -sc.vmax + (2 * sc.vmax * i / 10);
    stops += '<div style="flex:1;height:8px;background:' + _color(v, sc) + '"></div>';
  }
  var legHtml = '<div style="display:flex;flex-direction:column;gap:2px;font-size:9px;color:var(--text-2);font-family:var(--mono);margin-top:4px;max-width:240px">';
  legHtml += '<div style="display:flex">' + stops + '</div>';
  legHtml += '<div style="display:flex;justify-content:space-between">';
  legHtml += '<span>−' + sc.vmax.toFixed(3) + '</span>';
  legHtml += '<span>0</span>';
  legHtml += '<span>+' + sc.vmax.toFixed(3) + '</span>';
  legHtml += '</div></div>';
  document.getElementById('prism-primary-legend').innerHTML = legHtml;

  // Caption tail — append the explanation for non-primary views.
  var caption = document.getElementById('prism-primary-caption');
  if (caption) {
    var base = 'Each cell is a signed scalar — '
             + '<span style="color:#ff6060">red</span> = excited; '
             + '<span style="color:#6090ff">blue</span> = opposed. '
             + 'Click a cell to see its probe composition.';
    caption.innerHTML = base + (captionExtra
      ? '<br><span style="color:var(--text-3)">' + captionExtra + '</span>'
      : '');
  }

  // Button active-state
  var btns = document.querySelectorAll('.prism-view-btn');
  for (var bi = 0; bi < btns.length; bi++) {
    var b = btns[bi];
    if (b === btn) {
      b.style.background = 'var(--cyan)';
      b.style.color = '#000';
      b.style.border = 'none';
    } else {
      b.style.background = 'var(--bg-2)';
      b.style.color = 'var(--text-1)';
      b.style.border = '1px solid var(--border)';
    }
  }
}

// ─── Comparative Analysis Results Renderer ──────────────────────

function renderComparativeResults(r) {
  if (r.error) {
    return '<div class="mod-results"><div class="mod-results-body" style="padding:16px;color:var(--orange)">' + escHtml(r.error) + '</div></div>';
  }

  var h = '<div class="mod-results">';
  var agg = r.aggregate || {};
  var cats = agg.categories || {};
  var plots = r.plots || [];

  // Session summary badges
  h += '<div style="padding:10px 16px;border-bottom:1px solid var(--border)">';
  h += '<div class="metrics-grid">';
  h += mc('Prompts', r.n_prompts || 0, null, function(v){return v});
  (r.categories || []).forEach(function(cat) {
    var ci = cats[cat];
    h += mc(cat, ci ? ci.n : '?', null, function(v){return v});
  });
  h += '</div></div>';

  // Category summary table
  if (Object.keys(cats).length) {
    h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Category Summary</div>';
    h += '<div class="mod-results-body">';
    h += '<div style="font-size:11px;color:var(--text-2);margin-bottom:8px;line-height:1.5">Bootstrapped estimates (5000 resamples, 95% CI). ASM (disruption), LTP (selectivity), SFD (subspace).</div>';
    h += '<table class="mod-tbl"><thead><tr><th>Category</th><th class="num">N</th><th class="num">Stress</th><th class="num">Entropy</th><th class="num">Int%</th><th class="num">Net Corr</th><th class="num">Int CV</th><th class="num" style="color:var(--orange)">Density</th><th class="num" style="color:var(--orange)">Tau</th></tr></thead><tbody>';
    for (var catName in cats) {
      var s = cats[catName];
      var m = s.metrics || {};
      h += '<tr>';
      h += '<td><span class="pill ' + pillClass(catName) + '">' + catName + '</span></td>';
      h += '<td class="num">' + s.n + '</td>';
      h += '<td class="num">' + (m.stress_score ? m.stress_score.estimate.toFixed(4) : '--') + '</td>';
      h += '<td class="num">' + (m.entropy ? m.entropy.estimate.toFixed(4) : '--') + '</td>';
      h += '<td class="num">' + (m.middle_share ? (m.middle_share.estimate * 100).toFixed(1) + '%' : '--') + '</td>';
      h += '<td class="num">' + (m.net_correction ? m.net_correction.estimate.toFixed(5) : '--') + '</td>';
      h += '<td class="num">' + (m.interior_cv ? m.interior_cv.estimate.toFixed(4) : '--') + '</td>';
      h += '<td class="num">' + (m.sfd_density_mean ? m.sfd_density_mean.estimate.toFixed(4) : '--') + '</td>';
      h += '<td class="num">' + (m.rank_displacement_tau ? m.rank_displacement_tau.estimate.toFixed(3) : '--') + '</td>';
      h += '</tr>';
    }
    h += '</tbody></table></div>';
  }

  // ── Batch visualizations — popout list ──
  // Each entry in r.plots is {key, title, desc}, sourced from the
  // Python module's BATCH_PLOTS catalog (single source of truth for
  // both behavior and presentation). Order = catalog order.
  if (plots.length) {
    h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Visualizations (' + plots.length + ')</div>';
    h += '<div class="mod-results-body">';
    h += '<div style="font-size:11px;color:var(--text-2);margin-bottom:8px">Click any item to open in a new window.</div>';
    plots.forEach(function(p) {
      var sk = 'comp_' + p.key;
      storeViz(sk, p.title, _plotHtml(p.key, p.title, p.desc));
      h += vizLabel(sk, p.title, p.desc);
    });

    // Candidate graph aggregate (computed client-side, not in r.plots)
    if (dashResults.length > 1) {
      try {
        var cgAgg = _computeGraphsChunked(dashResults);
        if (cgAgg) {
          var sk2 = 'comp_candgraph';
          storeViz(sk2, 'Candidate Graph Aggregate', cgAgg);
          h += vizLabel(sk2, 'Candidate Graph Aggregate', 'Contested positions and role switches across batch', 'var(--purple)');
        }
      } catch(e) { console.error('Candidate graph aggregate:', e); }
    }
    h += '</div>';
  }

  // Separability details (if available)
  var sep = agg.separability || {};
  if (Object.keys(sep).length) {
    h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Separability (Cohen\'s d)</div>';
    h += '<div class="mod-results-body collapsed">';
    h += '<table class="mod-tbl"><thead><tr><th>Metric</th><th class="num">d</th><th class="num">CI Low</th><th class="num">CI High</th><th class="num">Accuracy</th></tr></thead><tbody>';
    var sepKeys = Object.keys(sep).sort(function(a, b) {
      return (sep[b].effect_size ? sep[b].effect_size.estimate : 0) - (sep[a].effect_size ? sep[a].effect_size.estimate : 0);
    });
    sepKeys.forEach(function(k) {
      var s = sep[k];
      var es = s.effect_size || {};
      var ci = es.ci || [];
      var th = s.threshold || {};
      var col = Math.abs(es.estimate || 0) >= 0.8 ? 'var(--green)' : Math.abs(es.estimate || 0) >= 0.5 ? 'var(--cyan)' : 'var(--text-2)';
      h += '<tr><td style="color:var(--cyan)">' + escHtml(k) + '</td>';
      h += '<td class="num" style="color:' + col + '">' + (es.estimate || 0).toFixed(3) + '</td>';
      h += '<td class="num">' + (ci[0] != null ? ci[0].toFixed(3) : '--') + '</td>';
      h += '<td class="num">' + (ci[1] != null ? ci[1].toFixed(3) : '--') + '</td>';
      h += '<td class="num">' + (th.accuracy ? (th.accuracy * 100).toFixed(1) + '%' : '--') + '</td>';
      h += '</tr>';
    });
    h += '</tbody></table></div>';
  }

  h += '</div>';
  return h;
}

// ─── Correction Field Topology Results Renderer ─────────────────

function renderCFTResults(r) {
  if (r.error) {
    return '<div class="mod-results"><div class="mod-results-body" style="padding:16px;color:var(--orange)">' + escHtml(r.error) + '</div></div>';
  }

  var h = '<div class="mod-results">';

  // Launch button
  var nUsable = (r.n_with_displacement || 0) + (r.n_with_ltp_fallback || 0);
  var lp = r.launch_params || {};
  h += '<div style="padding:12px 16px;display:flex;align-items:center;gap:12px;border-bottom:1px solid var(--border)">';
  h += '<button class="btn btn-primary btn-sm" onclick="_cftLaunchFromModule()" style="white-space:nowrap">↗ Launch Visualization</button>';
  h += '<span style="font-size:11px;color:var(--text-2)">' + nUsable + ' prompts ready';
  if (r.n_with_displacement > 0) h += ' · ' + r.n_with_displacement + ' displacement';
  if (r.n_with_ltp_fallback > 0) h += ' · ' + r.n_with_ltp_fallback + ' LTP fallback';
  h += '</span></div>';

  // Summary stats
  h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Topology Summary</div>';
  h += '<div class="mod-results-body"><div class="mod-summary">';
  h += tvStat('Usable Prompts', nUsable, r.n_skipped + ' skipped (no profiles)');
  h += tvStat('Base Candidates', r.n_with_base_candidates || 0, 'prompts with dual-bank data');
  var ts = r.token_stats || {};
  h += tvStat('Total Tokens', ts.total_tokens || 0, 'mean ' + (ts.mean_tokens_per_prompt || 0) + '/prompt, max ' + (ts.max_tokens || 0));
  h += '</div></div>';

  // Category breakdown
  if (r.by_category && Object.keys(r.by_category).length > 1) {
    h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Category Breakdown</div>';
    h += '<div class="mod-results-body collapsed">';
    h += '<table class="mod-tbl"><thead><tr><th>Category</th><th class="num">Prompts</th><th class="num">Mean Tokens</th></tr></thead><tbody>';
    Object.keys(r.by_category).sort().forEach(function(cat) {
      var ci = r.by_category[cat];
      h += '<tr><td style="color:var(--cyan)">' + escHtml(cat) + '</td>';
      h += '<td class="num">' + ci.count + '</td>';
      h += '<td class="num">' + ci.mean_tokens + '</td></tr>';
    });
    h += '</tbody></table></div>';
  }

  // Displacement stats
  if (r.displacement_stats) {
    var ds = r.displacement_stats;
    h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Displacement Magnitude</div>';
    h += '<div class="mod-results-body collapsed"><div class="mod-summary">';
    var ov = ds.overall || {};
    h += tvStat('Overall Mean', (ov.mean || 0).toFixed(6), 'std ' + (ov.std || 0).toFixed(6) + ', median ' + (ov.median || 0).toFixed(6));
    if (ds.by_category) {
      Object.keys(ds.by_category).sort().forEach(function(cat) {
        var cs = ds.by_category[cat];
        h += tvStat(cat, cs.mean.toFixed(6), 'std ' + cs.std.toFixed(6) + ', max ' + cs.max.toFixed(6));
      });
    }
    h += '</div></div>';
  }

  // Asymmetry stats
  if (r.asymmetry_stats) {
    var as = r.asymmetry_stats;
    h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Bank Asymmetry</div>';
    h += '<div class="mod-results-body collapsed"><div class="mod-summary">';
    var ao = as.overall || {};
    var biasLabel = ao.mean > 0.05 ? 'instruct-dominant' : ao.mean < -0.05 ? 'base-dominant' : 'balanced';
    h += tvStat('Overall Bias', biasLabel, 'mean asymmetry ' + (ao.mean || 0).toFixed(4) + ', ' + Math.round((ao.instruct_dominant_frac || 0) * 100) + '% instruct-dominant positions');
    if (as.by_category) {
      Object.keys(as.by_category).sort().forEach(function(cat) {
        var cs = as.by_category[cat];
        var catBias = cs.mean > 0.05 ? '↑ instruct' : cs.mean < -0.05 ? '↓ base' : '≈ balanced';
        h += tvStat(cat, catBias, 'mean ' + cs.mean.toFixed(4) + ', ' + Math.round(cs.instruct_dominant_frac * 100) + '% instruct-dominant');
      });
    }
    h += '</div></div>';
  }

  h += '</div>';
  return h;
}

// Store latest CFT launch params from module results
var _cftLaunchParams = null;

// Override fetchModuleResults to capture CFT params
var _origFetchModuleResults = fetchModuleResults;
fetchModuleResults = async function(name) {
  await _origFetchModuleResults(name);
  if (name === 'correction_field_topology') {
    try {
      var r = await fetch('/api/modules/correction_field_topology/results');
      var d = await r.json();
      if (d.ok && d.results && d.results.launch_params) {
        _cftLaunchParams = d.results.launch_params;
      }
    } catch(e) { /* silent */ }
  }
};

function _cftLaunchFromModule() {
  var p = _cftLaunchParams || {};
  // Fall back to module card param controls if no cached results
  if (!_cftLaunchParams) {
    var m = _moduleData['correction_field_topology'];
    if (m) p = getModuleParams('correction_field_topology');
  }
  var totalCount = sessionResults ? sessionResults.length : (dashResults ? dashResults.length : 0);
  if (!totalCount) {
    // Try from session info
    try { totalCount = parseInt(document.querySelector('#sessCount')?.textContent) || 0; } catch(e) {}
  }
  // Use dashboard info if available
  if (_promptTotal > 0) totalCount = _promptTotal;
  _lazyLoadTerrainPopout(totalCount, {
    category: p.category || 'all',
    recordLimit: p.record_limit || 100,
    tokenLimit: p.token_limit || 20,
    charLimit: p.char_limit || 50,
    autoRotate: !!p.auto_rotate,
    rotateSpeed: p.rotate_speed || 0.3,
  });
}

// ─── MI Instrumentation Results Renderer ────────────────────────

function renderMIInstrumentationResults(r) {
  if (r.error) {
    return '<div class="mod-results"><div class="mod-results-body" style="padding:16px;color:var(--orange)">' + escHtml(r.error) + '</div></div>';
  }

  var h = '<div class="mod-results">';
  var sm = r.summary || {};

  // ── Summary banner ──
  var rCol = sm.refusal_separates ? 'var(--green)' : 'var(--orange)';
  var sCol = sm.stress_separates ? 'var(--green)' : 'var(--orange)';
  h += '<div style="padding:12px 16px;border-bottom:1px solid var(--border)">';
  h += '<div class="metrics-grid">';
  h += mc('Prompts', r.n_prompts, null, function(v){return v});
  h += mc('Safe', r.n_safe, null, function(v){return v});
  h += mc('Risk', r.n_risk, null, function(v){return v});
  h += mc('Hidden Dim', r.hidden_dim, null, function(v){return v});
  h += '</div></div>';

  // ═══ 1. REFUSAL DIRECTION ═══
  var rd = r.refusal_direction || {};
  h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Refusal Direction</div>';
  h += '<div class="mod-results-body">';
  h += '<div style="font-size:11px;color:var(--text-2);margin-bottom:8px;line-height:1.5">';
  h += 'Empirical refusal direction extracted from mean hidden states (risk − safe). ';
  h += 'Cosine similarity of each prompt against this direction produces a continuous alignment score.</div>';
  h += '<div class="metrics-grid">';
  h += mc('Refusal AUROC', rd.auroc, null, function(v){return '<span style="color:'+rCol+'">'+v.toFixed(4)+'</span>'});
  h += mc('Stress AUROC', rd.stress_auroc, null, function(v){return '<span style="color:'+sCol+'">'+v.toFixed(4)+'</span>'});
  h += mc('Separation', rd.separation, null, function(v){var c=v>0?'var(--green)':'var(--orange)';return '<span style="color:'+c+'">'+v.toFixed(4)+'</span>'});
  h += mc('‖dir‖', rd.refusal_norm, null, function(v){return v.toFixed(4)});
  h += '</div>';

  // Cosine stats
  h += '<table class="mod-tbl" style="margin-top:8px"><thead><tr><th>Group</th><th class="num">Mean cos</th><th class="num">Std</th></tr></thead><tbody>';
  h += '<tr><td style="color:var(--green)">Safe</td><td class="num">' + (rd.mean_cosine_safe||0).toFixed(4) + '</td><td class="num">' + (rd.std_cosine_safe||0).toFixed(4) + '</td></tr>';
  h += '<tr><td style="color:var(--red)">Risk</td><td class="num">' + (rd.mean_cosine_risk||0).toFixed(4) + '</td><td class="num">' + (rd.std_cosine_risk||0).toFixed(4) + '</td></tr>';
  h += '</tbody></table>';

  // Per-category breakdown
  var pcats = rd.per_category || {};
  if (Object.keys(pcats).length > 2) {
    h += '<table class="mod-tbl" style="margin-top:8px"><thead><tr><th>Category</th><th class="num">Mean cos</th><th class="num">Std</th><th class="num">n</th></tr></thead><tbody>';
    for (var cat in pcats) {
      var pc = pcats[cat];
      h += '<tr><td><span class="pill ' + pillClass(cat) + '">' + cat + '</span></td>';
      h += '<td class="num">' + pc.mean.toFixed(4) + '</td>';
      h += '<td class="num">' + pc.std.toFixed(4) + '</td>';
      h += '<td class="num">' + pc.n + '</td></tr>';
    }
    h += '</tbody></table>';
  }
  h += '</div>';

  // ═══ 2. PER-LAYER AUROC ═══
  var pla = r.per_layer_auroc || {};
  var layers = pla.layers || [];
  if (layers.length) {
    h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Per-Layer AUROC (' + layers.length + ' layers)</div>';
    h += '<div class="mod-results-body">';
    h += '<div style="font-size:11px;color:var(--text-2);margin-bottom:8px;line-height:1.5">';
    h += 'Discrimination power of the weight-delta projection at each model layer. ';
    h += 'Shows where in the network the correction signal concentrates.</div>';

    // Visual bar chart
    h += '<div style="margin-bottom:8px">';
    layers.forEach(function(la) {
      var w = Math.max(2, (la.auroc - 0.4) / 0.6 * 100);
      var col = la.auroc > 0.8 ? 'var(--green)' : la.auroc > 0.65 ? 'var(--cyan)' : la.auroc > 0.55 ? 'var(--orange)' : 'var(--text-3)';
      h += '<div style="display:flex;align-items:center;gap:6px;margin-bottom:2px">';
      h += '<span style="width:32px;text-align:right;font-size:10px;color:var(--text-2)">L' + la.layer + '</span>';
      h += '<div style="flex:1;height:12px;background:var(--bg-3);border-radius:2px;overflow:hidden">';
      h += '<div style="width:' + w + '%;height:100%;background:' + col + ';border-radius:2px"></div></div>';
      h += '<span style="width:40px;font-size:10px;color:' + col + '">' + la.auroc.toFixed(3) + '</span>';
      h += '</div>';
    });
    h += '</div>';

    // Table
    h += '<table class="mod-tbl"><thead><tr><th>Layer</th><th class="num">AUROC</th><th class="num">Safe μ</th><th class="num">Risk μ</th><th class="num">Ratio</th></tr></thead><tbody>';
    layers.forEach(function(la) {
      var col = la.auroc > 0.8 ? 'var(--green)' : la.auroc > 0.65 ? 'var(--cyan)' : 'var(--text-2)';
      h += '<tr><td>L' + la.layer + '</td>';
      h += '<td class="num" style="color:' + col + '">' + la.auroc.toFixed(4) + '</td>';
      h += '<td class="num">' + la.mean_safe.toFixed(6) + '</td>';
      h += '<td class="num">' + la.mean_risk.toFixed(6) + '</td>';
      h += '<td class="num">' + la.ratio.toFixed(2) + 'x</td></tr>';
    });
    h += '</tbody></table>';
    h += '</div>';
  }

  // ═══ 3. PATCHING PRIORITY MAP ═══
  var pp = r.patching_priority || {};
  var topPts = pp.top_points || [];
  if (topPts.length) {
    h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Activation Patching Priority</div>';
    h += '<div class="mod-results-body">';
    h += '<div style="font-size:11px;color:var(--text-2);margin-bottom:8px;line-height:1.5">';
    h += 'Highest-value intervention points for activation patching experiments. ';
    h += 'Intensity = attribution × stress, aggregated across all prompts.</div>';

    // Layer marginal summary
    var lm = pp.layer_marginal || [];
    if (lm.length) {
      h += '<div style="margin-bottom:8px">';
      var maxInt = Math.max.apply(null, lm.map(function(x){return x.max_intensity})) || 1;
      lm.forEach(function(la) {
        var w = Math.max(2, la.max_intensity / maxInt * 100);
        h += '<div style="display:flex;align-items:center;gap:6px;margin-bottom:2px">';
        h += '<span style="width:32px;text-align:right;font-size:10px;color:var(--text-2)">L' + la.layer + '</span>';
        h += '<div style="flex:1;height:10px;background:var(--bg-3);border-radius:2px;overflow:hidden">';
        h += '<div style="width:' + w + '%;height:100%;background:var(--orange);border-radius:2px"></div></div>';
        h += '<span style="width:52px;font-size:10px;color:var(--orange)">' + la.max_intensity.toFixed(5) + '</span>';
        h += '</div>';
      });
      h += '</div>';
    }

    // Top points table
    h += '<table class="mod-tbl"><thead><tr><th>Rank</th><th>Layer</th><th>Position</th><th class="num">Intensity</th><th class="num">Obs</th></tr></thead><tbody>';
    topPts.slice(0, 15).forEach(function(pt, i) {
      h += '<tr><td>' + (i+1) + '</td><td>L' + pt.layer + '</td><td>P' + pt.position + '</td>';
      h += '<td class="num" style="color:var(--orange)">' + pt.intensity.toFixed(6) + '</td>';
      h += '<td class="num">' + pt.n_observations + '</td></tr>';
    });
    h += '</tbody></table>';
    h += '</div>';
  }

  // ═══ 4. RANDOM PROJECTION CONTROL ═══
  var rp = r.random_projection || {};
  h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Random Projection Control</div>';
  h += '<div class="mod-results-body">';
  h += '<div style="font-size:11px;color:var(--text-2);margin-bottom:8px;line-height:1.5">';
  h += 'Validates that the refusal direction and TAGM stress outperform random projections of the same hidden states. ';
  h += rp.n_trials + ' random unit vectors tested. p-value = fraction of random trials that match or exceed the real AUROC.</div>';

  h += '<div class="metrics-grid">';
  h += mc('Random μ', rp.random_mean_auroc, null, function(v){return v.toFixed(4)});
  h += mc('Random σ', rp.random_std_auroc, null, function(v){return v.toFixed(4)});
  h += mc('Random max', rp.random_max_auroc, null, function(v){return v.toFixed(4)});
  h += mc('Random p95', rp.random_p95_auroc, null, function(v){return v.toFixed(4)});
  h += '</div>';

  // Comparison table
  var rpRef = rp.refusal_direction || {};
  var rpStr = rp.tagm_stress || {};
  h += '<table class="mod-tbl" style="margin-top:8px"><thead><tr><th>Method</th><th class="num">AUROC</th><th class="num">Δ over random</th><th class="num">p-value</th><th>Significant</th></tr></thead><tbody>';

  var refSig = rpRef.empirical_p < 0.05;
  var strSig = rpStr.empirical_p < 0.05;
  h += '<tr><td style="color:var(--cyan)">Refusal Direction</td>';
  h += '<td class="num">' + (rpRef.auroc||0).toFixed(4) + '</td>';
  h += '<td class="num" style="color:var(--green)">+' + (rpRef.delta_over_random||0).toFixed(4) + '</td>';
  h += '<td class="num">' + (rpRef.empirical_p||1).toFixed(3) + '</td>';
  h += '<td style="color:' + (refSig?'var(--green)':'var(--orange)') + '">' + (refSig?'✓ yes':'✗ no') + '</td></tr>';

  h += '<tr><td style="color:var(--orange)">TAGM Stress</td>';
  h += '<td class="num">' + (rpStr.auroc||0).toFixed(4) + '</td>';
  h += '<td class="num" style="color:var(--green)">+' + (rpStr.delta_over_random||0).toFixed(4) + '</td>';
  h += '<td class="num">' + (rpStr.empirical_p||1).toFixed(3) + '</td>';
  h += '<td style="color:' + (strSig?'var(--green)':'var(--orange)') + '">' + (strSig?'✓ yes':'✗ no') + '</td></tr>';

  h += '</tbody></table>';

  // Histogram text summary
  var hist = rp.histogram || [];
  if (hist.length) {
    var below60 = hist.filter(function(v){return v<0.6}).length;
    var above70 = hist.filter(function(v){return v>0.7}).length;
    h += '<div style="font-size:10px;color:var(--text-3);margin-top:6px">';
    h += 'Distribution: ' + below60 + '/' + hist.length + ' trials below 0.6, ';
    h += above70 + '/' + hist.length + ' above 0.7';
    h += '</div>';
  }
  h += '</div>';

  h += '</div>';
  return h;
}

// ─── MI Readiness Results Renderer ──────────────────────────────

function renderMIResults(r) {
  if (r.error) {
    return '<div class="mod-results"><div class="mod-results-body" style="padding:16px;color:var(--orange)">' + escHtml(r.error) + '</div></div>';
  }

  var h = '<div class="mod-results">';

  // ── Readiness Banner ──
  var readyCol = r.overall_readiness === 'ready' ? 'var(--green)' : r.overall_readiness === 'gaps' ? 'var(--red)' : 'var(--orange)';
  var readyLabel = r.overall_readiness === 'ready' ? 'MI READY' : r.overall_readiness === 'gaps' ? 'GAPS IDENTIFIED' : 'NEAR READY';
  h += '<div style="padding:12px 16px;background:color-mix(in srgb,' + readyCol + ' 12%,transparent);border-left:3px solid ' + readyCol + ';margin:8px 0;font-family:var(--mono);font-size:12px">';
  h += '<span style="color:' + readyCol + ';font-weight:700">' + readyLabel + '</span>';
  h += '<span style="color:var(--text-1);margin-left:8px">' + escHtml(r.readiness_summary) + '</span>';
  h += '</div>';

  // ── Scorecard ──
  h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">MI Scorecard</div>';
  h += '<div class="mod-results-body"><div class="mod-summary">';
  if (r.scorecard) {
    r.scorecard.forEach(function(s) {
      var col = s.status === 'strong' ? 'var(--green)' : s.status === 'weak' ? 'var(--red)' : 'var(--orange)';
      var icon = s.status === 'strong' ? '●' : s.status === 'weak' ? '○' : '◐';
      h += '<div class="mod-stat"><div class="stat-label"><span style="color:' + col + '">' + icon + '</span> ' + escHtml(s.item) + '</div>';
      h += '<div class="stat-value" style="color:' + col + '">' + s.status + '</div>';
      h += '<div class="stat-detail">' + escHtml(s.detail) + '</div></div>';
    });
  }
  h += '</div></div>';

  // ── AUROC Comparison ──
  h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Discrimination Power (AUROC)</div>';
  h += '<div class="mod-results-body">';
  h += '<table class="mod-tbl"><thead><tr>';
  h += '<th>Method</th><th class="num">AUC</th><th class="num">± std</th><th>Notes</th>';
  h += '</tr></thead><tbody>';

  if (r.auroc_length_only) {
    h += '<tr><td>Sequence length only</td><td class="num">' + r.auroc_length_only.auc.toFixed(4) + '</td>';
    h += '<td class="num" style="color:var(--text-3)">' + r.auroc_length_only.std.toFixed(4) + '</td>';
    h += '<td style="color:var(--text-2);font-size:10px">Safe mean ' + r.mean_seq_len_safe + ' tok, risk mean ' + r.mean_seq_len_risk + ' tok</td></tr>';
  }
  if (r.auroc_full) {
    var aucCol = r.auroc_full.auc >= 0.95 ? 'var(--green)' : r.auroc_full.auc >= 0.80 ? 'var(--cyan)' : 'var(--orange)';
    h += '<tr><td style="font-weight:600">TAGM features (' + r.n_features + ' metrics)</td>';
    h += '<td class="num" style="color:' + aucCol + ';font-weight:700">' + r.auroc_full.auc.toFixed(4) + '</td>';
    h += '<td class="num" style="color:var(--text-3)">' + r.auroc_full.std.toFixed(4) + '</td>';
    h += '<td style="color:var(--text-2);font-size:10px">' + r.auroc_full.n_folds + '-fold CV</td></tr>';
  }
  if (r.auroc_residualized) {
    h += '<tr><td>TAGM features, length-residualized</td>';
    h += '<td class="num" style="font-weight:600">' + r.auroc_residualized.auc.toFixed(4) + '</td>';
    h += '<td class="num" style="color:var(--text-3)">' + r.auroc_residualized.std.toFixed(4) + '</td>';
    h += '<td style="color:var(--text-2);font-size:10px">Signal after removing length</td></tr>';
  }
  if (r.random_baseline) {
    h += '<tr><td>Random projection baseline</td>';
    h += '<td class="num" style="color:var(--text-3)">' + r.random_baseline.mean_auc.toFixed(4) + '</td>';
    h += '<td class="num" style="color:var(--text-3)">' + r.random_baseline.std_auc.toFixed(4) + '</td>';
    h += '<td style="color:var(--text-2);font-size:10px">' + r.random_baseline.n_trials + ' trials, Δ = +' + r.random_baseline.delta_above_random.toFixed(4) + '</td></tr>';
  }
  h += '</tbody></table></div>';

  // ── Per-Metric AUROC ──
  if (r.per_metric_auroc && r.per_metric_auroc.length) {
    h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Per-Metric AUROC</div>';
    h += '<div class="mod-results-body collapsed" style="max-height:350px">';
    h += '<table class="mod-tbl"><thead><tr><th>Metric</th><th class="num">AUROC</th><th>Direction</th></tr></thead><tbody>';
    r.per_metric_auroc.forEach(function(m) {
      var col = m.auroc >= 0.80 ? 'var(--green)' : m.auroc >= 0.65 ? 'var(--cyan)' : 'var(--text-2)';
      h += '<tr><td style="color:var(--cyan)">' + escHtml(m.metric) + '</td>';
      h += '<td class="num" style="color:' + col + '">' + m.auroc.toFixed(4) + '</td>';
      h += '<td style="color:var(--text-3);font-size:10px">' + m.direction + '</td></tr>';
    });
    h += '</tbody></table></div>';
  }

  // ── PCA ──
  if (r.pca && r.pca.components) {
    var pcaTarget = (r.config && r.config.pca_variance_target) ? Math.round(r.config.pca_variance_target * 100) : 95;
    h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">PCA: Effective Dimensionality (' + pcaTarget + '% target)</div>';
    h += '<div class="mod-results-body">';
    h += '<div style="padding:8px 0;font-size:11px;color:var(--text-1)">';
    h += '<span style="color:var(--cyan);font-weight:600">' + r.pca.effective_dimensionality + '</span> components explain ' + (r.pca.variance_target ? Math.round(r.pca.variance_target * 100) : 95) + '% variance ';
    h += '(from ' + r.pca.n_features_original + ' features)</div>';
    h += '<table class="mod-tbl"><thead><tr><th>Component</th><th>Axis Name</th><th class="num">Var %</th><th class="num">Cumul %</th><th>Top Loadings</th></tr></thead><tbody>';
    r.pca.components.forEach(function(c) {
      var bar = '<span style="display:inline-block;width:' + Math.round(c.variance_explained) + 'px;height:8px;background:var(--cyan);border-radius:2px;vertical-align:middle;margin-right:4px"></span>';
      var loads = (c.top_loadings || []).slice(0, 3).map(function(l) {
        return '<span style="color:var(--text-2)">' + l.metric + '</span> <span style="color:var(--text-3)">(' + l.loading.toFixed(2) + ')</span>';
      }).join(', ');
      h += '<tr><td>PC' + c.index + '</td><td style="color:var(--text-1)">' + escHtml(c.name) + '</td>';
      h += '<td class="num">' + bar + c.variance_explained + '%</td>';
      h += '<td class="num">' + c.cumulative + '%</td>';
      h += '<td style="font-size:10px">' + loads + '</td></tr>';
    });
    h += '</tbody></table></div>';
  }

  // ── Length Confounds ──
  if (r.length_confounds && r.length_confounds.length) {
    h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Length Confound Analysis</div>';
    h += '<div class="mod-results-body collapsed" style="max-height:350px">';
    h += '<table class="mod-tbl"><thead><tr><th>Metric</th><th class="num">r</th><th class="num">r²</th><th>Severity</th></tr></thead><tbody>';
    r.length_confounds.forEach(function(c) {
      var sCol = c.severity === 'heavily confounded' ? 'var(--red)' : c.severity === 'moderately confounded' ? 'var(--orange)' : c.severity === 'mildly confounded' ? 'var(--yellow, var(--orange))' : 'var(--green)';
      h += '<tr><td style="color:var(--cyan)">' + escHtml(c.metric) + '</td>';
      h += '<td class="num">' + c.r.toFixed(4) + '</td>';
      h += '<td class="num">' + c.r_squared.toFixed(4) + '</td>';
      h += '<td style="color:' + sCol + '">' + c.severity + '</td></tr>';
    });
    h += '</tbody></table></div>';
  }

  // ── Redundancy ──
  if (r.redundancy && r.redundancy.pairs && r.redundancy.pairs.length) {
    h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Metric Redundancy (|r| ≥ ' + r.redundancy.threshold + ')</div>';
    h += '<div class="mod-results-body collapsed">';
    h += '<table class="mod-tbl"><thead><tr><th>Metric A</th><th>Metric B</th><th class="num">r</th><th>Assessment</th></tr></thead><tbody>';
    r.redundancy.pairs.forEach(function(p) {
      var iCol = p.implication === 'functionally identical' ? 'var(--red)' : 'var(--orange)';
      h += '<tr><td style="color:var(--cyan)">' + escHtml(p.metric_a) + '</td>';
      h += '<td style="color:var(--cyan)">' + escHtml(p.metric_b) + '</td>';
      h += '<td class="num">' + p.r.toFixed(4) + '</td>';
      h += '<td style="color:' + iCol + '">' + p.implication + '</td></tr>';
    });
    h += '</tbody></table></div>';
  }

  // ── Category Stress ──
  if (r.category_stress) {
    h += '<div class="mod-results-header" onclick="this.nextElementSibling.classList.toggle(\'collapsed\')">Category Stress Distribution</div>';
    h += '<div class="mod-results-body collapsed"><div class="mod-summary">';
    var sep = r.category_stress._separation;
    if (sep) {
      h += tvStat('Safe Mean Stress', sep.safe_mean.toFixed(4), sep.n_safe + ' prompts');
      h += tvStat('Risk Mean Stress', sep.risk_mean.toFixed(4), sep.n_risk + ' prompts');
      h += tvStat("Cohen's d", sep.cohens_d.toFixed(3), 'supplementary metric');
    }
    Object.keys(r.category_stress).forEach(function(cat) {
      if (cat === '_separation') return;
      var cs = r.category_stress[cat];
      h += tvStat(cat, cs.mean_stress.toFixed(4) + ' ± ' + cs.std_stress.toFixed(4), 'n=' + cs.n);
    });
    h += '</div></div>';
  }

  // ── Data summary footer ──
  h += '<div style="padding:8px 16px;font-size:10px;color:var(--text-3);border-top:1px solid var(--border);margin-top:8px">';
  h += r.n_prompts + ' prompts (' + r.n_safe + ' safe, ' + r.n_risk + ' risk) · ';
  h += r.n_features + ' features · ';
  h += 'Categories: ' + (r.categories || []).join(', ');
  if (r.safe_categories) {
    h += ' · Safe=' + r.safe_categories.join(',');
  }
  if (r.config) {
    h += '<br>seed=' + r.config.random_seed + ' · ' + r.config.n_folds + '-fold CV · ';
    h += 'LR(lr=' + r.config.lr_learning_rate + ', iter=' + r.config.lr_iterations + ', λ=' + r.config.lr_regularization + ') · ';
    h += 'PCA@' + Math.round(r.config.pca_variance_target * 100) + '%';
  }
  h += '</div>';

  h += '</div>';
  return h;
}

(function() {
  var origSwitch = switchMainTab;
  var modulesLoaded = false;
  switchMainTab = function(el, id) {
    origSwitch(el, id);
    if (id === 'panel-modules' && !modulesLoaded) {
      modulesLoaded = true;
      loadModules();
    }
  };
})();

