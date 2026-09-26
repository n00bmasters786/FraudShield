// FraudShield Live — dashboard controller.
// Screen + system audio are captured here, handed to capture-worker.js (which streams them to the
// engine), and every message coming back is rendered: live score, overlay, evidence and events.

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const C = {
  low: '#34d399', mid: '#fbbf24', high: '#fb7185', accent: '#8b8cff',
  face: '#60a5fa', voice: '#c084fc', sync: '#2dd4bf',
  text: '#ececf1', muted: '#9d9daa', faint: '#6b6b78',
};
const VERDICT_COLOR = { genuine: C.low, suspicious: C.mid, deepfake: C.high };
const MOD_NAME = { face: 'Face', voice: 'Voice', sync: 'Lip-sync', fusion: 'Verdict' };
const MOD_STATUS = {
  face: { searching: 'Scanning', warming: 'Locking on', tracking: 'Tracking' },
  voice: { no_audio: 'Waiting', no_track: 'Audio off', silent: 'Silent', listening: 'Listening', analyzing: 'Analysing' },
  sync: { waiting: 'Waiting', listening: 'Listening', analyzing: 'Analysing' },
};
const SIG_STATUS = { alert: ['Alert', C.high], watch: ['Watch', C.mid], clear: ['Clear', C.low], 'n/a': ['N/A', C.faint] };
const SIG_ORDER = { alert: 0, watch: 1, clear: 2, 'n/a': 3 };
const GROUPS = [
  ['synthesis', 'Deepfake synthesis', 'Is the face / voice itself generated?'],
  ['liveness', 'Liveness', 'Is a live person in front of the camera?'],
];
const TARGET_FPS = 15;
const MAX_WIDTH = 1920;
const TIMELINE_SPAN_S = 90;

const riskColor = (v) => (v == null ? C.faint : v < 35 ? C.low : v < 65 ? C.mid : C.high);
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmtClock = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

const S = {
  phase: 'landing',
  stream: null,
  worker: null,
  audio: null,
  settings: null,
  startedAt: 0,
  stoppedAt: 0,
  timeline: [],
  events: [],
  last: null,
  ack: null,
  ackAt: 0,
  roi: null,
  selecting: false,
  drag: null,
  content: null,
  series: new Set([1, 2, 3, 4]),
  tab: 'face',
  verdict: 'idle',
  gauge: { shown: null, target: null },
  hud: null,
  hello: null,
  promptKey: '',
};

// =====================================================================
// Toasts
// =====================================================================
function toast(msg, ms = 4200) {
  const el = document.createElement('div');
  el.className = 'toast';
  el.innerHTML = msg;
  $('#toasts').append(el);
  setTimeout(() => { el.classList.add('out'); setTimeout(() => el.remove(), 450); }, ms);
}

// =====================================================================
// Capture lifecycle
// =====================================================================
function envCheck() {
  const w = $('#env-warn');
  let msg = '';
  if (!window.isSecureContext) {
    msg = `Screen capture only works on a secure page. Open <b>http://localhost:${location.port || 8000}</b> on this computer instead of <b>${esc(location.host)}</b>.`;
  } else if (!navigator.mediaDevices?.getDisplayMedia) {
    msg = 'This browser cannot share the screen. Use a recent Chrome or Edge.';
  } else if (!('MediaStreamTrackProcessor' in window)) {
    msg = 'This browser can only analyse while the dashboard tab is visible. Chrome or Edge keep analysing in the background.';
  }
  if (msg) { w.innerHTML = msg; w.hidden = false; }
}

async function start() {
  if (S.phase === 'live') return;
  if (!navigator.mediaDevices?.getDisplayMedia) { envCheck(); toast('Screen capture is not available on this page.'); return; }
  let stream;
  try {
    stream = await navigator.mediaDevices.getDisplayMedia({
      video: { displaySurface: 'monitor', frameRate: { ideal: 30, max: 30 } },
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, suppressLocalAudioPlayback: false },
      systemAudio: 'include',
      selfBrowserSurface: 'exclude',
      surfaceSwitching: 'include',
      monitorTypeSurfaces: 'include',
    });
  } catch (e) {
    toast(e.name === 'NotAllowedError' ? 'Screen sharing was cancelled.' : `Could not start screen capture: ${esc(e.message)}`);
    return;
  }

  S.stream = stream;
  const vTrack = stream.getVideoTracks()[0];
  const aTrack = stream.getAudioTracks()[0] || null;
  S.settings = vTrack.getSettings();
  vTrack.addEventListener('ended', () => stop('Screen sharing ended.'));

  const preview = $('#preview');
  preview.srcObject = new MediaStream([vTrack]);
  preview.play().catch(() => {});

  const worker = new Worker('capture-worker.js');
  S.worker = worker;
  worker.onmessage = (e) => onWorker(e.data);
  const wsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
  if ('MediaStreamTrackProcessor' in window) {
    const proc = new MediaStreamTrackProcessor({ track: vTrack });
    worker.postMessage({ type: 'start', wsUrl, readable: proc.readable, fps: TARGET_FPS, maxWidth: MAX_WIDTH }, [proc.readable]);
  } else {
    worker.postMessage({ type: 'start', wsUrl, fps: TARGET_FPS, maxWidth: MAX_WIDTH });
    bitmapLoop(worker);
  }
  if (aTrack) await startAudio(aTrack, worker);

  clearSession();
  S.startedAt = performance.now();
  setPhase('live');
  sendConfig({ has_audio: !!aTrack, roi: S.roi });

  const kind = { monitor: 'Entire screen', window: 'Window', browser: 'Browser tab' }[S.settings.displaySurface] || 'Screen';
  setPill('#pill-screen', 'on', `${kind} · ${S.settings.width}×${S.settings.height}`);
  setPill('#pill-audio', aTrack ? 'on' : 'off', aTrack ? 'System audio' : 'No audio');
  if (!aTrack) {
    toast('This share has <b>no system audio</b>, so voice and lip-sync checks are off. Stop and share again with <b>“Also share system audio”</b> ticked.', 8000);
  } else if (S.settings.displaySurface && S.settings.displaySurface !== 'monitor') {
    toast('Tip: share the <b>entire screen</b> so the customer stays in view whichever window is in front.', 6000);
  }
}

async function startAudio(track, worker) {
  try {
    const ctx = new AudioContext();
    await ctx.audioWorklet.addModule('pcm-worklet.js');
    const src = ctx.createMediaStreamSource(new MediaStream([track]));
    const node = new AudioWorkletNode(ctx, 'pcm-capture', { channelCount: 2, channelCountMode: 'explicit' });
    const ch = new MessageChannel();
    node.port.postMessage({ port: ch.port1 }, [ch.port1]);
    worker.postMessage({ type: 'audio-port', port: ch.port2 }, [ch.port2]);
    const mute = ctx.createGain();
    mute.gain.value = 0;                       // keep the graph pulling without echoing the call
    src.connect(node).connect(mute).connect(ctx.destination);
    const an = ctx.createAnalyser();
    an.fftSize = 2048;
    src.connect(an);
    if (ctx.state === 'suspended') await ctx.resume();
    S.audio = { ctx, an, buf: new Float32Array(an.fftSize), level: 0 };
  } catch (e) {
    toast(`Could not start audio analysis: ${esc(e.message)}`);
  }
}

function bitmapLoop(worker) {
  const v = $('#preview');
  const tick = async () => {
    if (S.worker !== worker || S.phase !== 'live') return;
    if (v.readyState >= 2) {
      try {
        const bmp = await createImageBitmap(v);
        worker.postMessage({ type: 'bitmap', bitmap: bmp, ts: performance.timeOrigin + performance.now() }, [bmp]);
      } catch { /* frame not ready */ }
    }
    setTimeout(tick, 1000 / TARGET_FPS);
  };
  tick();
}

function stop(reason) {
  if (S.phase !== 'live') return;
  const worker = S.worker;
  worker?.postMessage({ type: 'stop' });
  setTimeout(() => worker?.terminate(), 400);
  S.worker = null;
  S.stream?.getTracks().forEach((t) => t.stop());
  S.stream = null;
  S.audio?.ctx.close();
  S.audio = null;
  $('#preview').srcObject = null;
  S.stoppedAt = performance.now();
  S.ack = null;
  setSelecting(false);
  setPhase('stopped');
  $('#stage').dataset.face = 'stopped';
  $('#stage-msg').textContent = 'Monitoring stopped — the last results are kept.';
  setPill('#pill-engine', 'idle', 'Engine');
  setPill('#pill-screen', 'idle', 'Screen off');
  setPill('#pill-audio', 'idle', 'Audio off');
  $('#live-label').textContent = 'Stopped';
  if (reason) toast(reason);
}

function newCustomer() {
  if (S.phase !== 'live') return;
  S.worker?.postMessage({ type: 'reset' });
  clearSession();
  S.startedAt = performance.now();
  toast('New customer session started.');
}

function clearSession() {
  S.timeline = [];
  S.events = [];
  S.last = null;
  S.ack = null;
  S.gauge.target = null;
  S.promptKey = '';
  setVerdict('idle', 'Calibrating', 'Collecting evidence — keep the customer on camera and speaking.');
  $('#events').innerHTML = '<li class="ev-empty">Events appear here as the call is analysed.</li>';
  $('#ev-count').textContent = '0';
  $('#signals').innerHTML = '<div class="sig-empty">Checks appear once the customer\'s face is locked.</div>';
  $('#prompts').innerHTML = '<li class="pr-empty">Nothing to ask right now.</li>';
  $('#conf-bar').style.width = '0%';
  $('#conf-val').textContent = '0%';
  $('#live-label').textContent = 'Live';
  $$('.stream').forEach((el) => {
    el.dataset.risk = '';
    el.querySelector('.mring').style.setProperty('--p', 0);
    el.querySelector('.mnum').textContent = '—';
    el.querySelector('.stream-status').textContent = 'Waiting';
    el.querySelector('.stream-find').textContent = '—';
    el.querySelector('.bar span').style.width = '0%';
  });
  ['#tv-model', '#tv-blink', '#tv-audio', '#tv-pulse', '#tv-sync'].forEach((id) => { $(id).textContent = '—'; });
  ['#tc-model', '#tc-blink', '#tc-pulse', '#tc-sync'].forEach((id) => setupCanvas($(id)));
}

function setPhase(p) {
  S.phase = p;
  document.body.dataset.phase = p;
  if (p === 'live') requestAnimationFrame(positionTabInk);
}

function sendConfig(msg) {
  S.worker?.postMessage({ type: 'config', msg });
}

// =====================================================================
// Messages from the engine
// =====================================================================
function onWorker(m) {
  switch (m.type) {
    case 'conn':
      setPill('#pill-engine', m.state === 'open' ? 'on' : m.state === 'connecting' ? 'warn' : 'off',
        m.state === 'open' ? 'Engine live' : m.state === 'connecting' ? 'Connecting' : 'Engine offline');
      break;
    case 'hello':
      S.hello = m;
      break;
    case 'ack':
      if (!m.dropped && !m.error) { S.ack = m; S.ackAt = performance.now(); renderTarget(m); }
      break;
    case 'state':
      if (S.phase === 'live') onState(m);
      break;
    case 'worker-error':
      console.warn('capture worker:', m.message);
      break;
  }
}

function onState(st) {
  S.last = st;
  const t = (performance.now() - S.startedAt) / 1000;
  const [, o, f, v, s] = st.point;
  S.timeline.push([t, o, f, v, s]);
  if (S.timeline.length > 7200) S.timeline.shift();

  const ov = st.overall;
  setVerdict(ov.verdict, ov.label, ov.action);
  S.gauge.target = ov.score;
  $('#conf-val').textContent = `${Math.round(ov.confidence * 100)}%`;
  $('#conf-bar').style.width = `${ov.confidence * 100}%`;

  renderStreams(st.modules);
  renderSignals();
  renderPrompts(st.prompts);
  renderTelemetry(st.modules);
  renderStats(st);
  st.events.forEach(addEvent);
  updateHud();
}

function setVerdict(verdict, label, action) {
  const lab = $('#verdict-label');
  if (verdict !== S.verdict || lab.textContent !== label) {
    S.verdict = verdict;
    document.body.dataset.verdict = verdict;
    lab.textContent = label;
    lab.classList.remove('swap');
    void lab.offsetWidth;
    lab.classList.add('swap');
  }
  $('#verdict-action').textContent = action;
}

const verdictColor = () => VERDICT_COLOR[S.verdict] || C.accent;

function setPill(sel, state, text) {
  const el = $(sel);
  el.dataset.state = state;
  el.querySelector('span').textContent = text;
}

// =====================================================================
// Target / capture card
// =====================================================================
function renderTarget(a) {
  const stage = $('#stage');
  const state = a.tracking ? 'tracking' : a.target ? 'warming' : 'searching';
  stage.dataset.face = state;
  const chip = $('#target-chip');
  const n = a.faces_on_screen || 0;
  if (state === 'tracking') {
    chip.textContent = S.roi ? 'Locked · region' : 'Locked';
    chip.style.setProperty('--c', C.low);
    $('#ti-title').textContent = 'Customer face locked';
    $('#ti-sub').textContent = `Tracking 478 landmarks · ${Math.round(a.iod || 0)} px between the eyes${n > 1 ? ` · ${n} faces on screen, largest chosen` : ''}`;
    $('#kpi-face').textContent = `${Math.round(a.iod || 0)} px`;
  } else if (state === 'warming') {
    chip.textContent = 'Locking on';
    chip.style.setProperty('--c', C.mid);
    $('#ti-title').textContent = 'Locking on…';
    $('#ti-sub').textContent = 'Face found — fitting the landmark mesh.';
  } else {
    chip.textContent = S.roi ? 'Scanning region' : 'Scanning';
    chip.style.setProperty('--c', C.accent);
    $('#ti-title').textContent = 'No face locked';
    $('#ti-sub').textContent = S.roi
      ? 'No face inside the selected region. Move the video tile into it, or switch back to Auto.'
      : 'Open the video call on the shared screen. The largest face is locked automatically.';
    $('#kpi-face').textContent = '—';
  }
  $('#stage-msg').textContent = S.roi ? 'Scanning the selected region for a face…' : 'Scanning the screen for the customer\'s face…';
}

function renderStats(st) {
  const s = st.stats || {};
  $('#pill-fps').textContent = `${s.fps ? s.fps.toFixed(0) : '—'} fps`;
  $('#kpi-fps').textContent = s.fps ? `${s.fps.toFixed(0)} fps` : '—';
  $('#kpi-lat').textContent = s.proc_ms != null ? `${Math.round(s.proc_ms)} ms` : '—';
  const m = s.models || {};
  const states = [m.face, m.voice].map((v) => String(v || ''));
  const ready = states.filter((v) => v === 'ready').length;
  const state = ready === 2 ? 'on' : states.some((v) => v === 'loading' || v === 'not loaded') ? 'warn' : 'off';
  setPill('#pill-ai', state, ready === 2 ? 'AI models' : state === 'warn' ? 'AI loading' : `AI ${ready}/2`);
  $('#pill-ai').title = `Face detector: ${m.face || '—'}\nVoice detector: ${m.voice || '—'}`;
}

// =====================================================================
// Evidence streams, signals, prompts, events
// =====================================================================
function waitingText(k, m) {
  if (k === 'face') return m.status === 'warming' ? 'Locking on to the customer\'s face…' : 'No customer face on the shared screen yet.';
  if (k === 'voice') {
    return { no_track: 'Screen was shared without system audio.', silent: 'Call audio is silent.',
      listening: 'Waiting for the customer to speak.' }[m.status] || 'Waiting for call audio.';
  }
  return m.signals?.[0]?.message || 'Needs a tracked face and the customer speaking.';
}

function renderStreams(mods) {
  for (const k of ['face', 'voice', 'sync']) {
    const m = mods[k];
    const el = $(`.stream[data-mod="${k}"]`);
    const ring = el.querySelector('.mring');
    ring.style.setProperty('--p', m.score ?? 0);
    ring.style.setProperty('--rc', m.score == null ? 'rgba(255,255,255,0.18)' : riskColor(m.score));
    el.querySelector('.mnum').textContent = m.score == null ? '—' : Math.round(m.score);
    el.dataset.risk = m.score == null ? '' : m.score >= 65 ? 'high' : m.score >= 35 ? 'mid' : 'low';
    const st = MOD_STATUS[k][m.status] || m.status;
    el.querySelector('.stream-status').textContent = m.score != null ? `${st} · ${Math.round(m.confidence * 100)}%` : st;
    el.querySelector('.stream-find').textContent = (m.score != null && m.finding) || waitingText(k, m);
    el.querySelector('.bar span').style.width = `${(m.confidence || 0) * 100}%`;
  }
}

function renderSignals(fresh = false) {
  const box = $('#signals');
  const mod = S.last?.modules?.[S.tab];
  const sorted = (mod?.signals || []).slice()
    .sort((a, b) => SIG_ORDER[a.status] - SIG_ORDER[b.status] || b.risk * b.weight - a.risk * a.weight);
  const groups = mod?.details?.groups || {};
  const sigs = [];
  for (const [g, label, hint] of GROUPS) {
    const members = sorted.filter((s) => (s.group || 'liveness') === g);
    if (!members.length) continue;
    if (S.tab !== 'sync') sigs.push({ key: `hdr-${g}`, header: true, label, hint, score: groups[g]?.score, driver: groups.driver === g });
    sigs.push(...members);
  }
  if (fresh || !sigs.length) box.innerHTML = '';
  if (!sigs.length) {
    const msg = { face: 'Checks appear once the customer\'s face is locked.', voice: 'Checks appear once the customer speaks.',
      sync: 'Appears once there is a tracked face and speech.' }[S.tab];
    box.innerHTML = `<div class="sig-empty">${msg}</div>`;
    return;
  }
  box.querySelector('.sig-empty')?.remove();
  const existing = new Map($$('.sig, .sig-group', box).map((el) => [el.dataset.key, el]));
  sigs.forEach((s, i) => {
    let el = existing.get(s.key);
    if (s.header) {
      if (!el) {
        el = document.createElement('div');
        el.className = 'sig-group';
        el.dataset.key = s.key;
        el.innerHTML = '<div><span class="sg-name"></span><span class="sg-hint"></span></div><span class="chip"></span>';
      }
      existing.delete(s.key);
      el.querySelector('.sg-name').textContent = s.label;
      el.querySelector('.sg-hint').textContent = s.hint;
      const chip = el.querySelector('.chip');
      chip.textContent = s.score == null ? 'no evidence' : `${Math.round(s.score)}${s.driver ? ' · drives score' : ''}`;
      chip.style.setProperty('--c', s.score == null ? C.faint : riskColor(s.score));
      if (box.children[i] !== el) box.insertBefore(el, box.children[i] || null);
      return;
    }
    if (!el) {
      el = document.createElement('div');
      el.className = 'sig';
      el.dataset.key = s.key;
      el.style.animationDelay = `${i * 45}ms`;
      el.innerHTML = '<div class="sig-head"><span class="chip"></span><span class="sig-name"></span><span class="sig-val"></span></div><div class="bar"><span></span></div><div class="sig-msg"></div>';
    }
    existing.delete(s.key);
    const [lab, col] = SIG_STATUS[s.status];
    el.dataset.status = s.status;
    el.style.setProperty('--c', col);
    const chip = el.querySelector('.chip');
    chip.textContent = lab;
    chip.style.setProperty('--c', col);
    el.querySelector('.sig-name').textContent = s.label;
    el.querySelector('.sig-val').textContent = s.value;
    el.querySelector('.bar span').style.width = s.status === 'n/a' ? '0%' : `${Math.max(3, s.risk * 100)}%`;
    el.querySelector('.sig-msg').textContent = s.message;
    if (box.children[i] !== el) box.insertBefore(el, box.children[i] || null);
  });
  existing.forEach((el) => el.remove());
}

function renderPrompts(prompts) {
  const key = prompts.map((p) => p.title + p.text).join('|');
  if (key === S.promptKey) return;
  S.promptKey = key;
  const ul = $('#prompts');
  if (!prompts.length) { ul.innerHTML = '<li class="pr-empty">Nothing to ask right now — evidence is flowing.</li>'; return; }
  ul.innerHTML = prompts.map((p, i) => `
    <li style="animation-delay:${i * 60}ms"><span class="pi"><svg><use href="#i-${esc(p.icon)}"/></svg></span>
      <div><div class="pt">${esc(p.title)}</div><div class="px">${esc(p.text)}</div></div></li>`).join('');
}

function addEvent(ev) {
  const list = $('#events');
  list.querySelector('.ev-empty')?.remove();
  const t = (performance.now() - S.startedAt) / 1000;
  S.events.push({ ...ev, session_t: +t.toFixed(1) });
  const li = document.createElement('li');
  li.dataset.level = ev.level;
  li.innerHTML = `<span class="t">${fmtClock(t)}</span><span class="x"><b>${MOD_NAME[ev.module] || ev.module}</b> · ${esc(ev.text)}</span>`;
  list.prepend(li);
  while (list.children.length > 80) list.lastChild.remove();
  $('#ev-count').textContent = S.events.length;
  if (ev.module === 'fusion' && ev.level === 'alert') toast(`<b style="color:${C.high}">Deepfake alert</b> — ${esc(ev.text)}`, 6000);
}

// =====================================================================
// Canvas helpers
// =====================================================================
function setupCanvas(c) {
  const r = c.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(r.width * dpr));
  const h = Math.max(1, Math.round(r.height * dpr));
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  return [ctx, r.width, r.height];
}

function hexA(hex, a) {
  const h = hex.replace('#', '');
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  return `rgba(${r},${g},${b},${a})`;
}

/** Smooth polyline through points (null y = gap). Returns the drawn runs. */
function smoothPath(ctx, pts) {
  const runs = [];
  let run = [];
  for (const p of pts) {
    if (p[1] == null || !Number.isFinite(p[1])) { if (run.length) runs.push(run); run = []; } else run.push(p);
  }
  if (run.length) runs.push(run);
  ctx.beginPath();
  for (const r of runs) {
    ctx.moveTo(r[0][0], r[0][1]);
    if (r.length === 1) { ctx.lineTo(r[0][0] + 0.1, r[0][1]); continue; }
    for (let i = 1; i < r.length - 1; i++) {
      const mx = (r[i][0] + r[i + 1][0]) / 2;
      const my = (r[i][1] + r[i + 1][1]) / 2;
      ctx.quadraticCurveTo(r[i][0], r[i][1], mx, my);
    }
    const l = r[r.length - 1];
    ctx.lineTo(l[0], l[1]);
  }
  return runs;
}

function spark(canvas, values, { color, min, max, fill = true, marks = [], hline = null, x0 = 0, x1 = null, width = 1.6 } = {}) {
  const [ctx, W, H] = setupCanvas(canvas);
  const pad = 4;
  const xs = values.map((v) => v[0]);
  const lo = min ?? Math.min(...values.map((v) => v[1]).filter(Number.isFinite));
  const hi = max ?? Math.max(...values.map((v) => v[1]).filter(Number.isFinite));
  const xa = x0 ?? Math.min(...xs);
  const xb = x1 ?? Math.max(...xs);
  const X = (x) => pad + (W - 2 * pad) * ((x - xa) / (xb - xa || 1));
  const Y = (y) => H - pad - (H - 2 * pad) * clamp((y - lo) / (hi - lo || 1));
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, H / 2);
  ctx.lineTo(W - pad, H / 2);
  ctx.stroke();
  for (const m of marks) {
    ctx.fillStyle = hexA(C.low, 0.18);
    ctx.fillRect(X(m) - 2, pad, 4, H - 2 * pad);
  }
  if (hline != null) {
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = hexA(C.mid, 0.6);
    ctx.beginPath();
    ctx.moveTo(pad, Y(hline));
    ctx.lineTo(W - pad, Y(hline));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  const pts = values.map(([x, y]) => [X(x), y == null ? null : Y(y)]);
  const runs = smoothPath(ctx, pts);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.shadowColor = color;
  ctx.shadowBlur = 8;
  ctx.stroke();
  ctx.shadowBlur = 0;
  if (fill) {
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, hexA(color, 0.22));
    g.addColorStop(1, hexA(color, 0));
    ctx.fillStyle = g;
    for (const r of runs) {
      if (r.length < 2) continue;
      smoothPath(ctx, r);
      ctx.lineTo(r[r.length - 1][0], H);
      ctx.lineTo(r[0][0], H);
      ctx.closePath();
      ctx.fill();
    }
  }
  return { ctx, X, Y, W, H };
}

// =====================================================================
// Telemetry
// =====================================================================
function renderTelemetry(mods) {
  const f = mods.face.details || {};
  const md = f.model;
  if (md?.trace?.length > 1) {
    const col = riskColor(100 * (md.risk ?? md.p));
    const { ctx, X, Y, W } = spark($('#tc-model'), md.trace, { color: col, x0: -20, x1: 0, min: 0, max: 1, hline: 0.5 });
    ctx.fillStyle = C.faint;
    ctx.font = '500 9px "JetBrains Mono", monospace';
    ctx.fillText('fake', W - 30, Y(0.93));
    ctx.fillText('real', W - 30, Y(0.04));
    void X;
    const per = (md.per_model || []).map((v, i) => `${(md.names || [])[i] || i} ${v.toFixed(2)}`).join(' · ');
    $('#tv-model').textContent = `fake score ${md.p.toFixed(2)}${per ? ` — ${per}` : ''}`;
  } else {
    setupCanvas($('#tc-model'));
    $('#tv-model').textContent = md?.status === 'loading' ? 'loading…' : String(md?.status || '').startsWith('unavailable') ? 'not installed' : '—';
  }
  const b = f.blink;
  if (b?.ear?.length) {
    spark($('#tc-blink'), b.ear, { color: C.face, x0: -20, x1: 0, marks: b.events || [], hline: b.close_thr,
      min: Math.min(...b.ear.map((p) => p[1]).filter(Number.isFinite)) * 0.9,
      max: Math.max(...b.ear.map((p) => p[1]).filter(Number.isFinite)) * 1.05 });
    $('#tv-blink').textContent = b.rate != null ? `${b.rate.toFixed(0)} /min · ${b.count}` : '—';
  } else {
    setupCanvas($('#tc-blink'));
    $('#tv-blink').textContent = '—';
  }

  const p = f.pulse;
  if (p?.wave?.length > 8) {
    spark($('#tc-pulse'), p.wave.map((v, i) => [i, v]), { color: C.high, min: -2.6, max: 2.6, fill: false, width: 1.8 });
    $('#tv-pulse').textContent = p.bpm != null ? `${Math.round(p.bpm)} bpm` : '—';
  } else {
    setupCanvas($('#tc-pulse'));
    $('#tv-pulse').textContent = '—';
  }

  const sd = mods.sync.details || {};
  if (sd.curve?.length) {
    const n = sd.curve.length;
    const L = (n - 1) / 2;
    const { ctx, X, Y, H } = spark($('#tc-sync'), sd.curve.map((v, i) => [(i - L) * 40, v]), { color: C.sync, min: -0.6, max: 0.8 });
    const k = sd.curve.indexOf(Math.max(...sd.curve));
    ctx.fillStyle = C.sync;
    ctx.beginPath();
    ctx.arc(X((k - L) * 40), Y(sd.curve[k]), 3.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.beginPath();
    ctx.moveTo(X(0), 4);
    ctx.lineTo(X(0), H - 4);
    ctx.stroke();
    $('#tv-sync').textContent = `r ${sd.corr.toFixed(2)} · ${sd.lag_ms >= 0 ? '+' : ''}${Math.round(sd.lag_ms)} ms`;
  } else {
    setupCanvas($('#tc-sync'));
    $('#tv-sync').textContent = mods.sync.status === 'analyzing' ? 'lips static' : '—';
  }

  const vd = mods.voice.details || {};
  S.audioLabel = mods.voice.status === 'no_track' ? 'no audio'
    : vd.speech_s != null ? `speech ${vd.speech_s.toFixed(1)} s` : MOD_STATUS.voice[mods.voice.status] || '—';
}

function drawAudioScope() {
  const c = $('#tc-audio');
  const [ctx, W, H] = setupCanvas(c);
  const a = S.audio;
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.beginPath();
  ctx.moveTo(4, H / 2);
  ctx.lineTo(W - 4, H / 2);
  ctx.stroke();
  if (!a) { $('#tv-audio').textContent = S.phase === 'live' ? 'no audio' : '—'; return; }
  a.an.getFloatTimeDomainData(a.buf);
  let rms = 0;
  for (let i = 0; i < a.buf.length; i++) rms += a.buf[i] * a.buf[i];
  rms = Math.sqrt(rms / a.buf.length);
  a.level += (Math.min(1, rms * 6) - a.level) * 0.2;
  const g = ctx.createLinearGradient(0, 0, W, 0);
  g.addColorStop(0, hexA(C.voice, 0.2));
  g.addColorStop(0.5, C.voice);
  g.addColorStop(1, hexA(C.voice, 0.2));
  ctx.strokeStyle = g;
  ctx.lineWidth = 1.6;
  ctx.shadowColor = C.voice;
  ctx.shadowBlur = 6 + 14 * a.level;
  ctx.beginPath();
  const n = 256;
  const step = a.buf.length / n;
  for (let i = 0; i < n; i++) {
    const v = a.buf[Math.floor(i * step)];
    const x = 4 + (W - 8) * (i / (n - 1));
    const y = H / 2 - clamp(v * 3, -1, 1) * (H / 2 - 4);
    if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;
  $('#tv-audio').textContent = S.audioLabel || '—';
}

// =====================================================================
// Risk timeline
// =====================================================================
function drawTimeline() {
  const [ctx, W, H] = setupCanvas($('#timeline'));
  const L = 34, R = 14, T = 10, B = 22;
  const now = S.phase === 'live' ? (performance.now() - S.startedAt) / 1000
    : S.phase === 'stopped' ? (S.stoppedAt - S.startedAt) / 1000 : 0;
  const X = (t) => L + (W - L - R) * (1 - (now - t) / TIMELINE_SPAN_S);
  const Y = (v) => T + (H - T - B) * (1 - v / 100);

  // risk bands
  [[65, 100, C.high], [35, 65, C.mid], [0, 35, C.low]].forEach(([a, b, col]) => {
    ctx.fillStyle = hexA(col, 0.045);
    ctx.fillRect(L, Y(b), W - L - R, Y(a) - Y(b));
  });
  ctx.font = '500 10px "JetBrains Mono", monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  [0, 35, 65, 100].forEach((v) => {
    ctx.strokeStyle = v === 35 || v === 65 ? 'rgba(255,255,255,0.13)' : 'rgba(255,255,255,0.06)';
    ctx.setLineDash(v === 35 || v === 65 ? [4, 4] : []);
    ctx.beginPath();
    ctx.moveTo(L, Y(v));
    ctx.lineTo(W - R, Y(v));
    ctx.stroke();
    ctx.fillStyle = C.faint;
    ctx.fillText(String(v), L - 8, Y(v));
  });
  ctx.setLineDash([]);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  for (let s = 0; s <= TIMELINE_SPAN_S; s += 15) {
    const x = L + (W - L - R) * (1 - s / TIMELINE_SPAN_S);
    ctx.fillStyle = C.faint;
    ctx.fillText(s === 0 ? 'now' : `-${s}s`, x, H - 6);
  }

  const pts = S.timeline.filter((p) => p[0] >= now - TIMELINE_SPAN_S - 2);
  ctx.save();
  ctx.beginPath();
  ctx.rect(L, 0, W - L - R, H);
  ctx.clip();
  [[2, C.face], [3, C.voice], [4, C.sync]].forEach(([k, col]) => {
    if (!S.series.has(k)) return;
    smoothPath(ctx, pts.map((p) => [X(p[0]), p[k] == null ? null : Y(p[k])]));
    ctx.strokeStyle = hexA(col, 0.6);
    ctx.lineWidth = 1.3;
    ctx.stroke();
  });
  if (S.series.has(1)) {
    const grad = ctx.createLinearGradient(0, Y(100), 0, Y(0));
    grad.addColorStop(0, C.high);
    grad.addColorStop(0.35, C.high);
    grad.addColorStop(0.5, C.mid);
    grad.addColorStop(0.65, C.low);
    grad.addColorStop(1, C.low);
    const line = pts.map((p) => [X(p[0]), p[1] == null ? null : Y(p[1])]);
    const runs = smoothPath(ctx, line);
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2.6;
    ctx.shadowColor = verdictColor();
    ctx.shadowBlur = 12;
    ctx.stroke();
    ctx.shadowBlur = 0;
    const fillG = ctx.createLinearGradient(0, T, 0, H - B);
    fillG.addColorStop(0, hexA(verdictColor(), 0.22));
    fillG.addColorStop(1, hexA(verdictColor(), 0));
    ctx.fillStyle = fillG;
    for (const r of runs) {
      if (r.length < 2) continue;
      smoothPath(ctx, r);
      ctx.lineTo(r[r.length - 1][0], Y(0));
      ctx.lineTo(r[0][0], Y(0));
      ctx.closePath();
      ctx.fill();
    }
    const last = [...pts].reverse().find((p) => p[1] != null);
    if (last) {
      const x = X(last[0]);
      const y = Y(last[1]);
      ctx.setLineDash([2, 4]);
      ctx.strokeStyle = hexA(verdictColor(), 0.5);
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(X(now), y);
      ctx.stroke();
      ctx.setLineDash([]);
      const pulse = (performance.now() % 1600) / 1600;
      ctx.fillStyle = hexA(verdictColor(), 0.35 * (1 - pulse));
      ctx.beginPath();
      ctx.arc(x, y, 4 + 10 * pulse, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = verdictColor();
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
  if (!pts.length) {
    ctx.fillStyle = C.faint;
    ctx.font = '500 12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(S.phase === 'live' ? 'Collecting evidence…' : 'No data yet', (L + W - R) / 2, (T + H - B) / 2);
  }
}

// =====================================================================
// Live overlay on the screen preview
// =====================================================================
function contentRect(W, H) {
  const v = $('#preview');
  const vw = v.videoWidth || S.ack?.size?.[0];
  const vh = v.videoHeight || S.ack?.size?.[1];
  if (!vw || !vh) return null;
  const s = Math.min(W / vw, H / vh);
  return { ox: (W - vw * s) / 2, oy: (H - vh * s) / 2, cw: vw * s, ch: vh * s, vw, vh };
}

function brackets(ctx, x, y, w, h, col, len) {
  ctx.strokeStyle = col;
  ctx.lineWidth = 2.5;
  ctx.shadowColor = col;
  ctx.shadowBlur = 14;
  ctx.beginPath();
  [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]].forEach(([px, py, sx, sy]) => {
    ctx.moveTo(px, py + sy * len);
    ctx.lineTo(px, py);
    ctx.lineTo(px + sx * len, py);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;
}

function drawOverlay() {
  const [ctx, W, H] = setupCanvas($('#overlay'));
  const cr = contentRect(W, H);
  S.content = cr;
  if (!cr) return;
  const X = (n) => cr.ox + n * cr.cw;
  const Y = (n) => cr.oy + n * cr.ch;

  if (S.roi) {
    const [rx, ry, rw, rh] = S.roi;
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.beginPath();
    ctx.rect(cr.ox, cr.oy, cr.cw, cr.ch);
    ctx.rect(X(rx), Y(ry), rw * cr.cw, rh * cr.ch);
    ctx.fill('evenodd');
    ctx.setLineDash([6, 5]);
    ctx.strokeStyle = C.accent;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(X(rx), Y(ry), rw * cr.cw, rh * cr.ch);
    ctx.setLineDash([]);
  }
  if (S.drag) {
    const { x0, y0, x1, y1 } = S.drag;
    ctx.fillStyle = hexA(C.accent, 0.12);
    ctx.strokeStyle = C.accent;
    ctx.lineWidth = 1.5;
    const rx = X(Math.min(x0, x1));
    const ry = Y(Math.min(y0, y1));
    const rw = Math.abs(x1 - x0) * cr.cw;
    const rh = Math.abs(y1 - y0) * cr.ch;
    ctx.fillRect(rx, ry, rw, rh);
    ctx.strokeRect(rx, ry, rw, rh);
  }

  const a = S.ack;
  if (!a || S.phase !== 'live') return;
  const age = (performance.now() - S.ackAt) / 1000;
  const alpha = clamp(1 - (age - 0.8) / 0.8);
  if (!a.target || alpha <= 0) return;
  ctx.globalAlpha = alpha;
  const col = a.tracking ? verdictColor() : C.mid;
  const [bx, by, bw, bh] = a.target;
  const x = X(bx) - 6;
  const y = Y(by) - 6;
  const w = bw * cr.cw + 12;
  const h = bh * cr.ch + 12;

  if (a.contours) {
    ctx.strokeStyle = 'rgba(236,236,241,0.55)';
    ctx.lineWidth = 1;
    ctx.shadowColor = col;
    ctx.shadowBlur = 6;
    for (const [name, flat] of Object.entries(a.contours)) {
      ctx.beginPath();
      for (let i = 0; i < flat.length; i += 2) {
        const px = X(flat[i]);
        const py = Y(flat[i + 1]);
        if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      }
      if (!name.startsWith('brow') && name !== 'nose') ctx.closePath();
      ctx.stroke();
    }
    ctx.shadowBlur = 0;
    ctx.fillStyle = hexA(col, 0.75);
    const d = a.dots || [];
    for (let i = 0; i < d.length; i += 2) ctx.fillRect(X(d[i]) - 0.8, Y(d[i + 1]) - 0.8, 1.6, 1.6);
  }
  brackets(ctx, x, y, w, h, col, Math.max(10, Math.min(w, h) * 0.2));

  const score = S.gauge.shown;
  const label = a.tracking ? `CUSTOMER${score != null ? ` · RISK ${Math.round(score)}` : ''}` : 'LOCKING ON';
  ctx.font = '600 11px "JetBrains Mono", monospace';
  const tw = ctx.measureText(label).width + 16;
  const ly = y - 26 < cr.oy ? y + h + 6 : y - 26;
  ctx.fillStyle = 'rgba(8,8,12,0.78)';
  ctx.strokeStyle = hexA(col, 0.7);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, ly, tw, 20, 6);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = col;
  ctx.textBaseline = 'middle';
  ctx.fillText(label, x + 8, ly + 10.5);
  ctx.globalAlpha = 1;
}

function drawThumb() {
  const c = $('#thumb');
  const ctx = c.getContext('2d');
  const a = S.ack;
  const v = $('#preview');
  const fresh = a && a.target && S.phase === 'live' && performance.now() - S.ackAt < 1500 && v.videoWidth;
  if (!fresh) { ctx.clearRect(0, 0, c.width, c.height); return; }
  const [bx, by, bw, bh] = a.target;
  const side = 1.5 * Math.max(bw * v.videoWidth, bh * v.videoHeight);
  const cx = (bx + bw / 2) * v.videoWidth;
  const cy = (by + bh / 2) * v.videoHeight;
  ctx.drawImage(v, cx - side / 2, cy - side / 2, side, side, 0, 0, c.width, c.height);
}

// =====================================================================
// Gauge
// =====================================================================
const G = { cx: 120, cy: 112, r: 84, a0: 135, sweep: 270 };
const polar = (deg, r = G.r) => [G.cx + r * Math.cos((deg * Math.PI) / 180), G.cy + r * Math.sin((deg * Math.PI) / 180)];

function buildGauge() {
  const [sx, sy] = polar(G.a0);
  const [ex, ey] = polar(G.a0 + G.sweep);
  const d = `M ${sx} ${sy} A ${G.r} ${G.r} 0 1 1 ${ex} ${ey}`;
  ['#g-track', '#g-val', '#g-glow'].forEach((id) => $(id).setAttribute('d', d));
  G.len = $('#g-val').getTotalLength();
  ['#g-val', '#g-glow'].forEach((id) => {
    $(id).style.strokeDasharray = G.len;
    $(id).style.strokeDashoffset = G.len;
  });
  const ticks = [];
  for (let i = 0; i <= 20; i++) {
    const v = i * 5;
    const deg = G.a0 + (G.sweep * v) / 100;
    const major = v % 25 === 0;
    const [x1, y1] = polar(deg, G.r + 11);
    const [x2, y2] = polar(deg, G.r + (major ? 18 : 15));
    ticks.push(`<line class="g-tick${major ? ' major' : ''}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`);
    if (major && v > 0 && v < 100) {
      const [lx, ly] = polar(deg, G.r + 30);
      ticks.push(`<text class="g-lab" x="${lx}" y="${ly}">${v}</text>`);
    }
  }
  $('#g-ticks').innerHTML = ticks.join('');
}

function tweenGauge(dt) {
  const g = S.gauge;
  if (g.target == null) {
    g.shown = null;
  } else if (g.shown == null) {
    g.shown = 0;
  }
  if (g.shown != null) g.shown += (g.target - g.shown) * (1 - Math.exp(-dt / 0.45));
  const p = g.shown == null ? 0 : clamp(g.shown / 100);
  const off = G.len * (1 - p);
  $('#g-val').style.strokeDashoffset = off;
  $('#g-glow').style.strokeDashoffset = off;
  const [kx, ky] = polar(G.a0 + G.sweep * p);
  const knob = $('#g-knob');
  knob.setAttribute('cx', kx);
  knob.setAttribute('cy', ky);
  knob.style.opacity = g.shown == null ? 0 : 1;
  $('#g-num').textContent = g.shown == null ? '—' : Math.round(g.shown);
}

// =====================================================================
// Region selection
// =====================================================================
function setSelecting(on) {
  S.selecting = on;
  S.drag = null;
  $('#stage').classList.toggle('selecting', on);
  $('#btn-region').classList.toggle('on', on);
}

function toNorm(e) {
  const r = $('#stage').getBoundingClientRect();
  const cr = S.content;
  if (!cr) return null;
  return { x: clamp((e.clientX - r.left - cr.ox) / cr.cw), y: clamp((e.clientY - r.top - cr.oy) / cr.ch) };
}

function initRegion() {
  const stage = $('#stage');
  $('#btn-region').addEventListener('click', () => setSelecting(!S.selecting));
  $('#btn-auto').addEventListener('click', () => {
    S.roi = null;
    $('#btn-auto').hidden = true;
    sendConfig({ roi: null });
    toast('Back to automatic face finding across the whole screen.');
  });
  stage.addEventListener('pointerdown', (e) => {
    if (!S.selecting) return;
    const p = toNorm(e);
    if (!p) return;
    S.drag = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
    stage.setPointerCapture(e.pointerId);
  });
  stage.addEventListener('pointermove', (e) => {
    if (!S.drag) return;
    const p = toNorm(e);
    if (p) Object.assign(S.drag, { x1: p.x, y1: p.y });
  });
  stage.addEventListener('pointerup', () => {
    if (!S.drag) return;
    const { x0, y0, x1, y1 } = S.drag;
    const roi = [Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0)];
    setSelecting(false);
    if (roi[2] < 0.04 || roi[3] < 0.04) { toast('That region is too small — drag a box around the whole video tile.'); return; }
    S.roi = roi.map((v) => +v.toFixed(4));
    $('#btn-auto').hidden = false;
    sendConfig({ roi: S.roi });
    toast('Region locked — only faces inside it are analysed.');
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && S.selecting) setSelecting(false); });
}

// =====================================================================
// Tabs & legend
// =====================================================================
function positionTabInk() {
  const on = $('#sig-tabs .tab.on');
  const ink = $('#sig-tabs .tab-ink');
  if (!on || !ink) return;
  ink.style.width = `${on.offsetWidth}px`;
  ink.style.transform = `translateX(${on.offsetLeft - 4}px)`;
}

function initTabs() {
  $$('#sig-tabs .tab').forEach((b) => b.addEventListener('click', () => {
    $$('#sig-tabs .tab').forEach((x) => x.classList.toggle('on', x === b));
    S.tab = b.dataset.tab;
    positionTabInk();
    renderSignals(true);
  }));
  $$('#legend .lg').forEach((b) => b.addEventListener('click', () => {
    const k = +b.dataset.series;
    if (S.series.has(k)) S.series.delete(k); else S.series.add(k);
    b.classList.toggle('on', S.series.has(k));
  }));
  window.addEventListener('resize', positionTabInk);
}

// =====================================================================
// Pop-out HUD (Document Picture-in-Picture — floats over the video call)
// =====================================================================
async function openHud() {
  if (!('documentPictureInPicture' in window)) { toast('The pop-out HUD needs Chrome or Edge 116 or newer.'); return; }
  if (S.hud) return;
  let pip;
  try {
    pip = await documentPictureInPicture.requestWindow({ width: 330, height: 220 });
  } catch (e) {
    toast(`Could not open the HUD: ${esc(e.message)}`);
    return;
  }
  const d = pip.document;
  for (const href of [...$$('link[rel="stylesheet"]')].map((l) => l.href)) {
    const link = d.createElement('link');
    link.rel = 'stylesheet';
    link.href = href;
    d.head.append(link);
  }
  d.title = 'FraudShield HUD';
  d.body.className = 'hud';
  d.body.innerHTML = `
    <div class="hud-top">
      <div class="hud-gauge"><div class="mring" id="h-ring"><span class="mnum" id="h-num">—</span></div></div>
      <div><div class="hud-verdict" id="h-verdict">Calibrating</div><div class="hud-action" id="h-action"></div></div>
    </div>
    <div class="hud-rows">${['face', 'voice', 'sync'].map((k) => `
      <div class="hud-row" style="--c:var(--${k})"><span>${MOD_NAME[k]}</span><div class="bar"><span id="h-${k}"></span></div><b id="h-${k}-v">—</b></div>`).join('')}
    </div>`;
  pip.addEventListener('pagehide', () => { S.hud = null; });
  S.hud = pip;
  updateHud();
}

function updateHud() {
  const pip = S.hud;
  if (!pip || !S.last) return;
  const d = pip.document;
  const o = S.last.overall;
  d.body.dataset.verdict = o.verdict;
  const ring = d.getElementById('h-ring');
  ring.style.setProperty('--p', o.score ?? 0);
  ring.style.setProperty('--rc', VERDICT_COLOR[o.verdict] || C.accent);
  d.getElementById('h-num').textContent = o.score == null ? '—' : Math.round(o.score);
  d.getElementById('h-verdict').textContent = o.label;
  d.getElementById('h-action').textContent = o.action;
  for (const k of ['face', 'voice', 'sync']) {
    const m = S.last.modules[k];
    d.getElementById(`h-${k}`).style.width = `${m.score ?? 0}%`;
    d.getElementById(`h-${k}-v`).textContent = m.score == null ? '—' : Math.round(m.score);
  }
}

// =====================================================================
// Report export
// =====================================================================
function exportReport() {
  const st = S.last;
  if (!st) { toast('Nothing to export yet — let the call run for a few seconds.'); return; }
  const end = S.phase === 'stopped' ? S.stoppedAt : performance.now();
  const report = {
    product: 'FraudShield Live',
    generated_at: new Date().toISOString(),
    session_duration_s: +((end - S.startedAt) / 1000).toFixed(1),
    verdict: st.overall,
    modules: Object.fromEntries(Object.entries(st.modules).map(([k, m]) => [k, {
      status: m.status, score: m.score, confidence: m.confidence, top_finding: m.finding,
      checks: (m.signals || []).map((s) => ({ check: s.label, status: s.status, value: s.value, risk: s.risk, reliability: s.reliability, finding: s.message })),
    }])),
    timeline: S.timeline.map(([t, o, f, v, s]) => ({ t: +t.toFixed(1), overall: o, face: f, voice: v, lip_sync: s })),
    events: S.events,
    capture: { surface: S.settings?.displaySurface, width: S.settings?.width, height: S.settings?.height, region: S.roi },
    engine: S.hello?.backend,
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `fraudshield-report-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// =====================================================================
// Main loop
// =====================================================================
let lastFrame = performance.now();
function loop(now) {
  requestAnimationFrame(loop);
  const dt = Math.min(0.1, (now - lastFrame) / 1000);
  lastFrame = now;
  if (document.hidden || S.phase === 'landing') return;
  tweenGauge(dt);
  drawOverlay();
  drawThumb();
  drawTimeline();
  drawAudioScope();
  if (S.phase === 'live') $('#pill-timer').textContent = fmtClock((now - S.startedAt) / 1000);
}

function init() {
  envCheck();
  buildGauge();
  initRegion();
  initTabs();
  $('#btn-start').addEventListener('click', start);
  $('#btn-start-hero').addEventListener('click', start);
  $('#btn-stop').addEventListener('click', () => stop('Monitoring stopped.'));
  $('#btn-reset').addEventListener('click', newCustomer);
  $('#btn-hud').addEventListener('click', openHud);
  $('#btn-export').addEventListener('click', exportReport);
  $('#preview').addEventListener('loadedmetadata', (e) => {
    const v = e.target;
    if (v.videoWidth && v.videoHeight) $('#stage').style.aspectRatio = `${v.videoWidth} / ${v.videoHeight}`;
  });
  requestAnimationFrame(loop);
}

init();
