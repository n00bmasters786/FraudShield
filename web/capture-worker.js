// FraudShield capture worker.
// Owns the WebSocket to the analysis engine, JPEG-encodes screen frames and forwards
// system-audio PCM. It runs off the main thread on purpose: browsers throttle timers in
// hidden tabs, and the officer will usually have the video call — not this dashboard —
// in front. Frames arrive through a MediaStreamTrackProcessor stream, which isn't throttled.

const KIND_FRAME = 1;
const KIND_AUDIO = 2;

let ws = null;
let wsUrl = null;
let running = false;
let fps = 15;
let maxWidth = 1920;
let quality = 0.82;
let inFlight = false;
let sentAt = -1e9;
let canvas = null;
let ctx = null;
let reconnectTimer = null;
let lastConfig = null;
const meter = { frames: 0, bytes: 0, audioBytes: 0, since: performance.now() };

const epochNow = () => performance.timeOrigin + performance.now();

function packet(kind, extra, ts, body) {
  const out = new Uint8Array(16 + body.byteLength);
  const dv = new DataView(out.buffer);
  dv.setUint8(0, kind);
  dv.setUint32(4, extra >>> 0, true);
  dv.setFloat64(8, ts, true);
  out.set(new Uint8Array(body), 16);
  return out.buffer;
}

function sendJson(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}

function connect() {
  clearTimeout(reconnectTimer);
  postMessage({ type: 'conn', state: 'connecting' });
  ws = new WebSocket(wsUrl);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    inFlight = false;
    postMessage({ type: 'conn', state: 'open' });
    if (lastConfig) sendJson(lastConfig);
  };
  ws.onclose = () => {
    postMessage({ type: 'conn', state: 'closed' });
    if (running) reconnectTimer = setTimeout(connect, 1500);
  };
  ws.onmessage = (e) => {
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }
    if (msg.type === 'ack') inFlight = false;
    postMessage(msg);
  };
}

function readyForFrame(t) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return false;
  if (t - sentAt < 1000 / fps) return false;
  // one frame in flight at a time; give up waiting for a lost ack after 1.5 s
  return !(inFlight && t - sentAt < 1500);
}

async function encodeAndSend(source, w0, h0, ts) {
  const s = Math.min(1, maxWidth / w0);
  const w = Math.round(w0 * s);
  const h = Math.round(h0 * s);
  if (!canvas || canvas.width !== w || canvas.height !== h) {
    canvas = new OffscreenCanvas(w, h);
    ctx = canvas.getContext('2d', { alpha: false });
  }
  ctx.drawImage(source, 0, 0, w, h);
  const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality });
  const buf = await blob.arrayBuffer();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(packet(KIND_FRAME, 0, ts, buf));
    meter.frames += 1;
    meter.bytes += buf.byteLength;
  } else {
    inFlight = false;
  }
}

async function pump(readable) {
  const reader = readable.getReader();
  while (running) {
    let r;
    try { r = await reader.read(); } catch { break; }
    if (r.done) break;
    const frame = r.value;
    const t = performance.now();
    if (!readyForFrame(t)) { frame.close(); continue; }
    inFlight = true;
    sentAt = t;
    const ts = epochNow();
    try {
      await encodeAndSend(frame, frame.displayWidth, frame.displayHeight, ts);
    } catch (err) {
      inFlight = false;
      postMessage({ type: 'worker-error', message: String(err) });
    } finally {
      frame.close();
    }
  }
  try { reader.releaseLock(); } catch { /* stream already closed */ }
  postMessage({ type: 'capture-ended' });
}

function report() {
  const now = performance.now();
  const dt = (now - meter.since) / 1000;
  if (dt >= 1) {
    postMessage({
      type: 'wstats',
      fps: meter.frames / dt,
      kbps: (8 * (meter.bytes + meter.audioBytes)) / 1000 / dt,
    });
    meter.frames = meter.bytes = meter.audioBytes = 0;
    meter.since = now;
  }
}
setInterval(report, 1000);

self.onmessage = async (e) => {
  const m = e.data;
  switch (m.type) {
    case 'start':
      running = true;
      wsUrl = m.wsUrl;
      fps = m.fps || fps;
      maxWidth = m.maxWidth || maxWidth;
      connect();
      if (m.readable) pump(m.readable);
      break;
    case 'bitmap': {   // fallback path for browsers without MediaStreamTrackProcessor
      const t = performance.now();
      if (!readyForFrame(t)) { m.bitmap.close(); break; }
      inFlight = true;
      sentAt = t;
      try { await encodeAndSend(m.bitmap, m.bitmap.width, m.bitmap.height, m.ts); }
      catch { inFlight = false; }
      finally { m.bitmap.close(); }
      break;
    }
    case 'audio-port':
      m.port.onmessage = (ev) => {
        const { pcm, sr } = ev.data;
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(packet(KIND_AUDIO, sr, epochNow(), pcm));
          meter.audioBytes += pcm.byteLength;
        }
      };
      break;
    case 'config':
      lastConfig = { ...(lastConfig || {}), ...m.msg, type: 'config' };
      sendJson(lastConfig);
      break;
    case 'reset':
      sendJson({ type: 'reset' });
      break;
    case 'stop':
      running = false;
      clearTimeout(reconnectTimer);
      if (ws) ws.close();
      break;
  }
};
