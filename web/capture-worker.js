// FraudShield capture worker.
// Owns the WebSocket to the analysis engine, JPEG-encodes screen frames and forwards
// system-audio PCM. It runs off the main thread on purpose: browsers throttle timers in
// hidden tabs, and the officer will usually have the video call — not this dashboard —
// in front. Frames arrive through a MediaStreamTrackProcessor stream, which isn't throttled.
//
// To stay light, it sends what the engine asks for in each ack (`want`): the whole screen only
// while searching for the customer (a few times a second) and every ~2.5 s while tracking;
// otherwise just the region around the face, cropped at native resolution. Encoding, sending
// and decoding a ~400 px crop costs a fraction of a full 1080p frame, with the same face pixels.

const KIND_FRAME = 1;
const KIND_AUDIO = 2;

let ws = null;
let wsUrl = null;
let running = false;
let fps = 15;                     // upper bound; the engine's `want.fps` can ask for less
let maxWidth = 1920;
const FULL_QUALITY = 0.8;         // whole-screen frames (face search only)
const CROP_QUALITY = 0.82;        // face region: same quality the detectors were calibrated on (full frames were 0.82)
let want = { full: true, fps: 3, rect: null };
let inFlight = false;
let sentAt = -1e9;
let canvas = null;
let ctx = null;
let reconnectTimer = null;
let lastConfig = null;
const meter = { frames: 0, bytes: 0, audioBytes: 0, since: performance.now() };

const epochNow = () => performance.timeOrigin + performance.now();

function packet(kind, extra, ts, body, region = null) {
  const pre = region ? 8 : 0;
  const out = new Uint8Array(16 + pre + body.byteLength);
  const dv = new DataView(out.buffer);
  dv.setUint8(0, kind);
  dv.setUint32(4, (extra | (region ? 1 : 0)) >>> 0, true);
  dv.setFloat64(8, ts, true);
  if (region) region.forEach((v, i) => dv.setUint16(16 + 2 * i, v, true));   // x, y, screen W, screen H
  out.set(new Uint8Array(body), 16 + pre);
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
    want = { full: true, fps: 3, rect: null };   // a fresh engine session starts by searching
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
    if (msg.type === 'ack') {
      inFlight = false;
      if (msg.want) want = msg.want;
    }
    postMessage(msg);
  };
}

function readyForFrame(t) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return false;
  // 15 % tolerance: capture delivers frames at ~the same rate, and a frame arriving a hair early
  // must not be skipped (that would halve the analysed frame rate)
  if (t - sentAt < 850 / Math.min(fps, want.fps || fps)) return false;
  // one frame in flight at a time; give up waiting for a lost ack after 1.5 s
  return !(inFlight && t - sentAt < 1500);
}

async function encodeAndSend(source, w0, h0, ts) {
  // engine coordinates = the screen scaled to at most maxWidth
  const s = Math.min(1, maxWidth / w0);
  const W = Math.round(w0 * s);
  const H = Math.round(h0 * s);
  let region = null;
  let w = W;
  let h = H;
  if (!want.full && want.rect) {
    const [rx, ry, rw, rh] = want.rect;
    const x = Math.max(0, Math.round(rx * W));
    const y = Math.max(0, Math.round(ry * H));
    w = Math.min(W - x, Math.round(rw * W));
    h = Math.min(H - y, Math.round(rh * H));
    if (w >= 32 && h >= 32) region = [x, y, W, H];
    else { w = W; h = H; }
  }
  if (!canvas || canvas.width !== w || canvas.height !== h) {
    canvas = new OffscreenCanvas(w, h);
    ctx = canvas.getContext('2d', { alpha: false });
  }
  if (region) ctx.drawImage(source, region[0] / s, region[1] / s, w / s, h / s, 0, 0, w, h);
  else ctx.drawImage(source, 0, 0, w, h);
  const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: region ? CROP_QUALITY : FULL_QUALITY });
  const buf = await blob.arrayBuffer();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(packet(KIND_FRAME, 0, ts, buf, region));
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
