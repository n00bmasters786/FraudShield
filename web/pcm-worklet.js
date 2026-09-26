// Collects the shared system audio into ~43 ms mono Int16 chunks and hands them to the
// capture worker through a MessagePort (never touching the possibly-throttled main thread).
class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.size = 2048;
    this.buf = new Float32Array(this.size);
    this.n = 0;
    this.out = null;
    this.port.onmessage = (e) => { if (e.data && e.data.port) this.out = e.data.port; };
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input.length) {
      const a = input[0];
      const b = input[1];
      for (let i = 0; i < a.length; i++) {
        this.buf[this.n++] = b ? 0.5 * (a[i] + b[i]) : a[i];
        if (this.n === this.size) this.flush();
      }
    }
    return true;
  }

  flush() {
    const pcm = new Int16Array(this.size);
    for (let i = 0; i < this.size; i++) {
      const v = Math.max(-1, Math.min(1, this.buf[i]));
      pcm[i] = v < 0 ? v * 32768 : v * 32767;
    }
    this.n = 0;
    if (this.out) this.out.postMessage({ pcm: pcm.buffer, sr: sampleRate }, [pcm.buffer]);
  }
}

registerProcessor('pcm-capture', PcmCapture);
