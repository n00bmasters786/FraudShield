"""
FraudShield Live — real-time deepfake detection for video KYC.

The banking officer opens the dashboard, shares their entire screen (with
system audio), and runs the video KYC call as usual. The browser streams the
screen frames and the call audio over a WebSocket; this server finds the
customer's face on the screen, analyses face, voice and lip-sync, and pushes
a live deepfake-risk score back to the dashboard every second.

Run:
    venv\\Scripts\\python app.py            then open http://localhost:8000
"""

import argparse
import asyncio
import json
import struct
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from modules.live import LiveSession

ROOT = Path(__file__).parent
WEB = ROOT / "web"
ANALYSIS_PERIOD_S = 1.0

# binary packets from the browser: <u8 kind, 3 pad, u32 extra, f64 epoch-ms> + payload
HEADER = struct.Struct("<BxxxId")
KIND_FRAME, KIND_AUDIO = 1, 2

app = FastAPI(title="FraudShield Live")


@app.get("/api/health")
def health():
    return {"ok": True}


@app.websocket("/ws")
async def session_socket(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    # Face Mesh isn't thread-safe: every face call goes through one thread, analysis through another
    frame_pool = ThreadPoolExecutor(1, thread_name_prefix="fs-frame")
    analysis_pool = ThreadPoolExecutor(1, thread_name_prefix="fs-analysis")
    session = await loop.run_in_executor(frame_pool, LiveSession)
    outbox: asyncio.Queue = asyncio.Queue(maxsize=64)
    frame_busy = False

    async def sender():
        while True:
            msg = await outbox.get()
            await ws.send_text(json.dumps(msg, separators=(",", ":")))

    async def analyzer():
        while True:
            await asyncio.sleep(ANALYSIS_PERIOD_S)
            state = await loop.run_in_executor(analysis_pool, session.analyze)
            if state:
                await outbox.put(state)

    async def handle_frame(ts, jpeg):
        nonlocal frame_busy
        try:
            ack = await loop.run_in_executor(frame_pool, session.process_frame, ts, jpeg)
            await outbox.put(ack)
        except Exception as e:  # keep the session alive on a bad frame
            await outbox.put({"type": "ack", "error": str(e)})
        finally:
            frame_busy = False

    tasks = [asyncio.create_task(sender()), asyncio.create_task(analyzer())]
    await outbox.put({"type": "hello", "backend": session.face.backend, "challenge": session.challenge})
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data:
                if len(data) < HEADER.size:
                    continue
                kind, extra, ts = HEADER.unpack_from(data)
                payload = data[HEADER.size:]
                if kind == KIND_FRAME:
                    if frame_busy:   # client normally waits for the ack; drop if it didn't
                        await outbox.put({"type": "ack", "dropped": True})
                        continue
                    frame_busy = True
                    asyncio.create_task(handle_frame(ts, payload))
                elif kind == KIND_AUDIO:
                    session.add_audio(ts, extra, payload)
                continue
            text = msg.get("text")
            if not text:
                continue
            m = json.loads(text)
            if m.get("type") == "config":
                await loop.run_in_executor(frame_pool, session.configure, m)
            elif m.get("type") == "reset":
                await loop.run_in_executor(frame_pool, session.reset)
                await outbox.put({"type": "hello", "backend": session.face.backend, "challenge": session.challenge})
    except WebSocketDisconnect:
        pass
    finally:
        for t in tasks:
            t.cancel()
        frame_pool.submit(session.close)
        frame_pool.shutdown(wait=False)
        analysis_pool.shutdown(wait=False, cancel_futures=True)


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FraudShield Live dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    url = f"http://localhost:{args.port}"
    print(f"\n  FraudShield Live  →  {url}\n  (use Chrome or Edge; screen capture needs localhost or HTTPS)\n")
    if not args.no_browser:
        threading.Timer(1.2, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
