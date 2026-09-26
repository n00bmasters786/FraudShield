"""
Evaluate — and calibrate — FraudShield on labelled recordings.

Every clip runs through exactly the live pipeline (face finder → Face Mesh →
forensic checks → trained detector → liveness / synthesis group scoring).
The report shows, per check, how well it separates real from fake (ROC AUC)
and how the final face score lands against the dashboard thresholds.

    # any number of real / fake folders (sub-folders are scanned)
    venv\\Scripts\\python -m tools.evaluate --real clips\\real --fake clips\\fake

    # simulate the video-call path: 640 px stream, heavy JPEG, upscaled on screen
    venv\\Scripts\\python -m tools.evaluate --real ... --fake ... --degrade call

    # fit the detectors' calibration on these clips → models/calibration.json
    venv\\Scripts\\python -m tools.evaluate --real ... --fake ... --degrade call --calibrate

    # honest numbers: fit on every dataset but one, test on the held-out one
    venv\\Scripts\\python -m tools.evaluate --real d1\\real d2\\real --fake d1\\fake d2\\fake --cross

Per-clip results are cached (--cache), so re-calibrating doesn't re-run video.
Calibrate on recordings from your own KYC channel whenever you can: the fitted
mapping absorbs that channel's codec, resolution and lighting.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from modules import calibration
from modules.detectors import CALIBRATION_FILE, MODELS_DIR
from modules.scoring import Signal, group_score

VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def make_degrade(width, quality):
    """A `width`-px call stream at JPEG `quality`, upscaled back into the on-screen tile."""
    def degrade(frame):
        h, w = frame.shape[:2]
        if w > width:
            small = cv2.resize(frame, (width, max(2, round(h * width / w))), interpolation=cv2.INTER_AREA)
        else:
            small = frame
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.resize(cv2.imdecode(buf, cv2.IMREAD_COLOR), (w, h), interpolation=cv2.INTER_LINEAR)
    return degrade


DEGRADE = {
    "none": None,
    "call": make_degrade(1280, 70),       # a decent 720p video call
    "call-low": make_degrade(640, 40),    # poor bandwidth: 360p stream, blocky
}


def auc(scores, labels):
    """ROC AUC via the rank statistic (ties count half)."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def fit_logistic(P, y, prior_w, lam=8.0):
    """risk = sigmoid(Σ wᵢ·logit(pᵢ) + b) by Newton's method, L2-shrunk toward the default weights."""
    X = np.array([[calibration.logit(v) for v in row] for row in P])
    X = np.hstack([X, np.ones((len(X), 1))])
    y = np.asarray(y, float)
    prior = np.array(list(prior_w) + [0.0])
    th = prior.copy()
    for _ in range(100):
        q = 1 / (1 + np.exp(-np.clip(X @ th, -30, 30)))
        step = np.linalg.solve((X * (q * (1 - q))[:, None]).T @ X + lam * np.eye(len(th)),
                               X.T @ (q - y) + lam * (th - prior))
        th -= step
        if np.abs(step).max() < 1e-7:
            break
    return [float(v) for v in th[:-1]], float(th[-1])


def collect(dirs, exts, label):
    out = []
    for d in dirs or []:
        root = Path(d)
        dataset = root.parent.name if root.name.lower() in ("real", "fake") else root.name
        out += [(p, label, dataset) for p in sorted(root.rglob("*")) if p.suffix.lower() in exts]
    return out


# ---------------------------------------------------------------- running
def run_video(args, cache):
    from modules.face_module import score_video
    clips = collect(args.real, VIDEO, 0) + collect(args.fake, VIDEO, 1)
    degrade = DEGRADE[args.degrade]
    rows = []
    for i, (path, label, dataset) in enumerate(clips, 1):
        key = f"video|{path}|{args.degrade}|{args.max_seconds}|{os.environ.get('FRAUDSHIELD_FACE_MODELS', '')}"
        if key not in cache:
            t0 = time.time()
            try:
                _, _, res = score_video(str(path), max_seconds=args.max_seconds, degrade=degrade)
            except Exception as e:
                print(f"[{i}/{len(clips)}] {path.name}: failed ({e})")
                continue
            det = res.get("details") or {}
            cache[key] = {"tracked": res.get("status") == "tracking", "signals": res.get("signals", []),
                          "per_model": (det.get("model") or {}).get("per_model"),
                          "models": (det.get("model") or {}).get("names"),
                          "flicker": (det.get("flicker") or {}).get("ratio"), "secs": round(time.time() - t0, 1)}
        rows.append({"clip": str(path), "name": path.name, "label": label, "dataset": dataset, **cache[key]})
        r = rows[-1]
        pm = r["per_model"]
        print(f"[{i}/{len(clips)}] {dataset:<9} {'FAKE' if label else 'real'}  "
              f"ckpts {'—' if not pm else ' '.join(f'{v:.2f}' for v in pm)}  {path.name}  ({r['secs']}s)")
    return rows


def run_audio(args, cache):
    from modules.voice_module import score_audio
    clips = collect(args.real_audio, AUDIO, 0) + collect(args.fake_audio, AUDIO, 1)
    rows = []
    for path, label, dataset in clips:
        key = f"audio|{path}"
        if key not in cache:
            _, _, det = score_audio(str(path))
            md = det.get("model") or {}
            cache[key] = {"tracked": True, "signals": det.get("signals", []),
                          "per_model": [md["p"]] if md.get("p") is not None else None}
        rows.append({"clip": str(path), "name": path.name, "label": label, "dataset": dataset, **cache[key]})
        print(f"{'FAKE' if label else 'real'}  P(fake) {rows[-1]['per_model']}  {path.name}")
    return rows


# ---------------------------------------------------------------- scoring
def cal_for(row, model_key, cal):
    k = calibration.key(model_key, row.get("models"))
    c = cal.get(k)
    n = len(row["per_model"])
    return c if c and len(c["w"]) == n else {"w": [1.0 / n] * n, "b": 0.0}


def rescore(row, model_key, cal):
    """Final module score for a cached clip under calibration `cal` (the detector signal is re-mapped)."""
    sigs = []
    for d in row["signals"]:
        d = {k: v for k, v in d.items() if k != "status"}
        if d["key"] in ("model", "voice_model") and row.get("per_model"):
            c = cal_for(row, model_key, cal)
            z = sum(w * calibration.logit(p) for w, p in zip(c["w"], row["per_model"])) + c["b"]
            d["risk"] = float(1 / (1 + np.exp(-np.clip(z, -30, 30))))
        sigs.append(Signal(**d))
    score, *_ = group_score(sigs)
    return score


def detector_risk(row, model_key, cal):
    if not row.get("per_model"):
        return None
    c = cal_for(row, model_key, cal)
    z = sum(w * calibration.logit(p) for w, p in zip(c["w"], row["per_model"])) + c["b"]
    return float(1 / (1 + np.exp(-np.clip(z, -30, 30))))


def report(rows, model_key, cal, title):
    rows = [r for r in rows if r["tracked"]]
    if not rows:
        return {}
    y = [r["label"] for r in rows]
    print(f"\n==== {title}: {sum(y)} fake / {len(y) - sum(y)} real clips")
    cols = {"FINAL module score": [rescore(r, model_key, cal) for r in rows],
            "trained detector (calibrated)": [detector_risk(r, model_key, cal) for r in rows]}
    for k in range(max((len(r["per_model"]) for r in rows if r.get("per_model")), default=0)):
        cols[f"  checkpoint {k} raw P(fake)"] = [r["per_model"][k] if r.get("per_model") else None for r in rows]
    for key in sorted({s["key"] for r in rows for s in r["signals"]} - {"model", "voice_model"}):
        cols[f"check · {key}"] = [next((s["risk"] for s in r["signals"] if s["key"] == key and s["reliability"] >= 0.2),
                                       None) for r in rows]
    if any(r.get("flicker") is not None for r in rows):
        cols["raw · flicker ratio"] = [r.get("flicker") for r in rows]
    out = {}
    for name, vals in cols.items():
        pairs = [(v, l) for v, l in zip(vals, y) if v is not None]
        a = auc(*zip(*pairs)) if pairs else float("nan")
        out[name] = {"auc": a, "n": len(pairs)}
        print(f"  {name:<34} AUC {a:5.2f}   (n={len(pairs)})")
    final = [(s, l) for s, l in zip(cols["FINAL module score"], y) if s is not None]
    for thr, name in ((35, "at Suspicious (≥35)"), (65, "at Likely deepfake (≥65)")):
        tp = sum(1 for s, l in final if l and s >= thr)
        fn = sum(1 for s, l in final if l and s < thr)
        fp = sum(1 for s, l in final if not l and s >= thr)
        tn = sum(1 for s, l in final if not l and s < thr)
        print(f"  {name:<34} fakes caught {tp}/{tp + fn}   genuine flagged {fp}/{fp + tn}")
        out[name] = {"tp": tp, "fn": fn, "fp": fp, "tn": tn}
    return out


def fit(rows, model_key):
    pts = [(r["per_model"], r["label"]) for r in rows if r["tracked"] and r.get("per_model")]
    if len({l for _, l in pts}) < 2 or len(pts) < 8:
        return None
    n = len(pts[0][0])
    prior = calibration.DEFAULTS[model_key]["w"]
    prior = prior if len(prior) == n else [1.0 / n] * n
    w, b = fit_logistic([p for p, _ in pts], [l for _, l in pts], prior)
    return {"w": [round(v, 4) for v in w], "b": round(b, 4), "n": len(pts)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", nargs="*", help="folders of genuine videos")
    ap.add_argument("--fake", nargs="*", help="folders of deepfake videos")
    ap.add_argument("--real-audio", nargs="*", help="folders of genuine speech")
    ap.add_argument("--fake-audio", nargs="*", help="folders of synthetic / cloned speech")
    ap.add_argument("--degrade", choices=list(DEGRADE), default="none")
    ap.add_argument("--max-seconds", type=float, default=10.0)
    ap.add_argument("--cache", default=str(MODELS_DIR / "eval_cache.json"), help="per-clip result cache")
    ap.add_argument("--calibrate", action="store_true", help="fit the detectors' calibration and save it")
    ap.add_argument("--cross", action="store_true", help="leave-one-dataset-out calibration report")
    args = ap.parse_args()

    cache_path = Path(args.cache)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    try:
        vrows, arows = run_video(args, cache), run_audio(args, cache)
    finally:
        cache_path.parent.mkdir(exist_ok=True)
        cache_path.write_text(json.dumps(cache))

    current = calibration.params()
    for rows, key, title in ((vrows, "face_model", "face"), (arows, "voice_model", "voice")):
        report(rows, key, current, f"{title} · current calibration")
        if args.cross:
            for held in sorted({r["dataset"] for r in rows}):
                c = fit([r for r in rows if r["dataset"] != held], key)
                if c:
                    held_rows = [r for r in rows if r["dataset"] == held]
                    ck = calibration.key(key, held_rows[0].get("models")) if held_rows else key
                    report(held_rows, key, {**current, ck: c},
                           f"{title} · calibrated WITHOUT {held}, tested on {held}  (w={c['w']} b={c['b']})")

    if args.calibrate:
        cal = json.loads(CALIBRATION_FILE.read_text()) if CALIBRATION_FILE.exists() else {}
        for rows, key in ((vrows, "face_model"), (arows, "voice_model")):
            c = fit(rows, key)
            if c:
                ck = calibration.key(key, rows[0].get("models"))
                cal[ck] = {**c, "degrade": args.degrade, "fitted": time.strftime("%Y-%m-%d"),
                           "datasets": sorted({r["dataset"] for r in rows})}
                print(f"\ncalibrated {ck}: w={c['w']} b={c['b']} on {c['n']} clips")
                report(rows, key, {**current, ck: c}, f"{ck} · in-sample with new calibration")
        MODELS_DIR.mkdir(exist_ok=True)
        CALIBRATION_FILE.write_text(json.dumps(cal, indent=2))
        print(f"saved → {CALIBRATION_FILE}")


if __name__ == "__main__":
    main()
