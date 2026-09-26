"""
Generates a synthetic baseline of "normal" banking-app sessions for one user,
plus a handful of injected anomalous sessions to demo fraud detection.

Features (all numeric, ready for Isolation Forest):
- hour_of_day        : 0-23, when the session started
- is_new_device       : 0/1, device fingerprint never seen before for this user
- is_new_ip_country    : 0/1, IP geolocation country differs from the user's usual country
- ip_asn_risk         : 0-1, risk score of the network provider (e.g. datacenter/VPN ASN = high)
- session_latency_ms   : round-trip latency, bots/relays often show unusual latency
- mouse_entropy       : 0-1, irregularity of mouse movement (low = robotic/scripted)
- typing_cadence_cv    : coefficient of variation of keystroke intervals (low = scripted)
- touch_pressure_std   : std-dev of touch pressure on mobile (0 on desktop sessions)

Run:  python data/generate_device_sessions.py
Produces data/sessions.csv used by modules/device_module.py and app.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N_NORMAL = 220
N_FRAUD = 12

OUT_PATH = Path(__file__).parent / "sessions.csv"


def normal_sessions(n):
    return pd.DataFrame({
        "hour_of_day": RNG.normal(loc=13, scale=3.5, size=n).clip(0, 23).round().astype(int),
        "is_new_device": RNG.choice([0, 1], size=n, p=[0.92, 0.08]),
        "is_new_ip_country": RNG.choice([0, 1], size=n, p=[0.97, 0.03]),
        "ip_asn_risk": RNG.beta(1.5, 8, size=n),
        "session_latency_ms": RNG.normal(loc=80, scale=20, size=n).clip(10, None),
        "mouse_entropy": RNG.normal(loc=0.72, scale=0.08, size=n).clip(0, 1),
        "typing_cadence_cv": RNG.normal(loc=0.55, scale=0.1, size=n).clip(0, 1),
        "touch_pressure_std": RNG.normal(loc=0.35, scale=0.08, size=n).clip(0, 1),
        "label": 0,
    })


def fraud_sessions(n):
    return pd.DataFrame({
        "hour_of_day": RNG.choice([1, 2, 3, 4, 23], size=n),
        "is_new_device": RNG.choice([0, 1], size=n, p=[0.15, 0.85]),
        "is_new_ip_country": RNG.choice([0, 1], size=n, p=[0.25, 0.75]),
        "ip_asn_risk": RNG.beta(6, 2, size=n),
        "session_latency_ms": RNG.normal(loc=240, scale=60, size=n).clip(10, None),
        "mouse_entropy": RNG.normal(loc=0.15, scale=0.06, size=n).clip(0, 1),
        "typing_cadence_cv": RNG.normal(loc=0.08, scale=0.04, size=n).clip(0, 1),
        "touch_pressure_std": RNG.normal(loc=0.02, scale=0.02, size=n).clip(0, 1),
        "label": 1,
    })


if __name__ == "__main__":
    df = pd.concat([normal_sessions(N_NORMAL), fraud_sessions(N_FRAUD)], ignore_index=True)
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} sessions ({N_FRAUD} labelled fraud, used only for demo evaluation) -> {OUT_PATH}")
