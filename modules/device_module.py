"""
Module C - Device / Network Fingerprint Anomaly Detection.

Unsupervised: fits an IsolationForest on a baseline of "normal" sessions
(no fraud labels needed at fit time - realistic for a hackathon with no
real fraud data) and scores new sessions by how much of an outlier they are.
A light rule-based layer adds human-readable reasons on top of the ML score.

Usage:
    from modules.device_module import DeviceRiskModel
    model = DeviceRiskModel.fit_from_csv("data/sessions.csv")
    score, reasons = model.score(session_dict)
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES = [
    "hour_of_day", "is_new_device", "is_new_ip_country", "ip_asn_risk",
    "session_latency_ms", "mouse_entropy", "typing_cadence_cv", "touch_pressure_std",
]


@dataclass
class DeviceRiskModel:
    model: IsolationForest
    feature_means: dict = field(default_factory=dict)
    feature_stds: dict = field(default_factory=dict)

    @classmethod
    def fit_from_csv(cls, csv_path: str, contamination: float = 0.06):
        df = pd.read_csv(csv_path)
        X = df[FEATURES].values
        clf = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
        ).fit(X)
        return cls(
            model=clf,
            feature_means=df[FEATURES].mean().to_dict(),
            feature_stds=df[FEATURES].std().replace(0, 1).to_dict(),
        )

    def _rule_reasons(self, session: dict) -> list[str]:
        reasons = []
        if session.get("is_new_device"):
            reasons.append("New/unseen device fingerprint")
        if session.get("is_new_ip_country"):
            reasons.append("Login from an unfamiliar country/IP")
        if session.get("ip_asn_risk", 0) > 0.6:
            reasons.append("Network provider flagged as high-risk (VPN/proxy/datacenter ASN)")
        hour = session.get("hour_of_day", 12)
        if hour <= 5 or hour == 23:
            reasons.append(f"Unusual login hour ({hour}:00)")
        if session.get("mouse_entropy", 1) < 0.3:
            reasons.append("Mouse movement unnaturally uniform (possible automation)")
        if session.get("typing_cadence_cv", 1) < 0.2:
            reasons.append("Keystroke timing unnaturally regular (possible scripted input)")
        if session.get("session_latency_ms", 0) > 180:
            reasons.append("High session latency (possible relay/remote-access tooling)")
        return reasons

    def score(self, session: dict) -> tuple[float, list[str]]:
        x = np.array([[session.get(f, self.feature_means[f]) for f in FEATURES]])
        # decision_function: higher = more normal. Flip + rescale to 0-100 risk.
        raw = self.model.decision_function(x)[0]
        # empirically raw ranges roughly [-0.2, 0.25]; clip + rescale
        risk = float(np.clip((0.25 - raw) / 0.45, 0, 1) * 100)
        reasons = self._rule_reasons(session)
        if not reasons and risk > 40:
            reasons.append("Session pattern statistically unusual vs. this user's history")
        return round(risk, 1), reasons
