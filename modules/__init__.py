"""FraudShield engine modules."""

import os

# Same thread settings as app.py, for the offline tools (tools.evaluate etc.): small per-frame
# work, so cap the numeric libraries' thread pools and stop OpenMP workers from spin-waiting.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "2")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")
