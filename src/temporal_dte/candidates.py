from __future__ import annotations

import numpy as np
import pandas as pd

from .system import Dynamics, TRUE_DYNAMICS


def make_candidates(n: int = 100, seed: int = 7) -> list[Dynamics]:
    """Construct misspecified candidates; the exact oracle is deliberately excluded."""
    rng = np.random.default_rng(seed)
    models: list[Dynamics] = []
    for i in range(n):
        severity = rng.choice([0.06, 0.12, 0.22, 0.35], p=[0.2, 0.35, 0.3, 0.15])
        use_sin = bool(rng.random() > 0.18)
        use_state = bool(rng.random() > 0.22)
        use_lag = bool(rng.random() > 0.25)
        models.append(Dynamics(
            alpha=float(0.7 + rng.normal(0, severity)),
            sin_coef=float(0.2 + rng.normal(0, severity * 0.7)),
            treatment_base=float(0.2 + rng.normal(0, severity)),
            treatment_state=float(0.2 + rng.normal(0, severity)),
            lag_coef=float(0.1 + rng.normal(0, severity * 0.65)),
            state_scale=float(np.exp(rng.normal(0, severity * 0.8))),
            use_sin=use_sin,
            use_state_treatment=use_state,
            use_lag=use_lag,
        ))
    return models


def candidate_table(models: list[Dynamics]) -> pd.DataFrame:
    rows = []
    for i, model in enumerate(models):
        row = {"model_id": i, **model.to_dict()}
        row["is_true"] = False
        row["compressed"] = not model.use_lag
        rows.append(row)
    return pd.DataFrame(rows)
