from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .system import Dynamics, TRUE_DYNAMICS, rollout


def prediction_error(model: Dynamics, panel: dict[str, np.ndarray]) -> float:
    x, a = panel["x"], panel["a"]
    prev = np.concatenate([np.zeros((len(x), 1)), a[:, :-2]], axis=1)
    pred = model.transition(x[:, :-1], a[:, :-1], prev)
    return float(np.mean((pred - x[:, 1:]) ** 2))


def dynamics_error(model: Dynamics, seed: int = 991, n: int = 20000) -> float:
    """Independent reference distribution, balanced over current/lagged treatment."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.5, 2.5, size=n)
    a = rng.binomial(1, 0.5, size=n)
    prev = rng.binomial(1, 0.5, size=n)
    return float(np.mean((model.transition(x, a, prev) -
                          TRUE_DYNAMICS.transition(x, a, prev)) ** 2))


def make_policy_pair(n: int, horizon: int, kind: str, rng: np.random.Generator):
    baseline = np.zeros((n, horizon))
    treated = np.zeros((n, horizon))
    if kind == "in_policy":
        for i in range(n):
            start = rng.integers(0, max(1, horizon - 3))
            duration = int(rng.integers(1, min(4, horizon - start) + 1))
            treated[i, start:start + duration] = 1
    elif kind == "shift":
        offsets = rng.integers(0, 3, size=n)
        for i, offset in enumerate(offsets):
            treated[i, offset::3] = 1
    elif kind == "sustained":
        treated[:] = 1
    else:
        raise ValueError(kind)
    return baseline, treated


def interventional_metrics(model: Dynamics, contexts: dict[str, np.ndarray], horizon: int,
                           kind: str, seed: int = 1234) -> dict[str, float]:
    rng = np.random.default_rng(seed + horizon + sum(map(ord, kind)))
    x0, prev = contexts["x0"], contexts["a_prev"]
    p0, p1 = make_policy_pair(len(x0), horizon, kind, rng)
    # Paired stochastic potential outcomes: the same exogenous shocks drive both worlds.
    noise = rng.normal(0, 0.08, size=(len(x0), horizon))
    true0 = rollout(TRUE_DYNAMICS, x0, prev, p0, noise)
    true1 = rollout(TRUE_DYNAMICS, x0, prev, p1, noise)
    pred0 = rollout(model, x0, prev, p0)
    pred1 = rollout(model, x0, prev, p1)
    tau_true, tau_pred = true1[:, 1:] - true0[:, 1:], pred1[:, 1:] - pred0[:, 1:]
    dte = float(np.mean((tau_pred - tau_true) ** 2))
    intervention = float(0.5 * (np.mean((pred0[:, 1:] - true0[:, 1:]) ** 2) +
                                np.mean((pred1[:, 1:] - true1[:, 1:]) ** 2)))
    return {"dte_error": dte, "interventional_error": intervention}


def residualized_spearman(df: pd.DataFrame, x: str, y: str, control: str) -> float:
    """Rank-transform, linearly residualize x/y on the control, then correlate."""
    ranks = df[[x, y, control]].rank(method="average").to_numpy(float)
    z = np.column_stack([np.ones(len(df)), ranks[:, 2]])
    rx = ranks[:, 0] - z @ np.linalg.lstsq(z, ranks[:, 0], rcond=None)[0]
    ry = ranks[:, 1] - z @ np.linalg.lstsq(z, ranks[:, 1], rcond=None)[0]
    return float(spearmanr(rx, ry).statistic)
