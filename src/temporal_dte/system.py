from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class Dynamics:
    alpha: float = 0.7
    sin_coef: float = 0.2
    treatment_base: float = 0.2
    treatment_state: float = 0.2
    lag_coef: float = 0.1
    state_scale: float = 1.0
    use_sin: bool = True
    use_state_treatment: bool = True
    use_lag: bool = True

    def transition(self, x, a, a_prev):
        x = np.asarray(x)
        value = self.alpha * x
        if self.use_sin:
            value = value + self.sin_coef * np.sin(self.state_scale * x)
        effect = self.treatment_base
        if self.use_state_treatment:
            effect = effect + self.treatment_state * np.tanh(x)
        value = value + effect * np.asarray(a)
        if self.use_lag:
            value = value + self.lag_coef * np.asarray(a_prev)
        return value

    def to_dict(self):
        return asdict(self)


TRUE_DYNAMICS = Dynamics()


def rollout(model: Dynamics, x0: np.ndarray, a_prev: np.ndarray, policy: np.ndarray,
            noise: np.ndarray | None = None) -> np.ndarray:
    """Roll out batched trajectories. policy/noise shapes are (n, horizon)."""
    x0 = np.asarray(x0, dtype=float)
    policy = np.asarray(policy, dtype=float)
    n, horizon = policy.shape
    out = np.empty((n, horizon + 1), dtype=float)
    out[:, 0] = x0
    prev = np.asarray(a_prev, dtype=float).copy()
    if noise is None:
        noise = np.zeros((n, horizon), dtype=float)
    for h in range(horizon):
        out[:, h + 1] = model.transition(out[:, h], policy[:, h], prev) + noise[:, h]
        prev = policy[:, h]
    return out


def policy_actions(rng: np.random.Generator, x: np.ndarray, prev: np.ndarray,
                   policy: str) -> np.ndarray:
    if policy == "observational":
        logits = -1.0 + 1.4 * x + 1.0 * prev
        return rng.binomial(1, 1 / (1 + np.exp(-logits))).astype(float)
    if policy == "randomized":
        return rng.binomial(1, 0.35, size=x.shape).astype(float)
    raise ValueError(f"Unknown policy: {policy}")


def simulate_panel(n: int, length: int, seed: int, policy: str = "observational",
                   sigma: float = 0.08) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.empty((n, length), dtype=float)
    a = np.zeros((n, length), dtype=float)
    x[:, 0] = rng.normal(0, 0.8, size=n)
    prev = np.zeros(n)
    eps = rng.normal(0, sigma, size=(n, length - 1))
    for t in range(length - 1):
        a[:, t] = policy_actions(rng, x[:, t], prev, policy)
        x[:, t + 1] = TRUE_DYNAMICS.transition(x[:, t], a[:, t], prev) + eps[:, t]
        prev = a[:, t]
    a[:, -1] = policy_actions(rng, x[:, -1], prev, policy)
    return {"x": x, "a": a}
