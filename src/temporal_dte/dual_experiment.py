from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass(frozen=True)
class Model:
    alpha: float
    sin_coef: float
    treatment_base: float
    treatment_state: float
    lag_coef: float
    state_scale: float = 1.0
    intercept: float = 0.0

    def step(self, x, a, lag):
        return (self.intercept + self.alpha * x + self.sin_coef * np.sin(self.state_scale * x)
                + (self.treatment_base + self.treatment_state * np.tanh(x)) * a
                + self.lag_coef * lag)


SYSTEMS = {
    "Linear": Model(0.72, 0.0, 0.30, 0.0, 0.12),
    "Nonlinear": Model(0.70, 0.20, 0.20, 0.20, 0.10),
}


def rollout(model, x0, lag0, actions):
    n, horizon = actions.shape
    out = np.empty((n, horizon + 1))
    out[:, 0] = x0
    lag = lag0.copy()
    for h in range(horizon):
        out[:, h + 1] = model.step(out[:, h], actions[:, h], lag)
        lag = actions[:, h]
    return out


def simulate(true_model, n=1000, length=128, seed=42):
    rng = np.random.default_rng(seed)
    x = np.empty((n, length))
    a = np.zeros((n, length))
    x[:, 0] = rng.normal(0, 0.75, n)
    lag = np.zeros(n)
    for t in range(length - 1):
        # State-dependent observational policy with persistence and limited exploration.
        p = 1 / (1 + np.exp(-(-1.25 + 1.5 * x[:, t] + 1.2 * lag)))
        a[:, t] = rng.binomial(1, p)
        x[:, t + 1] = true_model.step(x[:, t], a[:, t], lag) + rng.normal(0, 0.08, n)
        lag = a[:, t]
    return x, a


def candidates(true_model, nonlinear, n=100, seed=7):
    rng = np.random.default_rng(seed)
    result = []
    while len(result) < n:
        scale = rng.choice([0.05, 0.10, 0.18, 0.28], p=[.15, .35, .35, .15])
        m = Model(
            alpha=float(true_model.alpha + rng.normal(0, scale)),
            sin_coef=float(true_model.sin_coef + (rng.normal(0, .75 * scale) if nonlinear else 0)),
            treatment_base=float(true_model.treatment_base + rng.normal(0, 1.1 * scale)),
            treatment_state=float(true_model.treatment_state + (rng.normal(0, scale) if nonlinear else 0)),
            lag_coef=float(true_model.lag_coef + rng.normal(0, .75 * scale)),
            state_scale=float(np.exp(rng.normal(0, .5 * scale))) if nonlinear else 1.0,
            # Treatment-independent drift is a controlled DTE blind spot.
            intercept=float(true_model.intercept + rng.normal(0, 1.25 * scale)),
        )
        # Exclude the oracle and grossly unstable candidates that trivialize long rollouts.
        if 0.25 <= m.alpha <= 0.94:
            result.append(m)
    return result


def prediction_error(model, x, a):
    lag = np.concatenate([np.zeros((len(x), 1)), a[:, :-2]], axis=1)
    pred = model.step(x[:, :-1], a[:, :-1], lag)
    return float(np.mean((pred - x[:, 1:]) ** 2))


def dynamics_error(model, truth, seed=991, n=30000):
    rng = np.random.default_rng(seed)
    # Independent reference distribution; treatments and lag are balanced by design.
    x = rng.uniform(-2.5, 2.5, n)
    a = rng.binomial(1, .5, n)
    lag = rng.binomial(1, .5, n)
    return float(np.mean((model.step(x, a, lag) - truth.step(x, a, lag)) ** 2))


def intervention_pair(n, horizon, kind="periodic_3", seed=123):
    rng = np.random.default_rng(seed + sum(map(ord, kind)))
    control = np.zeros((n, horizon))
    intervention = np.zeros((n, horizon))
    if kind.startswith("periodic_"):
        period = int(kind.split("_")[1])
        offsets = rng.integers(0, period, n)
        for i, offset in enumerate(offsets):
            intervention[i, offset::period] = 1
    elif kind == "single_early":
        intervention[:, min(1, horizon - 1)] = 1
    elif kind == "single_middle":
        intervention[:, horizon // 2] = 1
    elif kind == "single_late":
        intervention[:, max(0, horizon - 3)] = 1
    elif kind == "block_early":
        intervention[:, min(1, horizon - 1):min(5, horizon)] = 1
        if horizon == 1:
            intervention[:, 0] = 1
    elif kind == "block_late":
        intervention[:, max(0, horizon - 5):horizon - 1] = 1
        if horizon == 1:
            intervention[:, 0] = 1
    elif kind == "sustained":
        intervention[:] = 1
    else:
        raise ValueError(kind)
    return control, intervention


POLICIES = ("single_early", "single_middle", "single_late", "block_early",
            "block_late", "periodic_2", "periodic_3", "periodic_5", "sustained")


def dte_errors(model, truth, contexts, horizon=32):
    x0, lag0 = contexts
    errors = {}
    for policy in POLICIES:
        p0, p1 = intervention_pair(len(x0), horizon, policy)
        true_tau = rollout(truth, x0, lag0, p1)[:, 1:] - rollout(truth, x0, lag0, p0)[:, 1:]
        pred_tau = rollout(model, x0, lag0, p1)[:, 1:] - rollout(model, x0, lag0, p0)[:, 1:]
        errors[policy] = float(np.mean((pred_tau - true_tau) ** 2))
    values = np.asarray(list(errors.values()))
    return float(values.mean()), float(values.std()), errors


def rank_residual(x, control):
    xr = pd.Series(x).rank().to_numpy()
    cr = pd.Series(control).rank().to_numpy()
    z = np.column_stack([np.ones(len(cr)), cr])
    return xr - z @ np.linalg.lstsq(z, xr, rcond=None)[0]


def partial_spearman(frame):
    return float(spearmanr(rank_residual(frame.dte_error, frame.prediction_error),
                           rank_residual(frame.dynamics_error, frame.prediction_error)).statistic)


def bootstrap_partial_ci(frame, reps=2000, seed=2026):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        sample = frame.iloc[rng.integers(0, len(frame), len(frame))]
        vals.append(partial_spearman(sample))
    return np.quantile(vals, [.025, .975]).tolist()


def cv_incremental_r2(frame, repeats=100, folds=5, seed=11):
    """Repeated-CV delta R2: log ED ~ log EP + log DTE versus log EP alone."""
    eps = 1e-12
    ep = np.log10(frame.prediction_error.to_numpy() + eps)
    dte = np.log10(frame.dte_error.to_numpy() + eps)
    y = np.log10(frame.dynamics_error.to_numpy() + eps)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        ids = rng.permutation(len(y))
        chunks = np.array_split(ids, folds)
        y_all, p0_all, p1_all = [], [], []
        for test in chunks:
            train = np.setdiff1d(ids, test)
            x0_train = np.column_stack([np.ones(len(train)), ep[train]])
            x1_train = np.column_stack([np.ones(len(train)), ep[train], dte[train]])
            b0 = np.linalg.lstsq(x0_train, y[train], rcond=None)[0]
            b1 = np.linalg.lstsq(x1_train, y[train], rcond=None)[0]
            y_all.extend(y[test])
            p0_all.extend(np.column_stack([np.ones(len(test)), ep[test]]) @ b0)
            p1_all.extend(np.column_stack([np.ones(len(test)), ep[test], dte[test]]) @ b1)
        y_all, p0_all, p1_all = map(np.asarray, (y_all, p0_all, p1_all))
        denom = np.sum((y_all - y_all.mean()) ** 2)
        r0 = 1 - np.sum((y_all - p0_all) ** 2) / denom
        r1 = 1 - np.sum((y_all - p1_all) ** 2) / denom
        deltas.append(r1 - r0)
    return {"median": float(np.median(deltas)),
            "ci95": np.quantile(deltas, [.025, .975]).tolist(),
            "positive_fraction": float(np.mean(np.asarray(deltas) > 0))}


def matched_pair(frame, max_relative_gap=.08):
    """Find nearly EP-matched candidates with the largest DTE-error ratio."""
    best = None
    for i in range(len(frame)):
        for j in range(i + 1, len(frame)):
            ei, ej = frame.iloc[i].prediction_error, frame.iloc[j].prediction_error
            gap = abs(ei - ej) / ((ei + ej) / 2)
            if gap <= max_relative_gap:
                di, dj = frame.iloc[i].dte_error, frame.iloc[j].dte_error
                score = abs(np.log10(di + 1e-12) - np.log10(dj + 1e-12))
                if best is None or score > best[0]:
                    best = (score, frame.iloc[i], frame.iloc[j])
    if best is None:
        return matched_pair(frame, max_relative_gap * 2)
    return best[1], best[2]


def run(output_dir="results/core", figure_dir="figures/core"):
    output, figures = Path(output_dir), Path(figure_dir)
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    all_rows, pairs, stats = [], {}, {}
    rng = np.random.default_rng(100)
    contexts = (rng.uniform(-2, 2, 400), rng.binomial(1, .5, 400).astype(float))
    for name, truth in SYSTEMS.items():
        x, a = simulate(truth, seed=42 + (name == "Nonlinear"))
        models = candidates(truth, name == "Nonlinear", seed=7 + (name == "Nonlinear"))
        rows = []
        for model_id, model in enumerate(models):
            dte_mean, dte_std, policy_errors = dte_errors(model, truth, contexts)
            rows.append({"system": name, "model_id": model_id,
                         "prediction_error": prediction_error(model, x, a),
                         "dte_error": dte_mean,
                         "dte_policy_std": dte_std,
                         "dynamics_error": dynamics_error(model, truth),
                         **{f"dte_{key}": value for key, value in policy_errors.items()},
                         **asdict(model)})
        frame = pd.DataFrame(rows)
        all_rows.append(frame)
        pairs[name] = (matched_pair(frame), models, truth)
        stats[name] = {
            "partial_spearman": partial_spearman(frame),
            "partial_spearman_ci95": bootstrap_partial_ci(frame),
            "cv_delta_r2": cv_incremental_r2(frame),
        }
    data = pd.concat(all_rows, ignore_index=True)
    data.to_csv(output / "metrics.csv", index=False)
    (output / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    plot_figure1(pairs, figures / "figure1_matched_models.png")
    plot_figure2(data, stats, figures / "figure2_incremental_value.png")
    return data, stats


def plot_figure1(pairs, path):
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)
    x0, lag0, horizon = np.array([.7]), np.array([1.]), 32
    p0, p1 = intervention_pair(1, horizon, "periodic_3")
    for ax, (name, ((r1, r2), models, truth)) in zip(axes, pairs.items()):
        true_tau = (rollout(truth, x0, lag0, p1) - rollout(truth, x0, lag0, p0))[0, 1:]
        ax.plot(range(1, horizon + 1), true_tau, color="black", lw=3, label="Ground truth")
        for row, color in [(r1, "#2A9D8F"), (r2, "#E07A3F")]:
            model = models[int(row.model_id)]
            tau = (rollout(model, x0, lag0, p1) - rollout(model, x0, lag0, p0))[0, 1:]
            ax.plot(range(1, horizon + 1), tau, color=color, lw=2.5,
                    label=f"M{int(row.model_id)}: EP={row.prediction_error:.4f}, avg EDTE={row.dte_error:.3g}")
        gap = abs(r1.prediction_error-r2.prediction_error)/((r1.prediction_error+r2.prediction_error)/2)
        ax.set(title=f"{name} system (EP gap {100*gap:.1f}%)", xlabel="Horizon", ylabel="Dynamic treatment effect")
        ax.legend(fontsize=9)
    fig.suptitle("Figure 1 — Prediction-matched models can imply different intervention dynamics\n"
                 "Curves show periodic-3; EDTE labels average nine held-out policies", y=1.07)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_figure2(data, stats, path):
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.7))
    for ax, name in zip(axes, SYSTEMS):
        frame = data[data.system == name]
        # Plot rank residuals to make "controlling EP" literal and visible.
        xd = rank_residual(frame.dte_error, frame.prediction_error)
        yd = rank_residual(frame.dynamics_error, frame.prediction_error)
        sns.regplot(x=xd, y=yd, ax=ax, scatter_kws={"s": 55, "alpha": .75},
                    line_kws={"color": "#E07A3F", "lw": 3})
        s = stats[name]
        lo, hi = s["partial_spearman_ci95"]
        delta = s["cv_delta_r2"]
        ax.set(title=(f"{name}: partial ρ={s['partial_spearman']:.2f} [{lo:.2f}, {hi:.2f}]\n"
                      f"CV ΔR² median={delta['median']:.3f}; P(Δ>0)={delta['positive_fraction']:.2f}"),
               xlabel="DTE-error rank residual (after controlling prediction error)",
               ylabel="Dynamics-error rank residual")
    fig.suptitle("Figure 2 — Does DTE error add diagnostic information beyond prediction error?", y=1.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
