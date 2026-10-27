from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .combine import percentile_rank
from .dual_experiment import Model, dte_errors, partial_spearman
from .robustness import ROBUST_SYSTEMS


HORIZONS = (1, 4, 8, 16, 32)
MODEL_FIELDS = ("alpha", "sin_coef", "treatment_base", "treatment_state",
                "lag_coef", "state_scale", "intercept")


def evaluate(metrics_path="results/robustness/metrics.csv",
             output_dir="results/multistep", figure_dir="figures/multistep",
             prediction_quantile=.05):
    out, figs = Path(output_dir), Path(figure_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(metrics_path)
    rng = np.random.default_rng(100)
    contexts = (rng.uniform(-2, 2, 400), rng.binomial(1, .5, 400).astype(float))
    rows = []
    for system, frame0 in base.groupby("system", sort=False):
        truth = ROBUST_SYSTEMS[system][1]
        for _, row in frame0.iterrows():
            model = Model(**{field: float(row[field]) for field in MODEL_FIELDS})
            for horizon in HORIZONS:
                mean, std, policy = dte_errors(model, truth, contexts, horizon=horizon)
                rows.append({
                    "system": system, "family": row.family, "model_id": int(row.model_id),
                    "horizon": horizon, "prediction_error": row.prediction_error,
                    "dynamics_error": row.dynamics_error, "dte_error": mean,
                    "dte_policy_std": std,
                    **{f"dte_{key}": value for key, value in policy.items()},
                })
    detailed = pd.DataFrame(rows)
    selected_rows, diagnostic_rows = [], []
    for (system, horizon), frame0 in detailed.groupby(["system", "horizon"], sort=False):
        frame = frame0.reset_index(drop=True).copy()
        frame["prediction_rank"] = percentile_rank(frame.prediction_error)
        frame["dynamics_rank"] = percentile_rank(frame.dynamics_error)
        eligible = frame[frame.prediction_rank <= prediction_quantile]
        picked = eligible.loc[eligible.dte_error.idxmin()]
        selected_rows.append({
            "system": system, "family": picked.family, "horizon": horizon,
            "selected_model": int(picked.model_id),
            "selected_dynamics_percentile": picked.dynamics_rank,
            "selected_dynamics_error": picked.dynamics_error,
            "selected_dte_error": picked.dte_error,
        })
        diagnostic_rows.append({
            "system": system, "family": picked.family, "horizon": horizon,
            "partial_spearman": partial_spearman(frame),
        })
    selected, diagnostics = pd.DataFrame(selected_rows), pd.DataFrame(diagnostic_rows)
    summary = selected.groupby("horizon").agg(
        median_ED_percentile=("selected_dynamics_percentile", "median"),
        mean_ED_percentile=("selected_dynamics_percentile", "mean"),
        worst_ED_percentile=("selected_dynamics_percentile", "max"),
        top_5_systems=("selected_dynamics_percentile", lambda x: int((x <= .05).sum())),
        top_10_systems=("selected_dynamics_percentile", lambda x: int((x <= .10).sum())),
    ).reset_index()
    family_summary = selected.groupby(["family", "horizon"]).agg(
        median_ED_percentile=("selected_dynamics_percentile", "median"),
        mean_ED_percentile=("selected_dynamics_percentile", "mean"),
        worst_ED_percentile=("selected_dynamics_percentile", "max"),
    ).reset_index()
    detailed.to_csv(out / "candidate_horizon_metrics.csv", index=False)
    selected.to_csv(out / "selected_by_horizon.csv", index=False)
    diagnostics.to_csv(out / "diagnostic_by_horizon.csv", index=False)
    summary.to_csv(out / "horizon_summary.csv", index=False)
    family_summary.to_csv(out / "horizon_by_family.csv", index=False)
    plot(summary, family_summary, diagnostics, figs / "multistep_validation.png")
    return selected, diagnostics, summary, family_summary


def plot(summary, family, diagnostics, path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    axes[0].plot(summary.horizon, 100*summary.median_ED_percentile, marker="o", label="Median")
    axes[0].plot(summary.horizon, 100*summary.mean_ED_percentile, marker="s", label="Mean")
    axes[0].plot(summary.horizon, 100*summary.worst_ED_percentile, marker="^", label="Worst")
    axes[0].set(xlabel="Rollout horizon H", ylabel="Selected true ED percentile (%)",
                title="Selection quality")
    axes[0].legend()
    for name, color in [("Linear", "#4C78A8"), ("Nonlinear", "#E07A3F")]:
        part = family[family.family == name]
        axes[1].plot(part.horizon, 100*part.mean_ED_percentile, marker="o", color=color, label=name)
    axes[1].set(xlabel="Rollout horizon H", ylabel="Mean selected ED percentile (%)",
                title="Linear versus nonlinear")
    axes[1].legend()
    diag_summary = diagnostics.groupby(["family", "horizon"]).partial_spearman.median().reset_index()
    for name, color in [("Linear", "#4C78A8"), ("Nonlinear", "#E07A3F")]:
        part = diag_summary[diag_summary.family == name]
        axes[2].plot(part.horizon, part.partial_spearman, marker="o", color=color, label=name)
    axes[2].axhline(0, color="black", lw=1)
    axes[2].set(xlabel="Rollout horizon H", ylabel="Median partial Spearman",
                title="Incremental DTE diagnostic association")
    axes[2].legend()
    fig.suptitle("Multi-step rollout validation (prediction top 5%, then DTE)")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
