from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .dual_experiment import (Model, bootstrap_partial_ci, candidates,
                              cv_incremental_r2, dte_errors, dynamics_error,
                              partial_spearman, prediction_error, simulate)


ROBUST_SYSTEMS = {
    # (family, truth). Linear systems vary persistence, immediate effect, and lag.
    "L1_balanced": ("Linear", Model(.72, 0, .30, 0, .12)),
    "L2_persistent": ("Linear", Model(.88, 0, .18, 0, .05)),
    "L3_strong_action": ("Linear", Model(.45, 0, .45, 0, .20)),
    "L4_negative_lag": ("Linear", Model(.65, 0, .25, 0, -.12)),
    "L5_strong_lag": ("Linear", Model(.55, 0, .15, 0, .35)),
    # Nonlinear systems vary persistence, curvature, interaction sign/strength, and lag.
    "N1_balanced": ("Nonlinear", Model(.70, .20, .20, .20, .10, 1.0)),
    "N2_strong_interaction": ("Nonlinear", Model(.82, .10, .15, .30, .05, 1.5)),
    "N3_negative_interaction": ("Nonlinear", Model(.50, .30, .40, -.15, .20, .7)),
    "N4_opposing_curvature": ("Nonlinear", Model(.75, -.15, .25, .25, -.10, 1.2)),
    "N5_strong_lag": ("Nonlinear", Model(.60, .25, .10, .35, .30, .8)),
}


def run(output_dir="results/robustness", figure_dir="figures/robustness"):
    output, figures = Path(output_dir), Path(figure_dir)
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(100)
    contexts = (rng.uniform(-2, 2, 400), rng.binomial(1, .5, 400).astype(float))
    metric_rows, summaries = [], []
    for system_idx, (name, (family, truth)) in enumerate(ROBUST_SYSTEMS.items()):
        x, a = simulate(truth, seed=1000 + system_idx)
        models = candidates(truth, family == "Nonlinear", seed=2000 + system_idx)
        rows = []
        for model_id, model in enumerate(models):
            dte_mean, dte_std, policy_errors = dte_errors(model, truth, contexts)
            rows.append({
                "system": name, "family": family, "model_id": model_id,
                "prediction_error": prediction_error(model, x, a),
                "dte_error": dte_mean, "dte_policy_std": dte_std,
                "dynamics_error": dynamics_error(model, truth, seed=3000 + system_idx),
                **{f"dte_{key}": value for key, value in policy_errors.items()},
                **asdict(model),
            })
        frame = pd.DataFrame(rows)
        metric_rows.append(frame)
        rho = partial_spearman(frame)
        ci = bootstrap_partial_ci(frame, reps=1000, seed=4000 + system_idx)
        delta = cv_incremental_r2(frame, repeats=100, seed=5000 + system_idx)
        summaries.append({
            "system": name, "family": family, "partial_spearman": rho,
            "rho_ci_low": ci[0], "rho_ci_high": ci[1],
            "cv_delta_r2_median": delta["median"],
            "cv_delta_r2_ci_low": delta["ci95"][0],
            "cv_delta_r2_ci_high": delta["ci95"][1],
            "cv_positive_fraction": delta["positive_fraction"],
        })
    metrics, summary = pd.concat(metric_rows, ignore_index=True), pd.DataFrame(summaries)
    metrics.to_csv(output / "metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    family_summary = summary.groupby("family").agg(
        systems=("system", "count"),
        median_partial_spearman=("partial_spearman", "median"),
        positive_rho_systems=("partial_spearman", lambda x: int((x > 0).sum())),
        median_cv_delta_r2=("cv_delta_r2_median", "median"),
        positive_delta_systems=("cv_delta_r2_median", lambda x: int((x > 0).sum())),
    ).reset_index()
    family_summary.to_csv(output / "family_summary.csv", index=False)
    (output / "systems.json").write_text(json.dumps(
        {k: {"family": v[0], **asdict(v[1])} for k, v in ROBUST_SYSTEMS.items()}, indent=2),
        encoding="utf-8")
    plot(summary, figures / "robustness_forest.png")
    return summary, family_summary


def plot(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = summary.family.map({"Linear": "#4C78A8", "Nonlinear": "#E07A3F"})
    y = np.arange(len(summary))
    for i, row in summary.iterrows():
        axes[0].errorbar(row.partial_spearman, i,
                         xerr=[[row.partial_spearman-row.rho_ci_low],
                               [row.rho_ci_high-row.partial_spearman]],
                         fmt="none", ecolor=colors.iloc[i], elinewidth=2, capsize=4)
    axes[0].scatter(summary.partial_spearman, y, c=colors, s=65)
    axes[0].axvline(0, color="black", lw=1)
    axes[0].set(xlabel="Partial Spearman: EDTE vs ED | EP", yticks=y,
                yticklabels=summary.system, title="Rank-based incremental association")
    for i, row in summary.iterrows():
        axes[1].errorbar(row.cv_delta_r2_median, i,
                         xerr=[[row.cv_delta_r2_median-row.cv_delta_r2_ci_low],
                               [row.cv_delta_r2_ci_high-row.cv_delta_r2_median]],
                         fmt="none", ecolor=colors.iloc[i], elinewidth=2, capsize=4)
    axes[1].scatter(summary.cv_delta_r2_median, y, c=colors, s=65)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set(xlabel="Repeated-CV ΔR² from adding EDTE", yticks=y,
                yticklabels=[], title="Out-of-sample incremental prediction")
    fig.suptitle("DTE diagnostic value across ten controlled dynamic systems")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
