from __future__ import annotations

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .candidates import make_candidates
from .evaluate import make_policy_pair, residualized_spearman
from .system import TRUE_DYNAMICS, rollout


COLORS = {"navy": "#183153", "orange": "#E07A3F", "teal": "#2A9D8F", "gray": "#667085"}


def _save(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_figures(results_dir="results", figures_dir="figures"):
    sns.set_theme(style="whitegrid", context="talk")
    results, figures = Path(results_dir), Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(results / "candidate_metrics.csv")
    detail = pd.read_csv(results / "condition_metrics.csv")
    selection = pd.read_csv(results / "selection_bootstrap.csv")
    models = make_candidates(len(base), seed=7)

    # Figure 1: choose two models in the closest EP decile but separated in DTE.
    cutoff = base.prediction_error.quantile(0.25)
    pool = base[base.prediction_error <= cutoff]
    lo, hi = pool.loc[pool.dte_error.idxmin()], pool.loc[pool.dte_error.idxmax()]
    rng = np.random.default_rng(2026)
    x0, prev, horizon = np.array([0.7]), np.array([1.0]), 32
    p0, p1 = make_policy_pair(1, horizon, "shift", rng)
    true_tau = (rollout(TRUE_DYNAMICS, x0, prev, p1) - rollout(TRUE_DYNAMICS, x0, prev, p0))[0, 1:]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, horizon + 1), true_tau, color="black", lw=3, label="Ground-truth DTE")
    for row, color in [(lo, COLORS["teal"]), (hi, COLORS["orange"])]:
        m = models[int(row.model_id)]
        tau = (rollout(m, x0, prev, p1) - rollout(m, x0, prev, p0))[0, 1:]
        ax.plot(range(1, horizon + 1), tau, lw=2.5, color=color,
                label=f"Model {int(row.model_id)}: EP={row.prediction_error:.3g}, DTE={row.dte_error:.3g}")
    ax.set(xlabel="Horizon", ylabel="Treatment effect", title="Similar factual prediction, different intervention response")
    ax.legend(fontsize=10)
    _save(fig, figures / "figure1_prediction_blind_spot.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.scatterplot(base, x="dte_error", y="dynamics_error", hue="prediction_error",
                    palette="viridis_r", size="prediction_error", ax=axes[0], legend=False)
    rho = residualized_spearman(base, "dte_error", "dynamics_error", "prediction_error")
    axes[0].set(title=f"DTE diagnostic (partial Spearman={rho:.2f})", xscale="log", yscale="log")
    sns.scatterplot(base, x="interventional_error", y="dynamics_error", hue="prediction_error",
                    palette="viridis_r", size="prediction_error", ax=axes[1], legend=False)
    rho2 = residualized_spearman(base, "interventional_error", "dynamics_error", "prediction_error")
    axes[1].set(title=f"Strong baseline: full interventional error ({rho2:.2f})", xscale="log", yscale="log")
    fig.suptitle("Incremental diagnostic value after controlling factual prediction error")
    _save(fig, figures / "figure2_metric_validity.png")

    rows = []
    for (h, p, c), group in detail.groupby(["horizon", "policy", "compressed"]):
        if len(group) >= 8:
            rows.append({"horizon": h, "policy": p, "history": "Lag omitted" if c else "Full state",
                         "partial_spearman": residualized_spearman(group, "dte_error", "dynamics_error", "prediction_error")})
    corr = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, history in zip(axes, ["Full state", "Lag omitted"]):
        sns.lineplot(corr[corr.history == history], x="horizon", y="partial_spearman", hue="policy",
                     marker="o", linewidth=2.5, ax=ax)
        ax.axhline(0, color="black", lw=1)
        ax.set(title=history, ylabel="Partial Spearman: DTE vs dynamics error", ylim=(-0.15, 1.05))
    fig.suptitle("Diagnostic validity across horizon, policy shift, and information loss")
    _save(fig, figures / "figure3_conditions.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(selection, x="selector", y="dynamics_error", hue="selector", palette=[COLORS["gray"], COLORS["teal"]], ax=ax)
    ax.set(yscale="symlog", xlabel="", title="Which selection rule recovers the mechanism?", ylabel="Selected model dynamics error")
    _save(fig, figures / "figure4_selection.png")

    manifest = {
        "figure1": [int(lo.model_id), int(hi.model_id)],
        "note": "Figure 4 is the synthetic model-selection result. The optional foundation-model runner creates a separate external-validity figure without fabricating unavailable model outputs."
    }
    (figures / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
