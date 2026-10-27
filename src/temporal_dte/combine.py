from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def percentile_rank(series: pd.Series) -> pd.Series:
    # Low error is good: percentile 0 is the best candidate.
    return (series.rank(method="average") - 1) / max(1, len(series) - 1)


def add_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["prediction_rank"] = percentile_rank(out.prediction_error)
    out["dte_rank"] = percentile_rank(out.dte_error)
    out["mean_rank_score"] = (out.prediction_rank + out.dte_rank) / 2
    out["worst_rank_score"] = np.maximum(out.prediction_rank, out.dte_rank)
    return out


def select_two_stage(frame: pd.DataFrame, fraction=.25):
    cutoff = frame.prediction_error.quantile(fraction)
    eligible = frame[frame.prediction_error <= cutoff]
    return eligible.loc[eligible.dte_error.idxmin()]


def evaluate(metrics_path="results/robustness/metrics.csv",
             output_dir="results/combined", figure_dir="figures/combined"):
    output, figures = Path(output_dir), Path(figure_dir)
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(metrics_path)
    scored_parts, system_rows, selection_rows = [], [], []
    for system, frame0 in data.groupby("system", sort=False):
        frame = add_scores(frame0.reset_index(drop=True))
        frame["dynamics_rank"] = percentile_rank(frame.dynamics_error)
        scored_parts.append(frame)
        family = frame.family.iloc[0]
        scores = {
            "Prediction only": "prediction_rank",
            "DTE only": "dte_rank",
            "Mean rank": "mean_rank_score",
            "Worst rank": "worst_rank_score",
        }
        row = {"system": system, "family": family}
        for label, col in scores.items():
            row[label] = float(spearmanr(frame[col], frame.dynamics_error).statistic)
            picked = frame.loc[frame[col].idxmin()]
            selection_rows.append({
                "system": system, "family": family, "selector": label,
                "selected_model": int(picked.model_id),
                "selected_dynamics_error": float(picked.dynamics_error),
                "selected_dynamics_percentile": float(picked.dynamics_rank),
            })
        picked = select_two_stage(frame)
        selection_rows.append({
            "system": system, "family": family, "selector": "Prediction top 25% + DTE",
            "selected_model": int(picked.model_id),
            "selected_dynamics_error": float(picked.dynamics_error),
            "selected_dynamics_percentile": float(picked.dynamics_rank),
        })
        system_rows.append(row)
    scored = pd.concat(scored_parts, ignore_index=True)
    correlations = pd.DataFrame(system_rows)
    selections = pd.DataFrame(selection_rows)
    summary = selections.groupby(["family", "selector"]).agg(
        median_selected_ED_percentile=("selected_dynamics_percentile", "median"),
        mean_selected_ED_percentile=("selected_dynamics_percentile", "mean"),
        top_quartile_systems=("selected_dynamics_percentile", lambda x: int((x <= .25).sum())),
    ).reset_index()
    overall = selections.groupby("selector").agg(
        median_selected_ED_percentile=("selected_dynamics_percentile", "median"),
        mean_selected_ED_percentile=("selected_dynamics_percentile", "mean"),
        top_quartile_systems=("selected_dynamics_percentile", lambda x: int((x <= .25).sum())),
    ).sort_values("median_selected_ED_percentile").reset_index()
    scored.to_csv(output / "candidate_scores.csv", index=False)
    correlations.to_csv(output / "score_correlations.csv", index=False)
    selections.to_csv(output / "selected_models.csv", index=False)
    summary.to_csv(output / "family_summary.csv", index=False)
    overall.to_csv(output / "overall_summary.csv", index=False)
    (output / "definition.json").write_text(json.dumps({
        "mean_rank": "0.5 * percentile_rank(EP) + 0.5 * percentile_rank(EDTE)",
        "worst_rank": "max(percentile_rank(EP), percentile_rank(EDTE))",
        "two_stage": "minimize EDTE among candidates in the best EP quartile",
        "ground_truth_use": "ED is evaluation-only and is not used to define or tune a score",
    }, indent=2), encoding="utf-8")
    plot(selections, figures / "combined_selection.png")
    return correlations, selections, summary, overall


def plot(selections, path):
    order = ["Prediction only", "DTE only", "Mean rank", "Worst rank",
             "Prediction top 25% + DTE"]
    fig, ax = plt.subplots(figsize=(11, 6))
    positions = {name: i for i, name in enumerate(order)}
    rng = np.random.default_rng(9)
    for _, row in selections.iterrows():
        x = positions[row.selector] + rng.uniform(-.10, .10)
        color = "#4C78A8" if row.family == "Linear" else "#E07A3F"
        ax.scatter(x, row.selected_dynamics_percentile, color=color, alpha=.8, s=55)
    medians = selections.groupby("selector").selected_dynamics_percentile.median()
    for label in order:
        ax.plot([positions[label]-.22, positions[label]+.22], [medians[label]]*2,
                color="black", lw=4)
    ax.axhline(.25, color="gray", ls="--", label="Top-quartile ED threshold")
    ax.set(xticks=range(len(order)), xticklabels=order,
           ylabel="Selected model's true ED percentile (lower is better)",
           title="Combining prediction and DTE for model selection")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
