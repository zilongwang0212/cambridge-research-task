from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .combine import percentile_rank


THRESHOLDS = (0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50)


def select(frame: pd.DataFrame, q: float) -> pd.Series:
    """Prediction-admissibility screen followed by DTE minimization."""
    ranked = frame.copy()
    ranked["prediction_rank"] = percentile_rank(ranked.prediction_error)
    eligible = ranked[ranked.prediction_rank <= q]
    return eligible.loc[eligible.dte_error.idxmin()]


def evaluate(metrics_path="results/robustness/metrics.csv",
             output_dir="results/two_stage", figure_dir="figures/two_stage"):
    out, figs = Path(output_dir), Path(figure_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(metrics_path)
    selected_rows = []
    for system, frame0 in data.groupby("system", sort=False):
        frame = frame0.reset_index(drop=True).copy()
        frame["dynamics_rank"] = percentile_rank(frame.dynamics_error)
        family = frame.family.iloc[0]
        baselines = {
            "Prediction only": frame.loc[frame.prediction_error.idxmin()],
            "DTE only": frame.loc[frame.dte_error.idxmin()],
        }
        for rule, picked in baselines.items():
            selected_rows.append(_row(system, family, rule, np.nan, picked, len(frame)))
        for q in THRESHOLDS:
            picked = select(frame, q)
            n_eligible = int((percentile_rank(frame.prediction_error) <= q).sum())
            selected_rows.append(_row(system, family, f"Two-stage q={q:.2f}", q,
                                      picked, n_eligible))
    selected = pd.DataFrame(selected_rows)
    sensitivity = selected[selected.threshold.notna()].groupby("threshold").agg(
        median_ED_percentile=("dynamics_percentile", "median"),
        mean_ED_percentile=("dynamics_percentile", "mean"),
        worst_ED_percentile=("dynamics_percentile", "max"),
        top_10_systems=("dynamics_percentile", lambda x: int((x <= .10).sum())),
        top_25_systems=("dynamics_percentile", lambda x: int((x <= .25).sum())),
        median_eligible=("n_eligible", "median"),
    ).reset_index()
    by_family = selected[selected.threshold.notna()].groupby(["family", "threshold"]).agg(
        median_ED_percentile=("dynamics_percentile", "median"),
        mean_ED_percentile=("dynamics_percentile", "mean"),
        worst_ED_percentile=("dynamics_percentile", "max"),
    ).reset_index()
    loo_rows = []
    staged = selected[selected.threshold.notna()]
    for held_out in staged.system.unique():
        train = staged[staged.system != held_out]
        train_score = train.groupby("threshold").dynamics_percentile.mean()
        chosen_q = float(train_score.idxmin())
        test_row = staged[(staged.system == held_out) & staged.threshold.eq(chosen_q)].iloc[0]
        loo_rows.append({
            "held_out_system": held_out, "family": test_row.family,
            "chosen_threshold": chosen_q,
            "test_dynamics_percentile": test_row.dynamics_percentile,
            "test_selected_model": int(test_row.selected_model),
        })
    loo = pd.DataFrame(loo_rows)
    selected.to_csv(out / "selected_by_system.csv", index=False)
    sensitivity.to_csv(out / "threshold_sensitivity.csv", index=False)
    by_family.to_csv(out / "threshold_by_family.csv", index=False)
    loo.to_csv(out / "leave_one_system_out.csv", index=False)
    definition = {
        "score": "S_q(m)=rank_DTE(m) if rank_prediction(m)<=q, else +infinity",
        "initial_untuned_reference_q": 0.25,
        "recommended_q_from_leave_one_system_out": 0.05,
        "selection": "argmin S_q",
        "interpretation": "DTE is used only among prediction-admissible candidates",
        "ground_truth_use": "ED is evaluation-only; it does not set q or select a model",
        "tested_q": list(THRESHOLDS),
    }
    (out / "definition.json").write_text(json.dumps(definition, indent=2), encoding="utf-8")
    plot(sensitivity, selected, figs / "threshold_validation.png")
    return selected, sensitivity, by_family, loo


def _row(system, family, rule, threshold, picked, n_eligible):
    return {
        "system": system, "family": family, "rule": rule, "threshold": threshold,
        "selected_model": int(picked.model_id), "prediction_error": picked.prediction_error,
        "dte_error": picked.dte_error, "dynamics_error": picked.dynamics_error,
        "dynamics_percentile": picked.dynamics_rank, "n_eligible": n_eligible,
    }


def plot(sensitivity, selected, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(100*sensitivity.threshold, 100*sensitivity.median_ED_percentile,
                 marker="o", lw=2.5, label="Median")
    axes[0].plot(100*sensitivity.threshold, 100*sensitivity.mean_ED_percentile,
                 marker="s", lw=2.5, label="Mean")
    axes[0].plot(100*sensitivity.threshold, 100*sensitivity.worst_ED_percentile,
                 marker="^", lw=2.5, label="Worst system")
    axes[0].axvline(5, color="gray", ls="--", label="LOSO-selected q=5%")
    axes[0].set(xlabel="Prediction-admissibility threshold q (%)",
                ylabel="Selected model's true ED percentile (%)",
                title="Threshold sensitivity (lower is better)")
    axes[0].legend()
    primary = selected[selected.threshold.eq(.05)].copy()
    colors = primary.family.map({"Linear": "#4C78A8", "Nonlinear": "#E07A3F"})
    axes[1].barh(primary.system, 100*primary.dynamics_percentile, color=colors)
    axes[1].axvline(10, color="gray", ls="--", label="Top 10% ED")
    axes[1].set(xlabel="Selected model's true ED percentile (%)",
                title="LOSO-selected q=5% rule across 10 systems")
    axes[1].invert_yaxis()
    axes[1].legend()
    fig.suptitle("Validation of prediction-screened DTE selection")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
