from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from .candidates import candidate_table, make_candidates
from .evaluate import (dynamics_error, interventional_metrics, prediction_error,
                       residualized_spearman)
from .system import simulate_panel


def bootstrap_selection(df: pd.DataFrame, reps: int = 1000, seed: int = 99):
    """Compare EP-only versus EP-screen + DTE selection under candidate resampling."""
    rng = np.random.default_rng(seed)
    rows = []
    eligible = df[df.prediction_error <= df.prediction_error.quantile(0.25)].copy()
    for r in range(reps):
        sample = eligible.iloc[rng.integers(0, len(eligible), size=len(eligible))].reset_index(drop=True)
        ep_pick = sample.iloc[int(np.argmin(sample.prediction_error.to_numpy()))]
        dte_pick = sample.iloc[int(np.argmin(sample.dte_error.to_numpy()))]
        rows.extend([
            {"rep": r, "selector": "Prediction only", "dynamics_error": ep_pick.dynamics_error},
            {"rep": r, "selector": "Prediction screen + DTE", "dynamics_error": dte_pick.dynamics_error},
        ])
    return pd.DataFrame(rows)


def run(output_dir: str | Path = "results", n_series: int = 1000, length: int = 128,
        n_candidates: int = 100, n_contexts: int = 300, seed: int = 42):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    models = make_candidates(n_candidates, seed=7)
    panel = simulate_panel(n_series, length, seed=seed)
    rng = np.random.default_rng(seed + 1)
    contexts = {
        "x0": rng.uniform(-2.0, 2.0, size=n_contexts),
        "a_prev": rng.binomial(1, 0.5, size=n_contexts).astype(float),
    }
    base = candidate_table(models)
    base["prediction_error"] = [prediction_error(m, panel) for m in models]
    base["dynamics_error"] = [dynamics_error(m) for m in models]
    details = []
    for model_id, model in enumerate(models):
        for horizon in [1, 4, 8, 16, 32]:
            for policy in ["in_policy", "shift", "sustained"]:
                metrics = interventional_metrics(model, contexts, horizon, policy)
                details.append({"model_id": model_id, "horizon": horizon,
                                "policy": policy, **metrics})
    detail = pd.DataFrame(details).merge(
        base[["model_id", "prediction_error", "dynamics_error", "compressed"]],
        on="model_id", how="left")
    primary = detail[(detail.horizon == 32) & (detail.policy == "shift")]
    base = base.merge(primary[["model_id", "dte_error", "interventional_error"]],
                      on="model_id", how="left")
    selection = bootstrap_selection(base)
    summary = {
        "n_series": n_series,
        "length": length,
        "n_candidates": n_candidates,
        "primary_partial_spearman_dte_dynamics_given_prediction": residualized_spearman(
            base, "dte_error", "dynamics_error", "prediction_error"),
        "primary_partial_spearman_interventional_dynamics_given_prediction": residualized_spearman(
            base, "interventional_error", "dynamics_error", "prediction_error"),
        "median_selected_dynamics_error": selection.groupby("selector").dynamics_error.median().to_dict(),
    }
    base.to_csv(output / "candidate_metrics.csv", index=False)
    detail.to_csv(output / "condition_metrics.csv", index=False)
    selection.to_csv(output / "selection_bootstrap.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return base, detail, selection, summary
