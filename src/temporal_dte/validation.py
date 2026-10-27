from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from .combine import percentile_rank
from .dual_experiment import (POLICIES, candidates, dynamics_error, intervention_pair,
                              prediction_error, rollout, simulate)
from .robustness import ROBUST_SYSTEMS


GATES = (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def intervention_metrics(model, truth, contexts, horizon=16):
    """DTE and absolute interventional-forecast errors on exactly the same tasks."""
    x0, lag0 = contexts
    dte, forecast = [], []
    for policy in POLICIES:
        p0, p1 = intervention_pair(len(x0), horizon, policy)
        true0, true1 = rollout(truth, x0, lag0, p0)[:, 1:], rollout(truth, x0, lag0, p1)[:, 1:]
        pred0, pred1 = rollout(model, x0, lag0, p0)[:, 1:], rollout(model, x0, lag0, p1)[:, 1:]
        dte.append(np.mean(((pred1-pred0)-(true1-true0))**2))
        forecast.append(.5*(np.mean((pred0-true0)**2)+np.mean((pred1-true1)**2)))
    return float(np.mean(dte)), float(np.mean(forecast))


def run(repeats=20, n_candidates=100, n_contexts=250, horizon=16,
        output_dir="results/validation", figure_dir="figures/validation"):
    out, figs = Path(output_dir), Path(figure_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(600)
    contexts = (rng.uniform(-2, 2, n_contexts), rng.binomial(1, .5, n_contexts).astype(float))
    panels = {name: simulate(truth, seed=700+idx) for idx, (name, (_, truth)) in enumerate(ROBUST_SYSTEMS.items())}
    candidate_rows, selection_rows = [], []
    for repeat in range(repeats):
        for idx, (system, (family, truth)) in enumerate(ROBUST_SYSTEMS.items()):
            x, a = panels[system]
            models = candidates(truth, family == "Nonlinear", n_candidates,
                                seed=10000 + repeat*100 + idx)
            rows = []
            for model_id, model in enumerate(models):
                dte, ife = intervention_metrics(model, truth, contexts, horizon)
                rows.append({
                    "repeat": repeat, "system": system, "family": family,
                    "model_id": model_id, "prediction_error": prediction_error(model, x, a),
                    "dte_error": dte, "interventional_forecast_error": ife,
                    "dynamics_error": dynamics_error(model, truth, seed=800+idx),
                })
            frame = pd.DataFrame(rows)
            frame["prediction_rank"] = percentile_rank(frame.prediction_error)
            frame["dynamics_rank"] = percentile_rank(frame.dynamics_error)
            candidate_rows.append(frame)
            for label, column in [("Prediction only", "prediction_error"),
                                  ("DTE only", "dte_error"),
                                  ("IFE only", "interventional_forecast_error")]:
                picked = frame.loc[frame[column].idxmin()]
                selection_rows.append(_selection(repeat, system, family, label, np.nan, picked))
            for gate in GATES:
                eligible = frame[frame.prediction_rank <= gate]
                for label, column in [("Gate + DTE", "dte_error"),
                                      ("Gate + IFE", "interventional_forecast_error")]:
                    picked = eligible.loc[eligible[column].idxmin()]
                    selection_rows.append(_selection(repeat, system, family, label, gate, picked))
    candidates_df = pd.concat(candidate_rows, ignore_index=True)
    selections = pd.DataFrame(selection_rows)
    gate_summary = selections[selections.rule.eq("Gate + DTE")].groupby("gate").agg(
        median_ED_percentile=("dynamics_rank", "median"),
        mean_ED_percentile=("dynamics_rank", "mean"),
        q90_ED_percentile=("dynamics_rank", lambda x: float(np.quantile(x, .90))),
        worst_ED_percentile=("dynamics_rank", "max"),
        top_10_rate=("dynamics_rank", lambda x: float(np.mean(x <= .10))),
    ).reset_index()
    repeat_summary = selections[selections.gate.eq(.05)].groupby(["repeat", "rule"]).agg(
        mean_ED_percentile=("dynamics_rank", "mean"),
        median_ED_percentile=("dynamics_rank", "median"),
        worst_ED_percentile=("dynamics_rank", "max"),
    ).reset_index()
    comparison = selections[(selections.gate.eq(.05)) | selections.rule.isin(
        ["Prediction only", "DTE only", "IFE only"])].groupby("rule").agg(
        median_ED_percentile=("dynamics_rank", "median"),
        mean_ED_percentile=("dynamics_rank", "mean"),
        q90_ED_percentile=("dynamics_rank", lambda x: float(np.quantile(x, .90))),
        worst_ED_percentile=("dynamics_rank", "max"),
        top_10_rate=("dynamics_rank", lambda x: float(np.mean(x <= .10))),
    ).sort_values("mean_ED_percentile").reset_index()
    paired = selections[selections.gate.eq(.05)].pivot_table(
        index=["repeat", "system"], columns="rule", values="dynamics_rank")
    stat = wilcoxon(paired["Gate + DTE"], paired["Gate + IFE"], zero_method="zsplit")
    strong_baseline = {
        "paired_tasks": int(len(paired)),
        "gate_dte_wins": int((paired["Gate + DTE"] < paired["Gate + IFE"]).sum()),
        "ties": int((paired["Gate + DTE"] == paired["Gate + IFE"]).sum()),
        "gate_ife_wins": int((paired["Gate + IFE"] < paired["Gate + DTE"]).sum()),
        "wilcoxon_statistic": float(stat.statistic), "wilcoxon_pvalue": float(stat.pvalue),
    }
    # Full candidate table is useful but larger; keep it reproducible and explicit.
    candidates_df.to_csv(out / "candidate_metrics.csv", index=False)
    selections.to_csv(out / "selections.csv", index=False)
    gate_summary.to_csv(out / "gate_sensitivity.csv", index=False)
    repeat_summary.to_csv(out / "repeat_stability.csv", index=False)
    comparison.to_csv(out / "strong_baseline_comparison.csv", index=False)
    (out / "strong_baseline_test.json").write_text(json.dumps(strong_baseline, indent=2), encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps({
        "repeats": repeats, "systems": len(ROBUST_SYSTEMS), "candidates_per_pool": n_candidates,
        "contexts": n_contexts, "horizon": horizon, "policies": list(POLICIES), "gates": list(GATES)
    }, indent=2), encoding="utf-8")
    plot(gate_summary, repeat_summary, comparison, figs / "validation_suite.png")
    return gate_summary, repeat_summary, comparison, strong_baseline


def _selection(repeat, system, family, rule, gate, picked):
    return {"repeat": repeat, "system": system, "family": family, "rule": rule,
            "gate": gate, "model_id": int(picked.model_id),
            "dynamics_rank": float(picked.dynamics_rank),
            "dynamics_error": float(picked.dynamics_error),
            "prediction_error": float(picked.prediction_error),
            "dte_error": float(picked.dte_error),
            "interventional_forecast_error": float(picked.interventional_forecast_error)}


def plot(gate, repeat, comparison, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    axes[0].plot(100*gate.gate, 100*gate.median_ED_percentile, marker="o", label="Median")
    axes[0].plot(100*gate.gate, 100*gate.mean_ED_percentile, marker="s", label="Mean")
    axes[0].plot(100*gate.gate, 100*gate.q90_ED_percentile, marker="^", label="90th percentile")
    axes[0].set(xlabel="Prediction gate (%)", ylabel="Selected true ED percentile (%)",
                title="Gate sensitivity")
    axes[0].legend()
    for rule, color in [("Gate + DTE", "#E07A3F"), ("Gate + IFE", "#2A9D8F")]:
        part = repeat[repeat.rule == rule]
        axes[1].scatter(part.repeat, 100*part.mean_ED_percentile, color=color, label=rule)
    axes[1].set(xlabel="Candidate-pool resample", ylabel="Mean selected ED percentile (%)",
                title="Candidate resampling stability")
    axes[1].legend()
    comp = comparison.sort_values("mean_ED_percentile", ascending=True)
    axes[2].barh(comp.rule, 100*comp.mean_ED_percentile, color="#4C78A8")
    axes[2].set(xlabel="Mean selected ED percentile (%)", title="Strong-baseline comparison")
    fig.suptitle("Validation of prediction-gated intervention metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
