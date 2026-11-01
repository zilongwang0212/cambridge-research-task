from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .compositional_latent import evaluate
from .medical_latent import Config
from .temporal_jepa import observed_diagnostics, train_model


TRAINING_VARIANTS = {
    "h1_m099": ((1,), .99),
    "h8_m099": ((8,), .99),
    "multi_m000": ((1, 2, 4, 8), 0.),
    "multi_m090": ((1, 2, 4, 8), .9),
    "multi_m099": ((1, 2, 4, 8), .99),
    "multi_m0999": ((1, 2, 4, 8), .999),
}


def _training_ablations(cfg: Config):
    effects, diagnostics = [], []
    for label, (horizons, momentum) in TRAINING_VARIANTS.items():
        for seed in cfg.seeds:
            net = train_model(
                "jepa", seed, cfg, jepa_weight=1., latent_dim=8,
                jepa_horizons=horizons, ema_momentum=momentum,
            )
            effect = evaluate(net, cfg, seed)
            effect["variant"] = label
            effects.append(effect)
            diagnostic = observed_diagnostics(net, cfg, seed)
            diagnostic["variant"] = label
            diagnostics.append(diagnostic)
    return pd.concat(effects, ignore_index=True), pd.concat(diagnostics, ignore_index=True)


def _selection_sensitivity(main_diagnostics: pd.DataFrame):
    means = (
        main_diagnostics.groupby(["model_kind", "horizon"], as_index=False)
        [["observed_rollout_mse", "latent_closure_mse"]].mean()
    )
    rows = []
    for horizon in (1, 2, 4, 8, 16):
        at_h = means[means.horizon == horizon].set_index("model_kind")
        baseline = at_h.loc["multi_step", "observed_rollout_mse"]
        for gate in (1., 1.05, 1.10, 1.20):
            candidates = at_h.loc[[x for x in at_h.index if x.startswith("jepa_")]].copy()
            candidates["eligible"] = candidates.observed_rollout_mse <= gate * baseline
            eligible = candidates[candidates.eligible]
            selected = eligible.latent_closure_mse.idxmin() if len(eligible) else None
            rows.append({
                "closure_horizon": horizon,
                "prediction_gate": gate,
                "baseline_rollout_mse": baseline,
                "prediction_ceiling": gate * baseline,
                "eligible_models": ",".join(candidates.index[candidates.eligible]),
                "selected_model": selected,
            })
    return pd.DataFrame(rows)


def _lambda_summary(main_diagnostics: pd.DataFrame):
    selected = main_diagnostics[
        (main_diagnostics.horizon == 8)
        & main_diagnostics.model_kind.str.startswith("jepa_")
    ]
    summary = (
        selected.groupby("model_kind")
        [["observed_rollout_mse", "latent_closure_mse", "action_sensitivity"]]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def _plot(effect: pd.DataFrame, diagnostics: pd.DataFrame, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    h8 = diagnostics[diagnostics.horizon == 8].groupby("variant").mean(numeric_only=True)
    h8.observed_rollout_mse.plot.bar(ax=axes[0], color="#4C78A8")
    h8.latent_closure_mse.plot.bar(ax=axes[1], color="#F58518", logy=True)
    h16 = effect[effect.horizon == 16].groupby("variant").effect_mse.mean()
    h16.plot.bar(ax=axes[2], color="#54A24B", logy=True)
    axes[0].set(ylabel="Observed rollout MSE", title="H=8 factual fidelity")
    axes[1].set(ylabel="Normalized latent MSE", title="H=8 latent closure")
    axes[2].set(ylabel="Effect trajectory MSE", title="H=16 mean across schedules")
    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output / "temporal_jepa_ablation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(output_dir="results/temporal_jepa_ablation",
        figure_dir="figures/temporal_jepa_ablation"):
    cfg = Config()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    effect, diagnostics = _training_ablations(cfg)
    effect.to_csv(output / "effect_metrics.csv", index=False)
    diagnostics.to_csv(output / "observed_diagnostics.csv", index=False)

    main_path = Path("results/temporal_jepa/observed_diagnostics.csv")
    if not main_path.exists():
        raise FileNotFoundError("Run scripts/run_temporal_jepa.py before the ablation suite")
    main_diagnostics = pd.read_csv(main_path)
    selection = _selection_sensitivity(main_diagnostics)
    selection.to_csv(output / "selection_sensitivity.csv", index=False)
    _lambda_summary(main_diagnostics).to_csv(output / "lambda_sensitivity.csv", index=False)

    training_summary = (
        diagnostics[diagnostics.horizon == 8]
        .groupby("variant")[["observed_rollout_mse", "latent_closure_mse"]]
        .mean()
        .join(
            effect[effect.horizon == 16].groupby("variant").effect_mse.mean()
            .rename("h16_effect_mse_all_schedules")
        )
        .reset_index()
    )
    training_summary.to_csv(output / "training_ablation_summary.csv", index=False)
    metadata = {
        "seeds": list(cfg.seeds),
        "jepa_weight": 1.,
        "training_variants": {
            key: {"horizons": list(value[0]), "ema_momentum": value[1]}
            for key, value in TRAINING_VARIANTS.items()
        },
        "selection_metrics_use_counterfactuals": False,
        "test_effect_metrics_used_for_selection": False,
    }
    (output / "design.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _plot(effect, diagnostics, Path(figure_dir))
    return effect, diagnostics, selection, training_summary
