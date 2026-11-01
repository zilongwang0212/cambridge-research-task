#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from temporal_dte.medical_latent import Config, LatentWorldModel
from temporal_dte.temporal_jepa import interventional_trajectory_diagnostics


def plot_test_metrics(trajectory, effect, output_path):
    methods = ("one_step", "multi_step", "jepa_1")
    method_labels = ("One-step", "Multi-step", "Temporal JEPA")
    colors = ("#B8B8B8", "#4C78A8", "#2A9D8F")
    schedules = ("none", "seen_block", "alternating", "separated", "late_separated")
    schedule_labels = ("None", "Early", "Alternating", "Separated", "Late-separated")
    effect_schedules = schedules[1:]
    effect_labels = schedule_labels[1:]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    width = .24
    panels = (
        (axes[0], trajectory, schedules, schedule_labels,
         "x_trajectory_mse", "x_trajectory_se", r"$X$-trajectory MSE"),
        (axes[1], trajectory, schedules, schedule_labels,
         "state_trajectory_mse", "state_trajectory_se", r"$(X,C)$-trajectory MSE"),
        (axes[2], effect, effect_schedules, effect_labels,
         "effect_mse", "effect_se", "Treatment-effect MSE"),
    )
    for ax, data, scheds, labels, mean_col, se_col, title in panels:
        x = np.arange(len(scheds))
        for j, (method, method_label, color) in enumerate(zip(methods, method_labels, colors)):
            z = data[data.model_kind.eq(method)].set_index("schedule").loc[list(scheds)]
            ax.bar(
                x + (j - 1) * width,
                1000 * z[mean_col],
                width,
                yerr=1000 * z[se_col],
                capsize=2.5,
                color=color,
                edgecolor="white",
                linewidth=.6,
                label=method_label,
            )
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=22, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel(r"MSE $\times 10^{3}$")
        ax.grid(axis="y", alpha=.22, linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, .94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = Config()
    result_dir = Path("results/temporal_jepa")
    selected = ("one_step", "multi_step", "jepa_1")
    frames = []
    for model_kind in selected:
        for seed in cfg.seeds:
            model = LatentWorldModel(cfg, latent_dim=8)
            state = torch.load(
                result_dir / f"{model_kind}_seed{seed}.pt",
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state)
            model.eval()
            metrics = interventional_trajectory_diagnostics(model, cfg, seed)
            metrics["model_kind"] = model_kind
            frames.append(metrics)

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(result_dir / "interventional_trajectory_metrics.csv", index=False)
    summary = (
        results.groupby(["model_kind", "schedule", "horizon"])
        .agg(
            x_trajectory_mse=("x_trajectory_mse", "mean"),
            x_trajectory_se=("x_trajectory_mse", "sem"),
            c_trajectory_mse=("c_trajectory_mse", "mean"),
            c_trajectory_se=("c_trajectory_mse", "sem"),
            state_trajectory_mse=("state_trajectory_mse", "mean"),
            state_trajectory_se=("state_trajectory_mse", "sem"),
        )
        .reset_index()
    )
    summary.to_csv(result_dir / "interventional_trajectory_summary.csv", index=False)

    effect_raw = pd.read_csv(result_dir / "effect_metrics.csv")
    effect_raw = effect_raw[
        effect_raw.horizon.eq(16) & effect_raw.model_kind.isin(selected)
    ]
    effect_summary = (
        effect_raw.groupby(["model_kind", "schedule"])
        .effect_mse.agg(effect_mse="mean", effect_se="sem")
        .reset_index()
    )
    plot_test_metrics(
        summary[summary.horizon.eq(16)],
        effect_summary,
        Path("figures/temporal_jepa/interventional_test_metrics"),
    )

    h16 = summary[summary.horizon == 16]
    print("H=16 X-trajectory MSE")
    print(h16.pivot(index="model_kind", columns="schedule", values="x_trajectory_mse"))
    print("\nH=16 joint (X,C)-trajectory MSE")
    print(h16.pivot(index="model_kind", columns="schedule", values="state_trajectory_mse"))


if __name__ == "__main__":
    main()
