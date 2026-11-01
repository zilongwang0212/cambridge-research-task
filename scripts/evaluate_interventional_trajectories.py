#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
import torch

from temporal_dte.medical_latent import Config, LatentWorldModel
from temporal_dte.temporal_jepa import interventional_trajectory_diagnostics


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

    h16 = summary[summary.horizon == 16]
    print("H=16 X-trajectory MSE")
    print(h16.pivot(index="model_kind", columns="schedule", values="x_trajectory_mse"))
    print("\nH=16 joint (X,C)-trajectory MSE")
    print(h16.pivot(index="model_kind", columns="schedule", values="state_trajectory_mse"))


if __name__ == "__main__":
    main()
