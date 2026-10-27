#!/usr/bin/env python3
from temporal_dte.temporal_jepa import run

if __name__ == "__main__":
    effect,diag,selection=run()
    print("Selection:",selection)
    print("\nH=16 effect MSE")
    print(effect[effect.horizon==16].groupby(["model_kind","schedule"]).effect_mse.mean().unstack().to_string())
    print("\nH=16 observational diagnostics")
    print(diag[diag.horizon==16].groupby("model_kind")[["observed_rollout_mse","latent_closure_mse","action_sensitivity"]].mean().to_string())
