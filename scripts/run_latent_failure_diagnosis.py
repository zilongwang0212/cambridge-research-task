#!/usr/bin/env python3
from temporal_dte.latent_failure_diagnosis import run

if __name__ == "__main__":
    summary, _ = run()
    cols = ["tf_effect_mse", "free_effect_mse", "amplification", "c_use_mae", "memory_use_mae"]
    print(summary.groupby("latent_dim")[cols].mean().to_string())
    print("\nFailure signatures:\n", summary.failure_signature.value_counts().to_string())
