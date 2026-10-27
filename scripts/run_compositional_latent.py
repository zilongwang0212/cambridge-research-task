#!/usr/bin/env python3
from temporal_dte.compositional_latent import run

if __name__ == "__main__":
    df,coverage=run()
    print("H=16 means:\n",df[df.horizon==16].groupby(["model_kind","schedule"])[
        ["one_step_effect_mse","effect_mse","factual_mse"]].mean().to_string())
    print("\nCoverage:\n",coverage.to_string(index=False))
