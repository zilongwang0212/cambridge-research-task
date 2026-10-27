#!/usr/bin/env python3
from temporal_dte.compositional_latent import run_closure

if __name__ == "__main__":
    df,compare,selection=run_closure();print("Selection:",selection)
    chosen=compare[(compare.horizon==16)&compare.schedule.isin(["seen_block","separated","late_separated","alternating"])]
    print(chosen.groupby(["model_kind","schedule"])[["one_step_effect_mse","effect_mse","factual_mse"]].mean().to_string())
