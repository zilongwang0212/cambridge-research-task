#!/usr/bin/env python3
from temporal_dte.latent_test_validation import run

if __name__ == "__main__":
    models,selection=run()
    print("Family means:\n",models.groupby("family").mean(numeric_only=True).to_string())
    print("\nTop-3 selections:\n",selection[(selection.top_k==3)][
        ["gate_quantile","method","heldout_tail_ite_mse","heldout_h16_effect_mse"]].to_string(index=False))
