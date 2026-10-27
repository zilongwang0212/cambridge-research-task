#!/usr/bin/env python3
from temporal_dte.two_stage import evaluate

if __name__ == "__main__":
    selected, sensitivity, by_family, loo = evaluate()
    print("Threshold sensitivity\n", sensitivity.to_string(index=False))
    print("\nPrimary q=0.25\n", selected[selected.threshold.eq(.25)].to_string(index=False))
    print("\nLeave-one-system-out threshold validation\n", loo.to_string(index=False))
    print("LOSO median/mean/worst ED percentile:", loo.test_dynamics_percentile.median(),
          loo.test_dynamics_percentile.mean(), loo.test_dynamics_percentile.max())
