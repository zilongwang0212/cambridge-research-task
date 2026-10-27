#!/usr/bin/env python3
from temporal_dte.train_transformer import run_controlled, evaluate_controlled_rollouts, evaluate_controlled_ood

if __name__ == "__main__":
    _, summary = run_controlled()
    print(summary.to_string(index=False))
    _, rollout_summary = evaluate_controlled_rollouts()
    print("\nAutoregressive free rollout:\n", rollout_summary.to_string(index=False))
    _, ood_summary = evaluate_controlled_ood()
    print("\nOOD controlled dynamics:\n", ood_summary.to_string(index=False))
