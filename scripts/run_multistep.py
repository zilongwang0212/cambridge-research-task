#!/usr/bin/env python3
from temporal_dte.multistep import evaluate

if __name__ == "__main__":
    selected, diagnostics, summary, family = evaluate()
    print("Overall\n", summary.to_string(index=False))
    print("\nBy family\n", family.to_string(index=False))
