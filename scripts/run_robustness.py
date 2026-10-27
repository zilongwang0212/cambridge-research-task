#!/usr/bin/env python3
from temporal_dte.robustness import run

if __name__ == "__main__":
    summary, family = run()
    print(summary.to_string(index=False))
    print("\nFamily summary\n", family.to_string(index=False))
