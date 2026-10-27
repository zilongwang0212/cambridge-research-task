#!/usr/bin/env python3
from temporal_dte.validation import run

if __name__ == "__main__":
    gate, repeat, comparison, test = run()
    print("Gate sensitivity\n", gate.to_string(index=False))
    print("\nStrong baseline\n", comparison.to_string(index=False))
    print("\nPaired test\n", test)
