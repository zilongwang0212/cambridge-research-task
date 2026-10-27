#!/usr/bin/env python3
from temporal_dte.medical_part1 import run

if __name__ == "__main__":
    _, summary = run()
    print(summary.to_string(index=False))
