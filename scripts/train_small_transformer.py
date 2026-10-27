#!/usr/bin/env python3
from temporal_dte.train_transformer import run

if __name__ == "__main__":
    _, summary = run()
    print(summary.to_string(index=False))
