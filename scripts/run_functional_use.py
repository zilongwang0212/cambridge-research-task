#!/usr/bin/env python3
from temporal_dte.functional_use_experiment import run

if __name__ == "__main__":
    df=run()
    print(df.groupby("condition").mean(numeric_only=True).to_string())
