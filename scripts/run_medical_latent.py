#!/usr/bin/env python3
from temporal_dte.medical_latent import run

if __name__ == "__main__":
    summary,_=run()
    print(summary.groupby("latent_dim").mean(numeric_only=True).to_string())
