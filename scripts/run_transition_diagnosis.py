#!/usr/bin/env python3
from temporal_dte.transition_diagnosis import run

if __name__ == "__main__":
    df,_=run();q=df[(df.policy=="alternating")&(df.horizon==16)]
    print(q.groupby("family")[["encoded_effect_mse","rolled_effect_mse","closure_mse","amplification"]].mean().to_string())
