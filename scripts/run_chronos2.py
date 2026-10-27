#!/usr/bin/env python3
"""Zero-shot external-validity check for Chronos-2.

This script is optional because it downloads a large third-party checkpoint. It never
silently substitutes synthetic or mocked predictions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

from temporal_dte.evaluate import make_policy_pair
from temporal_dte.system import TRUE_DYNAMICS, rollout, simulate_panel


def load_pipeline(model_id: str):
    try:
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise SystemExit("Install the optional dependency: pip install '.[foundation]'") from exc
    return Chronos2Pipeline.from_pretrained(model_id, device_map="cpu")


def forecast(pipe, history_x, history_a, future_a, horizon, batch_size=32):
    inputs = []
    for x, a, fa in zip(history_x, history_a, future_a):
        inputs.append({
            "target": np.asarray(x, dtype=np.float32),
            "past_covariates": {"treatment": np.asarray(a, dtype=np.float32)},
            "future_covariates": {"treatment": np.asarray(fa, dtype=np.float32)},
        })
    preds = pipe.predict(inputs, prediction_length=horizon, batch_size=batch_size,
                         quantile_levels=[0.1, 0.5, 0.9])
    point = []
    for pred in preds:
        arr = pred.detach().cpu().numpy() if hasattr(pred, "detach") else np.asarray(pred)
        # Chronos returns (n_quantiles, horizon); index 1 is the requested median.
        point.append(arr[1] if arr.ndim == 2 else arr)
    return np.stack(point)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--n-series", type=int, default=64)
    parser.add_argument("--output", default="results/foundation")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    panel = simulate_panel(args.n_series, 160, seed=2027)
    records = []
    pipe = load_pipeline(args.model_id)
    for history_len in [32, 128]:
        history_x = panel["x"][:, :history_len]
        history_a = panel["a"][:, :history_len]
        x0, prev = history_x[:, -1], history_a[:, -1]
        for horizon in [16, 32]:
            rng = np.random.default_rng(700 + history_len + horizon)
            for policy in ["in_policy", "shift", "sustained"]:
                p0, p1 = make_policy_pair(args.n_series, horizon, policy, rng)
                pred0 = forecast(pipe, history_x, history_a, p0, horizon)
                pred1 = forecast(pipe, history_x, history_a, p1, horizon)
                true0 = rollout(TRUE_DYNAMICS, x0, prev, p0)[:, 1:]
                true1 = rollout(TRUE_DYNAMICS, x0, prev, p1)[:, 1:]
                records.append({
                    "model": args.model_id,
                    "history_length": history_len,
                    "horizon": horizon,
                    "policy": policy,
                    "dte_error": float(np.mean(((pred1 - pred0) - (true1 - true0)) ** 2)),
                    "interventional_error": float(0.5 * (np.mean((pred0 - true0) ** 2) + np.mean((pred1 - true1) ** 2))),
                })
    result = pd.DataFrame(records)
    result.to_csv(out / "chronos2_metrics.csv", index=False)
    (out / "chronos2_run.json").write_text(json.dumps({
        "model_id": args.model_id, "n_series": args.n_series, "status": "completed"
    }, indent=2), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
