from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .medical_latent import Config, LatentWorldModel, PatientSystem, fixed_actions, generate_batch


def state_rollout(x, c, d, u, lag, actions):
    """Return all deterministic simulator states, including the initial state."""
    xs, cs, ds, us = [x.copy()], [c.copy()], [d.copy()], [u.copy()]
    for k in range(actions.shape[1]):
        x, c, d, u = PatientSystem.step(x, c, d, u, actions[:, k], lag)
        xs.append(x.copy()); cs.append(c.copy()); ds.append(d.copy()); us.append(u.copy())
        lag = actions[:, k]
    return tuple(np.stack(v, 1) for v in (xs, cs, ds, us))


@torch.no_grad()
def teacher_forced_path(net, histories, actions):
    """One-step predictions after re-encoding the true branch history at every step."""
    x0, c0, u0, a0, d0, lag0 = histories
    states = state_rollout(x0[:, -1], c0[:, -1], d0, u0[:, -1], lag0, actions)
    sx, sc, _, su = states
    xhist, chist, uhist, ahist = x0.copy(), c0.copy(), u0.copy(), a0.copy()
    out = []
    for k in range(actions.shape[1]):
        # The original training convention places the action causing the next
        # transition on the last token and also passes it to latent advance.
        ahist[:, -1] = actions[:, k]
        z = net.encode(torch.tensor(xhist[:, -net_cfg.context:]),
                       torch.tensor(chist[:, -net_cfg.context:]),
                       torch.tensor(uhist[:, -net_cfg.context:]),
                       torch.tensor(ahist[:, -net_cfg.context:]))
        pred, _ = net.rollout(z, torch.tensor(actions[:, k:k+1]))
        out.append(pred[:, 0, 0].numpy())
        xhist = np.c_[xhist, sx[:, k+1]]
        chist = np.c_[chist, sc[:, k+1]]
        uhist = np.c_[uhist, su[:, k+1]]
        next_action = actions[:, min(k+1, actions.shape[1]-1)]
        ahist = np.c_[ahist, next_action]
    return np.stack(out, 1), sx[:, 1:]


def encode_np(net, x, c, u, a):
    return net.encode(*[torch.tensor(v, dtype=torch.float32) for v in (x, c, u, a)])


@torch.no_grad()
def free_path(net, z, actions):
    return net.rollout(z, torch.tensor(actions))[0][:, :, 0].numpy()


@torch.no_grad()
def input_use_tests(net, x, c, d, u, a, delta_c=.15):
    """Behavioral finite differences for C and treatment-memory use.

    These perturb inputs, not latent coordinates, avoiding the claim that a
    linear probe direction is a clean causal coordinate in an entangled Z.
    """
    n = len(x); zeros = torch.zeros(n); ones = torch.ones(n)

    def contrast(cx, aa):
        z = encode_np(net, x, cx, u, aa)
        y0 = net.rollout(z, zeros[:, None])[0][:, 0, 0]
        y1 = net.rollout(z, ones[:, None])[0][:, 0, 0]
        return (y1-y0).numpy()

    base = contrast(c, a)
    cplus = c.copy(); cplus[:, -1] += delta_c
    c_sensitivity = (contrast(cplus, a)-base)/delta_c
    true_c_sensitivity = -.10/(np.cosh(c[:, -1])**2)

    # Toggle A_{t-3}, holding observed X/C fixed. This asks whether the model
    # uses treatment history as memory. The controlled simulator derivative
    # through dose is -0.055 * 0.72^2 for next X (lag is unchanged).
    mem = a.copy(); old = mem[:, -3].copy(); mem[:, -3] = 1-old
    z0 = encode_np(net, x, c, u, a); z1 = encode_np(net, x, c, u, mem)
    y0 = net.rollout(z0, zeros[:, None])[0][:, 0, 0].numpy()
    y1 = net.rollout(z1, zeros[:, None])[0][:, 0, 0].numpy()
    signed_change = (y1-y0)*(1-2*old)
    true_mem_change = np.full(n, -.055*(.72**2), dtype=np.float32)
    return {
        "c_use_slope": float(c_sensitivity.mean()),
        "c_use_target": float(true_c_sensitivity.mean()),
        "c_use_mae": float(np.mean(np.abs(c_sensitivity-true_c_sensitivity))),
        "memory_use_change": float(signed_change.mean()),
        "memory_use_target": float(true_mem_change[0]),
        "memory_use_mae": float(np.mean(np.abs(signed_change-true_mem_change))),
    }


def evaluate_model(net, cfg, seed, n=384):
    rng = np.random.default_rng(23000+seed)
    x, c, d, u, a = generate_batch(n, cfg.context, rng, cfg.noise)
    histories = (x, c, u, a.copy(), d[:, -1], a[:, -2])
    z = encode_np(net, x, c, u, a)
    use = input_use_tests(net, x, c, d, u, a)
    rows = []
    for horizon in (1, 4, 8, 16, 32):
        for policy in ("alternating", "block", "sustained"):
            a0 = fixed_actions(n, horizon, "none")
            a1 = fixed_actions(n, horizon, policy)
            tf0, y0 = teacher_forced_path(net, histories, a0)
            tf1, y1 = teacher_forced_path(net, histories, a1)
            fr0, fr1 = free_path(net, z, a0), free_path(net, z, a1)
            true_tau, tf_tau, fr_tau = y1-y0, tf1-tf0, fr1-fr0
            effect_scale = np.mean(true_tau**2) + 1e-12
            tf_effect_mse = np.mean((tf_tau-true_tau)**2)
            free_effect_mse = np.mean((fr_tau-true_tau)**2)
            rows.append({
                "seed": seed, "horizon": horizon, "policy": policy,
                "tf_state_mse": .5*(np.mean((tf0-y0)**2)+np.mean((tf1-y1)**2)),
                "free_state_mse": .5*(np.mean((fr0-y0)**2)+np.mean((fr1-y1)**2)),
                "effect_scale": effect_scale,
                "tf_effect_mse": tf_effect_mse,
                "free_effect_mse": free_effect_mse,
                "tf_effect_nmse": tf_effect_mse/effect_scale,
                "free_effect_nmse": free_effect_mse/effect_scale,
                "amplification": free_effect_mse/(tf_effect_mse+1e-12),
                **use,
            })
    return pd.DataFrame(rows)


def add_failure_flags(summary):
    """Add non-exclusive failure signatures using fixed, interpretable thresholds."""
    summary = summary.copy()
    summary["c_use_relative_error"] = summary.c_use_mae/(summary.c_use_target.abs()+1e-12)
    summary["memory_use_relative_error"] = summary.memory_use_mae/(summary.memory_use_target.abs()+1e-12)
    summary["information_use_failure"] = (
        (summary.c_use_relative_error > .5) | (summary.memory_use_relative_error > .5))
    summary["local_response_failure"] = summary.tf_effect_nmse > .10
    summary["rollout_amplification_failure"] = summary.amplification > 2.0
    def signature(r):
        labels = []
        if r.information_use_failure: labels.append("use")
        if r.local_response_failure: labels.append("local")
        if r.rollout_amplification_failure: labels.append("amplification")
        return "+".join(labels) if labels else "none"
    summary["failure_signature"] = summary.apply(signature, axis=1)
    return summary


def plot(summary, curves, figure_dir):
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    g = summary.groupby("latent_dim").mean(numeric_only=True)
    axes[0].plot(g.index, g.c_use_mae, "o-", label="C effect-modifier")
    axes[0].plot(g.index, g.memory_use_mae, "o-", label="dose memory")
    axes[0].set(xlabel="Latent dimension", ylabel="finite-difference MAE", title="1. Is information used?")
    axes[0].legend()
    axes[1].plot(g.index, g.tf_effect_mse, "o-", label="Teacher-forced")
    axes[1].plot(g.index, g.free_effect_mse, "o-", label="Free rollout")
    axes[1].set(xlabel="Latent dimension", ylabel="Effect MSE (H=16)", yscale="log", title="2–3. Local error vs amplification")
    axes[1].legend()
    for dim, z in curves.groupby("latent_dim"):
        z = z.groupby("horizon").mean(numeric_only=True)
        axes[2].plot(z.index, z.free_effect_mse/(z.tf_effect_mse+1e-12), "o-", label=f"z={dim}")
    axes[2].set(xlabel="Horizon", ylabel="Free / teacher-forced effect MSE", yscale="log", title="Closed-loop amplification")
    axes[2].legend(ncol=2)
    fig.tight_layout(); fig.savefig(figure_dir/"failure_attribution.png", dpi=220, bbox_inches="tight"); plt.close(fig)


net_cfg = Config()


def run(model_dir="results/medical_latent", output_dir="results/latent_failure_diagnosis",
        figure_dir="figures/latent_failure_diagnosis"):
    cfg = net_cfg; model_dir, out = Path(model_dir), Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    parts = []
    for dim in cfg.latent_dims:
        for seed in cfg.seeds:
            net = LatentWorldModel(cfg, dim)
            net.load_state_dict(torch.load(model_dir/f"z{dim}_seed{seed}.pt", map_location="cpu", weights_only=True))
            net.eval(); result = evaluate_model(net, cfg, seed); result["latent_dim"] = dim; parts.append(result)
    curves = pd.concat(parts, ignore_index=True)
    summary = curves[(curves.horizon == 16) & (curves.policy == "alternating")].copy()
    summary = add_failure_flags(summary)
    curves.to_csv(out/"teacher_forcing_curves.csv", index=False)
    summary.to_csv(out/"model_failure_attribution.csv", index=False)
    (out/"config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    plot(summary, curves, Path(figure_dir))
    return summary, curves
