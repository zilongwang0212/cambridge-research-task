from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json, math, random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass
class Config:
    context: int = 16
    d_model: int = 32
    layers: int = 2
    heads: int = 4
    batch: int = 96
    steps: int = 450
    lr: float = 8e-4
    train_noise: float = 0.05
    latent_dims: tuple[int, ...] = (2, 4, 8, 16, 32)
    seeds: tuple[int, ...] = (0, 1, 2, 3)
    horizons: tuple[int, ...] = (1, 4, 8, 16, 32)
    shifts: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


class MedicalSystem:
    """Two deliberately small patient simulators with known intervention truth."""

    def __init__(self, kind: str):
        if kind not in {"simple", "moderate"}:
            raise ValueError(kind)
        self.kind = kind

    def propensity(self, x, c, lag, inverse=False):
        if self.kind == "simple":
            score = -0.5 + 1.0*x + 0.4*lag
        else:
            score = -0.4 + 0.8*x + 0.7*c + 0.4*lag
        if inverse:
            score = -score
        return 1/(1+np.exp(-np.clip(score, -12, 12)))

    def step(self, x, c, action, lag, noise_x=0., noise_c=0.):
        if self.kind == "simple":
            next_c = np.zeros_like(x)
            next_x = 0.72*x - 0.32*action + 0.12*lag + noise_x
        else:
            # Mild nonlinearity plus an observed, treatment-affected confounder.
            next_c = 0.65*c + 0.15*x + 0.18*action + noise_c
            effect = -(0.28 + 0.08*np.tanh(c))*action
            next_x = 0.68*x + 0.18*c + 0.10*np.sin(x) + effect + 0.10*lag + noise_x
        return next_x.astype(np.float32), next_c.astype(np.float32)


def histories(system, n, context, rng, noise=.05):
    x = np.zeros((n, context), np.float32)
    c = np.zeros((n, context), np.float32)
    a = np.zeros((n, context), np.float32)
    x[:, 0] = rng.normal(0, .8, n)
    if system.kind == "moderate":
        c[:, 0] = rng.normal(0, .7, n)
    lag = rng.binomial(1, .4, n).astype(np.float32)
    for t in range(context-1):
        p = system.propensity(x[:, t], c[:, t], lag)
        action = rng.binomial(1, p).astype(np.float32)
        a[:, t] = action
        nx = rng.normal(0, noise, n).astype(np.float32)
        nc = rng.normal(0, noise, n).astype(np.float32) if system.kind == "moderate" else 0.
        x[:, t+1], c[:, t+1] = system.step(x[:, t], c[:, t], action, lag, nx, nc)
        lag = action
    return x, c, a


class SmallMedicalTransformer(nn.Module):
    def __init__(self, cfg: Config, latent_dim: int):
        super().__init__()
        self.embed = nn.Linear(5, cfg.d_model)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.heads, cfg.d_model*3,
                                           dropout=.05, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.layers)
        self.to_latent = nn.Linear(cfg.d_model, latent_dim)
        self.head = nn.Sequential(nn.Tanh(), nn.Linear(latent_dim, 2))
        self.d_model = cfg.d_model

    def position(self, length, device):
        p = torch.arange(length, device=device).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2, device=device).float()
                        * (-math.log(10000)/self.d_model))
        pe = torch.zeros(length, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(p*div); pe[:, 1::2] = torch.cos(p*div)
        return pe

    def forward(self, x, c, past_a, current_a):
        b, length = x.shape
        time = torch.linspace(-1, 0, length, device=x.device).expand(b, length)
        hist = torch.stack([x, c, past_a, torch.zeros_like(x), time], -1)
        query = torch.stack([x[:, -1], c[:, -1], past_a[:, -2], current_a,
                             torch.ones_like(current_a)], -1)[:, None, :]
        tokens = torch.cat([hist, query], 1)
        encoded = self.encoder(self.embed(tokens) + self.position(length+1, x.device))
        return self.head(self.to_latent(encoded[:, -1]))


def train_one(system, latent_dim, seed, cfg):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    rng = np.random.default_rng(1000 + 97*seed + latent_dim)
    net = SmallMedicalTransformer(cfg, latent_dim)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    losses = []
    for step in range(cfg.steps):
        x, c, a = histories(system, cfg.batch, cfg.context, rng, cfg.train_noise)
        lag = a[:, -2]
        p = system.propensity(x[:, -1], c[:, -1], lag)
        action = rng.binomial(1, p).astype(np.float32)
        nx = rng.normal(0, cfg.train_noise, cfg.batch).astype(np.float32)
        nc = rng.normal(0, cfg.train_noise, cfg.batch).astype(np.float32) if system.kind == "moderate" else 0.
        tx, tc = system.step(x[:, -1], c[:, -1], action, lag, nx, nc)
        inputs = [torch.tensor(z, dtype=torch.float32) for z in (x,c,a,action)]
        target = torch.tensor(np.stack([tx,tc], 1), dtype=torch.float32)
        pred = net(*inputs)
        # c is absent in the simple system; do not let a trivial zero target dominate.
        loss = torch.mean((pred[:, 0]-target[:, 0])**2)
        if system.kind == "moderate":
            loss = loss + .5*torch.mean((pred[:, 1]-target[:, 1])**2)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % 50 == 0:
            losses.append((step, float(loss.detach())))
    return net.eval(), losses


@torch.no_grad()
def predict_step(net, x, c, a, action):
    tensors = [torch.tensor(z, dtype=torch.float32) for z in (x,c,a,action)]
    pred = net(*tensors).numpy()
    return pred[:, 0], pred[:, 1]


def policy_schedule(system, x0, c0, lag0, horizon, shift, rng):
    """Draw a policy between the historical and inverse-state treatment rules."""
    n = len(x0); actions = np.zeros((n, horizon), np.float32)
    x, c, lag = x0.copy(), c0.copy(), lag0.copy()
    uniforms = rng.uniform(size=(n, horizon))
    for t in range(horizon):
        pb = system.propensity(x, c, lag, inverse=False)
        pi = system.propensity(x, c, lag, inverse=True)
        p = (1-shift)*pb + shift*pi
        action = (uniforms[:, t] < p).astype(np.float32)
        actions[:, t] = action
        x, c = system.step(x, c, action, lag)
        lag = action
    return actions


def true_rollout(system, x0, c0, lag0, actions):
    xs, cs = [], []
    x, c, lag = x0.copy(), c0.copy(), lag0.copy()
    for t in range(actions.shape[1]):
        x, c = system.step(x, c, actions[:, t], lag)
        xs.append(x.copy()); cs.append(c.copy()); lag = actions[:, t]
    return np.stack(xs, 1), np.stack(cs, 1)


@torch.no_grad()
def model_rollout(net, x, c, a, actions):
    preds = []
    for t in range(actions.shape[1]):
        nx, nc = predict_step(net, x, c, a, actions[:, t])
        preds.append(nx.copy())
        x = np.concatenate([x[:, 1:], nx[:, None]], 1)
        c = np.concatenate([c[:, 1:], nc[:, None]], 1)
        a = np.concatenate([a[:, 1:-1], actions[:, t:t+1], np.zeros((len(x),1),np.float32)], 1)
    return np.stack(preds, 1)


@torch.no_grad()
def evaluate(net, system, cfg, seed, n=256):
    rng = np.random.default_rng(9000 + seed)
    x, c, a = histories(system, n, cfg.context, rng, cfg.train_noise)
    lag = a[:, -2]
    p = system.propensity(x[:, -1], c[:, -1], lag)
    action = rng.binomial(1, p).astype(np.float32)
    tx, tc = system.step(x[:, -1], c[:, -1], action, lag)
    px, _ = predict_step(net, x, c, a, action)
    factual = float(np.mean((px-tx)**2))
    rows = []
    for h in cfg.horizons:
        zero = np.zeros((n,h), np.float32)
        y0, _ = true_rollout(system, x[:,-1], c[:,-1], lag, zero)
        p0 = model_rollout(net, x.copy(), c.copy(), a.copy(), zero)
        for shift in cfg.shifts:
            target_actions = policy_schedule(system, x[:,-1], c[:,-1], lag, h, shift, rng)
            y1, _ = true_rollout(system, x[:,-1], c[:,-1], lag, target_actions)
            p1 = model_rollout(net, x.copy(), c.copy(), a.copy(), target_actions)
            cf = .5*(np.mean((p0-y0)**2)+np.mean((p1-y1)**2))
            effect = np.mean(((p1-p0)-(y1-y0))**2)
            rows.append({"factual_error":factual, "horizon":h, "policy_shift":shift,
                         "cf_error":float(cf), "effect_error":float(effect)})
    return pd.DataFrame(rows)


def summarize(metrics):
    rows = []
    for kind in ["simple", "moderate"]:
        terminal = metrics[(metrics.system==kind)&(metrics.horizon==16)&(metrics.policy_shift==1)]
        model = terminal.drop_duplicates(["latent_dim","seed"])
        cutoff = model.factual_error.quantile(.25)
        gated = model[model.factual_error <= cutoff]
        best_pred = model.loc[model.factual_error.idxmin()]
        matched = []
        values = model.reset_index(drop=True)
        for i in range(len(values)):
            for j in range(i+1, len(values)):
                a, b = values.loc[i], values.loc[j]
                gap = abs(a.factual_error-b.factual_error)/min(a.factual_error,b.factual_error)
                if gap <= .05:
                    matched.append((
                        max(a.cf_error,b.cf_error)/min(a.cf_error,b.cf_error),
                        max(a.effect_error,b.effect_error)/min(a.effect_error,b.effect_error),
                        gap,
                    ))
        best_cf = max(matched, key=lambda z:z[0])
        best_effect = max(matched, key=lambda z:z[1])
        rows.append({
            "system":kind,
            "spearman_prediction_cf":model.factual_error.corr(model.cf_error, method="spearman"),
            "spearman_prediction_effect":model.factual_error.corr(model.effect_error, method="spearman"),
            "prediction_gate_quantile":.25,
            "gated_models":len(gated),
            "gated_cf_max_min_ratio":gated.cf_error.max()/gated.cf_error.min(),
            "gated_effect_max_min_ratio":gated.effect_error.max()/gated.effect_error.min(),
            "prediction_selected_cf_regret":best_pred.cf_error-model.cf_error.min(),
            "prediction_selected_effect_regret":best_pred.effect_error-model.effect_error.min(),
            "matched_pair_tolerance":.05,
            "matched_cf_max_ratio":best_cf[0],
            "matched_cf_prediction_gap":best_cf[2],
            "matched_effect_max_ratio":best_effect[1],
            "matched_effect_prediction_gap":best_effect[2],
        })
    return pd.DataFrame(rows)


def plot(metrics, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, kind in zip(axes, ["simple", "moderate"]):
        q = metrics[(metrics.system==kind)&(metrics.horizon==16)&(metrics.policy_shift==1)].drop_duplicates(["latent_dim","seed"])
        for d, z in q.groupby("latent_dim"):
            ax.scatter(z.factual_error, z.cf_error, label=f"z={d}", s=55)
        ax.set(xlabel="Historical-policy one-step error", ylabel="Counterfactual rollout error",
               xscale="log", yscale="log", title=f"{kind.capitalize()} system: prediction vs intervention")
        ax.legend()
    fig.tight_layout(); fig.savefig(outdir/"figure_p1_decoupling.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, kind in zip(axes, ["simple", "moderate"]):
        q = metrics[(metrics.system==kind)&(metrics.policy_shift==1)]
        s = q.groupby(["latent_dim","horizon"]).effect_error.mean().reset_index()
        for d, z in s.groupby("latent_dim"):
            ax.plot(z.horizon, z.effect_error, marker="o", label=f"z={d}")
        ax.set(xlabel="Rollout horizon", ylabel="Treatment-effect error", yscale="log",
               title=f"{kind.capitalize()} system: rollout stress")
        ax.legend()
    fig.tight_layout(); fig.savefig(outdir/"figure_p1_rollout.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, kind in zip(axes, ["simple", "moderate"]):
        q = metrics[(metrics.system==kind)&(metrics.horizon==16)]
        s = q.groupby(["latent_dim","policy_shift"]).effect_error.mean().reset_index()
        for d, z in s.groupby("latent_dim"):
            ax.plot(z.policy_shift, z.effect_error, marker="o", label=f"z={d}")
        ax.set(xlabel="Policy shift (historical → inverse)", ylabel="Treatment-effect error",
               title=f"{kind.capitalize()} system: policy stress")
        ax.legend()
    fig.tight_layout(); fig.savefig(outdir/"figure_p1_policy_shift.png", dpi=220); plt.close(fig)

    # Full factorial view: every cell averages seeds at one horizon/shift pair.
    fig, axes = plt.subplots(2, 5, figsize=(19, 7), sharex=True, sharey="row")
    for row, kind in enumerate(["simple", "moderate"]):
        q = metrics[metrics.system==kind]
        for col, d in enumerate(sorted(q.latent_dim.unique())):
            table = q[q.latent_dim==d].groupby(["horizon","policy_shift"]).effect_error.mean().unstack()
            im = axes[row,col].imshow(np.log10(table.values+1e-12), origin="lower", aspect="auto",
                                      extent=[0,1,0,len(table.index)-1], cmap="viridis")
            axes[row,col].set_title(f"{kind}, z={d}")
            axes[row,col].set_yticks(range(len(table.index)), table.index)
            axes[row,col].set_xlabel("Policy shift")
            if col == 0: axes[row,col].set_ylabel("Horizon")
        fig.colorbar(im, ax=axes[row,:].tolist(), label="log10 treatment-effect error", shrink=.8)
    fig.subplots_adjust(wspace=.25, hspace=.35, right=.92)
    fig.savefig(outdir/"figure_p1_full_factorial.png", dpi=220, bbox_inches="tight"); plt.close(fig)


def run(output_dir="results/medical_part1", figure_dir="figures/medical_part1"):
    cfg = Config(); out, figs = Path(output_dir), Path(figure_dir)
    out.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    parts, logs = [], {}
    for kind in ["simple", "moderate"]:
        system = MedicalSystem(kind)
        for latent_dim in cfg.latent_dims:
            for seed in cfg.seeds:
                net, hist = train_one(system, latent_dim, seed, cfg)
                result = evaluate(net, system, cfg, seed)
                result["system"], result["latent_dim"], result["seed"] = kind, latent_dim, seed
                parts.append(result); logs[f"{kind}_z{latent_dim}_s{seed}"] = hist
                torch.save(net.state_dict(), out/f"{kind}_z{latent_dim}_seed{seed}.pt")
    metrics = pd.concat(parts, ignore_index=True); metrics.to_csv(out/"metrics.csv", index=False)
    summary = summarize(metrics); summary.to_csv(out/"summary.csv", index=False)
    full = metrics.assign(log_effect=np.log10(metrics.effect_error+1e-12)).groupby(
        ["system","latent_dim"]).agg(
            factual_mean=("factual_error","mean"),
            effect_geomean=("log_effect",lambda x:10**x.mean()),
            effect_median=("effect_error","median"),
            cf_median=("cf_error","median")).reset_index()
    full.to_csv(out/"full_grid_summary.csv", index=False)
    (out/"config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (out/"training_logs.json").write_text(json.dumps(logs), encoding="utf-8")
    plot(metrics, figs)
    return metrics, summary
