from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json, math, random
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

from .dual_experiment import Model, rollout


@dataclass
class Config:
    context: int = 16
    d_model: int = 48
    layers: int = 2
    heads: int = 4
    batch: int = 48
    steps: int = 350
    lr: float = 8e-4
    train_systems: int = 20
    test_systems: int = 10


def make_systems(n, seed):
    rng = np.random.default_rng(seed)
    systems = []
    for i in range(n):
        nonlinear = i % 2 == 1
        systems.append(Model(
            alpha=float(rng.uniform(.45, .86)),
            sin_coef=float(rng.uniform(-.22, .30)) if nonlinear else 0.,
            treatment_base=float(rng.uniform(.10, .45)),
            treatment_state=float(rng.uniform(-.18, .38)) if nonlinear else 0.,
            lag_coef=float(rng.uniform(-.12, .30)),
            state_scale=float(rng.uniform(.65, 1.45)) if nonlinear else 1.,
            intercept=float(rng.uniform(-.08, .08)),
        ))
    return systems


def observational_history(model, n, context, rng):
    x = np.empty((n, context), np.float32)
    a = np.zeros((n, context), np.float32)
    x[:, 0] = rng.normal(0, .75, n)
    lag = np.zeros(n)
    for t in range(context-1):
        p = 1/(1+np.exp(-(-1.1+1.3*x[:, t]+.9*lag)))
        a[:, t] = rng.binomial(1, p)
        x[:, t+1] = model.step(x[:, t], a[:, t], lag) + rng.normal(0, .05, n)
        lag = a[:, t]
    return x, a


def ood_history(model, n, context, rng, policy):
    """Generate valid histories under policies absent from Transformer training."""
    x = np.empty((n, context), np.float32)
    a = np.zeros((n, context), np.float32)
    x[:, 0] = rng.uniform(-2.5, 2.5, n)
    lag = rng.binomial(1, .5, n).astype(np.float32)
    for t in range(context - 1):
        if policy == "balanced_random":
            action = rng.binomial(1, .5, n)
        elif policy == "alternating":
            action = ((np.arange(n) + t) % 2).astype(np.float32)
        elif policy == "long_blocks":
            action = (((t // 4) + np.arange(n)) % 2).astype(np.float32)
        elif policy == "inverse_state":
            # Opposite of the confounded observational assignment: treatment is
            # most likely at low rather than high states.
            p = 1/(1+np.exp(-(0.2 - 1.5*x[:, t] - .7*lag)))
            action = rng.binomial(1, p)
        else:
            raise ValueError(policy)
        a[:, t] = action
        x[:, t+1] = model.step(x[:, t], action, lag) + rng.normal(0, .05, n)
        lag = action
    return x, a


def policies(n, h, rng, split="train"):
    p0 = np.zeros((n, h), np.float32)
    p1 = np.zeros((n, h), np.float32)
    if split == "train":
        kind = rng.choice(["pulse", "block", "period2", "random"])
        if kind == "pulse":
            for i in range(n): p1[i, rng.integers(0, h)] = 1
        elif kind == "block":
            width = min(h, int(rng.integers(2, 4)))
            for i in range(n):
                start = int(rng.integers(0, h-width+1)); p1[i, start:start+width] = 1
        elif kind == "period2":
            for i in range(n): p1[i, rng.integers(0, 2)::2] = 1
        else:
            p1 = rng.binomial(1, .35, (n, h)).astype(np.float32)
    else:
        kinds = ["period3", "period5", "sustained", "double_block"]
        for i in range(n):
            kind = kinds[i % len(kinds)]
            if kind == "period3": p1[i, i % 3::3] = 1
            elif kind == "period5": p1[i, i % 5::5] = 1
            elif kind == "sustained": p1[i] = 1
            else:
                p1[i, :min(2,h)] = 1; p1[i, max(0,h-3):max(0,h-1)] = 1
    return p0, p1


def observational_future(model, x0, lag0, horizon, rng):
    n=len(x0); actions=np.zeros((n,horizon),np.float32); ys=np.empty((n,horizon),np.float32)
    x=x0.copy(); lag=lag0.copy()
    for t in range(horizon):
        p=1/(1+np.exp(-(-1.1+1.3*x+.9*lag)))
        actions[:,t]=rng.binomial(1,p)
        x=model.step(x,actions[:,t],lag)
        ys[:,t]=x; lag=actions[:,t]
    return actions,ys


def batch_for(systems, cfg, rng, horizon, split="train"):
    # Mix systems inside each batch to encourage cross-system pretraining.
    xs, aas, f0s, f1s, y0s, y1s = [], [], [], [], [], []
    per = max(1, cfg.batch // min(len(systems), 8))
    chosen = rng.choice(len(systems), size=max(1, cfg.batch // per), replace=True)
    for sid in chosen:
        m = systems[int(sid)]
        x, a = observational_history(m, per, cfg.context, rng)
        f0, f1 = policies(per, horizon, rng, split)
        lag = a[:, -2] if cfg.context > 1 else np.zeros(per)
        y0 = rollout(m, x[:, -1], lag, f0)[:, 1:]
        y1 = rollout(m, x[:, -1], lag, f1)[:, 1:]
        xs.append(x); aas.append(a); f0s.append(f0); f1s.append(f1); y0s.append(y0); y1s.append(y1)
    arrays = [np.concatenate(z, 0)[:cfg.batch] for z in [xs, aas, f0s, f1s, y0s, y1s]]
    return [torch.tensor(z, dtype=torch.float32) for z in arrays]


class ActionTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Linear(4, cfg.d_model)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.heads, cfg.d_model*3,
                                           dropout=.05, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.layers)
        self.head = nn.Linear(cfg.d_model, 1)
        self.d_model = cfg.d_model

    def position(self, length, device):
        p = torch.arange(length, device=device).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2, device=device).float()*(-math.log(10000)/self.d_model))
        pe = torch.zeros(length, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(p*div); pe[:, 1::2] = torch.cos(p*div)
        return pe

    def forward(self, past_x, past_a, future_a):
        b, c = past_x.shape; h = future_a.shape[1]
        past = torch.stack([past_x, past_a, torch.zeros_like(past_x),
                            torch.linspace(-1, 0, c, device=past_x.device).expand(b, c)], -1)
        future = torch.stack([torch.zeros_like(future_a), future_a, torch.ones_like(future_a),
                              torch.linspace(0, 1, h, device=past_x.device).expand(b, h)], -1)
        z = self.proj(torch.cat([past, future], 1)) + self.position(c+h, past_x.device)
        return self.head(self.encoder(z)[:, c:]).squeeze(-1)


def train_one(objective, seed, cfg, train_systems):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    rng = np.random.default_rng(1000+seed)
    net = ActionTransformer(cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    history = []
    for step in range(cfg.steps):
        h = int(rng.choice([4, 8, 16]))
        x, a, f0, f1, y0, y1 = batch_for(train_systems, cfg, rng, h, "train")
        p0, p1 = net(x, a, f0), net(x, a, f1)
        # Treat one randomly selected intervention trajectory as the factual example.
        factual = bool(rng.integers(0, 2))
        lp = torch.mean((p1-y1)**2) if factual else torch.mean((p0-y0)**2)
        lint = .5*(torch.mean((p0-y0)**2)+torch.mean((p1-y1)**2))
        ldte = torch.mean(((p1-p0)-(y1-y0))**2)
        if objective == "prediction": loss = lp
        elif objective == "trajectory": loss = lp + lint
        elif objective == "dte": loss = lp + ldte
        elif objective == "joint": loss = lp + lint + ldte
        else: raise ValueError(objective)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % 50 == 0: history.append((step, float(lp), float(lint), float(ldte)))
    return net, history


def train_controlled(objective, seed, cfg, train_systems):
    """Equal-supervision comparison of trajectory and relational DTE losses.

    Every objective observes both labelled intervention trajectories.  This removes
    the extra-label confound in ``train_one``.  ``gated_dte`` first learns an
    interventional forecaster, then approximately minimizes DTE subject to keeping
    the trajectory loss within 5% of its warm-up moving average (a primal-dual
    relaxation of the prediction gate).
    """
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    rng = np.random.default_rng(3000 + seed)
    net = ActionTransformer(cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    history, ema, threshold, dual = [], None, None, 1.0
    warmup = cfg.steps // 2
    for step in range(cfg.steps):
        h = int(rng.choice([4, 8, 16]))
        x, a, f0, f1, y0, y1 = batch_for(train_systems, cfg, rng, h, "train")
        p0, p1 = net(x, a, f0), net(x, a, f1)
        lint = .5 * (torch.mean((p0-y0)**2) + torch.mean((p1-y1)**2))
        ldte = torch.mean(((p1-p0)-(y1-y0))**2)

        if objective == "forecast_equal":
            loss = lint
        elif objective == "forecast_dte_equal":
            # Both terms use the same labelled pair; DTE adds relational structure,
            # not additional observations.
            loss = lint + ldte
        elif objective == "gated_dte":
            if step < warmup:
                loss = lint
                value = float(lint.detach())
                ema = value if ema is None else .95 * ema + .05 * value
                if step == warmup - 1:
                    threshold = 1.05 * ema
            else:
                # Lagrangian for min L_DTE s.t. L_trajectory <= threshold.
                # Dividing by the fixed gate makes the multiplier dimensionless.
                violation = lint / threshold - 1.0
                loss = ldte + dual * violation
                dual = max(0.0, dual + .05 * float(violation.detach()))
        else:
            raise ValueError(objective)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % 50 == 0:
            history.append((step, float(lint.detach()), float(ldte.detach()), float(dual), threshold))
    return net, history


@torch.no_grad()
def autoregressive_predict(net, past_x, past_a, future_a):
    """Free rollout using only the model's one-step prediction at each step."""
    x, a = past_x.clone(), past_a.clone()
    predictions = []
    for t in range(future_a.shape[1]):
        current_a = future_a[:, t:t+1]
        nxt = net(x, a, current_a)[:, 0]
        predictions.append(nxt)
        x = torch.cat([x[:, 1:], nxt[:, None]], dim=1)
        # The final history action is a placeholder: actions at positions 0..C-2
        # are the transitions between the C observed states.
        a = torch.cat([a[:, 1:-1], current_a, torch.zeros_like(current_a)], dim=1)
    return torch.stack(predictions, dim=1)


@torch.no_grad()
def evaluate_autoregressive(net, systems, cfg, seed, n=48):
    rng, rows = np.random.default_rng(12000 + seed), []
    for sid, m in enumerate(systems):
        for h in [4, 8, 16, 32]:
            x, a = observational_history(m, n, cfg.context, rng)
            f0, f1 = policies(n, h, rng, "test"); lag = a[:, -2]
            y0 = rollout(m, x[:, -1], lag, f0)[:, 1:]
            y1 = rollout(m, x[:, -1], lag, f1)[:, 1:]
            tx, ta, tf0, tf1 = map(lambda z: torch.tensor(z, dtype=torch.float32), [x,a,f0,f1])
            p0 = autoregressive_predict(net, tx, ta, tf0).numpy()
            p1 = autoregressive_predict(net, tx, ta, tf1).numpy()
            rows.append({"system":sid, "horizon":h,
                "trajectory_error":.5*(np.mean((p0-y0)**2)+np.mean((p1-y1)**2)),
                "dte_error":np.mean(((p1-p0)-(y1-y0))**2)})
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate_ood_dynamics(net, systems, cfg, seed, n=256):
    """Independent controlled-transition error on valid policy-shift histories."""
    rng, rows = np.random.default_rng(15000 + seed), []
    probe_policies = ["balanced_random", "alternating", "long_blocks", "inverse_state"]
    for sid, m in enumerate(systems):
        family = "nonlinear" if sid % 2 else "linear"
        for policy in probe_policies:
            x, a = ood_history(m, n, cfg.context, rng, policy)
            lag = a[:, -2]
            for action in [0., 1.]:
                fa = np.full((n, 1), action, np.float32)
                target = m.step(x[:, -1], fa[:, 0], lag)
                pred = net(torch.tensor(x), torch.tensor(a), torch.tensor(fa)).numpy()[:, 0]
                rows.append({"system":sid, "family":family, "policy":policy,
                             "action":int(action), "ed_ood":np.mean((pred-target)**2),
                             "state_min":float(x[:,-1].min()), "state_max":float(x[:,-1].max())})
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate(net, systems, cfg, seed, n=48):
    rng = np.random.default_rng(9000+seed)
    rows = []
    for sid, m in enumerate(systems):
        for h in [1, 4, 8, 16, 32]:
            x, a = observational_history(m, n, cfg.context, rng)
            f0, f1 = policies(n, h, rng, "test")
            lag = a[:, -2]
            y0, y1 = rollout(m, x[:, -1], lag, f0)[:, 1:], rollout(m, x[:, -1], lag, f1)[:, 1:]
            tx, ta, tf0, tf1 = map(lambda z: torch.tensor(z, dtype=torch.float32), [x,a,f0,f1])
            p0, p1 = net(tx,ta,tf0).numpy(), net(tx,ta,tf1).numpy()
            fobs,yobs=observational_future(m,x[:,-1],lag,h,rng)
            pobs=net(tx,ta,torch.tensor(fobs)).numpy()
            rows.append({"system": sid, "horizon": h,
                         "factual_error": np.mean((pobs-yobs)**2),
                         "trajectory_error": .5*(np.mean((p0-y0)**2)+np.mean((p1-y1)**2)),
                         "dte_error": np.mean(((p1-p0)-(y1-y0))**2)})
        # Independent one-step controlled-transition probe over both current actions.
        x, a = observational_history(m, n, cfg.context, rng); lag = a[:, -2]
        errs=[]
        for action in [0.,1.]:
            fa=np.full((n,1),action,np.float32); target=m.step(x[:,-1],fa[:,0],lag)
            pred=net(torch.tensor(x),torch.tensor(a),torch.tensor(fa)).numpy()[:,0]
            errs.append(np.mean((pred-target)**2))
        rows.append({"system":sid,"horizon":0,"factual_error":np.nan,"trajectory_error":np.mean(errs),"dte_error":np.nan})
    return pd.DataFrame(rows)


def run(output_dir="results/transformer", figure_dir="figures/transformer", seeds=(0,1,2)):
    out, figs = Path(output_dir), Path(figure_dir); out.mkdir(parents=True,exist_ok=True); figs.mkdir(parents=True,exist_ok=True)
    cfg=Config(); train_sys=make_systems(cfg.train_systems, 123); test_sys=make_systems(cfg.test_systems, 987)
    all_results=[]; logs={}
    for objective in ["prediction","trajectory","dte","joint"]:
        for seed in seeds:
            net,hist=train_one(objective,seed,cfg,train_sys)
            result=evaluate(net,test_sys,cfg,seed); result["objective"]=objective; result["seed"]=seed
            all_results.append(result); logs[f"{objective}_{seed}"]=hist
            torch.save(net.state_dict(),out/f"{objective}_seed{seed}.pt")
    results=pd.concat(all_results,ignore_index=True); results.to_csv(out/"metrics.csv",index=False)
    summary=results.groupby(["objective","horizon"]).agg(
        factual_mean=("factual_error","mean"),factual_sem=("factual_error","sem"),
        trajectory_mean=("trajectory_error","mean"),trajectory_sem=("trajectory_error","sem"),
        dte_mean=("dte_error","mean"),dte_sem=("dte_error","sem")).reset_index()
    summary.to_csv(out/"summary.csv",index=False)
    (out/"config.json").write_text(json.dumps(cfg.__dict__,indent=2),encoding="utf-8")
    (out/"training_logs.json").write_text(json.dumps(logs),encoding="utf-8")
    plot(summary,figs/"training_objectives.png")
    return results,summary


def reevaluate_saved(output_dir="results/transformer", figure_dir="figures/transformer", seeds=(0,1,2)):
    out=Path(output_dir); figs=Path(figure_dir); cfg=Config(); test_sys=make_systems(cfg.test_systems,987)
    parts=[]
    for objective in ["prediction","trajectory","dte","joint"]:
        for seed in seeds:
            net=ActionTransformer(cfg); net.load_state_dict(torch.load(out/f"{objective}_seed{seed}.pt",weights_only=True)); net.eval()
            z=evaluate(net,test_sys,cfg,seed); z["objective"]=objective; z["seed"]=seed; parts.append(z)
    results=pd.concat(parts,ignore_index=True); results.to_csv(out/"metrics.csv",index=False)
    summary=results.groupby(["objective","horizon"]).agg(
        factual_mean=("factual_error","mean"),factual_sem=("factual_error","sem"),
        trajectory_mean=("trajectory_error","mean"),trajectory_sem=("trajectory_error","sem"),
        dte_mean=("dte_error","mean"),dte_sem=("dte_error","sem")).reset_index()
    summary.to_csv(out/"summary.csv",index=False); plot(summary,figs/"training_objectives.png")
    return results,summary


def run_controlled(output_dir="results/transformer_controlled",
                   figure_dir="figures/transformer_controlled", seeds=(0,1,2,3,4)):
    """Run the equal-label and approximate prediction-gated training ablation."""
    out, figs = Path(output_dir), Path(figure_dir)
    out.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    cfg = Config(); train_sys = make_systems(cfg.train_systems, 123); test_sys = make_systems(cfg.test_systems, 987)
    objectives = ["forecast_equal", "forecast_dte_equal", "gated_dte"]
    parts, logs = [], {}
    for objective in objectives:
        for seed in seeds:
            net, hist = train_controlled(objective, seed, cfg, train_sys)
            z = evaluate(net, test_sys, cfg, seed); z["objective"] = objective; z["seed"] = seed
            parts.append(z); logs[f"{objective}_{seed}"] = hist
            torch.save(net.state_dict(), out/f"{objective}_seed{seed}.pt")
    results = pd.concat(parts, ignore_index=True); results.to_csv(out/"metrics.csv", index=False)
    summary = results.groupby(["objective", "horizon"]).agg(
        factual_mean=("factual_error","mean"), factual_sem=("factual_error","sem"),
        trajectory_mean=("trajectory_error","mean"), trajectory_sem=("trajectory_error","sem"),
        dte_mean=("dte_error","mean"), dte_sem=("dte_error","sem")).reset_index()
    summary.to_csv(out/"summary.csv", index=False)
    (out/"config.json").write_text(json.dumps(cfg.__dict__ | {"seeds": list(seeds)}, indent=2), encoding="utf-8")
    (out/"training_logs.json").write_text(json.dumps(logs), encoding="utf-8")
    plot_controlled(summary, figs/"equal_supervision.png")
    return results, summary


def evaluate_controlled_rollouts(output_dir="results/transformer_controlled", seeds=(0,1,2,3,4)):
    out = Path(output_dir); cfg = Config(); test_sys = make_systems(cfg.test_systems, 987)
    parts = []
    for objective in ["forecast_equal", "forecast_dte_equal", "gated_dte"]:
        for seed in seeds:
            net = ActionTransformer(cfg)
            net.load_state_dict(torch.load(out/f"{objective}_seed{seed}.pt", weights_only=True)); net.eval()
            z = evaluate_autoregressive(net, test_sys, cfg, seed)
            z["objective"], z["seed"] = objective, seed; parts.append(z)
    result = pd.concat(parts, ignore_index=True); result.to_csv(out/"autoregressive_metrics.csv", index=False)
    summary = result.groupby(["objective","horizon"]).agg(
        trajectory_mean=("trajectory_error","mean"), trajectory_sem=("trajectory_error","sem"),
        dte_mean=("dte_error","mean"), dte_sem=("dte_error","sem")).reset_index()
    summary.to_csv(out/"autoregressive_summary.csv", index=False)
    return result, summary


def evaluate_controlled_ood(output_dir="results/transformer_controlled", seeds=(0,1,2,3,4)):
    out = Path(output_dir); cfg = Config(); test_sys = make_systems(cfg.test_systems, 987)
    parts = []
    for objective in ["forecast_equal", "forecast_dte_equal", "gated_dte"]:
        for seed in seeds:
            net = ActionTransformer(cfg)
            net.load_state_dict(torch.load(out/f"{objective}_seed{seed}.pt", weights_only=True)); net.eval()
            z = evaluate_ood_dynamics(net, test_sys, cfg, seed)
            z["objective"], z["seed"] = objective, seed; parts.append(z)
    result = pd.concat(parts, ignore_index=True); result.to_csv(out/"ood_dynamics_metrics.csv", index=False)
    summary = result.groupby(["objective","family","policy"]).agg(
        ed_ood_mean=("ed_ood","mean"), ed_ood_sem=("ed_ood","sem")).reset_index()
    summary.to_csv(out/"ood_dynamics_summary.csv", index=False)
    return result, summary


def plot_controlled(s, path):
    labels = {"forecast_equal":"Forecast", "forecast_dte_equal":"Forecast + DTE", "gated_dte":"Prediction-gated DTE"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for obj in labels:
        q = s[(s.objective == obj) & (s.horizon > 0)]
        axes[0].plot(q.horizon, q.factual_mean, marker="o", label=labels[obj])
        axes[1].plot(q.horizon, q.trajectory_mean, marker="o", label=labels[obj])
        axes[2].plot(q.horizon, q.dte_mean, marker="o", label=labels[obj])
    for ax, title, ylabel in zip(axes,
            ["Observational forecasting", "Unseen-policy trajectory", "Unseen-policy treatment effect"],
            ["Factual error", "Trajectory error", "DTE error"]):
        ax.set(xlabel="Horizon", ylabel=ylabel, yscale="log", title=title); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def plot(s,path):
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for obj in ["prediction","trajectory","dte","joint"]:
        q=s[(s.objective==obj)&(s.horizon>0)]
        axes[0].plot(q.horizon,q.trajectory_mean,marker="o",label=obj)
        axes[1].plot(q.horizon,q.dte_mean,marker="o",label=obj)
    probe=s[s.horizon==0].sort_values("trajectory_mean")
    axes[2].bar(probe.objective,probe.trajectory_mean)
    axes[0].set(xlabel="Held-out rollout horizon",ylabel="Trajectory error",yscale="log",title="Unseen-policy trajectory")
    axes[1].set(xlabel="Held-out rollout horizon",ylabel="DTE error",yscale="log",title="Unseen-policy effect")
    axes[2].set(ylabel="One-step transition probe error",title="Independent mechanism probe")
    axes[2].tick_params(axis="x",rotation=20); axes[0].legend(); axes[1].legend(); fig.tight_layout(); fig.savefig(path,dpi=220,bbox_inches="tight"); plt.close(fig)
