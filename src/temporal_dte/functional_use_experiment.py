from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json, random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .medical_latent import (Config, LatentWorldModel, PatientSystem, fixed_actions,
                             lagged_action_tokens, true_rollout)


def generate_policy_batch(n, length, rng, noise, policy):
    x=np.zeros((n,length),np.float32); c=np.zeros_like(x); d=np.zeros_like(x)
    u=np.zeros_like(x); a=np.zeros_like(x)
    x[:,0]=rng.normal(0,.75,n); c[:,0]=rng.normal(0,.65,n); u[:,0]=rng.normal(0,1,n)
    lag=rng.binomial(1,.4,n).astype(np.float32)
    for t in range(length-1):
        if policy == "observational":
            p=PatientSystem.propensity(x[:,t],c[:,t],lag,d[:,t])
        elif policy == "balanced":
            p=np.full(n,.5)
        else: raise ValueError(policy)
        action=rng.binomial(1,p).astype(np.float32); a[:,t]=action
        eps=[rng.normal(0,noise,n).astype(np.float32) for _ in range(3)]
        x[:,t+1],c[:,t+1],d[:,t+1],u[:,t+1]=PatientSystem.step(
            x[:,t],c[:,t],d[:,t],u[:,t],action,lag,*eps)
        lag=action
    return x,c,d,u,a


def train_condition(condition, seed, cfg, latent_dim=8, effect_weight=1.0):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    rng=np.random.default_rng(31000+101*seed)
    net=LatentWorldModel(cfg,latent_dim)
    opt=torch.optim.AdamW(net.parameters(),lr=cfg.lr,weight_decay=1e-4)
    policy = "balanced" if condition == "balanced_factual" else "observational"
    for _ in range(cfg.steps):
        x,c,d,u,a=generate_policy_batch(cfg.batch,cfg.context+cfg.train_horizon,rng,cfg.noise,policy)
        tx,tc,tu,ta=[torch.tensor(v) for v in (x,c,u,a)]
        z=net.encode(tx[:,:cfg.context],tc[:,:cfg.context],tu[:,:cfg.context],
                     lagged_action_tokens(ta[:,:cfg.context]))
        actions=ta[:,cfg.context-1:cfg.context-1+cfg.train_horizon]
        pred,_=net.rollout(z,actions)
        target=torch.stack([tx[:,cfg.context:cfg.context+cfg.train_horizon],
                            tc[:,cfg.context:cfg.context+cfg.train_horizon]],-1)
        weights=torch.linspace(1,1.5,cfg.train_horizon)[None,:,None]
        pred_loss=torch.mean(weights*(pred-target)**2)
        loss=pred_loss
        if condition == "observational_contrast":
            zeros=torch.zeros(cfg.batch,1); ones=torch.ones(cfg.batch,1)
            p0,_=net.rollout(z,zeros); p1,_=net.rollout(z,ones)
            true_tau=-(.24+.10*torch.tanh(tc[:,cfg.context-1]))
            effect_loss=torch.mean(((p1[:,0,0]-p0[:,0,0])-true_tau)**2)
            loss=pred_loss+effect_weight*effect_loss
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step()
    return net.eval()


@torch.no_grad()
def evaluate_functional_use(net, seed, cfg, n=1024, delta=.15):
    rng=np.random.default_rng(41000+seed)
    x,c,d,u,a=generate_policy_batch(n,cfg.context+1,rng,cfg.noise,"observational")
    hist=[torch.tensor(v[:,:cfg.context]) for v in (x,c,u)]
    hist.append(torch.tensor(lagged_action_tokens(a[:,:cfg.context])))
    z=net.encode(*hist); action=torch.tensor(a[:,cfg.context-1:cfg.context])
    factual=net.rollout(z,action)[0][:,0,0].numpy()
    factual_mse=float(np.mean((factual-x[:,cfg.context])**2))

    def tau(ch):
        zz=net.encode(torch.tensor(x[:,:cfg.context]),torch.tensor(ch),
                      torch.tensor(u[:,:cfg.context]),
                      torch.tensor(lagged_action_tokens(a[:,:cfg.context])))
        p0=net.rollout(zz,torch.zeros(n,1))[0][:,0,0]
        p1=net.rollout(zz,torch.ones(n,1))[0][:,0,0]
        return (p1-p0).numpy()
    base=tau(c[:,:cfg.context]); cp=c[:,:cfg.context].copy();cp[:,-1]+=delta
    slope=(tau(cp)-base)/delta
    target=-.10/(np.cosh(c[:,cfg.context-1])**2)
    one_step_tau=-(.24+.10*np.tanh(c[:,cfg.context-1]))
    one_step_effect_mse=float(np.mean((base-one_step_tau)**2))

    # Does fixing local effect use carry through a standard long rollout?
    hx,hc,hd,hu=x[:,cfg.context-1],c[:,cfg.context-1],d[:,cfg.context-1],u[:,cfg.context-1]
    lag=a[:,cfg.context-2]; h=16
    a0=fixed_actions(n,h,"none");a1=fixed_actions(n,h,"alternating")
    y0=true_rollout(hx,hc,hd,hu,lag,a0);y1=true_rollout(hx,hc,hd,hu,lag,a1)
    p0=net.rollout(z,torch.tensor(a0))[0][:,:,0].numpy()
    p1=net.rollout(z,torch.tensor(a1))[0][:,:,0].numpy()
    return {
        "factual_mse":factual_mse,
        "one_step_effect_mse":one_step_effect_mse,
        "sensitivity_mean":float(slope.mean()),
        "sensitivity_target":float(target.mean()),
        "sensitivity_mae":float(np.mean(np.abs(slope-target))),
        "h16_effect_mse":float(np.mean(((p1-p0)-(y1-y0))**2)),
    }


def plot_results(df, outdir):
    outdir.mkdir(parents=True,exist_ok=True)
    order=["observational_factual","balanced_factual","observational_contrast"]
    labels=["Obs.\nfactual","Balanced\nfactual","Obs. +\ncontrast"]
    g=df.groupby("condition").mean(numeric_only=True).loc[order]
    se=df.groupby("condition").sem(numeric_only=True).loc[order]
    fig,axes=plt.subplots(1,3,figsize=(14,4.3))
    axes[0].bar(labels,g.sensitivity_mean,yerr=se.sensitivity_mean,capsize=4)
    axes[0].axhline(g.sensitivity_target.mean(),color="black",ls="--",label="truth")
    axes[0].set(ylabel=r"$d\hat{\tau}/dC$",title="Effect-modifier sensitivity");axes[0].legend()
    axes[1].bar(labels,g.factual_mse,yerr=se.factual_mse,capsize=4)
    axes[1].set(ylabel="MSE",yscale="log",title="Factual prediction")
    axes[2].bar(labels,g.one_step_effect_mse,yerr=se.one_step_effect_mse,capsize=4,label="one-step")
    axes[2].plot(range(3),g.h16_effect_mse,"ko--",label="H=16")
    axes[2].set(ylabel="Effect MSE",yscale="log",title="Intervention fidelity");axes[2].legend()
    fig.tight_layout();fig.savefig(outdir/"coverage_objective_ablation.png",dpi=220,bbox_inches="tight");plt.close(fig)


def run(output_dir="results/functional_use",figure_dir="figures/functional_use"):
    cfg=Config();out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    conditions=("observational_factual","balanced_factual","observational_contrast")
    for condition in conditions:
        for seed in cfg.seeds:
            net=train_condition(condition,seed,cfg)
            rows.append({"condition":condition,"seed":seed,**evaluate_functional_use(net,seed,cfg)})
            torch.save(net.state_dict(),out/f"{condition}_seed{seed}.pt")
    df=pd.DataFrame(rows);df.to_csv(out/"metrics.csv",index=False)
    summary=df.groupby("condition").agg(["mean","sem"]);summary.to_csv(out/"summary.csv")
    (out/"config.json").write_text(json.dumps({**asdict(cfg),"latent_dim":8,"effect_weight":1.0},indent=2),encoding="utf-8")
    plot_results(df,Path(figure_dir));return df
