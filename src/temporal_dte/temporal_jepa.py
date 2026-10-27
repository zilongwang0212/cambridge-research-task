from __future__ import annotations

import copy, json, random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .compositional_latent import evaluate
from .functional_use_experiment import generate_policy_batch
from .medical_latent import Config, LatentWorldModel, lagged_action_tokens


HORIZONS = (1, 2, 4, 8)


def _ema(target, online, momentum=.99):
    with torch.no_grad():
        for pt, po in zip(target.parameters(), online.parameters()):
            pt.mul_(momentum).add_(po, alpha=1-momentum)


def _observational_batch(cfg, rng):
    return generate_policy_batch(cfg.batch, cfg.context+cfg.train_horizon,
                                 rng, cfg.noise, "observational")


def train_model(kind, seed, cfg, jepa_weight=1., latent_dim=8):
    """No counterfactual target or simulator state is used by any objective."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    rng=np.random.default_rng(121000+1009*seed)
    net=LatentWorldModel(cfg, latent_dim)
    target=copy.deepcopy(net).eval()
    for p in target.parameters(): p.requires_grad_(False)
    opt=torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    for _ in range(cfg.steps):
        x,c,d,u,a=_observational_batch(cfg,rng)
        tx,tc,tu,ta=[torch.tensor(v) for v in (x,c,u,a)]
        all_lag=lagged_action_tokens(ta)
        z0=net.encode(tx[:,:cfg.context],tc[:,:cfg.context],tu[:,:cfg.context],all_lag[:,:cfg.context])
        actions=ta[:,cfg.context-1:cfg.context-1+cfg.train_horizon]
        pred,z=net.rollout(z0,actions)
        outcome=torch.stack([tx[:,cfg.context:],tc[:,cfg.context:]],-1)
        if kind=="one_step":
            pred_loss=torch.mean((pred[:,:1]-outcome[:,:1])**2)
        else:
            weights=torch.linspace(1,1.5,cfg.train_horizon)[None,:,None]
            pred_loss=torch.mean(weights*(pred-outcome)**2)
        loss=pred_loss
        if kind=="jepa":
            rolled=z0; terms=[]
            for h in range(1,cfg.train_horizon+1):
                rolled=net.advance(rolled,actions[:,h-1])
                if h in HORIZONS:
                    with torch.no_grad():
                        zstar=target.encode(tx[:,h:cfg.context+h],tc[:,h:cfg.context+h],
                                            tu[:,h:cfg.context+h],all_lag[:,h:cfg.context+h])
                    terms.append(torch.mean((F.normalize(rolled,dim=1)-F.normalize(zstar,dim=1))**2))
            loss=pred_loss+jepa_weight*torch.stack(terms).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
        if kind=="jepa": _ema(target,net)
    return net.eval()


@torch.no_grad()
def observed_diagnostics(net,cfg,seed,n=768,h=16):
    rng=np.random.default_rng(131000+seed)
    x,c,d,u,a=generate_policy_batch(n,cfg.context+h,rng,cfg.noise,"observational")
    tx,tc,tu,ta=[torch.tensor(v) for v in (x,c,u,a)]
    lag=lagged_action_tokens(ta)
    z0=net.encode(tx[:,:cfg.context],tc[:,:cfg.context],tu[:,:cfg.context],lag[:,:cfg.context])
    actions=ta[:,cfg.context-1:cfg.context-1+h]
    pred,_=net.rollout(z0,actions)
    target=torch.stack([tx[:,cfg.context:cfg.context+h],tc[:,cfg.context:cfg.context+h]],-1)
    rows=[]; z=z0
    for step in range(1,h+1):
        z=net.advance(z,actions[:,step-1])
        if step in (1,2,4,8,16):
            zenc=net.encode(tx[:,step:cfg.context+step],tc[:,step:cfg.context+step],
                            tu[:,step:cfg.context+step],lag[:,step:cfg.context+step])
            closure=torch.mean((F.normalize(z,dim=1)-F.normalize(zenc,dim=1))**2).item()
            obs_mse=torch.mean((pred[:,:step]-target[:,:step])**2).item()
            rows.append({"seed":seed,"horizon":step,"observed_rollout_mse":obs_mse,
                         "latent_closure_mse":closure})
    shuffled=actions[torch.randperm(n)]
    za=net.rollout(z0,actions)[1]; zs=net.rollout(z0,shuffled)[1]
    sensitivity=torch.mean(torch.linalg.vector_norm(F.normalize(za,dim=1)-F.normalize(zs,dim=1),dim=1)).item()
    for row in rows: row["action_sensitivity"]=sensitivity
    return pd.DataFrame(rows)


def _plot(effect,diag,outdir):
    outdir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for kind,z in diag.groupby("model_kind"):
        q=z.groupby("horizon").latent_closure_mse.mean(); axes[0].plot(q.index,q.values,"o-",label=kind)
        q=z.groupby("horizon").observed_rollout_mse.mean(); axes[1].plot(q.index,q.values,"o-",label=kind)
    axes[0].set(xlabel="Horizon",ylabel="Normalized latent MSE",yscale="log",title="Observed transition closure")
    axes[1].set(xlabel="Horizon",ylabel="Outcome MSE",yscale="log",title="Observed rollout")
    q=effect[effect.horizon==16].groupby(["model_kind","schedule"]).effect_mse.mean().unstack()
    q.plot.bar(ax=axes[2]); axes[2].set(yscale="log",ylabel="Effect trajectory MSE",title="Counterfactual final test")
    axes[2].tick_params(axis="x",rotation=20); axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outdir/"temporal_jepa.png",dpi=220,bbox_inches="tight"); plt.close(fig)


def run(output_dir="results/temporal_jepa",figure_dir="figures/temporal_jepa"):
    cfg=Config(); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    effects=[]; diagnostics=[]; models={}
    specs=[("one_step",0.),("multi_step",0.)]+[("jepa",w) for w in (.1,1.,10.)]
    for kind,weight in specs:
        label=kind if kind!="jepa" else f"jepa_{weight:g}"
        for seed in cfg.seeds:
            net=train_model(kind,seed,cfg,weight)
            d=evaluate(net,cfg,seed); d["model_kind"]=label; effects.append(d)
            q=observed_diagnostics(net,cfg,seed); q["model_kind"]=label; diagnostics.append(q)
            torch.save(net.state_dict(),out/f"{label}_seed{seed}.pt"); models[(label,seed)]=net
    effect=pd.concat(effects,ignore_index=True); diag=pd.concat(diagnostics,ignore_index=True)
    effect.to_csv(out/"effect_metrics.csv",index=False); diag.to_csv(out/"observed_diagnostics.csv",index=False)
    # Select only with observational validation: lowest H=8 closure among JEPA
    # candidates whose H=8 outcome error is within 10% of multi-step prediction.
    h8=diag[diag.horizon==8].groupby("model_kind").mean(numeric_only=True)
    ceiling=1.10*h8.loc["multi_step","observed_rollout_mse"]
    eligible=h8.loc[[k for k in h8.index if k.startswith("jepa_")]]
    eligible=eligible[eligible.observed_rollout_mse<=ceiling]
    selected=(eligible.latent_closure_mse.idxmin() if len(eligible) else
              h8.loc[[k for k in h8.index if k.startswith("jepa_")]].observed_rollout_mse.idxmin())
    selection={"selected_model":selected,"criterion":"minimum H8 observational closure subject to <=10% H8 rollout-MSE degradation vs multi_step",
               "outcome_mse_ceiling":float(ceiling),"counterfactual_metrics_used_for_selection":False}
    (out/"selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
    keep=["one_step","multi_step",selected]
    ef=effect[effect.model_kind.isin(keep)]; dg=diag[diag.model_kind.isin(keep)]
    summary=ef.groupby(["model_kind","schedule","horizon"]).mean(numeric_only=True).reset_index()
    summary.to_csv(out/"selected_summary.csv",index=False)
    h16=ef[ef.horizon==16]
    findings={"selection":selection,"h16_effect_mse":h16.groupby(["model_kind","schedule"]).effect_mse.mean().unstack().to_dict(),
              "h16_observed":dg[dg.horizon==16].groupby("model_kind")[["observed_rollout_mse","latent_closure_mse","action_sensitivity"]].mean().to_dict("index")}
    (out/"findings.json").write_text(json.dumps(findings,indent=2),encoding="utf-8")
    _plot(ef,dg,Path(figure_dir)); return ef,dg,selection
