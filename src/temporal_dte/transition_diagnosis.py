from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .functional_use_experiment import generate_policy_batch
from .medical_latent import Config, LatentWorldModel, PatientSystem, lagged_action_tokens


def schedule(n,h,kind):
    a=np.zeros((n,h),np.float32)
    if kind=="pulse":a[:,0]=1
    elif kind=="block":a[:,:4]=1
    elif kind=="alternating":a[:,::2]=1
    else:raise ValueError(kind)
    return a


def extend_true_history(x,c,d,u,a,future):
    xs=[x];cs=[c];ds=[d];us=[u];acts=[a]
    cx,cc,cd,cu=x[:,-1],c[:,-1],d[:,-1],u[:,-1]
    lag=a[:,-2]
    for k in range(future.shape[1]):
        cx,cc,cd,cu=PatientSystem.step(cx,cc,cd,cu,future[:,k],lag)
        xs.append(cx[:,None]);cs.append(cc[:,None]);ds.append(cd[:,None]);us.append(cu[:,None])
        acts.append(future[:,k:k+1]);lag=future[:,k]
    return tuple(np.concatenate(v,1) for v in (xs,cs,ds,us,acts))


@torch.no_grad()
def encode_at(net,x,c,u,a,end,context):
    action_tokens=lagged_action_tokens(a[:,:end])[:,-context:]
    return net.encode(torch.tensor(x[:,:end][:,-context:]),torch.tensor(c[:,:end][:,-context:]),
                      torch.tensor(u[:,:end][:,-context:]),torch.tensor(action_tokens))


@torch.no_grad()
def effect_from_z(net,z):
    n=len(z);zero=torch.zeros(n,1);one=torch.ones(n,1)
    p0=net.rollout(z,zero)[0][:,0,0]
    p1=net.rollout(z,one)[0][:,0,0]
    return (p1-p0).numpy()


@torch.no_grad()
def diagnose(net,cfg,seed,n=512,max_h=16):
    rng=np.random.default_rng(61000+seed)
    x,c,d,u,a=generate_policy_batch(n,cfg.context,rng,cfg.noise,"observational")
    z0=net.encode(torch.tensor(x),torch.tensor(c),torch.tensor(u),
                  torch.tensor(lagged_action_tokens(a)))
    rows=[]
    for kind in ("pulse","block","alternating"):
        future=schedule(n,max_h,kind)
        xx,cc,dd,uu,aa=extend_true_history(x,c,d,u,a,future)
        zroll=z0.clone()
        for h in range(max_h+1):
            if h in (0,1,2,4,8,16):
                zenc=encode_at(net,xx,cc,uu,aa,cfg.context+h,cfg.context)
                eroll=effect_from_z(net,zroll);eenc=effect_from_z(net,zenc)
                truth=-(.24+.10*np.tanh(cc[:,cfg.context-1+h]))
                rows.append({"seed":seed,"policy":kind,"horizon":h,
                    "encoded_effect_mse":np.mean((eenc-truth)**2),
                    "rolled_effect_mse":np.mean((eroll-truth)**2),
                    "closure_mse":np.mean((eroll-eenc)**2),
                    "amplification":np.mean((eroll-truth)**2)/(np.mean((eenc-truth)**2)+1e-12)})
            if h<max_h:zroll=net.advance(zroll,torch.tensor(future[:,h]))
    return pd.DataFrame(rows)


def plot(df,outdir):
    outdir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(16,4.6))
    families=("factual_z8","contrast1_z8","contrast10_z8","contrast100_z8")
    colors=plt.cm.viridis(np.linspace(.05,.9,len(families)))
    for family,color in zip(families,colors):
        z=df[(df.family==family)&(df.policy=="alternating")].groupby("horizon").mean(numeric_only=True)
        axes[0].plot(z.index,z.encoded_effect_mse,"o--",color=color,alpha=.7,label=family+" encoded")
        axes[0].plot(z.index,z.rolled_effect_mse,"o-",color=color,label=family+" rolled")
        axes[1].plot(z.index,z.closure_mse,"o-",color=color,label=family)
        axes[2].plot(z.index,z.amplification,"o-",color=color,label=family)
    axes[0].set(xlabel="Steps before local response test",ylabel="One-step effect MSE",yscale="log",
                title="Re-encoded vs rolled state")
    axes[1].set(xlabel="Steps",ylabel="Response closure MSE",yscale="log",title="Transition closure failure")
    axes[2].set(xlabel="Steps",ylabel="Rolled / encoded effect MSE",yscale="log",title="Error amplification")
    axes[0].legend(fontsize=6,ncol=2);axes[1].legend(fontsize=7);axes[2].legend(fontsize=7)
    fig.tight_layout();fig.savefig(outdir/"transition_diagnosis.png",dpi=220,bbox_inches="tight");plt.close(fig)


def run(model_dir="results/latent_test",output_dir="results/transition_diagnosis",
        figure_dir="figures/transition_diagnosis"):
    cfg=Config();model_dir=Path(model_dir);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(model_dir/"model_metrics.csv");parts=[]
    for row in meta.itertuples():
        net=LatentWorldModel(cfg,row.latent_dim)
        net.load_state_dict(torch.load(model_dir/f"{row.model_id}.pt",map_location="cpu",weights_only=True));net.eval()
        d=diagnose(net,cfg,row.seed);d["model_id"]=row.model_id;d["family"]=row.family
        d["latent_dim"]=row.latent_dim;d["initial_ite_mse"]=row.heldout_tail_ite_mse
        d["prior_h16_effect_mse"]=row.heldout_h16_effect_mse;parts.append(d)
    df=pd.concat(parts,ignore_index=True);df.to_csv(out/"closure_metrics.csv",index=False)
    summary=df.groupby(["family","policy","horizon"]).mean(numeric_only=True).reset_index()
    summary.to_csv(out/"family_summary.csv",index=False)
    h16=df[(df.horizon==16)&(df.policy=="alternating")]
    findings={
      "median_amplification_h16":float(h16.amplification.median()),
      "fraction_amplification_gt_2":float((h16.amplification>2).mean()),
      "fraction_amplification_gt_5":float((h16.amplification>5).mean())}
    (out/"findings.json").write_text(json.dumps(findings,indent=2),encoding="utf-8")
    plot(df,Path(figure_dir));return df,summary
