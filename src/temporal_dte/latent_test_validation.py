from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .functional_use_experiment import generate_policy_batch, train_condition
from .medical_latent import (Config, LatentWorldModel, collect_latents, fixed_actions,
                             lagged_action_tokens, probe_latent, train_one, true_rollout)


@torch.no_grad()
def model_contrast(net, x, c, u, a):
    z=net.encode(torch.tensor(x),torch.tensor(c),torch.tensor(u),
                 torch.tensor(lagged_action_tokens(a)))
    n=len(x)
    p0=net.rollout(z,torch.zeros(n,1))[0][:,0,0]
    p1=net.rollout(z,torch.ones(n,1))[0][:,0,0]
    return (p1-p0).numpy(),z


@torch.no_grad()
def audit_and_validate(net,cfg,seed,n=3500,delta=.15):
    rng=np.random.default_rng(52000+seed)
    x,c,d,u,a=generate_policy_batch(n,cfg.context+1,rng,cfg.noise,"observational")
    hx=x[:,:cfg.context];hc=c[:,:cfg.context];hu=u[:,:cfg.context];ha=a[:,:cfg.context]
    z=net.encode(torch.tensor(hx),torch.tensor(hc),torch.tensor(hu),
                 torch.tensor(lagged_action_tokens(ha)))
    factual=net.rollout(z,torch.tensor(a[:,cfg.context-1:cfg.context]))[0][:,0,0].numpy()
    pred_mse=float(np.mean((factual-x[:,cfg.context])**2))

    tau,z=model_contrast(net,hx,hc,hu,ha)
    true_tau=-(.24+.10*np.tanh(hc[:,-1]))

    c_now=hc[:,-1]
    q15,q30,q70,q85=np.quantile(c_now,[.15,.30,.70,.85])
    central=(c_now>=q30)&(c_now<=q70)
    cp=hc[central].copy();cp[:,-1]+=delta
    tau_plus,_=model_contrast(net,hx[central],cp,hu[central],ha[central])
    slope=(tau_plus-tau[central])/delta
    slope_truth=-.10/(np.cosh(hc[central,-1])**2)
    sensitivity_mae=float(np.mean(np.abs(slope-slope_truth)))

    tails=(c_now<=q15)|(c_now>=q85)
    tail_ite_mse=float(np.mean((tau[tails]-true_tau[tails])**2))
    central_ite_mse=float(np.mean((tau[central]-true_tau[central])**2))

    # Secondary validation: an unseen alternating schedule at H=16.
    take=np.flatnonzero(tails)[:768]
    a0=fixed_actions(len(take),16,"none");a1=fixed_actions(len(take),16,"alternating")
    y0=true_rollout(hx[take,-1],hc[take,-1],d[take,cfg.context-1],hu[take,-1],
                    a[take,cfg.context-2],a0)
    y1=true_rollout(hx[take,-1],hc[take,-1],d[take,cfg.context-1],hu[take,-1],
                    a[take,cfg.context-2],a1)
    zt=z[take];p0=net.rollout(zt,torch.tensor(a0))[0][:,:,0].numpy()
    p1=net.rollout(zt,torch.tensor(a1))[0][:,:,0].numpy()
    h16_effect_mse=float(np.mean(((p1-p0)-(y1-y0))**2))
    return {"prediction_mse":pred_mse,"sensitivity_mae":sensitivity_mae,
            "central_ite_mse":central_ite_mse,"heldout_tail_ite_mse":tail_ite_mse,
            "heldout_h16_effect_mse":h16_effect_mse,"n_central":int(central.sum()),
            "n_tail":int(tails.sum())}


def selection_table(models):
    rows=[]
    for gate_q in (.2,.4,.6,1.0):
        cutoff=models.prediction_mse.quantile(gate_q)
        eligible=models[models.prediction_mse<=cutoff]
        methods={
            "prediction only":eligible.sort_values("prediction_mse"),
            "prediction + probe":eligible.sort_values("probe_c_r2",ascending=False),
        }
        probe_ok=eligible[eligible.probe_c_r2>=.8]
        if len(probe_ok)==0:probe_ok=eligible
        methods["prediction + probe + use"]=probe_ok.sort_values("sensitivity_mae")
        for k in (1,3,5):
            for method,ranked in methods.items():
                chosen=ranked.head(min(k,len(ranked)))
                rows.append({"gate_quantile":gate_q,"top_k":k,"method":method,
                             "n_eligible":len(eligible),"n_selected":len(chosen),
                             "heldout_tail_ite_mse":chosen.heldout_tail_ite_mse.mean(),
                             "heldout_h16_effect_mse":chosen.heldout_h16_effect_mse.mean(),
                             "selected_models":";".join(chosen.model_id)})
    return pd.DataFrame(rows)


def plot(models,selection,outdir):
    outdir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(16,4.6))
    sc=axes[0].scatter(models.prediction_mse,models.heldout_tail_ite_mse,
                       c=models.probe_c_r2,cmap="viridis",s=55)
    axes[0].set(xlabel="Factual prediction MSE",ylabel="Held-out tail ITE MSE",
                yscale="log",title="Prediction is not intervention fidelity")
    fig.colorbar(sc,ax=axes[0],label="C probe R²")
    axes[1].scatter(models.sensitivity_mae,models.heldout_tail_ite_mse,
                    c=models.probe_c_r2,cmap="viridis",s=55)
    axes[1].set(xlabel="Functional-use sensitivity MAE",ylabel="Held-out tail ITE MSE",
                xscale="log",yscale="log",title="Does the audit predict held-out ITE?")
    q=selection[(selection.top_k==3)&(selection.gate_quantile<1)]
    for method,z in q.groupby("method"):
        axes[2].plot(z.gate_quantile,z.heldout_tail_ite_mse,"o-",label=method)
    axes[2].set(xlabel="Prediction gate quantile",ylabel="Top-3 held-out ITE MSE",
                yscale="log",title="Model selection comparison");axes[2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(outdir/"latent_test_validation.png",dpi=220,bbox_inches="tight");plt.close(fig)


def run(output_dir="results/latent_test",figure_dir="figures/latent_test"):
    cfg=Config();out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    specs=[]
    for dim in cfg.latent_dims:
        for seed in cfg.seeds:specs.append((f"factual_z{dim}",dim,seed,"factual",0.))
    for seed in cfg.seeds:specs.append(("balanced_z8",8,seed,"balanced",0.))
    for weight in (1.,10.,100.):
        for seed in cfg.seeds:specs.append((f"contrast{weight:g}_z8",8,seed,"contrast",weight))

    for family,dim,seed,kind,weight in specs:
        if kind=="factual":net,_=train_one(dim,seed,cfg)
        elif kind=="balanced":net=train_condition("balanced_factual",seed,cfg,latent_dim=dim)
        else:net=train_condition("observational_contrast",seed,cfg,latent_dim=dim,effect_weight=weight)
        z,targets,states=collect_latents(net,cfg,seed,n=2500)
        probes=probe_latent(z,targets,states,cfg)
        metrics=audit_and_validate(net,cfg,seed)
        model_id=f"{family}_s{seed}"
        rows.append({"model_id":model_id,"family":family,"latent_dim":dim,"seed":seed,
                     "objective":kind,"effect_weight":weight,"probe_c_r2":probes["probe_c_r2"],
                     **metrics})
        torch.save(net.state_dict(),out/f"{model_id}.pt")
    models=pd.DataFrame(rows);selection=selection_table(models)
    models.to_csv(out/"model_metrics.csv",index=False);selection.to_csv(out/"selection_results.csv",index=False)
    summary=models.groupby("family").mean(numeric_only=True);summary.to_csv(out/"family_summary.csv")
    (out/"design.json").write_text(json.dumps({"latent_dims":cfg.latent_dims,"seeds":cfg.seeds,
      "primary_endpoint":"heldout_tail_ite_mse","audit_C_range":"30th--70th percentiles",
      "validation_C_range":"bottom/top 15 percent","probe_threshold":.8},indent=2),encoding="utf-8")
    plot(models,selection,Path(figure_dir));return models,selection
