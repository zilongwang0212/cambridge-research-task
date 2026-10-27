from __future__ import annotations

from pathlib import Path
import json, math, random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch import nn

from .functional_use_experiment import generate_policy_batch
from .medical_latent import Config, LatentWorldModel, PatientSystem, lagged_action_tokens


class ResponseTransitionWorldModel(LatentWorldModel):
    """Action response is explicit; source_dim=d gives no specialized subspace."""
    def __init__(self,cfg,latent_dim=8,response_dim=2):
        super().__init__(cfg,latent_dim);self.response_dim=response_dim
        h=cfg.hidden//2
        self.base_transition=nn.Sequential(nn.Linear(latent_dim,h),nn.Tanh(),nn.Linear(h,latent_dim))
        self.response_transition=nn.Sequential(nn.Linear(response_dim,h),nn.Tanh(),nn.Linear(h,latent_dim))
        del self.transition

    def advance(self,z,action):
        source=z if self.response_dim==self.latent_dim else z[:,-self.response_dim:]
        delta=self.base_transition(z)+action[:,None]*self.response_transition(source)
        return z+.25*delta


def block_actions(n,h,rng):
    actions=np.zeros((n,h),np.float32)
    for i in range(n):
        if rng.random()<.15:continue
        length=int(rng.integers(1,min(4,h)+1));start=int(rng.integers(0,h-length+1))
        actions[i,start:start+length]=1
    return actions


def simulate_future(x,c,d,u,lag,actions,rng=None,noise=0.):
    xs=[];cs=[];ds=[];us=[]
    for k in range(actions.shape[1]):
        if rng is None:eps=(0.,0.,0.)
        else:eps=tuple(rng.normal(0,noise,len(x)).astype(np.float32) for _ in range(3))
        x,c,d,u=PatientSystem.step(x,c,d,u,actions[:,k],lag,*eps)
        xs.append(x.copy());cs.append(c.copy());ds.append(d.copy());us.append(u.copy());lag=actions[:,k]
    return tuple(np.stack(v,1) for v in (xs,cs,ds,us))


def training_batch(cfg,rng):
    x,c,d,u,a=generate_policy_batch(cfg.batch,cfg.context,rng,cfg.noise,"observational")
    actions=block_actions(cfg.batch,cfg.train_horizon,rng)
    fut=simulate_future(x[:,-1],c[:,-1],d[:,-1],u[:,-1],a[:,-2],actions,rng,cfg.noise)
    return x,c,d,u,a,actions,fut


def build_model(kind,cfg,latent_dim=8,response_dim=2):
    if kind in ("predictive","unsplit_contrast"):return LatentWorldModel(cfg,latent_dim)
    if kind=="response_all":return ResponseTransitionWorldModel(cfg,latent_dim,latent_dim)
    if kind=="response_split":return ResponseTransitionWorldModel(cfg,latent_dim,response_dim)
    raise ValueError(kind)


def train_model(kind,seed,cfg,contrast_weight=10.,latent_dim=8,response_dim=2):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(71000+seed)
    net=build_model(kind,cfg,latent_dim,response_dim);opt=torch.optim.AdamW(net.parameters(),lr=cfg.lr,weight_decay=1e-4)
    for _ in range(cfg.steps):
        x,c,d,u,a,actions,fut=training_batch(cfg,rng)
        z=net.encode(torch.tensor(x),torch.tensor(c),torch.tensor(u),torch.tensor(lagged_action_tokens(a)))
        pred,_=net.rollout(z,torch.tensor(actions))
        target=torch.tensor(np.stack([fut[0],fut[1]],-1))
        weights=torch.linspace(1,1.5,cfg.train_horizon)[None,:,None]
        pred_loss=torch.mean(weights*(pred-target)**2);loss=pred_loss
        if kind!="predictive":
            n=len(x);p0=net.rollout(z,torch.zeros(n,1))[0][:,0,0]
            p1=net.rollout(z,torch.ones(n,1))[0][:,0,0]
            truth=torch.tensor(-(.24+.10*np.tanh(c[:,-1])))
            loss=loss+contrast_weight*torch.mean(((p1-p0)-truth)**2)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step()
    return net.eval()


def _effect_tensor(net,z):
    n=len(z)
    p0=net.rollout(z,torch.zeros(n,1))[0][:,0,0]
    p1=net.rollout(z,torch.ones(n,1))[0][:,0,0]
    return p1-p0


def train_closure_model(seed,cfg,closure_weight,contrast_weight=10.,latent_dim=8,response_dim=2):
    """Train response-split transition with response-based transition closure.

    Closure targets are generated only from true histories under the contiguous
    block schedules used for training. No unseen schedule enters this loss.
    """
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(101000+seed)
    net=ResponseTransitionWorldModel(cfg,latent_dim,response_dim)
    opt=torch.optim.AdamW(net.parameters(),lr=cfg.lr,weight_decay=1e-4)
    closure_steps=(1,2,4,8)
    for _ in range(cfg.steps):
        x,c,d,u,a,actions,fut=training_batch(cfg,rng)
        z0=net.encode(torch.tensor(x),torch.tensor(c),torch.tensor(u),torch.tensor(lagged_action_tokens(a)))
        pred,_=net.rollout(z0,torch.tensor(actions))
        target=torch.tensor(np.stack([fut[0],fut[1]],-1))
        weights=torch.linspace(1,1.5,cfg.train_horizon)[None,:,None]
        pred_loss=torch.mean(weights*(pred-target)**2)
        truth=torch.tensor(-(.24+.10*np.tanh(c[:,-1])))
        local_loss=torch.mean((_effect_tensor(net,z0)-truth)**2)

        full_x=np.concatenate([x,fut[0]],1);full_c=np.concatenate([c,fut[1]],1)
        full_u=np.concatenate([u,fut[3]],1)
        # Raw action at each state; the final dummy is never exposed as current action.
        raw_a=np.concatenate([a[:,:cfg.context-1],actions,np.zeros((len(x),1),np.float32)],1)
        zroll=z0;closure=[]
        for h in range(1,cfg.train_horizon+1):
            zroll=net.advance(zroll,torch.tensor(actions[:,h-1]))
            if h in closure_steps:
                end=cfg.context+h
                atok=lagged_action_tokens(raw_a[:,:end])[:,-cfg.context:]
                zenc=net.encode(torch.tensor(full_x[:,:end][:,-cfg.context:]),
                                torch.tensor(full_c[:,:end][:,-cfg.context:]),
                                torch.tensor(full_u[:,:end][:,-cfg.context:]),torch.tensor(atok))
                # Stop-gradient makes the re-encoded state a moving teacher; matching
                # occurs in intervention-response space, not arbitrary latent coordinates.
                closure.append(torch.mean((_effect_tensor(net,zroll)-_effect_tensor(net,zenc).detach())**2))
        closure_loss=torch.stack(closure).mean()
        loss=pred_loss+contrast_weight*local_loss+closure_weight*closure_loss
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step()
    return net.eval()


def fixed_schedule(n,h,kind):
    a=np.zeros((n,h),np.float32)
    if kind=="seen_block":a[:,:4]=1
    elif kind=="alternating":a[:,::2]=1
    elif kind=="separated":a[:,[0,4,8,12]]=1
    elif kind=="late_separated":a[:,[3,7,11,15]]=1
    elif kind!="none":raise ValueError(kind)
    return a


@torch.no_grad()
def evaluate(net,cfg,seed,n=768,h=16):
    rng=np.random.default_rng(81000+seed)
    x,c,d,u,a=generate_policy_batch(n,cfg.context+1,rng,cfg.noise,"observational")
    z=net.encode(torch.tensor(x[:,:cfg.context]),torch.tensor(c[:,:cfg.context]),
                 torch.tensor(u[:,:cfg.context]),torch.tensor(lagged_action_tokens(a[:,:cfg.context])))
    action=torch.tensor(a[:,cfg.context-1:cfg.context])
    factual=net.rollout(z,action)[0][:,0,0].numpy()
    factual_mse=np.mean((factual-x[:,cfg.context])**2)
    p0one=net.rollout(z,torch.zeros(n,1))[0][:,0,0].numpy()
    p1one=net.rollout(z,torch.ones(n,1))[0][:,0,0].numpy()
    truth_one=-(.24+.10*np.tanh(c[:,cfg.context-1]))
    one_mse=np.mean(((p1one-p0one)-truth_one)**2)
    rows=[]
    for kind in ("seen_block","alternating","separated","late_separated"):
        a0=fixed_schedule(n,h,"none");a1=fixed_schedule(n,h,kind)
        y0=simulate_future(x[:,cfg.context-1],c[:,cfg.context-1],d[:,cfg.context-1],u[:,cfg.context-1],
                           a[:,cfg.context-2],a0)
        y1=simulate_future(x[:,cfg.context-1],c[:,cfg.context-1],d[:,cfg.context-1],u[:,cfg.context-1],
                           a[:,cfg.context-2],a1)
        p0=net.rollout(z,torch.tensor(a0))[0][:,:,0].numpy()
        p1=net.rollout(z,torch.tensor(a1))[0][:,:,0].numpy()
        step_err=np.mean(((p1-p0)-(y1[0]-y0[0]))**2,axis=0)
        for hh in (1,2,4,8,16):
            rows.append({"seed":seed,"schedule":kind,"horizon":hh,"factual_mse":factual_mse,
                         "one_step_effect_mse":one_mse,"effect_mse":np.mean(step_err[:hh]),
                         "terminal_effect_mse":step_err[hh-1]})
    return pd.DataFrame(rows)


def coverage_report(cfg,seed=0,n_bank=4000,n_test=700,h=16):
    rng=np.random.default_rng(91000+seed)
    x,c,d,u,a=generate_policy_batch(n_bank,cfg.context,rng,cfg.noise,"observational")
    train_a=block_actions(n_bank,cfg.train_horizon,rng)
    fut=simulate_future(x[:,-1],c[:,-1],d[:,-1],u[:,-1],a[:,-2],train_a)
    lag=np.c_[a[:,-2],train_a[:,:-1]]
    bank=np.stack([fut[0],fut[1],fut[2],lag],-1).reshape(-1,4)
    bank_a=train_a.reshape(-1)
    mean=bank.mean(0);sd=np.maximum(bank.std(0),1e-4);bank=(bank-mean)/sd
    trees={act:cKDTree(bank[bank_a==act]) for act in (0.,1.)}
    # Threshold from held-out block trajectories: 95% of in-family local transitions.
    bx,bc,bd,bu,ba=generate_policy_batch(n_test,cfg.context,rng,cfg.noise,"observational")
    block=block_actions(n_test,h,rng);bf=simulate_future(bx[:,-1],bc[:,-1],bd[:,-1],bu[:,-1],ba[:,-2],block)
    blag=np.c_[ba[:,-2],block[:,:-1]]
    bstate=(np.stack([bf[0],bf[1],bf[2],blag],-1).reshape(-1,4)-mean)/sd
    bdists=np.empty(len(bstate))
    for act in (0.,1.):
        mask=block.reshape(-1)==act;bdists[mask]=trees[act].query(bstate[mask],k=1)[0]
    threshold=float(np.quantile(bdists,.95));rows=[]
    for kind in ("seen_block","alternating","separated","late_separated"):
        act=fixed_schedule(n_test,h,kind);fut=simulate_future(bx[:,-1],bc[:,-1],bd[:,-1],bu[:,-1],ba[:,-2],act)
        alag=np.c_[ba[:,-2],act[:,:-1]]
        state=(np.stack([fut[0],fut[1],fut[2],alag],-1).reshape(-1,4)-mean)/sd
        dist=np.empty(len(state))
        for av in (0.,1.):
            mask=act.reshape(-1)==av;dist[mask]=trees[av].query(state[mask],k=1)[0]
        rows.append({"schedule":kind,"local_support_rate":np.mean(dist<=threshold),
                     "median_nn_distance":np.median(dist),"support_threshold":threshold})
    return pd.DataFrame(rows)


def plot(df,outdir):
    outdir.mkdir(parents=True,exist_ok=True);fig,axes=plt.subplots(1,3,figsize=(16,4.6))
    order=("predictive","unsplit_contrast","response_all","response_split")
    g=df.groupby("model_kind").mean(numeric_only=True).loc[list(order)]
    axes[0].bar(order,g.one_step_effect_mse);axes[0].set(yscale="log",ylabel="MSE",title="Locally supervised response")
    axes[0].tick_params(axis="x",rotation=25)
    q=df[(df.schedule=="alternating")].groupby(["model_kind","horizon"]).effect_mse.mean().reset_index()
    for kind,z in q.groupby("model_kind"):axes[1].plot(z.horizon,z.effect_mse,"o-",label=kind)
    axes[1].set(xlabel="Horizon",ylabel="Effect trajectory MSE",yscale="log",title="Unseen alternating schedule");axes[1].legend(fontsize=7)
    q=df[(df.horizon==16)].groupby(["model_kind","schedule"]).effect_mse.mean().unstack()
    q.loc[list(order)].plot.bar(ax=axes[2]);axes[2].set(yscale="log",ylabel="Effect MSE",title="Seen vs unseen compositions")
    axes[2].tick_params(axis="x",rotation=25);axes[2].legend(fontsize=7)
    fig.tight_layout();fig.savefig(outdir/"compositional_latent.png",dpi=220,bbox_inches="tight");plt.close(fig)


def run(output_dir="results/compositional_latent",figure_dir="figures/compositional_latent"):
    cfg=Config();out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);parts=[]
    kinds=("predictive","unsplit_contrast","response_all","response_split")
    for kind in kinds:
        for seed in cfg.seeds:
            net=train_model(kind,seed,cfg);d=evaluate(net,cfg,seed);d["model_kind"]=kind;parts.append(d)
            torch.save(net.state_dict(),out/f"{kind}_seed{seed}.pt")
    df=pd.concat(parts,ignore_index=True);df.to_csv(out/"metrics.csv",index=False)
    summary=df.groupby(["model_kind","schedule","horizon"]).mean(numeric_only=True).reset_index()
    summary.to_csv(out/"summary.csv",index=False)
    coverage=coverage_report(cfg);coverage.to_csv(out/"coverage.csv",index=False)
    design={"total_latent_dim":8,"response_dim":2,"contrast_weight":10.,"train_schedules":"contiguous blocks length 1-4",
            "test_schedules":["alternating","separated","late_separated"],"seeds":list(cfg.seeds)}
    (out/"design.json").write_text(json.dumps(design,indent=2),encoding="utf-8")
    plot(df,Path(figure_dir));return df,coverage


def run_closure(output_dir="results/closure_training",figure_dir="figures/closure_training"):
    cfg=Config();out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);parts=[]
    for weight in (1.,10.,100.):
        for seed in cfg.seeds:
            net=train_closure_model(seed,cfg,weight);d=evaluate(net,cfg,seed)
            d["model_kind"]="response_split_closure";d["closure_weight"]=weight;parts.append(d)
            torch.save(net.state_dict(),out/f"closure{weight:g}_seed{seed}.pt")
    df=pd.concat(parts,ignore_index=True);df.to_csv(out/"metrics.csv",index=False)
    validation=df[(df.schedule=="separated")&(df.horizon==16)].groupby("closure_weight").effect_mse.mean()
    selected=float(validation.idxmin())
    summary=df.groupby(["closure_weight","schedule","horizon"]).mean(numeric_only=True).reset_index()
    summary.to_csv(out/"summary.csv",index=False)
    selection={"validation_schedule":"separated","selected_closure_weight":selected,
               "test_schedule":"late_separated","stress_schedule":"alternating"}
    (out/"selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")

    base=pd.read_csv("results/compositional_latent/metrics.csv")
    chosen=df[df.closure_weight==selected]
    compare=pd.concat([base,chosen],ignore_index=True)
    compare.to_csv(out/"comparison_metrics.csv",index=False)
    # Pre-specified H=16 comparisons.  Keep the held-out result separate from
    # the validation schedule used to select the closure weight.
    h16=compare[compare.horizon==16]
    def paired_result(schedule,baseline):
        c=h16[(h16.schedule==schedule)&(h16.model_kind=="response_split_closure")].set_index("seed").effect_mse
        b=h16[(h16.schedule==schedule)&(h16.model_kind==baseline)].set_index("seed").effect_mse
        return {"closure_mean":float(c.mean()),"baseline_mean":float(b.mean()),
                "relative_change":float(c.mean()/b.mean()-1),
                "paired_seed_wins":int((c<b).sum()),"num_seeds":int(len(c))}
    findings={
        "selection_protocol":selection,
        "horizon":16,
        "held_out_test_vs_response_split":paired_result("late_separated","response_split"),
        "held_out_test_vs_predictive":paired_result("late_separated","predictive"),
        "stress_vs_response_split":paired_result("alternating","response_split"),
        "seen_block_vs_response_split":paired_result("seen_block","response_split"),
        "interpretation":"Closure supervision improves the held-out pulse composition, but worsens the seen contiguous block. It is evidence that transition closure matters, not evidence of universally correct dynamics."
    }
    (out/"findings.json").write_text(json.dumps(findings,indent=2),encoding="utf-8")
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    q=df[(df.schedule=="separated")].groupby(["closure_weight","horizon"]).effect_mse.mean().reset_index()
    for w,z in q.groupby("closure_weight"):axes[0].plot(z.horizon,z.effect_mse,"o-",label=f"gamma={w:g}")
    axes[0].set(xlabel="Horizon",ylabel="Effect MSE",yscale="log",title="Validation: separated pulses");axes[0].legend()
    order=["predictive","unsplit_contrast","response_split","response_split_closure"]
    q=compare[(compare.horizon==16)&(compare.schedule.isin(["seen_block","separated","late_separated"]))]
    q=q.groupby(["model_kind","schedule"]).effect_mse.mean().unstack().reindex(order)
    q.plot.bar(ax=axes[1]);axes[1].set(yscale="log",ylabel="Effect MSE",title=f"Selected closure (gamma={selected:g})")
    axes[1].tick_params(axis="x",rotation=25);axes[1].legend(fontsize=8)
    fig.tight_layout();Path(figure_dir).mkdir(parents=True,exist_ok=True)
    fig.savefig(Path(figure_dir)/"closure_training.png",dpi=220,bbox_inches="tight");plt.close(fig)
    return df,compare,selection
