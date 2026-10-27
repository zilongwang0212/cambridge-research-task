from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json, math, random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import rankdata, spearmanr


@dataclass
class Config:
    context: int = 16
    train_horizon: int = 8
    d_model: int = 32
    layers: int = 2
    heads: int = 4
    hidden: int = 64
    batch: int = 96
    steps: int = 600
    lr: float = 8e-4
    noise: float = .04
    latent_dims: tuple[int, ...] = (1, 2, 4, 8, 16)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    eval_horizon: int = 16


class PatientSystem:
    """Interpretable longitudinal system with treatment memory and a nuisance."""

    @staticmethod
    def propensity(x, c, lag, dose):
        score = -.45 + .85*x + .65*c + .35*lag - .08*dose
        return 1/(1+np.exp(-np.clip(score, -12, 12)))

    @staticmethod
    def step(x, c, dose, nuisance, action, lag, noise_x=0., noise_c=0., noise_n=0.):
        next_dose = .72*dose + action
        next_c = .64*c + .13*x + .14*action + noise_c
        treatment = -(.24 + .10*np.tanh(c))*action
        next_x = .66*x + .18*c + .08*np.sin(x) + treatment + .10*lag - .055*dose + noise_x
        # Nuisance is observable and persistent but causally irrelevant to X/C/treatment response.
        next_nuisance = .88*nuisance + noise_n
        return tuple(np.asarray(v, dtype=np.float32) for v in
                     (next_x, next_c, next_dose, next_nuisance))


def lagged_action_tokens(actions):
    """Align action features with states: token s contains A_{s-1}, never A_s."""
    if torch.is_tensor(actions):
        out=torch.zeros_like(actions);out[:,1:]=actions[:,:-1];return out
    out=np.zeros_like(actions);out[:,1:]=actions[:,:-1];return out


def generate_batch(n, total_length, rng, noise=.04):
    x=np.zeros((n,total_length),np.float32); c=np.zeros_like(x); d=np.zeros_like(x); u=np.zeros_like(x)
    a=np.zeros_like(x)
    x[:,0]=rng.normal(0,.75,n); c[:,0]=rng.normal(0,.65,n); u[:,0]=rng.normal(0,1,n)
    lag=rng.binomial(1,.4,n).astype(np.float32)
    for t in range(total_length-1):
        p=PatientSystem.propensity(x[:,t],c[:,t],lag,d[:,t])
        action=rng.binomial(1,p).astype(np.float32); a[:,t]=action
        nx=rng.normal(0,noise,n).astype(np.float32); nc=rng.normal(0,noise,n).astype(np.float32)
        nn=rng.normal(0,noise,n).astype(np.float32)
        x[:,t+1],c[:,t+1],d[:,t+1],u[:,t+1]=PatientSystem.step(
            x[:,t],c[:,t],d[:,t],u[:,t],action,lag,nx,nc,nn)
        lag=action
    return x,c,d,u,a


class LatentWorldModel(nn.Module):
    def __init__(self,cfg,latent_dim):
        super().__init__(); self.d_model=cfg.d_model; self.latent_dim=latent_dim
        self.embed=nn.Linear(5,cfg.d_model)
        layer=nn.TransformerEncoderLayer(cfg.d_model,cfg.heads,cfg.d_model*3,dropout=.05,
                                         batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,cfg.layers)
        self.compress=nn.Linear(cfg.d_model,latent_dim)
        self.transition=nn.Sequential(nn.Linear(latent_dim+1,cfg.hidden),nn.Tanh(),
                                      nn.Linear(cfg.hidden,latent_dim))
        self.decoder=nn.Sequential(nn.Linear(latent_dim,cfg.hidden),nn.Tanh(),nn.Linear(cfg.hidden,2))

    def position(self,length,device):
        p=torch.arange(length,device=device).float().unsqueeze(1)
        div=torch.exp(torch.arange(0,self.d_model,2,device=device).float()*(-math.log(10000)/self.d_model))
        pe=torch.zeros(length,self.d_model,device=device); pe[:,0::2]=torch.sin(p*div); pe[:,1::2]=torch.cos(p*div)
        return pe

    def encode(self,x,c,nuisance,a):
        b,l=x.shape; time=torch.linspace(-1,0,l,device=x.device).expand(b,l)
        token=torch.stack([x,c,nuisance,a,time],-1)
        h=self.encoder(self.embed(token)+self.position(l,x.device))
        return self.compress(h[:,-1])

    def advance(self,z,action):
        # Residual transition makes repeated latent rollout stable enough to train,
        # while every future prediction still depends only on z and the action.
        return z + .25*self.transition(torch.cat([z,action[:,None]],1))

    def decode(self,z): return self.decoder(z)

    def rollout(self,z,actions):
        outputs=[]
        for t in range(actions.shape[1]):
            z=self.advance(z,actions[:,t]); outputs.append(self.decode(z))
        return torch.stack(outputs,1),z


class OracleStateModel(nn.Module):
    """Control model given the known sufficient state (x, c, dose, lag)."""
    def __init__(self,cfg):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(5,cfg.hidden),nn.Tanh(),nn.Linear(cfg.hidden,cfg.hidden),
                               nn.Tanh(),nn.Linear(cfg.hidden,4))

    def advance(self,state,action):
        return self.net(torch.cat([state,action[:,None]],1))

    def rollout(self,state,actions):
        xs=[]
        for t in range(actions.shape[1]):
            state=self.advance(state,actions[:,t]);xs.append(state[:,0])
        return torch.stack(xs,1)

    def rollout_states(self,state,actions):
        states=[]
        for t in range(actions.shape[1]):
            state=self.advance(state,actions[:,t]);states.append(state)
        return torch.stack(states,1)


def train_one(latent_dim,seed,cfg):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    rng=np.random.default_rng(2000+101*seed+latent_dim)
    net=LatentWorldModel(cfg,latent_dim); opt=torch.optim.AdamW(net.parameters(),lr=cfg.lr,weight_decay=1e-4)
    hist=[]; length=cfg.context+cfg.train_horizon
    for step in range(cfg.steps):
        x,c,d,u,a=generate_batch(cfg.batch,length,rng,cfg.noise)
        tensors=[torch.tensor(v,dtype=torch.float32) for v in (x,c,u,a)]
        tx,tc,tu,ta=tensors
        z=net.encode(tx[:,:cfg.context],tc[:,:cfg.context],tu[:,:cfg.context],
                     lagged_action_tokens(ta[:,:cfg.context]))
        actions=ta[:,cfg.context-1:cfg.context-1+cfg.train_horizon]
        pred,_=net.rollout(z,actions)
        target=torch.stack([tx[:,cfg.context:cfg.context+cfg.train_horizon],
                            tc[:,cfg.context:cfg.context+cfg.train_horizon]],-1)
        weights=torch.linspace(1,1.5,cfg.train_horizon)[None,:,None]
        loss=torch.mean(weights*(pred-target)**2)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
        if step%50==0: hist.append((step,float(loss.detach())))
    return net.eval(),hist


def train_oracle(seed,cfg):
    torch.manual_seed(seed);rng=np.random.default_rng(5000+seed)
    net=OracleStateModel(cfg);opt=torch.optim.AdamW(net.parameters(),lr=cfg.lr,weight_decay=1e-4)
    for _ in range(cfg.steps*5):
        x,c,d,u,a=generate_batch(cfg.batch,cfg.context+cfg.train_horizon,rng,cfg.noise)
        state=np.stack([x[:,cfg.context-1],c[:,cfg.context-1],d[:,cfg.context-1],a[:,cfg.context-2]],1)
        actions=torch.tensor(a[:,cfg.context-1:cfg.context-1+cfg.train_horizon])
        target=np.stack([x[:,cfg.context:cfg.context+cfg.train_horizon],
                         c[:,cfg.context:cfg.context+cfg.train_horizon],
                         d[:,cfg.context:cfg.context+cfg.train_horizon],
                         a[:,cfg.context-1:cfg.context-1+cfg.train_horizon]],-1)
        pred=net.rollout_states(torch.tensor(state),actions)
        scale=torch.tensor([1.,1.,.35,.5])[None,None,:]
        loss=torch.mean(scale*(pred-torch.tensor(target))**2)
        opt.zero_grad();loss.backward();opt.step()
    return net.eval()


@torch.no_grad()
def evaluate_oracle(net,cfg,seed,n=512):
    rng=np.random.default_rng(17000+seed);x,c,d,u,a=generate_batch(n,cfg.context,rng,cfg.noise)
    state=torch.tensor(np.stack([x[:,-1],c[:,-1],d[:,-1],a[:,-2]],1))
    rows=[]
    for h in [1,4,8,16,32]:
        for policy in ["alternating","block","sustained"]:
            a0=fixed_actions(n,h,"none");a1=fixed_actions(n,h,policy)
            y0=true_rollout(x[:,-1],c[:,-1],d[:,-1],u[:,-1],a[:,-2],a0)
            y1=true_rollout(x[:,-1],c[:,-1],d[:,-1],u[:,-1],a[:,-2],a1)
            p0=net.rollout(state,torch.tensor(a0)).numpy();p1=net.rollout(state,torch.tensor(a1)).numpy()
            rows.append({"seed":seed,"horizon":h,"policy":policy,
                         "cf_error":.5*(np.mean((p0-y0)**2)+np.mean((p1-y1)**2)),
                         "effect_error":np.mean(((p1-p0)-(y1-y0))**2)})
    return pd.DataFrame(rows)


def fixed_actions(n,h,kind):
    z=np.zeros((n,h),np.float32)
    if kind=="alternating": z[:,::2]=1
    elif kind=="block": z[:,:min(h,5)]=1
    elif kind=="sustained": z[:]=1
    elif kind!="none": raise ValueError(kind)
    return z


def true_rollout(x,c,d,u,lag,actions):
    xs=[]
    for t in range(actions.shape[1]):
        x,c,d,u=PatientSystem.step(x,c,d,u,actions[:,t],lag)
        xs.append(x.copy()); lag=actions[:,t]
    return np.stack(xs,1)


@torch.no_grad()
def collect_latents(net,cfg,seed,n=4000):
    rng=np.random.default_rng(11000+seed)
    x,c,d,u,a=generate_batch(n,cfg.context,rng,cfg.noise)
    z=net.encode(torch.tensor(x),torch.tensor(c),torch.tensor(u),
                 torch.tensor(lagged_action_tokens(a))).numpy()
    lag=a[:,-2]
    targets={"x":x[:,-1],"c":c[:,-1],"dose":d[:,-1],"nuisance":u[:,-1],"lag":lag}
    return z,targets,(x[:,-1],c[:,-1],d[:,-1],u[:,-1],lag)


def standardize_train_test(z,ntrain):
    z=np.asarray(z,dtype=np.float64)
    mean=z[:ntrain].mean(0,keepdims=True); sd=np.maximum(z[:ntrain].std(0,keepdims=True),1e-4)
    return np.clip((z[:ntrain]-mean)/sd,-20,20),np.clip((z[ntrain:]-mean)/sd,-20,20)


def ridge_predict(ztr,ytr,zte,alpha=1e-3):
    ztr=np.c_[np.ones(len(ztr)),ztr].astype(np.float64); zte=np.c_[np.ones(len(zte)),zte].astype(np.float64)
    ytr=np.asarray(ytr,dtype=np.float64)
    eye=np.eye(ztr.shape[1]); eye[0,0]=0
    coef=np.linalg.lstsq(np.vstack([ztr,np.sqrt(alpha)*eye]),
                          np.concatenate([ytr,np.zeros((eye.shape[0],)+ytr.shape[1:])]),rcond=None)[0]
    return np.einsum("ij,j...->i...",zte,coef,optimize=False)


def r2_score(y,p):
    return 1-np.sum((y-p)**2)/(np.sum((y-y.mean(0))**2)+1e-12)


def auc_score(y,score):
    y=np.asarray(y).astype(bool); n1=y.sum(); n0=(~y).sum()
    if not n1 or not n0:return np.nan
    ranks=rankdata(score); return (ranks[y].sum()-n1*(n1+1)/2)/(n1*n0)


def probe_latent(z,targets,states,cfg):
    ntr=int(.7*len(z)); ztr,zte=standardize_train_test(z,ntr); rows={}
    for name in ["x","c","dose","nuisance"]:
        y=targets[name]; pred=ridge_predict(ztr,y[:ntr],zte)
        rows[f"probe_{name}_r2"]=r2_score(y[ntr:],pred)
    y=targets["lag"]; pred=ridge_predict(ztr,y[:ntr],zte)
    rows["probe_lag_auc"]=auc_score(y[ntr:],pred)

    x,c,d,u,lag=states; h=cfg.eval_horizon
    a0=fixed_actions(len(z),h,"none"); a1=fixed_actions(len(z),h,"alternating")
    tau=true_rollout(x,c,d,u,lag,a1)-true_rollout(x,c,d,u,lag,a0)
    pred=ridge_predict(ztr,tau[:ntr],zte)
    rows["frozen_effect_probe_mse"]=float(np.mean((pred-tau[ntr:])**2))
    rows["frozen_effect_probe_r2"]=float(r2_score(tau[ntr:],pred))
    return rows


@torch.no_grad()
def evaluate(net,cfg,seed,n=512):
    rng=np.random.default_rng(15000+seed)
    x,c,d,u,a=generate_batch(n,cfg.context+1,rng,cfg.noise)
    z=net.encode(torch.tensor(x[:,:cfg.context]),torch.tensor(c[:,:cfg.context]),
                 torch.tensor(u[:,:cfg.context]),
                 torch.tensor(lagged_action_tokens(a[:,:cfg.context])))
    action=torch.tensor(a[:,cfg.context-1:cfg.context],dtype=torch.float32)
    pred,_=net.rollout(z,action)
    factual=float(np.mean((pred[:,0,0].numpy()-x[:,cfg.context])**2))
    # Deterministic intervention evaluation from an independent history endpoint.
    hx,hc,hd,hu=x[:,cfg.context-1],c[:,cfg.context-1],d[:,cfg.context-1],u[:,cfg.context-1]
    lag=a[:,cfg.context-2]; rows=[]
    for h in [1,4,8,16,32]:
        for policy in ["alternating","block","sustained"]:
            a0=fixed_actions(n,h,"none"); a1=fixed_actions(n,h,policy)
            y0=true_rollout(hx,hc,hd,hu,lag,a0); y1=true_rollout(hx,hc,hd,hu,lag,a1)
            p0,_=net.rollout(z,torch.tensor(a0)); p1,_=net.rollout(z,torch.tensor(a1))
            p0,p1=p0[:,:,0].numpy(),p1[:,:,0].numpy()
            rows.append({"horizon":h,"policy":policy,"factual_error":factual,
                         "cf_error":.5*(np.mean((p0-y0)**2)+np.mean((p1-y1)**2)),
                         "effect_error":np.mean(((p1-p0)-(y1-y0))**2)})
    return pd.DataFrame(rows)


def plot(model_summary,metrics,outdir):
    outdir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    grouped=model_summary.groupby("latent_dim")
    axes[0].plot(grouped.factual_error.mean().index,grouped.factual_error.mean(),marker="o",label="Factual")
    axes[0].plot(grouped.effect_error.mean().index,grouped.effect_error.mean(),marker="o",label="Effect H=16")
    axes[0].set(xlabel="Latent dimension",ylabel="Error",yscale="log",title="Prediction vs intervention");axes[0].legend()
    for key,label in [("probe_x_r2","X"),("probe_c_r2","C"),("probe_dose_r2","Dose"),("probe_nuisance_r2","Nuisance")]:
        axes[1].plot(grouped[key].mean().index,grouped[key].mean(),marker="o",label=label)
    axes[1].plot(grouped.probe_lag_auc.mean().index,grouped.probe_lag_auc.mean(),marker="o",label="Lag AUC")
    axes[1].set(xlabel="Latent dimension",ylabel="Probe score",title="What is retained in Z?");axes[1].legend()
    axes[2].scatter(model_summary.probe_dose_r2,model_summary.effect_error,c=model_summary.latent_dim,cmap="viridis",s=60)
    axes[2].set(xlabel="Dose-memory probe R²",ylabel="Effect error (H=16)",yscale="log",title="Information vs rollout fidelity")
    fig.tight_layout();fig.savefig(outdir/"latent_information_diagnosis.png",dpi=220,bbox_inches="tight");plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,5))
    q=metrics.groupby(["latent_dim","horizon"]).effect_error.mean().reset_index()
    for d,z in q.groupby("latent_dim"):ax.plot(z.horizon,z.effect_error,marker="o",label=f"z={d}")
    ax.set(xlabel="Horizon",ylabel="Effect error",yscale="log",title="True latent-state rollout");ax.legend()
    fig.tight_layout();fig.savefig(outdir/"latent_rollout.png",dpi=220,bbox_inches="tight");plt.close(fig)


def run(output_dir="results/medical_latent",figure_dir="figures/medical_latent"):
    cfg=Config();out,figs=Path(output_dir),Path(figure_dir);out.mkdir(parents=True,exist_ok=True)
    parts=[];models=[];logs={}
    for dim in cfg.latent_dims:
        for seed in cfg.seeds:
            net,hist=train_one(dim,seed,cfg); result=evaluate(net,cfg,seed)
            result["latent_dim"],result["seed"]=dim,seed;parts.append(result)
            z,targets,states=collect_latents(net,cfg,seed); probes=probe_latent(z,targets,states,cfg)
            h16=result[(result.horizon==16)&(result.policy=="alternating")].iloc[0]
            models.append({"latent_dim":dim,"seed":seed,"factual_error":h16.factual_error,
                           "cf_error":h16.cf_error,"effect_error":h16.effect_error,**probes})
            logs[f"z{dim}_s{seed}"]=hist;torch.save(net.state_dict(),out/f"z{dim}_seed{seed}.pt")
    metrics=pd.concat(parts,ignore_index=True);summary=pd.DataFrame(models)
    oracle=[]
    for seed in cfg.seeds:
        oracle_net=train_oracle(seed,cfg);oracle.append(evaluate_oracle(oracle_net,cfg,seed))
        torch.save(oracle_net.state_dict(),out/f"oracle_seed{seed}.pt")
    oracle=pd.concat(oracle,ignore_index=True);oracle.to_csv(out/"oracle_metrics.csv",index=False)
    metrics.to_csv(out/"rollout_metrics.csv",index=False);summary.to_csv(out/"model_diagnostics.csv",index=False)
    corr=[]
    for col in ["factual_error","probe_x_r2","probe_c_r2","probe_dose_r2","probe_nuisance_r2","probe_lag_auc","frozen_effect_probe_mse"]:
        corr.append({"variable":col,"spearman_with_effect_error":spearmanr(summary[col],summary.effect_error).statistic})
    pd.DataFrame(corr).to_csv(out/"correlations.csv",index=False)
    (out/"config.json").write_text(json.dumps(asdict(cfg),indent=2),encoding="utf-8")
    (out/"training_logs.json").write_text(json.dumps(logs),encoding="utf-8")
    plot(summary,metrics,figs)
    return summary,metrics
