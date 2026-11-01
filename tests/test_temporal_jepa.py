import numpy as np
import torch

from temporal_dte.medical_latent import Config
from temporal_dte.temporal_jepa import train_model, observed_diagnostics


def test_temporal_jepa_smoke():
    cfg=Config(context=8,train_horizon=4,batch=8,steps=2,seeds=(0,))
    net=train_model("jepa",0,cfg,jepa_weight=.1,latent_dim=4)
    out=observed_diagnostics(net,cfg,0,n=12,h=4)
    assert set(out.horizon)=={1,2,4}
    assert np.isfinite(out.latent_closure_mse).all()


def test_single_horizon_and_no_ema_smoke():
    cfg=Config(context=8,train_horizon=4,batch=8,steps=2,seeds=(0,))
    net=train_model("jepa",0,cfg,jepa_weight=.1,latent_dim=4,
                    jepa_horizons=(1,),ema_momentum=0.)
    out=observed_diagnostics(net,cfg,0,n=12,h=4)
    assert np.isfinite(out.observed_rollout_mse).all()
