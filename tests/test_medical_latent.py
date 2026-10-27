import numpy as np
import torch

from temporal_dte.medical_latent import (Config, LatentWorldModel, PatientSystem,
                                         generate_batch, lagged_action_tokens)


def test_action_tokens_never_include_current_action():
    actions=np.array([[1,0,1,1]],np.float32)
    assert np.array_equal(lagged_action_tokens(actions),[[0,1,0,1]])


def test_nuisance_does_not_change_patient_dynamics():
    x=np.array([.5],np.float32);c=np.array([-.2],np.float32);d=np.array([1.],np.float32)
    a=np.array([1.],np.float32);lag=np.array([0.],np.float32)
    first=PatientSystem.step(x,c,d,np.array([-3.],np.float32),a,lag)
    second=PatientSystem.step(x,c,d,np.array([4.],np.float32),a,lag)
    assert np.allclose(first[:3],second[:3])


def test_true_latent_rollout_shapes():
    cfg=Config(batch=4,context=8,train_horizon=3)
    x,c,d,u,a=generate_batch(4,11,np.random.default_rng(0))
    net=LatentWorldModel(cfg,latent_dim=2)
    z=net.encode(torch.tensor(x[:,:8]),torch.tensor(c[:,:8]),torch.tensor(u[:,:8]),
                 torch.tensor(lagged_action_tokens(a[:,:8])))
    pred,last=net.rollout(z,torch.tensor(a[:,7:10]))
    assert z.shape==(4,2) and pred.shape==(4,3,2) and last.shape==(4,2)
