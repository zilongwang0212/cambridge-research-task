import numpy as np
import torch

from temporal_dte.medical_part1 import Config, MedicalSystem, SmallMedicalTransformer, histories


def test_simple_treatment_reduces_next_state():
    system = MedicalSystem("simple")
    x = np.array([1.], dtype=np.float32); c = np.zeros(1, dtype=np.float32); lag = np.zeros(1, dtype=np.float32)
    untreated, _ = system.step(x, c, np.zeros(1, dtype=np.float32), lag)
    treated, _ = system.step(x, c, np.ones(1, dtype=np.float32), lag)
    assert treated[0] < untreated[0]


def test_history_and_transformer_shapes():
    cfg = Config(batch=4, context=8)
    system = MedicalSystem("moderate")
    x, c, a = histories(system, 4, 8, np.random.default_rng(0))
    net = SmallMedicalTransformer(cfg, latent_dim=4)
    out = net(torch.tensor(x), torch.tensor(c), torch.tensor(a), torch.zeros(4))
    assert out.shape == (4, 2)
