import numpy as np
from temporal_dte.dual_experiment import Model, rollout


def test_linear_rollout_tracks_delayed_action():
    model = Model(alpha=0, sin_coef=0, treatment_base=1, treatment_state=0, lag_coef=.5)
    got = rollout(model, np.array([0.]), np.array([0.]), np.array([[1., 0.]]))
    np.testing.assert_allclose(got, [[0., 1., .5]])
