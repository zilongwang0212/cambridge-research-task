import numpy as np

from temporal_dte.system import Dynamics, TRUE_DYNAMICS, rollout


def test_true_transition_matches_specification():
    x, a, prev = np.array([0.4]), np.array([1.0]), np.array([1.0])
    expected = 0.7*x + 0.2*np.sin(x) + (0.2+0.2*np.tanh(x))*a + 0.1*prev
    np.testing.assert_allclose(TRUE_DYNAMICS.transition(x, a, prev), expected)


def test_rollout_uses_lagged_action():
    m = Dynamics(alpha=0, sin_coef=0, treatment_base=0, treatment_state=0,
                 lag_coef=1, use_sin=False, use_state_treatment=False, use_lag=True)
    got = rollout(m, np.array([0.]), np.array([0.]), np.array([[1., 0.]]))
    np.testing.assert_allclose(got, [[0., 0., 1.]])
