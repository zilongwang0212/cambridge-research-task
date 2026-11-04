# Learning Causally Relevant Temporal States in Medical World Models

Code for a focused proof of concept on treatment-conditioned temporal dynamics in
medical world models.

## Central claim

Accurate prediction of observed patient trajectories does not by itself guarantee
accurate long-horizon treatment-effect simulation. A finite-capacity latent state
may omit treatment-relevant history, and a learned transition may fail to preserve
such information over time even when it is initially encoded. This repository
studies these two failure modes and evaluates **Temporal JEPA**, an
action-conditioned objective that learns multi-horizon latent dynamics from
observed longitudinal trajectories without counterfactual labels.

## Code structure

```text
src/temporal_dte/
  medical_latent.py             simulator and latent world model
  functional_use_experiment.py  observational trajectory generation
  latent_test_validation.py     latent-capacity model bank
  transition_diagnosis.py       representation-versus-transition audit
  compositional_latent.py       intervention schedules and rollout utilities
  temporal_jepa.py              predictive baselines and Temporal JEPA
  temporal_jepa_ablation.py     training and selection ablations

scripts/
  run_medical_latent.py
  run_latent_test.py
  run_transition_diagnosis.py
  run_temporal_jepa.py
  evaluate_interventional_trajectories.py
  run_temporal_jepa_ablation.py

tests/                           simulator and training smoke tests
```

Generated metrics, checkpoints, and figures are written to `results/` and
`figures/`. These directories are intentionally not tracked because the complete
pipeline recreates them.

## Installation

Python 3.10 or newer is required. The experiments run on CPU; a GPU only reduces
training time.

```bash
git clone https://github.com/zilongwang0212/cambridge-research-task.git
cd cambridge-research-task
python3 -m venv .venv
source .venv/bin/activate
make install
make test
```

## Reproduction

Run the full pipeline from the repository root:

```bash
make reproduce
```

The pipeline runs in the following order:

1. latent-capacity analysis;
2. transition-propagation diagnosis;
3. one-step, multi-step, and Temporal JEPA training;
4. interventional trajectory evaluation;
5. Temporal JEPA ablations.

The Makefile handles dependencies automatically. In particular,
`transition-diagnosis` uses the model bank produced by `latent-test`, and
`trajectories` uses the checkpoints produced by `temporal-jepa`.

Individual stages can also be run separately:

```bash
make medical-latent
make transition-diagnosis
make temporal-jepa
make trajectories
make temporal-jepa-ablation
```

To remove generated outputs and Python caches:

```bash
make clean
```

## License

MIT
