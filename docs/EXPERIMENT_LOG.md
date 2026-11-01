# Forecasting Is Not Temporal Understanding

This repository tests one falsifiable capability: **predicting dynamic treatment
effects under held-out intervention policies**. It asks whether this diagnostic adds
information about recovery of a controlled temporal mechanism beyond factual
prediction error.

## Main experiment

The ground-truth simulator is

\[
X_{t+1}=0.7X_t+0.2\sin(X_t)+(0.2+0.2\tanh(X_t))A_t+0.1A_{t-1}+\epsilon_t.
\]

We generate 1,000 length-128 observational trajectories and construct 100
misspecified candidate dynamics. The exact oracle is deliberately not among the
candidates. Every candidate is evaluated with:

- factual one-step prediction error, under the behavior policy;
- held-out-policy dynamic treatment effect error;
- full interventional trajectory error, a strong baseline;
- transition-function error on an independent, treatment-balanced state-action grid.

Candidate selection first screens to the best factual-prediction quartile. We then
compare selecting by prediction error alone against selecting by DTE error. See
[THEORY.md](THEORY.md) for the observational non-identifiability argument.

## Reproduce

```bash
python3 -m pip install -e .
make test
make synthetic
```

For the focused two-system experiment (the recommended starting point):

```bash
make core
```

This creates `results/core/metrics.csv`, `results/core/summary.json`, and the two
figures in `figures/core/`.

## Focused result: one linear and one nonlinear system

Each system has 100 manually parameter-perturbed candidates; neither candidate set
contains the exact oracle. We calculate exactly three errors: factual prediction
error \(E_P\), held-out-policy dynamic-treatment-effect error \(E_{DTE}\), and
independent transition-function error \(E_D\).

The result is deliberately mixed:

- **Nonlinear system:** after controlling \(E_P\), partial Spearman
  \(\rho=0.343\), bootstrap 95% CI `[0.126, 0.518]`. Adding \(E_{DTE}\) improves
  repeated five-fold out-of-sample \(R^2\) by a median `0.121` in all 100 repeats.
- **Linear system:** partial Spearman \(\rho=-0.378\), bootstrap 95% CI
  `[-0.559, -0.128]`; median CV \(\Delta R^2=-0.028\). Under the chosen whole-function
  definition of \(E_D\), DTE has no incremental diagnostic value.

The linear negative result is interpretable: differencing two potential-outcome
trajectories cancels action-independent state-propagation error, whereas \(E_D\)
penalizes it. DTE is therefore a targeted diagnostic of intervention-sensitive
dynamics, not a universal surrogate for complete transition-function recovery.

Here \(E_{DTE}\) is an equal-weight average over nine held-out policies: early,
middle, and late single pulses; early and late treatment blocks; periodic schedules
with periods 2, 3, and 5; and sustained treatment. The CSV retains every
policy-specific error and the across-policy standard deviation.

The checked-in results were generated with 1,000 series, 100 candidates, five
horizons, and three policy conditions. The run is deterministic.

## Current result

In the primary held-out intermittent-policy condition at horizon 32:

- partial Spearman correlation between DTE error and independent dynamics error,
  controlling factual prediction error: **0.309**;
- corresponding correlation for full interventional forecasting error: **0.412**;
- median dynamics error of the model selected by factual prediction alone: **0.0130**;
- median dynamics error after prediction screening and DTE selection: **0.00255**.

Thus DTE has incremental diagnostic value in this construction, but it does **not**
beat the stronger full-interventional-trajectory baseline. This qualification is an
important part of the result.

## Figures

1. `figures/figure1_prediction_blind_spot.png`: candidates with adequate factual
   prediction but different treatment-response curves.
2. `figures/figure2_metric_validity.png`: DTE and full-interventional diagnostics
   versus independent dynamics error.
3. `figures/figure3_conditions.png`: horizon, policy shift, and lag-information
   ablation.
4. `figures/figure4_selection.png`: mechanism error under the two selection rules.

## Optional Chronos-2 external-validity check

The runner uses treatment as a known future covariate and compares identical
histories under two future treatment schedules:

```bash
python3 -m pip install -e '.[foundation]'
PYTHONPATH=src python3 scripts/run_chronos2.py
```

This download-dependent experiment is kept separate from the primary metric-validity
experiment. Failure on synthetic data is not, by itself, evidence that a foundation
model lacks temporal understanding; it may reflect pretraining distribution mismatch.
The script never creates mocked foundation-model results.

## Scope and limitations

- The lag omission is a controlled proxy for information loss, not a direct probe of
  a neural network's latent dimensionality.
- Ground-truth individual counterfactuals are available only because this is a known
  simulator with paired exogenous noise.
- DTE tests a particular intervention-sensitive capability; it is evidence about,
  not a complete definition of, temporal understanding.
- The reported association is specific to the chosen candidate distribution. Robust
  conclusions require repeating the experiment across simulator and candidate
  families.

## Repeated-candidate validation

A larger validation reruns 20 independent candidate pools for each of 10 systems
(20,000 candidates total), uses nine policies and horizon 16, and adds full
interventional forecasting error (IFE) as a strong baseline. The original 1.8%
mean selected-ED percentile does not persist exactly: prediction-gated DTE averages
4.7% across resamples (SD across pool-level means 2.1%). Gates at 2% and 5% are
statistically indistinguishable, while gates at 25% are significantly worse than
5%. Gated IFE is slightly better on average (4.0% versus 4.7%), but the paired
comparison is not significant (Wilcoxon p=0.415). These results support a strict
prediction gate, but do not support superiority of DTE over direct interventional
forecasting.

## Equal-supervision Transformer training ablation

The small action-conditioned Transformer experiment now includes a stricter
control for supervision quantity. `forecast_equal` and `forecast_dte_equal` see
the same paired intervention trajectories, use the same architecture, optimizer,
steps, train systems, and random seeds; the only difference is the additional
relational DTE loss in the latter. Results use five seeds and ten held-out systems.

At horizon 16, adding DTE changes mean held-out errors as follows:

- observational factual error: `0.1162 -> 0.0901` (-22.5%);
- unseen-policy trajectory error: `0.0550 -> 0.0422` (-23.4%);
- unseen-policy DTE error: `0.0915 -> 0.0701` (-23.4%);
- independent one-step transition probe: `0.0313 -> 0.0252` (-19.4%).

The benefit also persists under genuine autoregressive free rollout. At horizons
16 and 32, trajectory error falls by 38.3% and 42.5%, while DTE error falls by
23.1% and 22.4%, respectively. The horizon-32 evaluation is extrapolation because
training horizons stop at 16. Per-system/seed paired tests are exploratory because
observations sharing a system or seed are not fully independent; more independent
systems and seeds are required for a final statistical claim.

An approximate primal-dual implementation of the literal prediction-gated DTE
objective is a negative result: it is unstable and substantially worsens free
rollout. Thus the current evidence supports a combined forecast + DTE loss, not
the particular hard-gating optimizer attempted here. Outputs are in
`results/transformer_controlled/` and the comparison plot is in
`figures/transformer_controlled/equal_supervision.png`.

### Independent OOD dynamics probe

We additionally generate valid histories from the known simulator under four
policies absent from training: balanced random actions, alternating actions,
long treatment blocks, and inverse state-dependent assignment. At each resulting
state, both current actions are queried and the deterministic one-step prediction
is compared directly with the true transition function. This evaluation is never
used for training, checkpoint selection, or hyperparameter tuning.

This stricter test is negative. Averaged over five seeds, ten held-out systems,
four policies, and two current actions, the equal-supervision forecast baseline has
`E_D^OOD = 0.01917`, whereas forecast + DTE has `0.02155` (+12.4%). The degradation
appears in both linear (+11.6%) and nonlinear (+13.3%) systems. The hard-gated
variant is substantially worse (`0.05728`). Consequently, the present experiment
supports improved in-distribution transition accuracy and unseen-policy trajectory
prediction, but **does not support general OOD recovery of the true controlled
dynamics**. Results are in `results/transformer_controlled/ood_dynamics_metrics.csv`.

## Medical world-model Part 1

`scripts/run_medical_part1.py` studies whether factual one-step accuracy is
sufficient evidence for counterfactual rollout fidelity. It uses two transparent
patient simulators: a simple linear treatment-lag system and a moderately complex
system with an observed treatment-affected confounder and mild nonlinearity.
Forty small Transformers (five latent dimensions, four seeds, two systems) are
trained only on factual transitions generated by a state-dependent historical
clinician policy. No counterfactual or treatment-effect target enters training.

At horizon 16 under the maximally shifted policy, strictly prediction-matched
model pairs (factual errors within 5%) can still differ in treatment-effect error
by 60.6x in the simple system and 28.4x in the moderate system. This is a proof of
concept that historical-policy forecasting accuracy need not determine treatment
simulation accuracy. Horizon and policy shift expose the separation; latent size
helps explain heterogeneity but is not monotonically equivalent to mechanism
quality. Results are in `results/medical_part1/` and figures in
`figures/medical_part1/`.

### Why does latent dimension matter?

`scripts/run_medical_latent.py` replaces the output-head bottleneck with a genuine
latent dynamics model, `Z[t+1] = F(Z[t], A[t])`, and diagnoses what is linearly
recoverable from the frozen state. With 25 factual-only models, a one-dimensional
state retains current severity (`R²=0.93`) but largely loses the treatment-effect
modifier (`R²=0.07`); its intervention-effect error is 2.6x that of dimension two.
From dimension two onward, modifier, dose-memory, and treatment-lag probes are
strong. The frozen effect probe improves monotonically (`R²=0.54 -> 0.99`), but
the world-model rollout does not. In the held-out alternating-policy test at
horizon 16, dimension 16 has 12% better factual error than dimension 4 while its
effect error is about 2.0x worse. Thus small-state failure is partly information
loss, whereas larger-state failures arise from how the learned transition uses
and propagates available information. Larger states also increasingly encode the
causally irrelevant nuisance (`R²≈0 -> 0.98`), showing that raw capacity is not a
mechanism-quality measure. Outputs are in `results/medical_latent/`.

## Temporal JEPA ablations and selection sensitivity

`scripts/run_temporal_jepa_ablation.py` completes the training-design and
model-selection sensitivity analyses. Each training variant uses the same model,
data generator, prediction objective, JEPA weight (`lambda_J=1`), and five seeds.

At horizon 8, one-step JEPA, endpoint-only JEPA, and multi-horizon JEPA with the
original EMA momentum (`m=0.99`) obtain closure MSEs of `0.01563`, `0.00244`, and
`0.000384`, respectively. Their mean horizon-16 effect MSEs across four schedules
are `0.001404`, `0.001543`, and `0.001360`. Multi-horizon matching therefore
strongly improves closure, but one-step JEPA remains competitive for effect
estimation; closure is a diagnostic and selection proxy, not an effect guarantee.

For multi-horizon JEPA, EMA momenta `0`, `0.9`, `0.99`, and `0.999` produce mean
horizon-16 effect MSEs of `0.001905`, `0.001289`, `0.001360`, and `0.001849`.
Moderate smoothing is favored, with `m=0.9` best in this finite grid. The original
`m=0.99` conclusion remains positive relative to the multi-step predictive baseline,
but it is not the optimal EMA value found by the ablation.

The observational selection rule was recomputed for closure horizons
`H_c in {1,2,4,8,16}` and prediction gates
`gamma in {1.00,1.05,1.10,1.20}` without retraining or using counterfactual
metrics. Under the prespecified `gamma=1.10`, horizons 2, 4, and 8 all select
`lambda_J=1`. At horizon 8, `gamma=1.05`, `1.10`, and `1.20` select weights
`0.1`, `1`, and `10`, respectively; no JEPA candidate meets the strict `1.00`
gate. The gate therefore represents a real factual-fidelity tolerance and must be
reported transparently.

Outputs are stored in `results/temporal_jepa_ablation/`, with the summary figure in
`figures/temporal_jepa_ablation/temporal_jepa_ablation.png`.

## License

MIT
