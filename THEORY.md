# Why observational RMSE is insufficient

Consider the controlled transition

\[
X_{t+1}=\alpha X_t+\beta A_t
\]

observed only under the behavior policy \(A_t=\kappa X_t\). The observed process is

\[
X_{t+1}=(\alpha+\kappa\beta)X_t.
\]

For every constant \(c\), all parameter pairs satisfying
\(\alpha+\kappa\beta=c\) induce the same observational trajectory distribution.
Consequently, even infinite observational data and zero forecasting RMSE cannot
identify \(\beta\).

Under an intervention, however,

\[
X_{t+h}(\bar a)=\alpha^hX_t+
\beta\sum_{j=0}^{h-1}\alpha^{h-1-j}a_{t+j},
\]

so observationally equivalent systems generally disagree on both one-step and
long-horizon dynamic treatment effects. Therefore no bound of the form
`DTE error <= g(observational RMSE)`, with `g(0)=0`, can hold without additional
identification assumptions or policy variation.

This is an insufficiency result, not a claim that DTE estimation alone proves full
temporal understanding. The empirical claim tested here is narrower: after factual
prediction adequacy is established, held-out-policy DTE error can provide incremental
evidence about recovery of the controlled transition mechanism.
