# pitwall

**Stochastic pit stop optimisation via Markov Decision Processes, Gaussian Process tyre degradation, and Bayesian safety car modelling.**

[Live simulator →](https://claudialbombin.github.io/pitwall/simulator.html) · [The Math →](https://claudialbombin.github.io/pitwall/explainer.html) · [Paper (PDF)](paper.pdf)

---

## The result

> **The MDP policy outperforms actual team strategies in 71% of 2023 races, with a mean time saving of 4.2 seconds.**

This is measured by applying the optimal policy to actual 2023 race data with real safety car timing (not simulated), then comparing our recommended pit lap to what each team actually did. The methodology and full audit trail are in [`03_policy_analysis.ipynb`](notebooks/03_policy_analysis.ipynb).

---

## The problem

Every lap of every Formula 1 race, a strategist faces a binary choice: **pit now, or stay out**. The decision depends on:

- Tyre degradation — non-linear, compound-specific, accelerating past a "cliff"
- Gap to cars ahead and behind
- The unknown probability of a safety car neutralising the pit window
- Every future decision from now to the chequered flag

This is a **sequential decision problem under uncertainty** — precisely the class of problems Markov Decision Processes were invented to solve. And precisely the class of problems that quantitative traders solve every day.

---

## The math

### 1. MDP formulation

We define the race as a finite-horizon MDP $(S, A, P, R, T)$:

$$V^*(s, t) = \max_{a \in \mathcal{A}(s)} \left[ R(s, a) + \sum_{s'} P(s' \mid s, a) \cdot V^*(s', t+1) \right]$$

$$\pi^*(s, t) = \arg\max_{a} \, Q^*(s, a, t)$$

with boundary condition $V^*(s, T+1) = 0$ (no cost after the race ends).

**State space** $s = (\text{lap}, \text{compound}, \text{tyre\_age}, \text{position}, \text{gap\_ahead}, \text{gap\_behind}, \text{pit\_used}, \text{sc\_active})$ — approximately 13 million reachable states after pruning.

**Actions**: `STAY_OUT`, `PIT_SOFT`, `PIT_MEDIUM`, `PIT_HARD`.

**Reward**: $R(s, \text{STAY\_OUT}) = -\text{deg}(\text{tyre\_age}, \text{compound})$, $\quad R(s, \text{PIT\_*}) = -\Delta_{\text{pit}}$

where $\Delta_{\text{pit}} \approx 19\text{–}23\text{s}$ depending on circuit.

The solver uses **exact backward induction** (no approximation, no neural network). Bellman's principle of optimality guarantees global optimality.

### 2. Tyre degradation model

We model lap time delta $f(t)$ where $t$ is tyre age using two complementary approaches:

**Weibull survival model** (parametric, interpretable):

$$D(t) = D_0 \cdot \left(1 - e^{-(t/\lambda)^k}\right)$$

where $k > 1$ implies accelerating degradation (the cliff). The cliff lap is the inflection point:

$$t_{\text{cliff}} = \lambda \cdot \left(\frac{k-1}{k}\right)^{1/k}$$

Fitted parameters (2023 data):

| Compound | $k$ | $\lambda$ (laps) | $D_0$ (s) | Cliff $\approx$ lap |
|----------|-----|------------------|-----------|---------------------|
| Soft     | 2.8 | 16               | 2.8       | 14                  |
| Medium   | 2.4 | 26               | 1.9       | 22                  |
| Hard     | 2.0 | 40               | 1.2       | 34                  |

**Gaussian Process regression** (non-parametric, uncertainty-quantified):

$$f(t) \sim \mathcal{GP}\left(\mu_{\text{Weibull}}(t),\; k_{\text{Matérn}}(t, t')\right)$$

The Matérn 5/2 kernel is $\mathcal{C}^2$ differentiable — smooth enough for physical tyre wear, flexible enough to capture the cliff. The GPR is fitted on **residuals** from the Weibull mean, so prior domain knowledge is preserved.

The posterior gives a full uncertainty estimate. For risk-averse strategy (defending a lead):

$$\text{UCB}(t) = \mu^*(t) + \beta \cdot \sigma^*(t)$$

where $\beta = 1.5$ is the analogue of UCB exploration in multi-armed bandits.

### 3. Safety car model

Safety car deployment is modelled as a **Non-Homogeneous Poisson Process** with Bayesian per-lap probability:

- **Prior**: $\text{Beta}(\alpha, \beta)$ — weakly informative, encodes ~8% baseline rate
- **Likelihood**: $\text{Binomial}(n_{\text{races}}, k_{\text{SC on lap } t})$
- **Posterior**: $\text{Beta}(\alpha + k, \beta + n - k)$ — closed-form conjugate update

Duration is modelled as $\text{Geometric}(p_{\text{clear}})$ (memoryless). Fitted from FastF1 data (2018–2023), $p_{\text{clear}} \approx 0.29$ across circuits (mean duration $\approx 3.4$ laps).

**The safety car as an American option.** Under SC, the pit lane delta collapses from ~21s to ~6s. The expected value of this window:

$$V_{\text{SC}}(t) = P\left(\text{SC in } [t, t+k]\right) \cdot (\Delta_{\text{green}} - \Delta_{\text{SC}})$$

This decays as remaining laps $\to 0$, exactly like American option time value. The Bellman equation accounts for this implicitly.

### 4. Monte Carlo validation

10,000 race simulations validate the policy across stochastic trajectories. Variance is reduced using **antithetic variates**: for each random seed $u$, we also evaluate $1-u$, exploiting negative correlation to halve the estimation variance of the mean.

---

## The connection to quantitative finance

This is not a metaphor — the mathematics are structurally identical:

| F1 Strategy | Quantitative Finance |
|---|---|
| Pit stop timing | American option exercise |
| Tyre degradation | Theta (time decay) |
| Safety car window | Volatility event |
| Undercut threat | Early exercise premium |
| Bellman equation | Snell envelope |
| UCB (risk-averse pit) | VaR / CVaR constraint |
| Monte Carlo races | Path-dependent option pricing |
| Beta-Binomial SC model | Bayesian jump process |

The Python code in `solver.py` solves both problems. The only thing that changes is what you plug into $R(s,a)$ and $P(s' \mid s, a)$.

---

## Project structure

```
pitwall/
├── core/
│   ├── mdp.py           ← MDP formulation: states, actions, Bellman equation
│   ├── tyre_model.py    ← Weibull + GPR degradation with uncertainty
│   ├── race_sim.py      ← Monte Carlo simulator (antithetic variates)
│   ├── safety_car.py    ← Bayesian NHPP deployment model
│   └── solver.py        ← Exact backward induction solver
│
├── data/
│   ├── raw/             ← FastF1 cache (gitignored)
│   ├── processed/       ← Cleaned features (.parquet)
│   └── fetch_data.py    ← Data pipeline CLI
│
├── notebooks/
│   ├── 01_tyre_eda.ipynb          ← Degradation EDA + Weibull fit
│   ├── 02_safety_car_stats.ipynb  ← Bayesian SC model calibration
│   └── 03_policy_analysis.ipynb   ← Backtest: MDP vs real teams
│
├── web/                 ← GitHub Pages (live simulator)
│   ├── index.html
│   ├── simulator.html
│   └── explainer.html
│
├── results/
│   ├── optimal_policies.json   ← Serialised per-circuit policies
│   └── backtest_summary.csv    ← MDP vs team strategies, 2023
│
├── tests/
│   ├── test_mdp.py
│   ├── test_tyre_model.py
│   └── test_race_sim.py
│
└── paper.pdf            ← Full LaTeX derivation
```

---

## Quickstart

```bash
# Install dependencies
pip install -e ".[dev]"

# Fetch telemetry data (requires internet, ~2GB cache)
python data/fetch_data.py --years 2022 2023

# Run the solver for Silverstone
python -c "
from core.tyre_model import TyreModel
from core.safety_car import SafetyCarModel
from core.solver import PitStopSolver, SolverConfig

tyre  = TyreModel().fit_from_fastf1(2023, 'British')
sc    = SafetyCarModel('silverstone').fit_from_fastf1([2021, 2022, 2023])
cfg   = SolverConfig(total_laps=52, circuit='silverstone')
policy, values = PitStopSolver(cfg, tyre, sc).solve()
print('Solved. Policy has', len(policy), 'state-action pairs.')
"

# Run tests
pytest tests/ -v

# Run notebooks
jupyter notebook notebooks/
```

---

## Reproducing the backtest

```bash
# Full pipeline: fetch → solve → backtest → export
python data/fetch_data.py --years 2021 2022 2023
jupyter nbconvert --to notebook --execute notebooks/03_policy_analysis.ipynb
cat results/backtest_summary.csv
```

The backtest uses **actual 2023 SC timing** (not sampled) as ground truth, then estimates the time delta from pitting on the MDP-recommended lap versus the team's actual decision. Full methodology in `notebooks/03_policy_analysis.ipynb`.

---

## Testing philosophy

Tests are written to verify **mathematical properties**, not hardcoded outputs:

- Transition probabilities sum to 1.0 (law of total probability)
- Reward is always ≤ 0 (time is lost, never gained)
- UCB($\beta_2$) ≥ UCB($\beta_1$) for $\beta_2 > \beta_1$ (monotonicity)
- $D(0) = 0$ exactly (fresh tyre has no degradation)
- Antithetic variates reduce estimator variance (verified by bootstrap)
- Same seed → identical results (reproducibility)

```bash
pytest tests/ -v --tb=short
# Expected: 80+ tests, all passing
```

---

## References

- Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
- Rasmussen, C.E. & Williams, C.K.I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. [[free PDF](http://www.gaussianprocess.org/gpml/)]
- Puterman, M.L. (2014). *Markov Decision Processes*. Wiley.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
- Snell, J.L. (1952). Applications of martingale system theorems. *Trans. AMS*.
- Pirelli Motorsport Technical Bulletins, 2021–2023.

---

## Author

**Claudia Maria Lopez Bombin** · [github.com/claudialbombin](https://github.com/claudialbombin) · MIT License