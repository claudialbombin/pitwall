"""
Stochastic Safety Car Model: Bayesian Inference on Historical Deployment
========================================================================

THE SAFETY CAR PROBLEM:
In F1 strategy, the safety car (SC) and virtual safety car (VSC) are the
single greatest source of randomness. A well-timed safety car can transform
a losing strategy into a race win — ask Lewis Hamilton at Silverstone 2019,
or conversely, Max Verstappen at Abu Dhabi 2021. A poorly timed one can
destroy a comfortable lead.

The strategist's dilemma: pitting under a safety car costs ~5-8 seconds
instead of the usual 20-25 second pit lane delta. But you cannot know
in advance when — or whether — a safety car will appear. You are making
decisions under genuine, irreducible uncertainty.

This module models that uncertainty formally, giving the MDP solver a
principled probability distribution over safety car events rather than
a deterministic assumption (the usual failure mode of simpler strategy tools).

WHAT WE MODEL:
  1. P(SC deploys on lap t) — the probability a safety car appears on any
     given lap, conditioned on race lap number and circuit characteristics.

  2. P(SC clears on lap t | SC active) — the probability an active safety
     car period ends on lap t, given it is currently deployed.

  3. P(SC duration = d laps) — the distribution over safety car lengths,
     used to model the strategic window for pit stops.

THE STATISTICAL APPROACH: NON-HOMOGENEOUS POISSON PROCESS
Safety car events are rare, discrete, and arrive randomly in time — the
classic setup for a Poisson process. However, the deployment rate is NOT
constant across a race:

  - Laps 1-3: highest risk (first corner incidents, cold tyres)
  - Mid-race:  lower baseline risk
  - Lap 1:     approximately 3-4x higher than average lap probability
  - Wet conditions: 5-8x higher rate

We model this as a Non-Homogeneous Poisson Process (NHPP) with a
lap-varying intensity function λ(t):

  P(SC on lap t) ≈ 1 - exp(-λ(t))  ≈ λ(t)  for small λ(t)

We estimate λ(t) from historical data using a Bayesian approach:
  - Prior: Beta(α, β) on each lap's deployment probability
  - Likelihood: Binomial(n_races, k_sc_on_lap_t)
  - Posterior: Beta(α + k, β + n - k)  [conjugate update]

The Beta-Binomial conjugacy means the posterior has a closed form —
no MCMC needed. The posterior mean is our point estimate; the posterior
variance gives us uncertainty bounds that feed into the MDP solver.

CIRCUIT-SPECIFIC CALIBRATION:
Safety car rates vary dramatically by circuit:
  Monaco:      ~85% of races see a SC  (narrow, no run-off)
  Monza:       ~45%  (fast, but wide with good run-off)
  Silverstone: ~50%
  Spa:         ~55%  (weather + Raidillon)

We fit separate models per circuit from FastF1 historical data (2018-2023).
For circuits without historical data, we use the global prior.

DURATION MODEL:
Once deployed, how long does a safety car last? We fit a Geometric
distribution to historical SC duration data:

  P(SC ends on lap t | SC has lasted d laps) = p_clear
  P(duration = d) = p_clear · (1 - p_clear)^(d-1)  [Geometric]

The Geometric distribution is the discrete analogue of the Exponential —
it has the memoryless property, meaning the probability of clearing on the
next lap does not depend on how long the SC has been out. This is a
simplification (in reality, FIA tries to clear quickly), but it fits the
data reasonably well and keeps the model tractable.

The strategic implication: there is always a "window" to pit under SC.
The size of that window, and the probability of it remaining open for
one more lap, is exactly what this model quantifies.

Author: Claudia Maria Lopez Bombin
GitHub: https://github.com/claudialbombin/pitwall
License: MIT

References:
  Cox, D.R. & Lewis, P.A.W. (1966). The Statistical Analysis of Series
    of Events. Methuen. [NHPP foundations]
  Gelman, A. et al. (2013). Bayesian Data Analysis, 3rd Ed. CRC Press.
  FastF1 safety car data, seasons 2018-2023.
  FIA Formula 1 Sporting Regulations, Article 41 (Safety Car Procedure).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Default priors (Beta distribution parameters)
# ---------------------------------------------------------------------------
# Prior belief: safety car appears on ~8% of laps on average.
# Beta(2, 23) has mean = 2/(2+23) ≈ 0.08 and is weakly informative.

_PRIOR_ALPHA = 2.0  # pseudo-successes (SC deployments)
_PRIOR_BETA = 23.0  # pseudo-failures (no SC)

# Default p_clear per lap (Geometric duration model).
# Historical average: SC lasts ~3.5 laps → p_clear ≈ 1/3.5 ≈ 0.29
_DEFAULT_P_CLEAR = 0.29

# Lap 1 multiplier: first lap is ~4x more dangerous
_LAP1_MULTIPLIER = 4.0
# Laps 2-3 multiplier
_EARLY_MULTIPLIER = 2.0


# ---------------------------------------------------------------------------
# Per-circuit historical data (simplified — full data in data/processed/)
# ---------------------------------------------------------------------------
# Format: {circuit_name: {"n_races": int, "sc_by_lap": list[int]}}
# sc_by_lap[i] = number of races where SC was deployed on lap i+1
# Source: FastF1 sessions 2018-2023

_CIRCUIT_DATA: dict[str, dict] = {
    "monaco": {
        "n_races": 6,
        "p_deploy_baseline": 0.12,
        "p_clear_per_lap": 0.22,  # SC lasts longer in Monaco
        "mean_duration": 4.5,
    },
    "monza": {
        "n_races": 6,
        "p_deploy_baseline": 0.06,
        "p_clear_per_lap": 0.33,
        "mean_duration": 3.0,
    },
    "silverstone": {
        "n_races": 7,
        "p_deploy_baseline": 0.07,
        "p_clear_per_lap": 0.30,
        "mean_duration": 3.3,
    },
    "spa": {
        "n_races": 6,
        "p_deploy_baseline": 0.09,
        "p_clear_per_lap": 0.28,
        "mean_duration": 3.6,
    },
    "default": {
        "n_races": 0,
        "p_deploy_baseline": 0.08,
        "p_clear_per_lap": _DEFAULT_P_CLEAR,
        "mean_duration": 3.5,
    },
}


# ---------------------------------------------------------------------------
# Bayesian Beta-Binomial model for deployment probability
# ---------------------------------------------------------------------------


@dataclass
class BayesianLapProbability:
    """
    Bayesian estimate of P(SC on lap t) using Beta-Binomial conjugacy.

    Posterior: Beta(α + k, β + n - k)
      α, β = prior parameters
      n    = number of observed races
      k    = number of races with SC on this lap
    """

    alpha: float = _PRIOR_ALPHA
    beta: float = _PRIOR_BETA

    def update(self, n_races: int, n_sc_events: int) -> "BayesianLapProbability":
        """Update posterior with observed data."""
        return BayesianLapProbability(
            alpha=self.alpha + n_sc_events,
            beta=self.beta + (n_races - n_sc_events),
        )

    @property
    def mean(self) -> float:
        """Posterior mean: E[p] = α / (α + β)"""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Posterior variance."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """
        Bayesian credible interval (not a frequentist confidence interval).
        P(p ∈ [lo, hi]) = level under the posterior.
        """
        lo = (1 - level) / 2
        hi = 1 - lo
        dist = stats.beta(self.alpha, self.beta)
        return float(dist.ppf(lo)), float(dist.ppf(hi))


# ---------------------------------------------------------------------------
# Safety Car Model
# ---------------------------------------------------------------------------


@dataclass
class SafetyCarModel:
    """
    Full stochastic model for SC/VSC deployment and duration.

    Combines:
      - Non-homogeneous Poisson Process for deployment timing
      - Geometric distribution for duration
      - Bayesian updating from historical data

    Interface consumed by solver.py and race_sim.py.

    Usage:
        model = SafetyCarModel("silverstone")
        model.fit_from_fastf1(year=2023)

        p = model.p_deploy(lap=3)     # P(SC on lap 3)
        p = model.p_clear(lap=5)      # P(SC clears on lap 5)
        d = model.sample_duration()   # Random SC duration (laps)
    """

    circuit: str = "default"
    total_laps: int = 66

    # Per-lap Bayesian probability models (populated during fit or from defaults)
    _lap_probs: list[BayesianLapProbability] = field(
        default_factory=list, init=False, repr=False
    )
    _p_clear: float = field(default=_DEFAULT_P_CLEAR, init=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        circuit_data = _CIRCUIT_DATA.get(self.circuit, _CIRCUIT_DATA["default"])
        self._p_clear = circuit_data["p_clear_per_lap"]

        # Initialise per-lap probs from circuit baseline + lap-position priors
        baseline = circuit_data["p_deploy_baseline"]
        self._lap_probs = []

        for lap in range(1, self.total_laps + 1):
            # Apply lap-position multipliers to baseline
            if lap == 1:
                p = min(0.95, baseline * _LAP1_MULTIPLIER)
            elif lap <= 3:
                p = min(0.95, baseline * _EARLY_MULTIPLIER)
            else:
                # Slight decay towards end of race (fewer cars = fewer incidents)
                decay = 1.0 - 0.003 * max(0, lap - 40)
                p = baseline * max(0.5, decay)

            # Convert point estimate to Beta prior (method of moments)
            # α/(α+β) = p, α+β = 10 (concentration)
            concentration = 10.0
            alpha = p * concentration
            beta = (1 - p) * concentration
            self._lap_probs.append(BayesianLapProbability(alpha=alpha, beta=beta))

    # ------------------------------------------------------------------
    # Fitting from FastF1 data
    # ------------------------------------------------------------------

    def fit_from_fastf1(self, year: int | list[int]) -> "SafetyCarModel":
        """
        Update Bayesian priors with observed SC deployments from FastF1.

        Loads race session data for the given year(s) and circuit,
        records on which lap each SC/VSC was deployed, and updates
        the per-lap Beta posteriors via conjugate update.

        Parameters
        ----------
        year : int or list[int] — seasons to include (e.g. [2021, 2022, 2023])
        """
        try:
            import fastf1  # type: ignore[import]
        except ImportError:
            raise ImportError("fastf1 required. Install with: pip install fastf1")

        years = [year] if isinstance(year, int) else year
        fastf1.Cache.enable_cache("data/raw")

        all_sc_laps: list[list[int]] = []  # sc_laps per race

        for yr in years:
            try:
                session = fastf1.get_session(yr, self.circuit, "R")
                session.load(laps=True, telemetry=False, weather=False)
                sc_laps = self._extract_sc_laps(session)
                all_sc_laps.append(sc_laps)
            except Exception as e:
                warnings.warn(
                    f"Could not load {self.circuit} {yr}: {e}", RuntimeWarning
                )

        if not all_sc_laps:
            warnings.warn("No sessions loaded. Using prior only.", RuntimeWarning)
            return self

        n_races = len(all_sc_laps)

        # Update per-lap posteriors
        for lap_idx in range(self.total_laps):
            lap_num = lap_idx + 1
            n_sc_on_this_lap = sum(
                1 for race_sc_laps in all_sc_laps if lap_num in race_sc_laps
            )
            self._lap_probs[lap_idx] = self._lap_probs[lap_idx].update(
                n_races, n_sc_on_this_lap
            )

        # Update p_clear from observed durations
        durations = self._extract_durations(all_sc_laps)
        if durations:
            # MLE for Geometric: p_clear = 1 / mean_duration
            self._p_clear = 1.0 / np.mean(durations)

        self._fitted = True
        return self

    @staticmethod
    def _extract_sc_laps(session) -> list[int]:
        """Extract lap numbers with SC/VSC from a FastF1 session."""
        sc_laps = []
        try:
            # FastF1 stores track status per lap: "4" = SC, "5" = VSC, "6" = VSC ending
            laps = session.laps
            sc_mask = laps["TrackStatus"].isin(["4", "5", "6"])
            sc_lap_nums = laps[sc_mask]["LapNumber"].dropna().astype(int).unique()
            sc_laps = sorted(sc_lap_nums.tolist())
        except Exception:
            pass
        return sc_laps

    @staticmethod
    def _extract_durations(all_sc_laps: list[list[int]]) -> list[int]:
        """
        Compute consecutive run lengths of SC laps (duration in laps).
        Used to fit the Geometric clearance model.
        """
        durations = []
        for sc_laps in all_sc_laps:
            if not sc_laps:
                continue
            sc_set = sorted(set(sc_laps))
            # Find consecutive runs
            run = 1
            for i in range(1, len(sc_set)):
                if sc_set[i] == sc_set[i - 1] + 1:
                    run += 1
                else:
                    durations.append(run)
                    run = 1
            durations.append(run)
        return durations

    # ------------------------------------------------------------------
    # Probability queries (used by solver.py)
    # ------------------------------------------------------------------

    def p_deploy(self, lap: int) -> float:
        """
        P(SC deploys on lap t | SC not currently active).

        Returns posterior mean deployment probability for this lap.
        For the risk-averse variant, use p_deploy_ucb().
        """
        if lap < 1 or lap > self.total_laps:
            return 0.0
        return self._lap_probs[lap - 1].mean

    def p_deploy_ucb(self, lap: int, confidence: float = 0.9) -> float:
        """
        Upper credible bound on P(SC on lap t).
        Used by the risk-averse solver variant.
        """
        if lap < 1 or lap > self.total_laps:
            return 0.0
        _, hi = self._lap_probs[lap - 1].credible_interval(level=confidence)
        return hi

    def p_clear(self, lap: int) -> float:
        """
        P(SC clears on lap t | SC currently active).

        Geometric model: constant clearance probability per lap.
        Ignores lap argument in base model (memoryless property).
        Can be extended to non-constant p_clear if data supports it.
        """
        return self._p_clear

    def expected_sc_laps(self) -> float:
        """
        Expected number of SC laps in the race.
        E[laps under SC] = Σ_t P(SC on lap t) · E[duration]
        """
        mean_duration = 1.0 / self._p_clear
        return sum(p.mean for p in self._lap_probs) * mean_duration

    # ------------------------------------------------------------------
    # Sampling (used by race_sim.py Monte Carlo)
    # ------------------------------------------------------------------

    def sample_sc_events(
        self, rng: Optional[np.random.Generator] = None
    ) -> list[tuple[int, int]]:
        """
        Sample a complete race's SC events from the stochastic model.

        Returns list of (start_lap, end_lap) tuples for each SC period.
        Used by race_sim.py to generate Monte Carlo race trajectories.

        Parameters
        ----------
        rng : numpy random Generator (pass for reproducibility)
        """
        if rng is None:
            rng = np.random.default_rng()

        events: list[tuple[int, int]] = []
        sc_active = False
        sc_start = 0

        for lap in range(1, self.total_laps + 1):
            if sc_active:
                # SC is out — does it clear this lap?
                if rng.random() < self._p_clear:
                    events.append((sc_start, lap - 1))
                    sc_active = False
            else:
                # SC is not out — does it deploy this lap?
                p = self._lap_probs[lap - 1].mean
                if rng.random() < p:
                    sc_active = True
                    sc_start = lap

        # Handle SC active at end of race
        if sc_active:
            events.append((sc_start, self.total_laps))

        return events

    def sample_duration(self, rng: Optional[np.random.Generator] = None) -> int:
        """
        Sample a single SC duration from the Geometric model.

        Returns number of laps the SC is deployed.
        """
        if rng is None:
            rng = np.random.default_rng()
        # Geometric: number of trials until first success
        return int(rng.geometric(self._p_clear))

    # ------------------------------------------------------------------
    # Strategic analysis utilities
    # ------------------------------------------------------------------

    def pit_window_value(
        self,
        lap: int,
        pit_delta_sc: float = 6.0,
        pit_delta_normal: float = 21.0,
        laps_remaining: int = 30,
    ) -> float:
        """
        Expected value of having a SC window available at lap t.

        The value of the "SC option" at lap t is the expected savings from
        pitting under SC vs normal conditions, weighted by the probability
        that a SC actually appears:

          V_SC(t) = P(SC in [t, t+k]) · (pit_delta_normal - pit_delta_sc)

        where k = laps_remaining. This is directly analogous to the
        intrinsic value of an American call option — and like an option,
        it decays as laps_remaining → 0.

        Parameters
        ----------
        lap              : current lap
        pit_delta_sc     : pit stop time cost under SC (seconds)
        pit_delta_normal : pit stop time cost under green flag (seconds)
        laps_remaining   : number of laps left to benefit from SC
        """
        # Probability of at least one SC in the remaining window
        p_no_sc = np.prod(
            [
                1.0 - self._lap_probs[t - 1].mean
                for t in range(lap, min(lap + laps_remaining, self.total_laps) + 1)
            ]
        )
        p_sc_in_window = 1.0 - p_no_sc

        savings = pit_delta_normal - pit_delta_sc
        return p_sc_in_window * savings

    def deployment_heatmap(self) -> np.ndarray:
        """
        Return per-lap deployment probabilities as a 1D array.
        Serialised to optimal_policies.json for the D3 visualisation.
        """
        return np.array([p.mean for p in self._lap_probs])

    def summary(self) -> dict:
        """Human-readable model summary."""
        return {
            "circuit": self.circuit,
            "total_laps": self.total_laps,
            "fitted": self._fitted,
            "p_clear_per_lap": round(self._p_clear, 3),
            "mean_sc_duration": round(1.0 / self._p_clear, 1),
            "expected_sc_laps": round(self.expected_sc_laps(), 1),
            "peak_risk_lap": int(np.argmax([p.mean for p in self._lap_probs])) + 1,
            "lap1_p_deploy": round(self._lap_probs[0].mean, 3),
        }
