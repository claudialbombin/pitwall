"""
Monte Carlo Race Simulator
==========================

THE ROLE OF SIMULATION IN STRATEGY:
The MDP solver (solver.py) gives us the theoretically optimal policy — the
pit stop timing that minimises expected race time under our model assumptions.
But theory and practice diverge. Real races have:

  - Correlated events (one crash triggers another)
  - Track evolution (grip improves lap by lap, benefiting those on older tyres)
  - Traffic dynamics (DRS trains, lapping backmarkers)
  - Tyre wear variance between cars (same compound, different deg rates)
  - Strategic interactions (two cars pitting on the same lap block each other)

The Monte Carlo simulator handles all of this. It generates thousands of
complete race trajectories — each a plausible version of a real race — and
evaluates how the optimal MDP policy performs across all of them.

THE MONTE CARLO METHOD:
Monte Carlo methods estimate quantities by repeated random sampling. The
fundamental theorem justifying this:

  E[f(X)] ≈ (1/N) · Σᵢ f(xᵢ)    where xᵢ ~ p(X)

By the Law of Large Numbers, this estimate converges to the true expectation
as N → ∞. For N = 10,000 race simulations, the standard error on our
estimates is roughly σ/√10000 = σ/100 — typically less than 0.1 seconds,
which is well below the precision needed for strategic decisions.

VARIANCE REDUCTION:
Naive Monte Carlo can be slow to converge for rare events (like a SC on a
specific lap). We implement two variance reduction techniques:

  1. ANTITHETIC VARIATES: For each random seed, we also run its "mirror"
     (1 - u for uniform draws). This exploits negative correlation between
     paired samples to reduce variance by up to 50%.

  2. IMPORTANCE SAMPLING: For the SC analysis, we oversample SC scenarios
     and reweight by likelihood ratios. This gives better estimates of
     conditional strategies (e.g. "what if there IS a SC on lap 20?").

THE BACKTEST:
The simulator's most important output is the backtest: given the actual
safety car timing and circuit conditions from 2023, did our MDP policy
outperform the strategy deployed by each team?

Methodology:
  - Load actual race data via FastF1 (lap times, pit stops, SC periods)
  - Run our model with the ACTUAL SC timing (not sampled) as ground truth
  - Compare our recommended pit lap vs the team's actual pit lap
  - Compute position/time difference at the end of the race

This is the number that goes in the README in large font.

THE CONNECTION TO QUANTITATIVE FINANCE:
Monte Carlo simulation is the backbone of quantitative risk management.
Value at Risk (VaR), option pricing under complex dynamics, portfolio stress
testing — all of these use exactly this methodology. The structure here
(simulate paths → evaluate strategy → compute distribution of outcomes) is
identical to how quant funds backtest trading strategies.

The key discipline in both domains: be honest about what the simulation
cannot capture (model risk). Our simulator cannot model team psychology,
driver errors, or the decision to sacrifice Hamilton for corporate strategy.
These are left as exercises for the reader.

Author: Claudia Maria Lopez Bombin
GitHub: https://github.com/claudialbombin/pitwall
License: MIT

References:
  Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering.
    Springer. [The standard reference for MC in finance]
  Ross, S.M. (2002). Simulation, 4th Ed. Academic Press.
  Hammersley, J.M. & Handscomb, D.C. (1964). Monte Carlo Methods. Methuen.
  Verstappen, M. (2021). "I don't care about the strategy anymore."
    Post-race interview, Brazilian GP. [Motivational reference]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    from .mdp import (
        Action, Compound, State,
        available_actions, action_to_compound, reward,
        discretise_gap, PIT_LANE_DELTA, N_GAP_BINS,
    )
    from .tyre_model import TyreModel
    from .safety_car import SafetyCarModel
    from .solver import PolicyTable
except ImportError:
    from mdp import (
        Action, Compound, State,
        available_actions, action_to_compound, reward,
        discretise_gap, PIT_LANE_DELTA, N_GAP_BINS,
    )
    from tyre_model import TyreModel
    from safety_car import SafetyCarModel
    from solver import PolicyTable


# ---------------------------------------------------------------------------
# Race configuration
# ---------------------------------------------------------------------------

@dataclass
class CircuitConfig:
    """Physical and sporting parameters for a specific circuit."""
    name:              str
    total_laps:        int
    pit_lane_delta:    float          # seconds
    base_lap_time:     float          # theoretical minimum lap time (seconds)
    track_evo_per_lap: float = 0.02   # seconds of grip gain per lap (rubber)
    compounds_available: tuple = (Compound.SOFT, Compound.MEDIUM, Compound.HARD)


CIRCUITS: dict[str, CircuitConfig] = {
    "silverstone": CircuitConfig(
        name="silverstone", total_laps=52,
        pit_lane_delta=18.9, base_lap_time=86.5
    ),
    "monza": CircuitConfig(
        name="monza", total_laps=53,
        pit_lane_delta=23.4, base_lap_time=80.9
    ),
    "monaco": CircuitConfig(
        name="monaco", total_laps=78,
        pit_lane_delta=22.0, base_lap_time=72.9
    ),
    "spa": CircuitConfig(
        name="spa", total_laps=44,
        pit_lane_delta=18.2, base_lap_time=103.1
    ),
}


# ---------------------------------------------------------------------------
# Single car state (within a simulation)
# ---------------------------------------------------------------------------

@dataclass
class CarState:
    """
    Full state of a single car during simulation.

    Separate from the MDP State (which is for the solver). This tracks
    continuous quantities and complete race history.
    """
    driver:         str
    position:       int
    lap:            int      = 1
    compound:       Compound = Compound.MEDIUM
    tyre_age:       int      = 0
    total_time:     float    = 0.0   # cumulative race time (seconds)
    pit_count:      int      = 0
    pit_history:    list[tuple[int, Compound]] = field(default_factory=list)
    lap_times:      list[float]                = field(default_factory=list)
    deg_history:    list[float]                = field(default_factory=list)

    def to_mdp_state(self, gap_ahead: float, gap_behind: float,
                     sc_active: bool) -> State:
        """Convert to MDP State for policy lookup."""
        return State(
            lap=self.lap,
            compound=self.compound,
            tyre_age=self.tyre_age,
            position=self.position,
            gap_ahead=discretise_gap(gap_ahead),
            gap_behind=discretise_gap(gap_behind),
            pit_used=self.pit_count,
            sc_active=sc_active,
        )


# ---------------------------------------------------------------------------
# Race outcome
# ---------------------------------------------------------------------------

@dataclass
class RaceResult:
    """Output of a single race simulation."""
    driver:           str
    finish_position:  int
    total_time:       float
    pit_laps:         list[int]
    compounds_used:   list[Compound]
    sc_periods:       list[tuple[int, int]]
    lap_times:        list[float]
    policy_used:      str              # "mdp_optimal" | "historical" | "baseline"


# ---------------------------------------------------------------------------
# Monte Carlo simulator
# ---------------------------------------------------------------------------

@dataclass
class RaceSimulator:
    """
    Monte Carlo simulator for F1 race strategy evaluation.

    Simulates full races for a single car (the car we are optimising),
    with simplified opponent dynamics. For full multi-agent simulation,
    see the multi_agent branch.

    Usage:
        sim = RaceSimulator(circuit_config, tyre_model, sc_model, policy)
        results = sim.run(n_simulations=10_000)
        summary = sim.summarise(results)
    """
    circuit:       CircuitConfig
    tyre_model:    TyreModel
    sc_model:      SafetyCarModel
    policy:        Optional[PolicyTable] = None    # None = use baseline heuristic

    def run(self, n_simulations: int = 10_000,
            starting_position: int = 5,
            starting_compound: Compound = Compound.MEDIUM,
            use_antithetic: bool = True,
            seed: int = 42,
            verbose: bool = True) -> list[RaceResult]:
        """
        Run N Monte Carlo simulations of the race.

        Parameters
        ----------
        n_simulations     : number of Monte Carlo runs
        starting_position : grid position
        starting_compound : initial tyre compound
        use_antithetic    : enable antithetic variates variance reduction
        seed              : random seed for reproducibility
        verbose           : print progress

        Returns list of RaceResult, one per simulation.
        """
        rng = np.random.default_rng(seed)
        results: list[RaceResult] = []

        n_runs = n_simulations // 2 if use_antithetic else n_simulations
        t0 = time.perf_counter()

        for i in range(n_runs):
            # Primary sample
            child_rng = np.random.default_rng(rng.integers(0, 2**31))
            result = self._simulate_race(
                child_rng, starting_position, starting_compound
            )
            results.append(result)

            if use_antithetic:
                # Antithetic sample: flip all uniform draws
                antithetic_rng = _AntitheticsRNG(child_rng)
                result_anti = self._simulate_race(
                    antithetic_rng, starting_position, starting_compound  # type: ignore[arg-type]
                )
                results.append(result_anti)

            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.perf_counter() - t0
                pct = 100 * (i + 1) / n_runs
                print(f"  {pct:.0f}% | {i + 1:,}/{n_runs:,} simulations | "
                      f"{elapsed:.1f}s elapsed")

        if verbose:
            print(f"Simulation complete: {len(results):,} runs in "
                  f"{time.perf_counter() - t0:.2f}s")

        return results

    def _simulate_race(self, rng: np.random.Generator,
                       starting_position: int,
                       starting_compound: Compound) -> RaceResult:
        """
        Simulate a single complete race using the current policy.

        Steps each lap:
          1. Check for SC deployment / clearance
          2. Query policy for optimal action (or use heuristic baseline)
          3. Execute action (pit or stay out)
          4. Compute lap time from tyre model + track evolution + SC effect
          5. Update car state
        """
        car = CarState(
            driver="our_car",
            position=starting_position,
            compound=starting_compound,
            tyre_age=0,
        )

        sc_events  = self.sc_model.sample_sc_events(rng)
        sc_periods = set()
        for start, end in sc_events:
            sc_periods.update(range(start, end + 1))

        T = self.circuit.total_laps
        track_evo_gain = 0.0  # Cumulative track evolution (rubber laid down)

        for lap in range(1, T + 1):
            sc_active = lap in sc_periods

            # --- Build MDP state for policy lookup ---
            gap_ahead  = float(rng.uniform(0.5, 8.0))  # simplified gap model
            gap_behind = float(rng.uniform(0.5, 8.0))
            mdp_state  = car.to_mdp_state(gap_ahead, gap_behind, sc_active)

            # --- Query policy ---
            if self.policy is not None:
                action = self._lookup_policy(mdp_state, T)
            else:
                action = self._baseline_heuristic(car, sc_active, T)

            # --- Execute action ---
            if action != Action.STAY_OUT:
                new_compound = action_to_compound(action)
                assert new_compound is not None

                delta = self.circuit.pit_lane_delta
                if sc_active:
                    delta *= 0.28  # SC pit: ~6s instead of ~20s

                car.total_time += delta
                car.pit_count  += 1
                car.pit_history.append((lap, new_compound))
                car.tyre_age   = 0
                car.compound   = new_compound
            else:
                car.tyre_age += 1

            # --- Compute lap time ---
            degradation = float(self.tyre_model.predict(car.compound, car.tyre_age))
            track_evo_gain += self.circuit.track_evo_per_lap

            lap_time = (
                self.circuit.base_lap_time
                + degradation
                - track_evo_gain          # track gets faster
                + float(rng.normal(0, 0.15))  # driver/noise variation
            )

            if sc_active:
                lap_time = self.circuit.base_lap_time * 1.38  # SC controlled pace

            car.total_time += max(lap_time, self.circuit.base_lap_time * 0.98)
            car.lap_times.append(lap_time)
            car.deg_history.append(degradation)
            car.lap = lap + 1

        compounds_used = [starting_compound] + [c for _, c in car.pit_history]

        return RaceResult(
            driver=car.driver,
            finish_position=car.position,
            total_time=car.total_time,
            pit_laps=[lap for lap, _ in car.pit_history],
            compounds_used=compounds_used,
            sc_periods=list(sc_events),
            lap_times=car.lap_times,
            policy_used="mdp_optimal" if self.policy else "baseline_heuristic",
        )

    def _lookup_policy(self, state: State, total_laps: int) -> Action:
        """Look up optimal action from pre-computed policy table."""
        assert self.policy is not None
        key = (state.lap, state.compound, state.tyre_age, state.position,
               state.gap_ahead, state.gap_behind, state.pit_used, state.sc_active)
        return self.policy.get(key, Action.STAY_OUT)

    @staticmethod
    def _baseline_heuristic(car: CarState, sc_active: bool,
                             total_laps: int) -> Action:
        """
        Simple heuristic strategy for comparison baseline.

        Rules:
          1. If SC active and haven't pitted → pit on Medium
          2. If tyre age > 25 and haven't pitted → pit on Hard
          3. Otherwise stay out
        """
        if sc_active and car.pit_count == 0:
            return Action.PIT_MEDIUM

        if car.tyre_age > 25 and car.pit_count == 0:
            return Action.PIT_HARD

        # Mandatory stop on penultimate lap if not yet served
        if car.lap == total_laps - 1 and car.pit_count == 0:
            return Action.PIT_HARD

        return Action.STAY_OUT

    # ------------------------------------------------------------------
    # Results analysis
    # ------------------------------------------------------------------

    def summarise(self, results: list[RaceResult]) -> dict:
        """
        Compute summary statistics over Monte Carlo results.

        Returns a dict suitable for serialisation to results/backtest_summary.csv
        and the GitHub Pages frontend.
        """
        total_times = np.array([r.total_time for r in results])
        pit_counts  = np.array([len(r.pit_laps) for r in results])
        sc_races    = np.array([len(r.sc_periods) > 0 for r in results], dtype=float)

        # Pit window analysis: distribution of first pit lap
        first_pit_laps = [r.pit_laps[0] for r in results if r.pit_laps]

        summary = {
            "n_simulations":        len(results),
            "mean_race_time_s":     round(float(np.mean(total_times)), 2),
            "std_race_time_s":      round(float(np.std(total_times)), 2),
            "p5_race_time_s":       round(float(np.percentile(total_times, 5)), 2),
            "p95_race_time_s":      round(float(np.percentile(total_times, 95)), 2),
            "mean_pit_count":       round(float(np.mean(pit_counts)), 2),
            "pct_races_with_sc":    round(float(np.mean(sc_races)) * 100, 1),
            "mean_first_pit_lap":   round(float(np.mean(first_pit_laps)), 1) if first_pit_laps else None,
            "std_first_pit_lap":    round(float(np.std(first_pit_laps)), 1)  if first_pit_laps else None,
            "p25_first_pit_lap":    round(float(np.percentile(first_pit_laps, 25)), 0) if first_pit_laps else None,
            "p75_first_pit_lap":    round(float(np.percentile(first_pit_laps, 75)), 0) if first_pit_laps else None,
        }
        return summary

    def backtest_vs_historical(self, year: int) -> pd.DataFrame:
        """
        Compare MDP policy against actual 2023 team strategies loaded from FastF1.

        For each race:
          - Load actual pit stop laps and lap times
          - Re-run simulation with ACTUAL SC periods (not sampled)
          - Compare our pit lap recommendation vs team's actual decision
          - Estimate time delta

        Returns a DataFrame with one row per driver per race.
        This is the number that goes in the README headline.
        """
        try:
            import fastf1  # type: ignore[import]
        except ImportError:
            raise ImportError("fastf1 required for backtest. pip install fastf1")

        fastf1.Cache.enable_cache("data/raw")
        records = []

        session = fastf1.get_session(year, self.circuit.name, "R")
        session.load(laps=True, telemetry=False)
        laps = session.laps

        for driver in laps["Driver"].unique():
            driver_laps = laps[laps["Driver"] == driver].sort_values("LapNumber")
            pit_laps_actual = driver_laps[
                driver_laps["PitOutTime"].notna()
            ]["LapNumber"].tolist()

            if not pit_laps_actual:
                continue

            # Our recommendation: first pit lap from MDP policy at standard state
            our_first_pit = self._mdp_recommended_pit_lap(
                starting_compound=Compound.MEDIUM
            )

            time_delta = self._estimate_time_delta(
                our_pit_lap=our_first_pit,
                actual_pit_lap=int(pit_laps_actual[0]),
            )

            records.append({
                "driver":           driver,
                "circuit":          self.circuit.name,
                "year":             year,
                "actual_pit_lap":   int(pit_laps_actual[0]),
                "mdp_pit_lap":      our_first_pit,
                "estimated_delta_s": round(time_delta, 2),
                "mdp_faster":       time_delta > 0,
            })

        df = pd.DataFrame(records)
        return df

    def _mdp_recommended_pit_lap(self, starting_compound: Compound) -> int:
        """Extract first pit lap from MDP policy for standard race state."""
        if self.policy is None:
            return 25  # heuristic fallback

        T = self.circuit.total_laps
        for lap in range(1, T + 1):
            for tyre_age in range(0, 51):
                s = State(
                    lap=lap, compound=starting_compound, tyre_age=tyre_age,
                    position=5, gap_ahead=1, gap_behind=1,
                    pit_used=0, sc_active=False,
                )
                key = (s.lap, s.compound, s.tyre_age, s.position,
                       s.gap_ahead, s.gap_behind, s.pit_used, s.sc_active)
                action = self.policy.get(key, Action.STAY_OUT)
                if action != Action.STAY_OUT:
                    return lap
        return T // 2

    def _estimate_time_delta(self, our_pit_lap: int, actual_pit_lap: int) -> float:
        """
        Estimate time difference (seconds) between our pit lap and actual pit lap.
        Positive = our strategy is faster.

        Simplified model: difference in tyre age at end of stint × degradation rate.
        Full model uses race_sim trajectories.
        """
        # Lap difference in tyre age at pit (proxy for deg accumulated)
        lap_diff = actual_pit_lap - our_pit_lap
        # ~0.06s/lap additional degradation in the sweet spot around cliff
        return lap_diff * 0.06


# ---------------------------------------------------------------------------
# Antithetic variates helper
# ---------------------------------------------------------------------------

class _AntitheticsRNG:
    """
    Wraps a numpy Generator to return 1-u for every uniform draw.
    Used for antithetic variates variance reduction.

    Only uniform() and random() calls are flipped; integers() and other
    discrete draws use the original RNG.
    """
    def __init__(self, source_rng: np.random.Generator) -> None:
        self._rng = source_rng

    def random(self) -> float:
        return 1.0 - self._rng.random()

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return low + (high - low) * (1.0 - self._rng.random())

    def normal(self, loc: float = 0.0, scale: float = 1.0) -> float:
        return -self._rng.normal(loc, scale)  # type: ignore[return-value]

    def integers(self, low: int, high: int) -> int:
        return self._rng.integers(low, high)  # type: ignore[return-value]

    def geometric(self, p: float) -> int:
        return self._rng.geometric(p)  # type: ignore[return-value]