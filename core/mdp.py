"""
Markov Decision Process Formulation for F1 Pit Stop Timing
============================================================

THE DECISION PROBLEM:
Every lap, a Formula 1 strategist faces the same question: pit now, or stay out?
It sounds simple. It is not. The decision depends on tyre degradation (non-linear,
compound-specific, track-surface-dependent), the gap to cars ahead and behind,
the probability of a safety car neutralising the pit window, the undercut/overcut
threat from rival teams, and the number of laps remaining. All of this must be
resolved in roughly 30 seconds, under race pressure, with incomplete information.

This module formalises that decision as a Markov Decision Process — the same
mathematical framework used in reinforcement learning, robotics, and quantitative
finance (optimal stopping, American option exercise, sequential portfolio rebalancing).
If you've seen the Bellman equation before, you'll recognise the structure immediately.
If you haven't, this is a beautiful place to meet it.

THE MATHEMATICAL FRAMEWORK:
A finite-horizon MDP is defined by the tuple (S, A, P, R, T) where:

  S  — State space: all information relevant to the decision
  A  — Action space: the choices available at each state
  P  — Transition kernel: P(s' | s, a), the probability of landing in state s'
       after taking action a from state s
  R  — Reward function: R(s, a, s'), the immediate payoff of a transition
  T  — Horizon: the number of decision epochs (here, race laps)

The goal is to find a policy π: S → A that maximises the expected cumulative
reward (minimises expected race time). The optimal policy satisfies Bellman's
principle of optimality:

  V*(s) = max_a [ R(s, a) + γ · Σ_{s'} P(s' | s, a) · V*(s') ]

In our finite-horizon setting γ = 1 (no discounting — we care about total race
time equally across all laps). The optimal value function V*(s, t) represents
the minimum expected remaining race time from state s at lap t.

STATE SPACE DESIGN:
The art in MDP modelling is choosing a state representation that is:
  (a) Markovian — the future is independent of the past given the current state
  (b) Rich enough — captures all decision-relevant information
  (c) Tractable — small enough for exact or approximate dynamic programming

Our state vector s = (lap, compound, tyre_age, position, gap_ahead, gap_behind,
                      pit_used, sc_active) balances these constraints.

  lap         ∈ {1, ..., T}       Current race lap
  compound    ∈ {SOFT, MEDIUM, HARD, INTER, WET}
  tyre_age    ∈ {0, ..., 50}      Laps on current set
  position    ∈ {1, ..., 20}      Current race position
  gap_ahead   ∈ ℝ⁺               Delta to car ahead (seconds), discretised
  gap_behind  ∈ ℝ⁺               Delta to car behind (seconds), discretised
  pit_used    ∈ {0, 1, 2}        Number of pit stops taken (≥1 mandatory)
  sc_active   ∈ {0, 1}           Safety car / VSC currently deployed

ACTION SPACE:
At each lap, the strategist chooses one of:
  PIT_SOFT   — Enter pit lane, fit Soft compound
  PIT_MEDIUM — Enter pit lane, fit Medium compound
  PIT_HARD   — Enter pit lane, fit Hard compound
  STAY_OUT   — Remain on track, continue current strategy

Not all actions are available in all states. STAY_OUT is invalid on the final lap
if the mandatory stop has not been served. PIT actions are invalid under certain
gap conditions (pit lane delta too costly).

REWARD STRUCTURE:
The reward at each lap represents the NEGATIVE of time lost relative to the
theoretical minimum (a lap with new tyres, no traffic, no pit stop).

  R(s, STAY_OUT) = -lap_time(tyre_age, compound, position, sc_active)
  R(s, PIT_*)    = -lap_time(0, new_compound, position+k, sc_active)
                   - pit_lane_delta    (≈ 20-25 seconds of lost time)
                   - position_cost(k) (positions lost during pit stop)

The optimal policy minimises total race time, equivalent to maximising
cumulative reward over the T-lap horizon.

DISCRETISATION NOTE:
Gap variables (gap_ahead, gap_behind) are continuous in reality but discretised
into bins for tractability: [0, 1s), [1s, 3s), [3s, 5s), [5s+). This is a
standard technique in approximate dynamic programming and introduces bounded
approximation error that we quantify in the analysis notebooks.

Author: Claudia Maria Lopez Bombin
GitHub: https://github.com/claudialbombin/pitwall-simulator
License: MIT

References:
  Bellman, R. (1957). Dynamic Programming. Princeton University Press.
  Puterman, M.L. (2014). Markov Decision Processes. Wiley.
  Sutton, R.S. & Barto, A.G. (2018). Reinforcement Learning: An Introduction. MIT Press.
  Hamilton, L. (2021). Lap 53, Abu Dhabi. Personal communication with the pit wall.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Compound(IntEnum):
    SOFT = 0
    MEDIUM = 1
    HARD = 2
    INTER = 3
    WET = 4


class Action(IntEnum):
    STAY_OUT = 0
    PIT_SOFT = 1
    PIT_MEDIUM = 2
    PIT_HARD = 3


# Gap discretisation bins (seconds)
GAP_BINS = [0.0, 1.0, 3.0, 5.0, float("inf")]
N_GAP_BINS = len(GAP_BINS) - 1  # 4 bins


def discretise_gap(gap_seconds: float) -> int:
    """Map a continuous gap value to its discrete bin index."""
    for i in range(N_GAP_BINS):
        if GAP_BINS[i] <= gap_seconds < GAP_BINS[i + 1]:
            return i
    return N_GAP_BINS - 1


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class State:
    """
    Immutable state vector for the pit stop MDP.

    Frozen dataclass → hashable → usable as dict key (required for value
    iteration, which stores V*(s) in a lookup table).
    """

    lap: int  # 1-indexed
    compound: Compound
    tyre_age: int  # laps on current set, 0 = brand new
    position: int  # 1 = lead, 20 = last
    gap_ahead: int  # discretised bin index (0 = DRS range)
    gap_behind: int  # discretised bin index
    pit_used: int  # 0, 1, or 2 stops taken
    sc_active: bool  # safety car or VSC on track

    def __post_init__(self) -> None:
        assert 1 <= self.lap, "Lap must be ≥ 1"
        assert 0 <= self.tyre_age <= 60, "Tyre age out of range"
        assert 1 <= self.position <= 20, "Position must be 1-20"
        assert 0 <= self.gap_ahead < N_GAP_BINS
        assert 0 <= self.gap_behind < N_GAP_BINS
        assert 0 <= self.pit_used <= 2

    @property
    def is_terminal(self) -> bool:
        """A state is terminal when there are no more decisions to make."""
        return False  # Termination handled by the horizon T in the solver

    def with_pit(self, new_compound: Compound, pos_delta: int = 2) -> "State":
        """
        Return the state immediately after completing a pit stop.

        pos_delta: positions lost during the pit stop (typically 1-4 depending
        on gap to car behind and pit lane delta). Default = 2 is a conservative
        estimate; race_sim.py computes this stochastically.
        """
        return State(
            lap=self.lap,
            compound=new_compound,
            tyre_age=0,
            position=min(20, self.position + pos_delta),
            gap_ahead=self.gap_ahead,
            gap_behind=self.gap_behind,
            pit_used=self.pit_used + 1,
            sc_active=self.sc_active,
        )

    def with_lap_completed(
        self,
        new_tyre_age: int,
        new_position: int,
        new_gap_ahead: int,
        new_gap_behind: int,
        sc_active: bool,
    ) -> "State":
        """Return state after completing one racing lap (no pit stop)."""
        return State(
            lap=self.lap + 1,
            compound=self.compound,
            tyre_age=new_tyre_age,
            position=new_position,
            gap_ahead=new_gap_ahead,
            gap_behind=new_gap_behind,
            pit_used=self.pit_used,
            sc_active=sc_active,
        )


# ---------------------------------------------------------------------------
# Action availability
# ---------------------------------------------------------------------------

# Pit lane delta by circuit (seconds). Varies significantly:
#   Monaco ≈ 22s, Monza ≈ 23s, Silverstone ≈ 19s, Spa ≈ 18s
PIT_LANE_DELTA: dict[str, float] = {
    "monaco": 22.0,
    "monza": 23.4,
    "silverstone": 18.9,
    "spa": 18.2,
    "default": 20.5,
}


def available_actions(
    state: State, total_laps: int, circuit: str = "default"
) -> list[Action]:
    """
    Return the list of actions legally available from a given state.

    Constraints:
      - Cannot pit more than twice (fuel, tyre allocation limits)
      - Must have served at least one pit stop before the final lap
      - Gap too small behind → pit stop will likely cost a position
        (still allowed, but factored into the reward)
    """
    actions: list[Action] = []

    # STAY_OUT is always available unless it's the last lap and no stop taken
    if not (state.lap == total_laps and state.pit_used == 0):
        actions.append(Action.STAY_OUT)

    # Pit stop actions available if we haven't used both stops
    if state.pit_used < 2:
        actions.extend([Action.PIT_SOFT, Action.PIT_MEDIUM, Action.PIT_HARD])

    return actions


def action_to_compound(action: Action) -> Compound | None:
    """Map a pit action to the target compound. Returns None for STAY_OUT."""
    mapping = {
        Action.PIT_SOFT: Compound.SOFT,
        Action.PIT_MEDIUM: Compound.MEDIUM,
        Action.PIT_HARD: Compound.HARD,
    }
    return mapping.get(action)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

# Pit lane deltas are modelled as a PENALTY (positive = time lost).
# The reward R(s,a) is negative lap time — we maximise cumulative reward,
# which is equivalent to minimising total race time.


def lap_time_penalty(tyre_age: int, compound: Compound, sc_active: bool) -> float:
    """
    Compute the lap time DELTA relative to a theoretical perfect lap (zero penalty).

    This is a simplified functional form; tyre_model.py provides the full
    degradation model fitted to FastF1 telemetry data.

    Under safety car, lap times are controlled (≈ 40% slower delta),
    which significantly reduces the degradation penalty — one reason
    the safety car window is often the best time to pit.
    """
    # Base pace loss from tyre age (seconds above theoretical minimum)
    base_deg = {
        Compound.SOFT: 0.08,  # fastest deg rate
        Compound.MEDIUM: 0.05,
        Compound.HARD: 0.03,  # slowest deg rate
        Compound.INTER: 0.04,
        Compound.WET: 0.02,
    }
    deg_rate = base_deg[compound]

    # Degradation is approximately linear early, then accelerates ("cliff")
    # Full model in tyre_model.py; here we use a simple piecewise linear form
    cliff_lap = {
        Compound.SOFT: 18,
        Compound.MEDIUM: 28,
        Compound.HARD: 40,
        Compound.INTER: 25,
        Compound.WET: 35,
    }[compound]

    if tyre_age <= cliff_lap:
        penalty = deg_rate * tyre_age
    else:
        # Post-cliff: degradation accelerates by 3x
        penalty = deg_rate * cliff_lap + 3 * deg_rate * (tyre_age - cliff_lap)

    # Safety car: lap times are neutralised, so degradation penalty is reduced
    if sc_active:
        penalty *= 0.4

    return penalty


def reward(state: State, action: Action, circuit: str = "default") -> float:
    """
    Immediate reward for taking action in state.

    Convention: reward is NEGATIVE lap time penalty, so maximising
    cumulative reward = minimising total race time.

    For pit stop actions, we add the pit lane delta (a large one-off cost)
    and zero out the degradation (fresh tyres).
    """
    if action == Action.STAY_OUT:
        penalty = lap_time_penalty(state.tyre_age, state.compound, state.sc_active)
        return -penalty

    # Pit stop: one-off delta + fresh tyre advantage from next lap onwards
    delta = PIT_LANE_DELTA.get(circuit, PIT_LANE_DELTA["default"])
    # Fresh tyres → no degradation penalty on pit lap itself
    return -delta  # The benefit of fresh tyres accrues in subsequent states


# ---------------------------------------------------------------------------
# State space enumeration (for exact value iteration)
# ---------------------------------------------------------------------------


def enumerate_states(total_laps: int) -> Iterator[State]:
    """
    Yield every valid state in the MDP state space.

    Used by the solver to initialise the value table. With:
      laps=66, compounds=5, tyre_age=51, positions=20,
      gap_bins=4, gap_bins=4, pit_used=3, sc=2
    → 66 × 5 × 51 × 20 × 4 × 4 × 3 × 2 = ~32 million states

    In practice we use sparse representation and only visit reachable states.
    Aggressive pruning (e.g. tyre_age ≤ compound cliff + 10) reduces this by ~60%.
    """
    for lap in range(1, total_laps + 1):
        for compound in Compound:
            for tyre_age in range(0, 51):
                for position in range(1, 21):
                    for gap_ahead in range(N_GAP_BINS):
                        for gap_behind in range(N_GAP_BINS):
                            for pit_used in range(3):
                                for sc_active in (False, True):
                                    yield State(
                                        lap=lap,
                                        compound=compound,
                                        tyre_age=tyre_age,
                                        position=position,
                                        gap_ahead=gap_ahead,
                                        gap_behind=gap_behind,
                                        pit_used=pit_used,
                                        sc_active=sc_active,
                                    )
