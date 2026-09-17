"""
live/advisor.py
================
The bridge between "what the live feed says is happening" and "what your
actual academic model says you should do about it".

Nothing here reimplements strategy logic — it's a thin adapter around
core.mdp / core.tyre_model / core.safety_car / core.solver, the exact same
modules the notebooks and the exported results/optimal_policies.json use
for the offline simulator. The live path just builds a core.mdp.State from
a live DriverSnapshot and calls PitStopSolver.query() instead of querying
with slider values.

COST NOTE: exact backward induction over the full state space (lap x
compound x tyre_age x position x gap_ahead x gap_behind x pit_used x
sc_active) takes real wall-clock time in pure Python — potentially a few
minutes for a full 20-car field. It only depends on the circuit + race
distance, not on anything that changes lap to lap, so it is solved ONCE
per (circuit, total_laps) and cached to disk under
results/live_policy_cache/ — a whole race weekend (FP1..Race, same
circuit) only pays that cost the first time live/run.py starts for it.

FALLBACK FOR UNMODELLED CIRCUITS: core/race_sim.py only hand-tunes 4
circuits (monaco, monza, silverstone, spa). Every other circuit — i.e.
most of the calendar — falls back to the generic "default" pit lane delta
and default tyre/safety-car parameters, exactly the same fallback already
accepted by .github/workflows/deploy.yml when FastF1 cache data isn't
available. This is a known simplification, not a bug: circuit-specific
tuning is future work, not something the live pipeline needed to invent.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

from core.mdp import PIT_LANE_DELTA, Action, Compound, State, discretise_gap
from core.safety_car import SafetyCarModel
from core.solver import PitStopSolver, SolverConfig, pit_window
from core.tyre_model import TyreModel

from live.state import DriverSnapshot, RaceState

log = logging.getLogger("live.advisor")

CACHE_DIR = Path(__file__).resolve().parent.parent / "results" / "live_policy_cache"

# Real F1 compound names -> the 5-way enum used across core/.
_COMPOUND_MAP = {
    "SOFT": Compound.SOFT,
    "MEDIUM": Compound.MEDIUM,
    "HARD": Compound.HARD,
    "INTERMEDIATE": Compound.INTER,
    "INTER": Compound.INTER,
    "WET": Compound.WET,
}

_ACTION_LABELS_ES = {
    Action.STAY_OUT: "Quedarse fuera",
    Action.PIT_SOFT: "Entrar a boxes — blandos",
    Action.PIT_MEDIUM: "Entrar a boxes — medios",
    Action.PIT_HARD: "Entrar a boxes — duros",
}

# n_positions used when pre-solving. Measured on a plain laptop core:
# ~18,000 states/s at n_positions=2, scaling ~linearly with n_positions
# (the state space is n_positions x 51 tyre ages x 5 compounds x 4x4 gap
# bins x 3 pit counts x 2 SC states, per lap). n_positions=10 lands a full
# 70-lap race around 15 minutes to solve — comfortably inside the
# scheduler's pre-session buffer (see live/schedule.py) and only paid
# once per circuit thanks to the disk cache below. deploy.yml's CI
# pipeline uses n_positions=5 to stay fast on every push; 10 buys more
# resolution across the midfield since here we're not racing a CI timeout.
# Positions beyond this are clamped to the last modelled bucket — a
# documented simplification, same spirit as the existing default-circuit
# fallback.
N_POSITIONS = 10


def _cache_key(circuit: str, total_laps: int) -> str:
    raw = f"{circuit}|{total_laps}|{N_POSITIONS}|v1"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


@dataclass
class Advice:
    driver_number: str
    action: Action
    action_label: str
    expected_loss_s: float
    pit_window_laps: list

    def to_json(self) -> dict:
        return {
            "action": self.action.name,
            "action_label": self.action_label,
            "expected_loss_s": round(self.expected_loss_s, 2),
            "pit_window_laps": self.pit_window_laps,
        }


class LiveAdvisor:
    """One instance per (circuit, total_laps) — solve once, query many times."""

    def __init__(self, circuit: str, total_laps: int, use_cache: bool = True):
        self.circuit = circuit
        self.total_laps = max(1, total_laps)
        self.solver: PitStopSolver | None = None
        self._load_or_solve(use_cache)

    def _load_or_solve(self, use_cache: bool) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"{_cache_key(self.circuit, self.total_laps)}.pkl"

        tyre = TyreModel()  # default parameters — see module docstring above
        sc = SafetyCarModel(circuit=self.circuit, total_laps=self.total_laps)
        cfg = SolverConfig(
            total_laps=self.total_laps,
            circuit=self.circuit if self.circuit in PIT_LANE_DELTA else "default",
            n_positions=N_POSITIONS,
            verbose=True,
        )
        solver = PitStopSolver(cfg, tyre, sc)

        if use_cache and cache_file.exists():
            log.info("Loading cached policy for %s (%d laps)", self.circuit, self.total_laps)
            with cache_file.open("rb") as f:
                solver.pi, solver.V = pickle.load(f)
        else:
            log.info(
                "Solving pit stop MDP for %s (%d laps, n_positions=%d) — this can "
                "take a few minutes the first time for a given circuit.",
                self.circuit,
                self.total_laps,
                N_POSITIONS,
            )
            solver.solve()
            if use_cache:
                with cache_file.open("wb") as f:
                    pickle.dump((solver.pi, solver.V), f)

        self.solver = solver

    # ------------------------------------------------------------------
    def advise(self, driver: DriverSnapshot, race: RaceState) -> Advice | None:
        if self.solver is None or driver.position is None or driver.retired:
            return None

        compound = _COMPOUND_MAP.get(driver.compound, Compound.MEDIUM)
        position = max(1, min(N_POSITIONS, driver.position))
        gap_ahead = discretise_gap(
            driver.interval_ahead_s if driver.interval_ahead_s is not None else 99.0
        )
        gap_behind = discretise_gap(
            driver.gap_behind_s if driver.gap_behind_s is not None else 99.0
        )
        lap = max(1, min(self.total_laps, race.current_lap or 1))
        pit_used = max(0, min(2, driver.stops))

        state = State(
            lap=lap,
            compound=compound,
            tyre_age=max(0, min(60, driver.tyre_age)),
            position=position,
            gap_ahead=gap_ahead,
            gap_behind=gap_behind,
            pit_used=pit_used,
            sc_active=race.sc_active,
        )

        action, value = self.solver.query(state)

        window = pit_window(
            self.solver.pi,
            compound=compound,
            position=position,
            gap_ahead=gap_ahead,
            gap_behind=gap_behind,
            pit_used=pit_used,
            sc_active=race.sc_active,
            total_laps=self.total_laps,
        )
        window = [lap_no for lap_no in window if lap_no >= lap]  # only what's left to come

        return Advice(
            driver_number=driver.number,
            action=action,
            action_label=_ACTION_LABELS_ES[action],
            expected_loss_s=value,
            pit_window_laps=window[:5],
        )
