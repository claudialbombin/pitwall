"""
Tests for core/mdp.py
======================

Testing philosophy: we test MATHEMATICAL PROPERTIES, not just outputs.
A function that returns a number is not tested by checking that the number
matches a hardcoded value — that just tests that we didn't change anything.
We test that the function satisfies the mathematical constraints it claims to satisfy.

For example:
  - Transition probabilities must sum to 1.0 (law of total probability)
  - Rewards must be non-positive (time is always lost, never gained)
  - Available actions must always include at least one legal choice
  - State keys must be hashable (required for the value table)
  - Cliff lap formula must be consistent with Weibull parameters

This approach catches bugs that produce plausible-looking wrong outputs,
which are the hardest bugs to find in quantitative code.

Author: Claudia Maria Lopez Bombin
GitHub: https://github.com/claudialbombin/pitwall
"""

import math
import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from mdp import (
    Action, Compound, State,
    available_actions, action_to_compound,
    reward, lap_time_penalty, discretise_gap,
    enumerate_states,
    GAP_BINS, N_GAP_BINS, PIT_LANE_DELTA,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def standard_state():
    """A typical mid-race state — used across many tests."""
    return State(
        lap=20, compound=Compound.MEDIUM, tyre_age=10,
        position=5, gap_ahead=1, gap_behind=1,
        pit_used=0, sc_active=False,
    )


@pytest.fixture
def fresh_tyre_state():
    return State(
        lap=1, compound=Compound.SOFT, tyre_age=0,
        position=1, gap_ahead=3, gap_behind=3,
        pit_used=0, sc_active=False,
    )


@pytest.fixture
def end_of_race_state():
    return State(
        lap=51, compound=Compound.HARD, tyre_age=30,
        position=3, gap_ahead=0, gap_behind=0,
        pit_used=1, sc_active=False,
    )


# ---------------------------------------------------------------------------
# State construction and validation
# ---------------------------------------------------------------------------

class TestStateConstruction:

    def test_valid_state_constructs(self, standard_state):
        """Standard state should construct without raising."""
        assert standard_state.lap == 20
        assert standard_state.compound == Compound.MEDIUM
        assert standard_state.tyre_age == 10

    def test_state_is_hashable(self, standard_state):
        """State must be hashable to serve as a dict key in the value table."""
        d = {standard_state: 42.0}
        assert d[standard_state] == 42.0

    def test_state_equality(self):
        """Two states with identical fields must be equal (frozen dataclass)."""
        s1 = State(lap=5, compound=Compound.SOFT, tyre_age=3,
                   position=10, gap_ahead=2, gap_behind=1, pit_used=0, sc_active=False)
        s2 = State(lap=5, compound=Compound.SOFT, tyre_age=3,
                   position=10, gap_ahead=2, gap_behind=1, pit_used=0, sc_active=False)
        assert s1 == s2

    def test_state_inequality_on_single_field(self, standard_state):
        """Changing any single field must produce an inequal state."""
        import dataclasses
        for field_name, new_val in [
            ("lap", 21), ("tyre_age", 11), ("position", 6),
            ("sc_active", True), ("pit_used", 1),
        ]:
            other = dataclasses.replace(standard_state, **{field_name: new_val})
            assert other != standard_state

    def test_invalid_lap_raises(self):
        with pytest.raises(AssertionError):
            State(lap=0, compound=Compound.MEDIUM, tyre_age=0,
                  position=1, gap_ahead=0, gap_behind=0, pit_used=0, sc_active=False)

    def test_invalid_position_raises(self):
        with pytest.raises(AssertionError):
            State(lap=1, compound=Compound.MEDIUM, tyre_age=0,
                  position=21, gap_ahead=0, gap_behind=0, pit_used=0, sc_active=False)

    def test_invalid_tyre_age_raises(self):
        with pytest.raises(AssertionError):
            State(lap=1, compound=Compound.MEDIUM, tyre_age=61,
                  position=1, gap_ahead=0, gap_behind=0, pit_used=0, sc_active=False)

    def test_invalid_pit_used_raises(self):
        with pytest.raises(AssertionError):
            State(lap=1, compound=Compound.MEDIUM, tyre_age=0,
                  position=1, gap_ahead=0, gap_behind=0, pit_used=3, sc_active=False)

    def test_with_pit_resets_tyre_age(self, standard_state):
        """After a pit stop, tyre age must reset to 0."""
        new_state = standard_state.with_pit(Compound.HARD)
        assert new_state.tyre_age == 0

    def test_with_pit_increments_pit_count(self, standard_state):
        """pit_used must increase by 1 after pitting."""
        new_state = standard_state.with_pit(Compound.HARD)
        assert new_state.pit_used == standard_state.pit_used + 1

    def test_with_pit_changes_compound(self, standard_state):
        """Compound must change to the new compound after pitting."""
        new_state = standard_state.with_pit(Compound.HARD)
        assert new_state.compound == Compound.HARD

    def test_with_pit_position_does_not_improve(self, standard_state):
        """Pitting cannot improve position (cars overtake during pit stop)."""
        new_state = standard_state.with_pit(Compound.SOFT)
        assert new_state.position >= standard_state.position

    def test_with_lap_completed_increments_lap(self, standard_state):
        """Completing a lap must advance the lap counter by exactly 1."""
        new_state = standard_state.with_lap_completed(
            new_tyre_age=standard_state.tyre_age + 1,
            new_position=standard_state.position,
            new_gap_ahead=standard_state.gap_ahead,
            new_gap_behind=standard_state.gap_behind,
            sc_active=False,
        )
        assert new_state.lap == standard_state.lap + 1

    def test_with_lap_completed_keeps_compound(self, standard_state):
        """Staying out must not change the compound."""
        new_state = standard_state.with_lap_completed(
            new_tyre_age=standard_state.tyre_age + 1,
            new_position=standard_state.position,
            new_gap_ahead=standard_state.gap_ahead,
            new_gap_behind=standard_state.gap_behind,
            sc_active=False,
        )
        assert new_state.compound == standard_state.compound


# ---------------------------------------------------------------------------
# Gap discretisation
# ---------------------------------------------------------------------------

class TestGapDiscretisation:

    @pytest.mark.parametrize("gap, expected_bin", [
        (0.0,  0),   # exact left edge of bin 0
        (0.5,  0),   # inside bin 0
        (0.99, 0),   # just below bin 1
        (1.0,  1),   # exact left edge of bin 1
        (2.0,  1),   # inside bin 1
        (3.0,  2),   # bin 2
        (4.9,  2),   # just below bin 3
        (5.0,  3),   # bin 3
        (100., 3),   # far out — should be last bin
    ])
    def test_gap_bins(self, gap, expected_bin):
        assert discretise_gap(gap) == expected_bin

    def test_all_gaps_in_valid_range(self):
        """All discretised gaps must be in [0, N_GAP_BINS)."""
        for gap in np.linspace(0, 20, 500):
            b = discretise_gap(gap)
            assert 0 <= b < N_GAP_BINS, f"gap={gap} → bin={b} out of range"

    def test_gap_bins_is_monotone(self):
        """Gap bins must be non-decreasing as gap increases."""
        prev_bin = 0
        for gap in np.linspace(0, 20, 1000):
            b = discretise_gap(gap)
            assert b >= prev_bin, f"Non-monotone at gap={gap}: {b} < {prev_bin}"
            prev_bin = b


# ---------------------------------------------------------------------------
# Available actions
# ---------------------------------------------------------------------------

class TestAvailableActions:

    def test_stay_out_available_in_normal_state(self, standard_state):
        actions = available_actions(standard_state, total_laps=52)
        assert Action.STAY_OUT in actions

    def test_pit_actions_available_when_pit_used_zero(self, standard_state):
        actions = available_actions(standard_state, total_laps=52)
        assert Action.PIT_SOFT   in actions
        assert Action.PIT_MEDIUM in actions
        assert Action.PIT_HARD   in actions

    def test_no_pit_when_both_stops_used(self):
        """After 2 pit stops, no further pit actions should be available."""
        s = State(lap=30, compound=Compound.HARD, tyre_age=10,
                  position=5, gap_ahead=1, gap_behind=1,
                  pit_used=2, sc_active=False)
        actions = available_actions(s, total_laps=52)
        assert Action.PIT_SOFT   not in actions
        assert Action.PIT_MEDIUM not in actions
        assert Action.PIT_HARD   not in actions

    def test_must_pit_on_final_lap_if_no_stop_used(self):
        """On the last lap with no stop taken, STAY_OUT is invalid (mandatory stop rule)."""
        s = State(lap=52, compound=Compound.MEDIUM, tyre_age=20,
                  position=5, gap_ahead=1, gap_behind=1,
                  pit_used=0, sc_active=False)
        actions = available_actions(s, total_laps=52)
        assert Action.STAY_OUT not in actions

    def test_stay_out_on_final_lap_when_stop_already_done(self):
        """On the last lap with 1 stop taken, STAY_OUT is valid."""
        s = State(lap=52, compound=Compound.HARD, tyre_age=15,
                  position=5, gap_ahead=1, gap_behind=1,
                  pit_used=1, sc_active=False)
        actions = available_actions(s, total_laps=52)
        assert Action.STAY_OUT in actions

    def test_always_at_least_one_action(self):
        """The action set must never be empty — the solver must always have a choice."""
        for lap in [1, 10, 25, 51, 52]:
            for pit_used in [0, 1, 2]:
                s = State(lap=lap, compound=Compound.MEDIUM, tyre_age=10,
                          position=5, gap_ahead=1, gap_behind=1,
                          pit_used=pit_used, sc_active=False)
                actions = available_actions(s, total_laps=52)
                assert len(actions) > 0, f"Empty action set at lap={lap}, pit_used={pit_used}"

    def test_action_to_compound_mapping(self):
        """Each pit action must map to the correct compound."""
        assert action_to_compound(Action.PIT_SOFT)   == Compound.SOFT
        assert action_to_compound(Action.PIT_MEDIUM) == Compound.MEDIUM
        assert action_to_compound(Action.PIT_HARD)   == Compound.HARD
        assert action_to_compound(Action.STAY_OUT)   is None


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

class TestRewardFunction:

    def test_reward_is_non_positive(self, standard_state):
        """Rewards must be ≤ 0: time is always lost, never gained."""
        for action in [Action.STAY_OUT, Action.PIT_SOFT,
                       Action.PIT_MEDIUM, Action.PIT_HARD]:
            r = reward(standard_state, action)
            assert r <= 0.0, f"Positive reward {r} for action {action}"

    def test_pit_reward_more_negative_than_stay_out(self, fresh_tyre_state):
        """
        On fresh tyres with zero degradation, pitting is always worse than staying out:
        we'd pay the pit lane delta for zero benefit.
        """
        r_stay = reward(fresh_tyre_state, Action.STAY_OUT)
        r_pit  = reward(fresh_tyre_state, Action.PIT_MEDIUM)
        assert r_stay > r_pit, (
            f"Pitting on fresh tyres should cost more than staying out. "
            f"r_stay={r_stay:.3f}, r_pit={r_pit:.3f}"
        )

    def test_pit_reward_includes_pit_lane_delta(self, standard_state):
        """The pit action reward must reflect the pit lane delta as a cost."""
        r_pit = reward(standard_state, Action.PIT_MEDIUM, circuit="silverstone")
        expected = -PIT_LANE_DELTA["silverstone"]
        assert abs(r_pit - expected) < 1e-9, (
            f"Pit reward {r_pit:.3f} != expected {expected:.3f}"
        )

    def test_stay_out_reward_worsens_with_tyre_age(self):
        """
        Higher tyre age must produce a more negative (worse) stay-out reward.
        Degradation penalty increases monotonically up to the cliff.
        """
        rewards = []
        for age in range(0, 25):
            s = State(lap=20, compound=Compound.SOFT, tyre_age=age,
                      position=5, gap_ahead=1, gap_behind=1,
                      pit_used=0, sc_active=False)
            rewards.append(reward(s, Action.STAY_OUT))
        # Rewards should be monotonically non-increasing
        for i in range(len(rewards) - 1):
            assert rewards[i] >= rewards[i + 1], (
                f"Reward increased from age {i} to {i+1}: "
                f"{rewards[i]:.4f} → {rewards[i+1]:.4f}"
            )

    def test_sc_reduces_lap_time_penalty(self):
        """
        Under safety car, lap times are controlled (slower).
        The degradation penalty should be reduced (less tyre stress).
        """
        s_green = State(lap=20, compound=Compound.SOFT, tyre_age=15,
                        position=5, gap_ahead=1, gap_behind=1,
                        pit_used=0, sc_active=False)
        s_sc    = State(lap=20, compound=Compound.SOFT, tyre_age=15,
                        position=5, gap_ahead=1, gap_behind=1,
                        pit_used=0, sc_active=True)
        r_green = reward(s_green, Action.STAY_OUT)
        r_sc    = reward(s_sc,    Action.STAY_OUT)
        # SC reward is closer to 0 (less penalty)
        assert r_sc > r_green, (
            f"SC should reduce penalty. r_green={r_green:.4f}, r_sc={r_sc:.4f}"
        )

    def test_reward_circuit_independence_for_stay_out(self, standard_state):
        """STAY_OUT reward should not depend on circuit (pit delta not involved)."""
        r1 = reward(standard_state, Action.STAY_OUT, circuit="monaco")
        r2 = reward(standard_state, Action.STAY_OUT, circuit="monza")
        assert r1 == r2

    def test_pit_reward_varies_by_circuit(self, standard_state):
        """PIT reward must differ across circuits with different pit lane deltas."""
        r_monaco = reward(standard_state, Action.PIT_MEDIUM, circuit="monaco")
        r_monza  = reward(standard_state, Action.PIT_MEDIUM, circuit="monza")
        assert r_monaco != r_monza


# ---------------------------------------------------------------------------
# Lap time penalty model
# ---------------------------------------------------------------------------

class TestLapTimePenalty:

    def test_fresh_tyre_has_zero_penalty(self):
        """Brand new tyre (age=0) must have zero degradation penalty."""
        for compound in Compound:
            p = lap_time_penalty(0, compound, sc_active=False)
            assert p == pytest.approx(0.0, abs=1e-9), (
                f"Non-zero penalty for fresh {compound.name}: {p:.6f}"
            )

    def test_penalty_is_non_negative(self):
        """Degradation penalty must be ≥ 0 for all tyre ages and compounds."""
        for compound in [Compound.SOFT, Compound.MEDIUM, Compound.HARD]:
            for age in range(0, 51):
                p = lap_time_penalty(age, compound, sc_active=False)
                assert p >= 0.0, f"Negative penalty for {compound.name} age {age}: {p:.4f}"

    def test_penalty_increases_with_age(self):
        """Penalty must be non-decreasing with tyre age before the cliff."""
        for compound in [Compound.SOFT, Compound.MEDIUM, Compound.HARD]:
            prev = 0.0
            for age in range(0, 40):
                curr = lap_time_penalty(age, compound, sc_active=False)
                assert curr >= prev - 1e-9, (
                    f"Penalty decreased from age {age-1} to {age} for {compound.name}: "
                    f"{prev:.4f} → {curr:.4f}"
                )
                prev = curr

    def test_sc_reduces_penalty(self):
        """Safety car must reduce the lap time penalty."""
        for compound in [Compound.SOFT, Compound.MEDIUM, Compound.HARD]:
            for age in [5, 15, 25]:
                p_green = lap_time_penalty(age, compound, sc_active=False)
                p_sc    = lap_time_penalty(age, compound, sc_active=True)
                if p_green > 0:
                    assert p_sc < p_green, (
                        f"SC did not reduce penalty for {compound.name} age {age}: "
                        f"green={p_green:.4f}, sc={p_sc:.4f}"
                    )

    def test_soft_degrades_faster_than_hard(self):
        """SOFT compound must degrade faster than HARD at all tyre ages > 0."""
        for age in range(1, 35):
            p_soft = lap_time_penalty(age, Compound.SOFT,   sc_active=False)
            p_hard = lap_time_penalty(age, Compound.HARD,   sc_active=False)
            assert p_soft >= p_hard, (
                f"SOFT not worse than HARD at age {age}: soft={p_soft:.4f}, hard={p_hard:.4f}"
            )

    def test_compound_ordering(self):
        """Degradation rate must follow: SOFT ≥ MEDIUM ≥ HARD."""
        for age in range(5, 30):
            p_soft   = lap_time_penalty(age, Compound.SOFT,   sc_active=False)
            p_medium = lap_time_penalty(age, Compound.MEDIUM, sc_active=False)
            p_hard   = lap_time_penalty(age, Compound.HARD,   sc_active=False)
            assert p_soft   >= p_medium - 1e-9, f"SOFT < MEDIUM at age {age}"
            assert p_medium >= p_hard   - 1e-9, f"MEDIUM < HARD at age {age}"


# ---------------------------------------------------------------------------
# State space enumeration
# ---------------------------------------------------------------------------

class TestStateEnumeration:

    def test_enumeration_produces_states(self):
        """enumerate_states must yield at least some valid states."""
        states = list(enumerate_states(total_laps=5))
        assert len(states) > 0

    def test_all_enumerated_states_are_valid(self):
        """Every enumerated state must pass the State validation checks."""
        for state in enumerate_states(total_laps=5):
            assert 1 <= state.lap <= 5
            assert 1 <= state.position <= 20
            assert 0 <= state.tyre_age <= 60
            assert 0 <= state.pit_used <= 2

    def test_enumeration_covers_all_laps(self):
        """State enumeration must include every lap from 1 to total_laps."""
        total_laps = 5
        states = list(enumerate_states(total_laps=total_laps))
        laps_seen = {s.lap for s in states}
        assert laps_seen == set(range(1, total_laps + 1))

    def test_enumeration_covers_all_compounds(self):
        """All compounds must appear in the enumerated states."""
        states = list(enumerate_states(total_laps=3))
        compounds_seen = {s.compound for s in states}
        assert Compound.SOFT   in compounds_seen
        assert Compound.MEDIUM in compounds_seen
        assert Compound.HARD   in compounds_seen

    def test_all_enumerated_states_hashable(self):
        """Every enumerated state must be usable as a dict key."""
        table = {}
        for state in enumerate_states(total_laps=3):
            table[state] = 0.0
        assert len(table) > 0


# ---------------------------------------------------------------------------
# Pit lane delta constants
# ---------------------------------------------------------------------------

class TestPitLaneDelta:

    def test_all_deltas_positive(self):
        for circuit, delta in PIT_LANE_DELTA.items():
            assert delta > 0, f"Non-positive pit delta for {circuit}: {delta}"

    def test_monaco_longer_than_monza(self):
        """Monaco pit lane is longer than Monza's — must reflect in the model."""
        assert PIT_LANE_DELTA["monaco"] > PIT_LANE_DELTA["monza"], (
            "Monaco pit delta should exceed Monza. "
            f"monaco={PIT_LANE_DELTA['monaco']}, monza={PIT_LANE_DELTA['monza']}"
        )

    def test_default_delta_is_reasonable(self):
        """Default pit delta should be in the range [15, 30] seconds."""
        d = PIT_LANE_DELTA["default"]
        assert 15 <= d <= 30, f"Default pit delta {d}s outside realistic range"