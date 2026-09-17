"""
Tests for live/advisor.py
===========================

These exercise the live path end-to-end against the REAL PitStopSolver —
no mocking the model itself, since the whole point of live/advisor.py is
"don't reimplement the strategy logic, just feed it live inputs". They're
marked `slow` because solving the MDP, even for a tiny toy race, takes
real wall-clock time (see the cost note in live/advisor.py) — excluded
from a quick local `pytest -m "not slow"`, but this repo's CI does not
currently filter on that marker (only `not integration`), so it still
runs there. Use a very small total_laps to keep it reasonably fast.
"""

import pytest
from core.mdp import Action
from live.advisor import _ACTION_LABELS_ES, LiveAdvisor
from live.state import DriverSnapshot, RaceState

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def advisor():
    # use_cache=False: a fresh, disposable circuit name avoids clashing
    # with (or depending on) any cache left by other tests/runs.
    return LiveAdvisor(circuit="pytest_fixture_circuit", total_laps=3, use_cache=False)


def test_advise_returns_a_valid_action(advisor):
    race = RaceState(circuit="pytest_fixture_circuit", total_laps=3, current_lap=2)
    driver = DriverSnapshot(
        number="44",
        tla="HAM",
        position=3,
        interval_ahead_s=1.2,
        gap_behind_s=4.0,
        lap=2,
        compound="MEDIUM",
        tyre_age=20,
        stops=0,
    )
    race.drivers["44"] = driver

    advice = advisor.advise(driver, race)

    assert advice is not None
    assert isinstance(advice.action, Action)
    assert advice.action_label == _ACTION_LABELS_ES[advice.action]
    assert advice.expected_loss_s <= 0  # reward convention: time lost is <= 0
    assert all(lap_no >= race.current_lap for lap_no in advice.pit_window_laps)


def test_advise_returns_none_for_retired_or_unplaced_drivers(advisor):
    race = RaceState(circuit="pytest_fixture_circuit", total_laps=3, current_lap=2)

    retired = DriverSnapshot(number="1", position=1, retired=True)
    no_position = DriverSnapshot(number="16", position=None)

    assert advisor.advise(retired, race) is None
    assert advisor.advise(no_position, race) is None


def test_advise_tolerates_an_unknown_compound_string(advisor):
    race = RaceState(circuit="pytest_fixture_circuit", total_laps=3, current_lap=1)
    driver = DriverSnapshot(number="99", position=5, compound="SUPERSOFT_2027_PROTOTYPE")

    # Must not raise — falls back to a default compound mapping instead.
    advice = advisor.advise(driver, race)
    assert advice is not None


def test_advise_clamps_out_of_range_positions(advisor):
    race = RaceState(circuit="pytest_fixture_circuit", total_laps=3, current_lap=1)
    last_place = DriverSnapshot(number="88", position=20, compound="HARD", tyre_age=5)

    # N_POSITIONS is smaller than 20 by design (see live/advisor.py) — this
    # must clamp rather than KeyError.
    advice = advisor.advise(last_place, race)
    assert advice is not None
