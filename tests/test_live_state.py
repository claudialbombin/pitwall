"""
Tests for live/state.py
=========================

Testing philosophy: this module's whole job is to survive an unofficial,
undocumented feed without crashing a live GitHub Actions run. So beyond
"does it parse the fields we expect", we specifically test that malformed
or unexpected payloads degrade gracefully instead of raising.
"""

from live.state import RaceState, _to_float_gap


def test_to_float_gap_parses_plain_deltas():
    assert _to_float_gap("+1.234") == 1.234
    assert _to_float_gap(0.5) == 0.5


def test_to_float_gap_handles_leader_and_lapped_cars():
    assert _to_float_gap("") is None  # leader has no gap
    assert _to_float_gap(None) is None
    assert _to_float_gap("LAP 1") is None  # still on the leader's first lap
    assert _to_float_gap("1L") is None  # lapped — not a time gap
    assert _to_float_gap("garbage") is None  # unparseable, must not raise


def test_driver_list_and_timing_data_populate_a_driver():
    race = RaceState()
    race.apply(
        "DriverList",
        {"44": {"Tla": "HAM", "TeamName": "Mercedes", "TeamColour": "27F4D2"}},
    )
    race.apply(
        "TimingData",
        {
            "Lines": {
                "44": {
                    "Position": "2",
                    "GapToLeader": "+3.456",
                    "IntervalToPositionAhead": {"Value": "+1.200"},
                    "NumberOfLaps": 10,
                    "NumberOfPitStops": 1,
                    "InPit": False,
                }
            }
        },
    )

    d = race.drivers["44"]
    assert d.tla == "HAM"
    assert d.team == "Mercedes"
    assert d.team_colour == "27F4D2"
    assert d.position == 2
    assert d.gap_to_leader_s == 3.456
    assert d.interval_ahead_s == 1.2
    assert d.lap == 10
    assert d.stops == 1
    assert d.in_pit is False


def test_timing_app_data_uses_the_latest_stint():
    race = RaceState()
    race.apply(
        "TimingAppData",
        {
            "Lines": {
                "44": {
                    "Stints": {
                        "0": {"Compound": "SOFT", "TotalLaps": 10},
                        "1": {"Compound": "MEDIUM", "TotalLaps": 3},
                    }
                }
            }
        },
    )
    d = race.drivers["44"]
    assert d.compound == "MEDIUM"
    assert d.tyre_age == 3


def test_track_status_maps_to_safety_car_flag():
    race = RaceState()
    race.apply("TrackStatus", {"Status": "4"})
    assert race.sc_active is True
    assert race.to_json()["track_status_label"] == "safety_car"

    race.apply("TrackStatus", {"Status": "1"})
    assert race.sc_active is False
    assert race.to_json()["track_status_label"] == "green"


def test_lap_count_and_laps_remaining():
    race = RaceState()
    race.apply("LapCount", {"CurrentLap": 10, "TotalLaps": 51})
    assert race.current_lap == 10
    assert race.total_laps == 51
    assert race.laps_remaining == 41


def test_leaderboard_derives_gap_behind_from_the_next_car():
    race = RaceState()
    race.apply(
        "TimingData",
        {
            "Lines": {
                "1": {"Position": "1"},
                "44": {"Position": "2", "IntervalToPositionAhead": {"Value": "+0.800"}},
                "16": {"Position": "3", "IntervalToPositionAhead": {"Value": "+1.500"}},
            }
        },
    )
    board = {d.number: d for d in race.leaderboard()}
    # Gap behind P1 is P2's interval to the car ahead (P1).
    assert board["1"].gap_behind_s == 0.8
    assert board["44"].gap_behind_s == 1.5
    assert board["16"].gap_behind_s is None  # nobody behind the last car we know of


def test_apply_never_raises_on_malformed_payloads():
    race = RaceState()
    # Each of these is a plausible way the feed could hand us something we
    # don't expect — none of them should raise.
    race.apply("TimingData", {"Lines": {"44": "not-a-dict"}})
    race.apply("TimingData", {"Lines": None})
    race.apply("TimingAppData", {"Lines": {"44": {"Stints": {"x": "oops"}}}})
    race.apply("TrackStatus", {})
    race.apply("LapCount", {"CurrentLap": "not-a-number"})
    race.apply("UnknownTopic", {"anything": "goes"})
    race.apply("Heartbeat", None)
    # Reaching this line at all is the test passing.
    assert True
