"""
live/state.py
=============
Turns the raw topic messages from live/signalr_client.py into one flat,
JSON-serialisable snapshot of "what is happening in the race right now":
per driver — position, gaps, tyre, lap, pit stops — plus session-wide
context (lap count, track status / safety car).

Field names below follow the community-documented shape of F1's live
timing feed (the same one FastF1 is built on — track status codes here
are taken directly from fastf1.livetiming.data's own mapping). Every
accessor is defensive (.get() with fallbacks) on purpose: this feed is
unofficial, field shapes have drifted slightly between seasons in the
wild, and one unexpected message must never crash the listener mid-race.
When something looks off, RaceState logs a warning and moves on instead
of raising — so a GitHub Actions run keeps going, and the parsing can be
patched before the next session instead of losing a whole race's data.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("live.state")

# TrackStatus.Status codes — confirmed against FastF1's own mapping
# (fastf1.livetiming.data._track_status_mapping), since FastF1 is already
# a dependency of this repo and is the most authoritative free reference
# for this undocumented feed.
TRACK_STATUS_SC = {"4", "6"}  # Safety Car deployed, Virtual Safety Car deployed
TRACK_STATUS_LABELS = {
    "1": "green",
    "2": "yellow",
    "4": "safety_car",
    "5": "red",
    "6": "vsc",
    "7": "vsc_ending",
}


def _to_float_gap(value: Any) -> float | None:
    """
    Parse a gap/interval value like '+1.234', '1L' (lapped), 'LAP 1'
    (still on the leader's first lap), or '' (leader, no gap) into
    seconds — or None when it doesn't represent a plain time gap.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.upper().startswith("LAP"):
        return None
    if text.upper().endswith("L"):  # e.g. "1L" = one lap behind, not a time gap
        return None
    try:
        return float(text.replace("+", ""))
    except ValueError:
        return None


@dataclass
class DriverSnapshot:
    number: str
    tla: str = "???"
    team: str = ""
    team_colour: str = "888880"  # hex, no '#' — matches DriverList.TeamColour
    position: int | None = None
    gap_to_leader_s: float | None = None
    interval_ahead_s: float | None = None
    gap_behind_s: float | None = None  # derived, see RaceState.leaderboard()
    lap: int = 0
    compound: str = "UNKNOWN"
    tyre_age: int = 0
    stops: int = 0
    in_pit: bool = False
    retired: bool = False

    def to_json(self) -> dict:
        return {
            "number": self.number,
            "tla": self.tla,
            "team": self.team,
            "team_colour": self.team_colour,
            "position": self.position,
            "gap_to_leader_s": self.gap_to_leader_s,
            "interval_ahead_s": self.interval_ahead_s,
            "gap_behind_s": self.gap_behind_s,
            "lap": self.lap,
            "compound": self.compound,
            "tyre_age": self.tyre_age,
            "stops": self.stops,
            "in_pit": self.in_pit,
            "retired": self.retired,
        }


@dataclass
class RaceState:
    circuit: str = "default"
    display_name: str = ""
    session_type: str = ""
    current_lap: int = 0
    total_laps: int = 0
    track_status: str = "1"
    drivers: dict = field(default_factory=dict)
    last_update_monotonic: float = field(default_factory=time.monotonic)

    # -- derived ------------------------------------------------------
    @property
    def sc_active(self) -> bool:
        return self.track_status in TRACK_STATUS_SC

    @property
    def laps_remaining(self) -> int:
        return max(0, self.total_laps - self.current_lap)

    def _driver(self, number: str) -> DriverSnapshot:
        return self.drivers.setdefault(number, DriverSnapshot(number=number))

    # -- ingestion ------------------------------------------------------
    def apply(self, topic: str, payload: Any) -> None:
        if payload is None:
            return
        try:
            handler = getattr(self, f"_apply_{topic.lower()}", None)
            if handler is not None:
                handler(payload)
                self.last_update_monotonic = time.monotonic()
        except Exception:
            log.warning("Failed to apply topic %s", topic, exc_info=True)

    def _apply_driverlist(self, payload: dict) -> None:
        for number, info in payload.items():
            if not isinstance(info, dict):
                continue
            d = self._driver(number)
            d.tla = info.get("Tla", d.tla)
            d.team = info.get("TeamName", info.get("Team", d.team))
            colour = info.get("TeamColour")
            if colour:
                d.team_colour = str(colour).lstrip("#")

    def _apply_timingdata(self, payload: dict) -> None:
        for number, line in payload.get("Lines", {}).items():
            if not isinstance(line, dict):
                continue
            d = self._driver(number)
            pos = line.get("Position")
            if pos is not None:
                with contextlib.suppress(TypeError, ValueError):
                    d.position = int(pos)
            if "GapToLeader" in line:
                d.gap_to_leader_s = _to_float_gap(line.get("GapToLeader"))
            interval = line.get("IntervalToPositionAhead")
            if isinstance(interval, dict):
                d.interval_ahead_s = _to_float_gap(interval.get("Value"))
            elif interval is not None:
                d.interval_ahead_s = _to_float_gap(interval)
            if "NumberOfLaps" in line:
                with contextlib.suppress(TypeError, ValueError):
                    d.lap = int(line["NumberOfLaps"])
            if "NumberOfPitStops" in line:
                with contextlib.suppress(TypeError, ValueError):
                    d.stops = int(line["NumberOfPitStops"])
            if "InPit" in line:
                d.in_pit = bool(line["InPit"])
            if "Retired" in line or "Stopped" in line:
                d.retired = bool(line.get("Retired") or line.get("Stopped"))

    def _apply_timingappdata(self, payload: dict) -> None:
        for number, line in payload.get("Lines", {}).items():
            if not isinstance(line, dict):
                continue
            d = self._driver(number)
            stints = line.get("Stints")
            if not stints:
                continue
            # Stints is a dict keyed by stint index ("0", "1", ...) —
            # the highest index is the stint currently being run.
            try:
                current_key = max(stints.keys(), key=lambda k: int(k))
            except ValueError:
                continue
            current = stints[current_key]
            if not isinstance(current, dict):
                continue
            if current.get("Compound"):
                d.compound = str(current["Compound"]).upper()
            age = current.get("TotalLaps")
            if age is not None:
                with contextlib.suppress(TypeError, ValueError):
                    d.tyre_age = int(age)

    def _apply_trackstatus(self, payload: dict) -> None:
        status = payload.get("Status")
        if status is not None:
            self.track_status = str(status)

    def _apply_lapcount(self, payload: dict) -> None:
        if "CurrentLap" in payload:
            with contextlib.suppress(TypeError, ValueError):
                self.current_lap = int(payload["CurrentLap"])
        if "TotalLaps" in payload:
            with contextlib.suppress(TypeError, ValueError):
                self.total_laps = int(payload["TotalLaps"])

    def _apply_sessioninfo(self, payload: dict) -> None:
        meeting = payload.get("Meeting", {})
        if isinstance(meeting, dict) and meeting.get("Name"):
            self.display_name = meeting["Name"]
        self.session_type = payload.get("Type") or payload.get("Name") or self.session_type

    # -- output ------------------------------------------------------
    def leaderboard(self) -> list:
        """
        Drivers ranked by position, with gap_behind_s filled in — the feed
        only gives each driver its OWN interval to the car ahead, so the
        gap behind driver N is simply driver N+1's interval_ahead_s.
        """
        ranked = sorted(
            (d for d in self.drivers.values() if d.position is not None),
            key=lambda d: d.position,
        )
        for i, d in enumerate(ranked):
            d.gap_behind_s = ranked[i + 1].interval_ahead_s if i + 1 < len(ranked) else None
        return ranked

    def to_json(self) -> dict:
        return {
            "circuit": self.circuit,
            "display_name": self.display_name,
            "session_type": self.session_type,
            "current_lap": self.current_lap,
            "total_laps": self.total_laps,
            "laps_remaining": self.laps_remaining,
            "track_status": self.track_status,
            "track_status_label": TRACK_STATUS_LABELS.get(self.track_status, "unknown"),
            "sc_active": self.sc_active,
            "drivers": [d.to_json() for d in self.leaderboard()],
        }
