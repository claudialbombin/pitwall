"""
live/schedule.py
=================
A hand-maintained calendar of 2026 F1 session start times, and the logic
that decides "is there a session live right now?".

WHY HARDCODED INSTEAD OF FETCHED FROM AN API:
The calendar is public, published months in advance, and barely ever
changes once the season starts. Hardcoding it means one less moving part
(one less network call that can fail inside a scheduled job) and it costs
nothing to maintain: update this file once a year, in about five minutes,
using any public F1 calendar site. If you'd rather fetch it, the free
Jolpica-F1 API (Ergast's spiritual successor) exposes a schedule endpoint —
left as a documented option, not wired in, to keep the live pipeline's
only network dependency being the timing feed itself.

HOW THIS FEEDS THE REST OF THE PIPELINE:
`is_live_now()` is deliberately generous (buffers before/after the nominal
start time) because the real source of truth is the live feed itself, not
this file. If the calendar is a few minutes off, or a session over/under-
runs, `live/run.py` just finds nothing streaming yet/anymore and retries or
exits quietly — it never trusts this file blindly for anything except
"is it worth trying to connect right now" and "what circuit + how many laps
should I pre-solve the strategy model for".

UPDATING FOR A NEW SEASON:
Replace the SEASON block below. For each Grand Prix weekend, one call to
`_gp(circuit_slug, display_name, total_laps, {session_code: utc_datetime})`.
`circuit_slug` should match a key in `core.mdp.PIT_LANE_DELTA` when one
exists (monaco/monza/silverstone/spa) so the live advisor picks up the
real pit lane delta instead of the generic default — for every other
circuit the generic default is used, which is a known, accepted
simplification already used elsewhere in this repo (see deploy.yml).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

UTC = timezone.utc

# How early/late we treat a scheduled session as "worth connecting for".
# Generous on purpose: F1 sessions routinely start late, and connecting a
# little early just means live/run.py waits for the feed to go green.
# 35 min covers a full-field MDP solve (~15 min worst case, see
# live/advisor.py) plus the SignalR negotiate/connect handshake, with
# margin. Solving is cached per circuit, so only the FIRST session of a
# race weekend actually needs this much lead time — you can also trigger
# live-listener.yml manually (workflow_dispatch) ahead of a weekend to
# warm the cache early if you want extra safety margin.
BUFFER_BEFORE = timedelta(minutes=35)
BUFFER_AFTER = timedelta(minutes=15)

# Fallback session duration used only to bound how long a listener run is
# allowed to keep retrying before giving up (real end-of-session detection
# comes from the live TrackStatus / SessionInfo topics, not from this).
_DURATION = {
    "FP1": timedelta(hours=1, minutes=30),
    "FP2": timedelta(hours=1, minutes=30),
    "FP3": timedelta(hours=1, minutes=30),
    "SQ": timedelta(hours=1, minutes=15),
    "SPRINT": timedelta(hours=1, minutes=15),
    "Q": timedelta(hours=1, minutes=15),
    "R": timedelta(hours=3, minutes=30),
}


@dataclass(frozen=True)
class SessionWindow:
    circuit: str  # slug — matched against core.mdp.PIT_LANE_DELTA when possible
    display_name: str  # human readable GP name, shown on the live page
    session: str  # "FP1" | "FP2" | "FP3" | "SQ" | "SPRINT" | "Q" | "R"
    start: datetime  # UTC, nominal scheduled start
    total_laps: int  # expected race distance; overridden by the live feed's
    # own SessionInfo.TotalLaps as soon as the session reports one

    @property
    def window_start(self) -> datetime:
        return self.start - BUFFER_BEFORE

    @property
    def window_end(self) -> datetime:
        duration = _DURATION.get(self.session, timedelta(hours=2))
        return self.start + duration + BUFFER_AFTER

    def contains(self, now: datetime) -> bool:
        return self.window_start <= now <= self.window_end


SEASON_2026: list[SessionWindow] = []


def _gp(circuit: str, display_name: str, total_laps: int, sessions: dict) -> None:
    for code, start in sessions.items():
        SEASON_2026.append(SessionWindow(circuit, display_name, code, start, total_laps))


# ── Remaining 2026 rounds (from the point this file was written, Sep 2026) ──
# Times from the public calendar; Las Vegas in particular runs past local
# midnight and is easy to get wrong when converting — double check it closer
# to the date if you're relying on this for that weekend specifically.

_gp(
    "baku",
    "Azerbaijan Grand Prix",
    51,
    {
        "FP1": datetime(2026, 9, 24, 8, 30, tzinfo=UTC),
        "FP2": datetime(2026, 9, 24, 12, 0, tzinfo=UTC),
        "FP3": datetime(2026, 9, 25, 8, 30, tzinfo=UTC),
        "Q": datetime(2026, 9, 25, 12, 0, tzinfo=UTC),
        "R": datetime(2026, 9, 26, 11, 0, tzinfo=UTC),
    },
)

_gp(
    "sakhir",
    "Bahrain Grand Prix",
    57,
    {
        "FP1": datetime(2026, 10, 2, 4, 30, tzinfo=UTC),
        "FP2": datetime(2026, 10, 2, 8, 0, tzinfo=UTC),
        "FP3": datetime(2026, 10, 3, 4, 30, tzinfo=UTC),
        "Q": datetime(2026, 10, 3, 8, 0, tzinfo=UTC),
        "R": datetime(2026, 10, 4, 7, 0, tzinfo=UTC),
    },
)

_gp(
    "marina_bay",
    "Singapore Grand Prix",
    62,
    {
        "FP1": datetime(2026, 10, 9, 8, 30, tzinfo=UTC),
        "SQ": datetime(2026, 10, 9, 12, 30, tzinfo=UTC),
        "SPRINT": datetime(2026, 10, 10, 9, 0, tzinfo=UTC),
        "Q": datetime(2026, 10, 10, 13, 0, tzinfo=UTC),
        "R": datetime(2026, 10, 11, 12, 0, tzinfo=UTC),
    },
)

_gp(
    "cota",
    "United States Grand Prix",
    56,
    {
        "FP1": datetime(2026, 10, 23, 17, 30, tzinfo=UTC),
        "FP2": datetime(2026, 10, 23, 21, 0, tzinfo=UTC),
        "FP3": datetime(2026, 10, 24, 17, 30, tzinfo=UTC),
        "Q": datetime(2026, 10, 24, 21, 0, tzinfo=UTC),
        "R": datetime(2026, 10, 25, 19, 0, tzinfo=UTC),
    },
)

_gp(
    "mexico_city",
    "Mexico City Grand Prix",
    71,
    {
        "FP1": datetime(2026, 10, 30, 17, 30, tzinfo=UTC),
        "FP2": datetime(2026, 10, 30, 21, 0, tzinfo=UTC),
        "FP3": datetime(2026, 10, 31, 16, 30, tzinfo=UTC),
        "Q": datetime(2026, 10, 31, 20, 0, tzinfo=UTC),
        "R": datetime(2026, 11, 1, 19, 0, tzinfo=UTC),
    },
)

_gp(
    "interlagos",
    "São Paulo Grand Prix",
    71,
    {
        "FP1": datetime(2026, 11, 6, 14, 30, tzinfo=UTC),
        "FP2": datetime(2026, 11, 6, 18, 0, tzinfo=UTC),
        "FP3": datetime(2026, 11, 7, 13, 30, tzinfo=UTC),
        "Q": datetime(2026, 11, 7, 17, 0, tzinfo=UTC),
        "R": datetime(2026, 11, 8, 16, 0, tzinfo=UTC),
    },
)

_gp(
    "las_vegas",
    "Las Vegas Grand Prix",
    50,
    {
        # NOTE: local-midnight session, verify against the official calendar
        # closer to the date.
        "FP1": datetime(2026, 11, 20, 3, 0, tzinfo=UTC),
        "FP2": datetime(2026, 11, 21, 3, 0, tzinfo=UTC),
        "Q": datetime(2026, 11, 22, 3, 0, tzinfo=UTC),
        "R": datetime(2026, 11, 23, 3, 0, tzinfo=UTC),
    },
)

_gp(
    "lusail",
    "Qatar Grand Prix",
    57,
    {
        "FP1": datetime(2026, 11, 27, 12, 30, tzinfo=UTC),
        "FP2": datetime(2026, 11, 27, 16, 0, tzinfo=UTC),
        "FP3": datetime(2026, 11, 28, 13, 30, tzinfo=UTC),
        "Q": datetime(2026, 11, 28, 17, 0, tzinfo=UTC),
        "R": datetime(2026, 11, 29, 15, 0, tzinfo=UTC),
    },
)

_gp(
    "yas_marina",
    "Abu Dhabi Grand Prix",
    58,
    {
        "FP1": datetime(2026, 12, 4, 8, 30, tzinfo=UTC),
        "FP2": datetime(2026, 12, 4, 12, 0, tzinfo=UTC),
        "FP3": datetime(2026, 12, 5, 9, 30, tzinfo=UTC),
        "Q": datetime(2026, 12, 5, 13, 0, tzinfo=UTC),
        "R": datetime(2026, 12, 6, 12, 0, tzinfo=UTC),
    },
)

SEASON_2026.sort(key=lambda w: w.start)


def is_live_now(now: datetime | None = None) -> SessionWindow | None:
    """Return the SessionWindow active at `now` (default: current UTC time), or None."""
    now = now or datetime.now(UTC)
    for window in SEASON_2026:
        if window.contains(now):
            return window
    return None


def next_session(now: datetime | None = None) -> SessionWindow | None:
    """Return the next upcoming SessionWindow after `now`, or None if the season is over."""
    now = now or datetime.now(UTC)
    upcoming = [w for w in SEASON_2026 if w.start >= now]
    return upcoming[0] if upcoming else None


if __name__ == "__main__":
    # Used by .github/workflows/live-scheduler.yml — plain, greppable stdout,
    # always exits 0 (a calendar miss should never fail the workflow).
    now = datetime.now(UTC)
    live = is_live_now(now)
    if live:
        print(
            f"LIVE circuit={live.circuit} session={live.session} "
            f"name={live.display_name!r} total_laps={live.total_laps}"
        )
    else:
        nxt = next_session(now)
        if nxt:
            mins = int((nxt.start - now).total_seconds() // 60)
            print(
                f"IDLE next={nxt.start.isoformat()} in_minutes={mins} "
                f"session={nxt.session} name={nxt.display_name!r}"
            )
        else:
            print("IDLE no_more_sessions_this_season")
