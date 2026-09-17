"""
live/run.py
============
Entry point: python -m live.run

Orchestrates the whole live pipeline for ONE session:
  1. figure out what's live right now (or accept a manual override — see
     the CLI flags, used for pre-warming the strategy cache or local testing)
  2. pre-solve (or load the cached) strategy model for that circuit
  3. connect to the F1 timing feed and keep a RaceState up to date
  4. every few seconds, combine RaceState + LiveAdvisor into one JSON
     payload and publish it via LiveDataPublisher (see live/publish.py —
     this is what makes the whole thing free: no server, just a git push)

Exits cleanly (code 0) whenever there's nothing to do right now — this is
what lets .github/workflows/live-scheduler.yml dispatch this workflow
opportunistically every few minutes without worrying about "wasting" a run.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from live.advisor import LiveAdvisor
from live.publish import LiveDataPublisher
from live.schedule import SessionWindow, is_live_now
from live.signalr_client import PitwallLiveClient
from live.state import RaceState

log = logging.getLogger("live.run")

REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLISH_INTERVAL_S = 5.0
IDLE_TIMEOUT_S = 6 * 60  # no feed updates for 6 min -> assume the session ended
MAX_RUNTIME_S = 4 * 60 * 60  # hard safety cutoff regardless of anything else
RECONNECT_ATTEMPTS = 5
RECONNECT_BACKOFF_S = 15


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="pitwall live timing listener")
    p.add_argument("--circuit", help="Force a circuit slug instead of reading the calendar")
    p.add_argument("--total-laps", type=int, help="Force total laps (used with --circuit)")
    p.add_argument("--display-name", default="", help="Optional display name override")
    p.add_argument("--session", default="", help="Optional session code override (FP1/Q/R/...)")
    p.add_argument(
        "--prewarm-only",
        action="store_true",
        help="Just solve+cache the strategy model for --circuit/--total-laps and exit, "
        "without connecting to the live feed. Handy to warm the cache ahead of a race "
        "weekend via a manual workflow_dispatch run.",
    )
    return p.parse_args()


def _resolve_window(args: argparse.Namespace) -> SessionWindow | None:
    if args.circuit and args.total_laps:
        return SessionWindow(
            circuit=args.circuit,
            display_name=args.display_name or args.circuit,
            session=args.session or "MANUAL",
            start=datetime.now(timezone.utc),
            total_laps=args.total_laps,
        )
    return is_live_now()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    args = _parse_args()
    window = _resolve_window(args)

    if window is None:
        log.info("No live session right now and no --circuit override — nothing to do.")
        return 0

    log.info(
        "Target session: %s %s at %s (%d laps)",
        window.display_name,
        window.session,
        window.circuit,
        window.total_laps,
    )

    t0 = time.monotonic()
    advisor = LiveAdvisor(circuit=window.circuit, total_laps=window.total_laps)
    log.info("Strategy model ready in %.1fs", time.monotonic() - t0)

    if args.prewarm_only:
        log.info("Prewarm-only run — exiting without connecting to the live feed.")
        return 0

    race = RaceState(
        circuit=window.circuit,
        display_name=window.display_name or window.circuit,
        session_type=window.session,
        total_laps=window.total_laps,
    )

    publisher = LiveDataPublisher(REPO_ROOT, min_interval_s=PUBLISH_INTERVAL_S)
    publisher.setup()

    try:
        asyncio.run(_listen_and_publish(race, advisor, publisher, window))
    finally:
        payload = _build_payload(race, advisor, window, live=False)
        publisher.close(final_payload=payload)

    return 0


async def _listen_and_publish(
    race: RaceState,
    advisor: LiveAdvisor,
    publisher: LiveDataPublisher,
    window: SessionWindow,
) -> None:
    start = time.monotonic()
    attempt = 0

    while True:
        if time.monotonic() - start > MAX_RUNTIME_S:
            log.info("Hit the hard safety cutoff (%.0f min) — stopping.", MAX_RUNTIME_S / 60)
            return

        try:
            async with PitwallLiveClient() as client:
                attempt = 0  # reset backoff after a successful connect
                async for msg in client.messages():
                    race.apply(msg.topic, msg.data)
                    payload = _build_payload(race, advisor, window, live=True)
                    publisher.publish(payload)

                    idle_for = time.monotonic() - race.last_update_monotonic
                    if idle_for > IDLE_TIMEOUT_S:
                        log.info("No updates for %.0fs — assuming the session ended.", idle_for)
                        return
        except (ConnectionError, OSError) as exc:
            attempt += 1
            if attempt > RECONNECT_ATTEMPTS:
                log.error("Giving up after %d reconnect attempts: %s", attempt, exc)
                return
            log.warning(
                "Feed connection dropped (%s) — reconnecting in %ds (attempt %d/%d)",
                exc,
                RECONNECT_BACKOFF_S,
                attempt,
                RECONNECT_ATTEMPTS,
            )
            await asyncio.sleep(RECONNECT_BACKOFF_S)


def _build_payload(
    race: RaceState, advisor: LiveAdvisor, window: SessionWindow, live: bool
) -> dict:
    data = race.to_json()
    data["live"] = live
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    for driver_entry in data["drivers"]:
        snapshot = race.drivers.get(driver_entry["number"])
        advice = advisor.advise(snapshot, race) if snapshot else None
        driver_entry["advice"] = advice.to_json() if advice else None
    return data


if __name__ == "__main__":
    sys.exit(main())
