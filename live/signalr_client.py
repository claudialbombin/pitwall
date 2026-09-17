"""
live/signalr_client.py
=======================
Minimal client for F1's live timing feed (livetiming.formula1.com).

THIS IS UNOFFICIAL. There is no published spec — everything here follows
the reverse-engineering already done by the community (the same feed
FastF1's own `fastf1.livetiming` module is built on; see also the
"F1-SignalR" and "f1-telemetry" projects referenced in this repo's README).
It can change or disappear without notice. `PitwallLiveClient` is written
to fail soft: a malformed or unexpected message is logged and skipped, it
never takes down the whole listener process over one bad frame.

PROTOCOL, IN SHORT:
This is old ASP.NET SignalR (not the newer ASP.NET Core SignalR), which
uses a two-step handshake:

  1. negotiate  — plain HTTPS GET, returns a ConnectionToken + a cookie.
  2. connect    — a WebSocket using that token, over which we send one
                  "Subscribe" message naming the topics we want, then just
                  read frames forever.

Two message shapes come back on the socket:
  - The very first frame is an "R" (reference) message: the FULL current
    value of every subscribed topic, keyed by topic name. This is our
    starting snapshot.
  - After that, "M" (method) messages carry incremental updates: each is
    [topic_name, patch, timestamp]; `patch` should be merged into whatever
    we already know about that topic (dict-merge, not replace).

Some topics (their name ends in ".z") arrive raw-deflate-compressed and
base64-encoded, because they're high-frequency (Position.z, CarData.z).
Everything else is plain JSON.
"""

from __future__ import annotations

import base64
import json
import logging
import zlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests
import websockets

log = logging.getLogger("live.signalr_client")

NEGOTIATE_URL = "https://livetiming.formula1.com/signalr/negotiate"
CONNECT_URL = "wss://livetiming.formula1.com/signalr/connect"
HUB = "Streaming"
CLIENT_PROTOCOL = "1.5"

# Headers matter here: the feed 500s on some header casing, and rejects
# requests without a browser-shaped User-Agent. "BestHTTP" is the value
# every known working client uses (it's what the .NET client library that
# F1's own broadcast tooling is built on identifies itself as).
_HEADERS = {
    "User-Agent": "BestHTTP",
    "Accept-Encoding": "gzip,identity",
}

# Topics we actually need for the live leaderboard + strategy advisor.
# (Deliberately excludes CarData.z / telemetry channels — high bandwidth,
# not needed for pit strategy, and the whole point is keeping this cheap
# and reliable on a GitHub Actions runner.)
DEFAULT_TOPICS = [
    "Heartbeat",
    "SessionInfo",
    "TrackStatus",
    "LapCount",
    "DriverList",
    "TimingData",
    "TimingAppData",
    "RaceControlMessages",
    "WeatherData",
]


@dataclass
class LiveMessage:
    topic: str
    data: dict
    is_snapshot: bool = False  # True for the initial "R" full-state dump


def _connection_data(topics_hub: str = HUB) -> str:
    return json.dumps([{"name": topics_hub}], separators=(",", ":"))


def _decode_z(value: str) -> Any:
    """Decode one of the '.z'-suffixed raw-deflate + base64 payloads."""
    raw = base64.b64decode(value)
    decompressed = zlib.decompress(raw, -zlib.MAX_WBITS)
    return json.loads(decompressed)


def _unwrap(topic: str, payload: Any) -> Any:
    """Undo '.z' compression when present; leave everything else untouched."""
    if topic.endswith(".z") and isinstance(payload, str):
        try:
            return _decode_z(payload)
        except Exception:
            log.warning("Failed to decode compressed topic %s", topic, exc_info=True)
            return None
    return payload


class PitwallLiveClient:
    """
    Usage:
        async with PitwallLiveClient() as client:
            async for msg in client.messages():
                ...
    """

    def __init__(self, topics: list[str] | None = None, timeout: float = 10.0):
        self.topics = topics or DEFAULT_TOPICS
        self.timeout = timeout
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._cookie: str | None = None

    # ------------------------------------------------------------------
    def negotiate(self) -> tuple[str, str]:
        """Return (connection_token, cookie) from the negotiate handshake."""
        params = {
            "clientProtocol": CLIENT_PROTOCOL,
            "connectionData": _connection_data(),
        }
        resp = requests.get(NEGOTIATE_URL, params=params, headers=_HEADERS, timeout=self.timeout)
        resp.raise_for_status()
        token = resp.json()["ConnectionToken"]
        cookie = "; ".join(f"{k}={v}" for k, v in resp.cookies.get_dict().items())
        return token, cookie

    async def connect(self) -> None:
        token, cookie = self.negotiate()
        self._cookie = cookie
        url = (
            f"{CONNECT_URL}?transport=webSockets"
            f"&connectionToken={quote(token)}"
            f"&connectionData={quote(_connection_data())}"
            f"&clientProtocol={CLIENT_PROTOCOL}"
        )
        headers = dict(_HEADERS)
        if cookie:
            headers["Cookie"] = cookie
        log.info("Connecting to F1 live timing feed...")
        self._ws = await websockets.connect(
            url, extra_headers=headers, ping_interval=15, ping_timeout=self.timeout
        )
        subscribe = {"H": HUB, "M": "Subscribe", "A": [self.topics], "I": 1}
        await self._ws.send(json.dumps(subscribe))
        log.info("Subscribed to topics: %s", ", ".join(self.topics))

    async def messages(self) -> AsyncIterator[LiveMessage]:
        """Yield LiveMessage objects forever, until the socket closes."""
        assert self._ws is not None, "call connect() first"
        async for raw in self._ws:
            if not raw or raw == "{}":
                continue  # keep-alive frame
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("Non-JSON frame from feed, skipping: %r", raw[:200])
                continue

            # Initial full-state snapshot, keyed by topic name.
            if "R" in frame and isinstance(frame["R"], dict):
                for topic, payload in frame["R"].items():
                    yield LiveMessage(topic, _unwrap(topic, payload), is_snapshot=True)
                continue

            # Incremental updates.
            for entry in frame.get("M", []):
                args = entry.get("A", [])
                if len(args) < 2:
                    continue
                topic, payload = args[0], args[1]
                yield LiveMessage(topic, _unwrap(topic, payload))

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def __aenter__(self) -> PitwallLiveClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
