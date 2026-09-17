# Live timing — how it works, and why it costs €0

This is the write-up for the "watch the current F1 session live and get a
real strategy recommendation" feature: `web/live.html` plus everything in
this `live/` directory. It's kept separate from the top-level `README.md`
deliberately — that file currently describes a different project entirely
(a blackjack Monte Carlo solver, not pitwall) and rewriting it wasn't part
of this task, so this doc stands alone until that gets sorted out.

## The constraint that shaped everything here

Real-time F1 data has exactly one mainstream provider aimed at hobby
projects — [OpenF1](https://openf1.org) — and as of writing, its live tier
is €9.90/month. That was ruled out: this had to cost nothing, ever.

So instead, `live/signalr_client.py` talks directly to the same
undocumented feed FastF1 itself is built on (`livetiming.formula1.com`).
It's free and requires no login, but it is **not an official, supported
API** — it's the reverse-engineering the whole hobbyist F1-timing
community relies on (see the comments in that file for links). It can
change shape or disappear without notice. Everything downstream is
written to fail soft rather than take down a scheduled run over it.

## The architecture, in one paragraph

Every 5 minutes, `.github/workflows/live-scheduler.yml` (free, unlimited
minutes on a public repo) checks the hand-maintained calendar in
`live/schedule.py`. When a session is about to go live, it dispatches
`.github/workflows/live-listener.yml`, which runs `live/run.py` for the
duration of that one session: connect to the feed, keep a `RaceState`
(`live/state.py`) up to date lap by lap, and every few seconds ask
`live/advisor.py` — a thin wrapper around the *exact same*
`core.solver.PitStopSolver` the offline simulator uses — what it would
recommend given the real gaps, tyres, and safety car status right now.
The combined snapshot gets written to `live.json` and pushed (single
amended commit, not one per tick — see `live/publish.py`) to an orphan
`live-data` branch. `web/live.html` polls that file straight from
`raw.githubusercontent.com`, which is a free, CORS-enabled public CDN.
No server, no database, no paid API, no secrets beyond the token GitHub
already injects into every Action run.

## What to check before/after the first real session

The very first live session (the next one after this was built) is the
real test — nothing here has been run against an actual live F1 feed yet,
only against synthetic data. Specifically worth watching in the
`live-listener.yml` run logs:

- Whether the SignalR negotiate/connect handshake in `signalr_client.py`
  still works as documented — header casing and the exact hub name are
  the parts most likely to have drifted.
- Whether the field names assumed in `state.py` (`GapToLeader`,
  `IntervalToPositionAhead`, `Stints`, etc.) still match. Every parser in
  that file logs a warning instead of crashing when something looks off,
  so a bad assumption shows up as a gap in the data, not a dead workflow.
- Whether the MDP solve finishes comfortably inside the 35-minute buffer
  in `schedule.py` — measured at ~13.6s per lap at `n_positions=10` on a
  GitHub-hosted runner's CPU (see `live/advisor.py`); a 70-lap race is
  ~16 minutes, cached afterwards for the rest of that race weekend.

If the feed's shape has changed, the fix is almost always contained to
one `_apply_*` method in `state.py` — there's no need to touch the
solver, the publishing logic, or the frontend.

## Updating the calendar for a new season

`live/schedule.py` is a plain list of session windows, built with one
`_gp(...)` call per Grand Prix weekend. Replace the `SEASON_2026` block
with the new season's dates from any public F1 calendar — five minutes of
copy-paste, no code changes needed elsewhere. `deploy.yml` re-exports it
to `results/schedule.json` on every deploy, so `web/live.html`'s
"next session in..." countdown picks up the change automatically.

## Known simplifications (already consistent with the rest of this repo)

- Only 4 circuits (`monaco`, `monza`, `silverstone`, `spa`) have hand-tuned
  pit lane deltas in `core/mdp.py`. Every other circuit — most of the
  calendar — falls back to the generic default, the same fallback
  `.github/workflows/deploy.yml` already accepts for circuits without a
  cached FastF1 dataset.
- The strategy model solves at `n_positions=10` for the live path (see
  `live/advisor.py` for the reasoning) rather than the full 20-car grid,
  trading a little resolution in the midfield for a solve time that
  reliably fits the pre-session buffer.
- Gap/interval values are discretised into 4 bins, same as the offline
  model (`core/mdp.py::GAP_BINS`) — nothing new introduced for live.
- "Session ended" is detected by a data-idle timeout (6 minutes of
  silence from the feed), not a specific end-of-session message, so it
  works the same way regardless of how a given session actually ends
  (chequered flag, red flag, abandoned session, etc).

## Local testing without a live session

```bash
pip install -e ".[dev]"
pip install -r requirements-live.txt

# Solve + cache the strategy model for a tiny fake race, no network needed:
python -m live.run --circuit test_circuit --total-laps 6 --prewarm-only

# Unit tests (fast — synthetic feed payloads, no solving):
pytest tests/test_live_state.py -v

# Slower — actually runs the MDP solver end to end (marked `slow`):
pytest tests/test_live_advisor.py -v
```

There is currently no offline "replay a recorded session" harness for
`live/run.py` end-to-end (i.e. the SignalR client + publish loop
together) — the unit tests cover `state.py` and `advisor.py` directly
instead. Worth adding once a real session's raw feed has been captured
once, if this needs deeper regression coverage later.
