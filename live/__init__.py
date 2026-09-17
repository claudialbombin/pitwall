"""
live/
=====
Live timing + real-time strategy advisor for pitwall.

This package is deliberately separate from `core/` (the offline MDP solver,
tyre model and safety car model used by the notebooks, tests and the static
GitHub Pages simulator). `live/` is the *online* half: it connects to a
real F1 session while it is happening, keeps a running snapshot of the race,
and asks the exact same `core.solver.PitStopSolver` for a recommendation —
this package does not reimplement the strategy model, it just feeds it live
inputs instead of slider values.

Design constraint: the whole thing must cost EUR 0. See README.md, section
"Live timing", for the full write-up. Short version:

  - No paid live-timing subscription (OpenF1's live tier is EUR 9.90/month —
    ruled out). Instead we speak directly to the same undocumented,
    unauthenticated SignalR feed that FastF1 itself is built on
    (livetiming.formula1.com). It is free, but unofficial and can change
    or break without notice — this code fails soft (retries, then gives up
    cleanly) rather than crashing a scheduled job.
  - No always-on server. GitHub Actions on a public repo gives unlimited
    free minutes, so a workflow (live-listener.yml) simply runs for the
    duration of one session and then exits.
  - No paid storage/CDN. The listener writes live.json and force-pushes it
    (single amended commit, not one commit per tick) to an orphan branch
    called `live-data`. GitHub Pages doesn't need to know about that branch
    at all — the static site just fetches the file straight from
    raw.githubusercontent.com, which serves public repo content with
    permissive CORS headers, for free.
"""
