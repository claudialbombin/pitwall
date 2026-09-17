"""
live/publish.py
================
The "server" in this project's zero-cost architecture: instead of hosting
an API, the live listener just keeps overwriting one file — live.json —
in a dedicated `live-data` branch of this same repo, via `git commit
--amend` + `git push --force`. One amended commit, not one commit per
tick, so a 2-hour race doesn't leave hundreds of commits in the history.

The static site (web/live.html) then reads that file straight from
raw.githubusercontent.com/<owner>/<repo>/live-data/live.json — which is a
public, CORS-enabled, free CDN GitHub already runs. No paid hosting, no
server, no third-party service: just this repo talking to itself.

This module never touches the checkout that's actually running the
Python listener — it works in a separate `git worktree`, so the branch
that holds the running code (main, or whatever ref triggered the
workflow) is never modified.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

log = logging.getLogger("live.publish")

BRANCH = "live-data"
FILE_NAME = "live.json"


def _run(cmd: list, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


class LiveDataPublisher:
    """
    Usage:
        pub = LiveDataPublisher(repo_root)
        pub.setup()
        pub.publish(payload)   # call as often as you like — internally throttled
        pub.close()
    """

    def __init__(
        self,
        repo_root: Path,
        branch: str = BRANCH,
        file_name: str = FILE_NAME,
        min_interval_s: float = 5.0,
        remote: str = "origin",
    ):
        self.repo_root = Path(repo_root)
        self.branch = branch
        self.file_name = file_name
        self.min_interval_s = min_interval_s
        self.remote = remote
        self.worktree_dir = self.repo_root / ".live-worktree"
        self._last_push = 0.0
        self._first_commit_done = False

    # ------------------------------------------------------------------
    def setup(self) -> None:
        """Create (or reuse) a worktree checked out to an orphan `live-data` branch."""
        _run(["git", "fetch", self.remote, self.branch], cwd=self.repo_root, check=False)

        if self.worktree_dir.exists():
            log.info("Reusing existing worktree at %s", self.worktree_dir)
            return

        remote_ref = f"{self.remote}/{self.branch}"
        has_remote_branch = (
            _run(
                ["git", "rev-parse", "--verify", remote_ref],
                cwd=self.repo_root,
                check=False,
            ).returncode
            == 0
        )

        if has_remote_branch:
            log.info("Checking out existing %s into a worktree", self.branch)
            _run(
                ["git", "worktree", "add", "-B", self.branch, str(self.worktree_dir), remote_ref],
                cwd=self.repo_root,
            )
        else:
            log.info("Creating orphan branch %s", self.branch)
            _run(
                ["git", "worktree", "add", "--detach", str(self.worktree_dir)],
                cwd=self.repo_root,
            )
            _run(["git", "checkout", "--orphan", self.branch], cwd=self.worktree_dir)
            # Wipe anything carried over from the detached commit — this
            # branch should contain nothing but live.json.
            _run(["git", "rm", "-rf", "--quiet", "."], cwd=self.worktree_dir, check=False)

    def publish(self, payload: dict, force_push: bool = False) -> bool:
        """
        Write `payload` to live.json and push if enough time has passed since
        the last push (or `force_push=True`, e.g. for the very first/last write
        of a run). Returns True if a push happened.
        """
        now = time.monotonic()
        if not force_push and (now - self._last_push) < self.min_interval_s:
            # Still write the file locally so the next throttled call has the
            # freshest data queued up, just don't push yet.
            self._write(payload)
            return False

        self._write(payload)
        self._commit_and_push()
        self._last_push = now
        return True

    def close(self, final_payload: dict | None = None) -> None:
        """Flush a final update (e.g. tagged as session-ended) and remove the worktree."""
        if final_payload is not None:
            self._write(final_payload)
            self._commit_and_push()
        _run(
            ["git", "worktree", "remove", "--force", str(self.worktree_dir)],
            cwd=self.repo_root,
            check=False,
        )

    # ------------------------------------------------------------------
    def _write(self, payload: dict) -> None:
        target = self.worktree_dir / self.file_name
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _commit_and_push(self) -> None:
        _run(["git", "add", self.file_name], cwd=self.worktree_dir)
        # Nothing to commit if the content is byte-identical to last time.
        diff = _run(["git", "diff", "--cached", "--quiet"], cwd=self.worktree_dir, check=False)
        if diff.returncode == 0 and self._first_commit_done:
            return

        if self._first_commit_done:
            _run(["git", "commit", "--amend", "--no-edit", "--quiet"], cwd=self.worktree_dir)
        else:
            _run(
                ["git", "commit", "-m", "live: update live.json", "--quiet"],
                cwd=self.worktree_dir,
            )
            self._first_commit_done = True

        result = _run(
            ["git", "push", "--force", self.remote, self.branch],
            cwd=self.worktree_dir,
            check=False,
        )
        if result.returncode != 0:
            log.warning("Push to %s failed: %s", self.branch, result.stderr.strip())
