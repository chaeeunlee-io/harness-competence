"""`exp daytona-sweep`: find and delete Daytona sandboxes that no run is using any more.

Harbor deletes a trial's sandbox when the trial ends, but not when the Harbor process itself is
killed (Ctrl-C of a job, a stopped queue): those sandboxes keep running with no auto-stop and no
auto-delete, and every one keeps its disk. The organisation's disk limit (300 GiB on the tier these runs used; a
MedAgentBench sandbox takes 9 GiB, a Reasoning Gym one 1 GiB) then refuses new sandboxes, and
every trial started meanwhile fails with `DaytonaValidationError: Total disk limit exceeded`.

A sandbox counts as idle when Daytona has recorded no activity on it for `--idle-min` minutes.
A live trial touches its sandbox at every agent step, so idle sandboxes belong to stopped runs.
The default only lists; `--delete` deletes the idle ones.

`exp daytona-wait --free-gib N` blocks until the organisation has N GiB of sandbox disk free,
for a run that must not start into a full organisation (a GRPO run: a sandbox that cannot be
created becomes a zero-reward rollout). Both use DAYTONA_API_KEY (and DAYTONA_API_URL, if set)
from the environment, as Harbor does.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
import urllib.request

API = os.environ.get("DAYTONA_API_URL", "https://app.daytona.io/api").rstrip("/")


def _call(method: str, path: str):
    req = urllib.request.Request(API + path, method=method,
                                 headers={"Authorization": "Bearer " + os.environ["DAYTONA_API_KEY"]})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
    return json.loads(body) if body else None


def sandboxes() -> list[dict]:
    data = _call("GET", "/sandbox?limit=200")
    return data if isinstance(data, list) else data.get("items", [])


def _idle_minutes(s: dict, now: dt.datetime) -> float:
    last = dt.datetime.fromisoformat((s.get("lastActivityAt") or s["createdAt"]).replace("Z", "+00:00"))
    return (now - last).total_seconds() / 60


def main(a) -> int:
    if not os.environ.get("DAYTONA_API_KEY"):
        raise SystemExit("DAYTONA_API_KEY is not set (on EIDF: . setup/eidf/session.sh)")
    now = dt.datetime.now(dt.timezone.utc)
    items = sandboxes()
    idle = [s for s in items if _idle_minutes(s, now) > a.idle_min]
    disk = lambda xs: sum(int(s.get("disk") or 0) for s in xs)
    print(f"{len(items)} sandboxes, {disk(items)} GiB; idle for more than {a.idle_min} min: {len(idle)}, {disk(idle)} GiB")
    for s in sorted(idle, key=lambda s: s["createdAt"]):
        print(f"  {s['id']}  created {s['createdAt'][:16]}  last activity {(s.get('lastActivityAt') or '')[:16]}"
              f"  {s.get('disk')} GiB  {s.get('state')}")
    if not a.delete:
        print("(listing only; --delete deletes them)" if idle else "")
        return 0
    failed = 0
    for s in idle:
        try:
            _call("DELETE", f"/sandbox/{s['id']}")
        except Exception as e:   # one sandbox that cannot be deleted does not stop the sweep
            failed += 1
            print(f"  could not delete {s['id']}: {e}")
    print(f"deleted {len(idle) - failed} of {len(idle)}")
    return 1 if failed else 0


def wait_free(a) -> int:
    """Block until at least a.free_gib of the a.limit_gib sandbox disk is free."""
    if not os.environ.get("DAYTONA_API_KEY"):
        raise SystemExit("DAYTONA_API_KEY is not set (on EIDF: . setup/eidf/session.sh)")
    last = None
    while True:
        try:
            used = sum(int(s.get("disk") or 0) for s in sandboxes())
        except Exception as e:   # an API hiccup is not a reason to start the run
            print(f"daytona-wait: could not list sandboxes ({e}); retrying", flush=True)
            used = None
        if used is not None:
            if a.limit_gib - used >= a.free_gib:
                print(f"daytona-wait: {a.limit_gib - used} GiB free of {a.limit_gib} (needed {a.free_gib})", flush=True)
                return 0
            if used != last:
                print(f"daytona-wait: {used} of {a.limit_gib} GiB in use, waiting for {a.free_gib} GiB free", flush=True)
                last = used
        time.sleep(a.poll)
