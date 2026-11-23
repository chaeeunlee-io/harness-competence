"""`exp failures`: automatic failure-type proxies for an experiment's cells.

For every finished trial of each cell (the same cells `exp table` reads), counts what can be read
mechanically from the trial folder, without judging the content:

- failed:        reward below 1
- timeout:       the agent hit its time budget (AgentTimeoutError)
- infra:         any other recorded exception (sandbox or environment fault; excluded from rates)
- re-prompts:    Terminus-2 episodes whose prompt is the harness asking for valid output again
                 ("parsing errors"): unparseable or wrong-format actions
- empty replies: Terminus-2 episodes with an empty response
- no action:     BFCL assistant turns that end without a tool call before the conversation is over
- steps:         median episodes (Terminus-2) or messages (BFCL) per trial

These are the "invalid calls / termination failures" proxies behind the failure-type breakdown:
interface failures show up as re-prompts, empty replies, no-action turns and
timeouts; a trial that submits cleanly and still fails is a task failure. Rates are per failed
trial, so arms with different success rates stay comparable. The human annotation on a sample
calibrates them.
"""
from __future__ import annotations

import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from pathlib import Path

from exp import config as C, render as R
from exp import table as T


def _tail(path: str, n: int) -> str:
    """The last n bytes of a file as text (one small read, not the whole episode)."""
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - n))
        return fh.read().decode("utf-8", errors="replace")


def trial_signals(trial: Path) -> dict | None:
    """Mechanical signals of one trial; None if it has no result.json."""
    try:
        r = json.load(open(trial / "result.json"))
    except (OSError, ValueError):
        return None
    exc = (r.get("exception_info") or {}).get("exception_type")
    reward = None
    rp = trial / "verifier" / "reward.txt"
    if rp.is_file():
        try:
            reward = float(rp.read_text().strip())
        except ValueError:
            pass
    s = {"reward": reward, "timeout": exc == "AgentTimeoutError", "infra": bool(exc) and exc != "AgentTimeoutError",
         "steps": 0, "reprompts": 0, "empty": 0, "noaction": 0}
    episodes = sorted(glob(str(trial / "agent" / "episode-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    if episodes:   # Terminus-2
        s["steps"] = len(episodes)
        for e in episodes:
            try:
                if "parsing errors" in _tail(f"{e}/prompt.txt", 600):
                    s["reprompts"] += 1
                if os.path.getsize(f"{e}/response.txt") < 8 or not _tail(f"{e}/response.txt", 64).strip():
                    s["empty"] += 1
            except OSError:
                pass
        return s
    mt = trial / "agent" / "mt_transcript.json"
    if mt.is_file():   # BFCL multi-turn
        try:
            msgs = json.load(open(mt))["messages"]
        except (OSError, ValueError, KeyError):
            return s
        s["steps"] = len(msgs)
        for i, m in enumerate(msgs):
            if m.get("role") == "assistant" and not m.get("tool_calls"):
                # a turn with no call is normal only as the last reply to a user turn; count it when the
                # user had asked for an action that never came (another user turn or a tool follows)
                later = msgs[i + 1:i + 2]
                if later and later[0].get("role") in ("user", "tool"):
                    s["noaction"] += 1
    return s


# Trial folders live on a network file system: each stat/read is a round trip, so trials are
# scanned concurrently (I/O bound) and every cell row is flushed as soon as it is ready.
READERS = 16


def cell_signals(job_dir: Path) -> dict | None:
    trials = [Path(t) for t in sorted(glob(str(job_dir / "*/"))) if os.path.isdir(t)]
    with ThreadPoolExecutor(READERS) as ex:
        rows = list(ex.map(trial_signals, trials))
    rows = [x for x in rows if x and (x["reward"] is not None or x["infra"] or x["timeout"])]
    if not rows:
        return None
    scored = [x for x in rows if not x["infra"]]
    failed = [x for x in scored if (x["reward"] or 0) < 1]
    med = lambda xs: statistics.median(xs) if xs else 0
    pct = lambda n, d: 100.0 * n / d if d else 0.0
    return {"trials": len(scored), "infra": sum(x["infra"] for x in rows), "failed": len(failed),
            "fail_pct": pct(len(failed), len(scored)),
            "timeout_pct": pct(sum(x["timeout"] for x in failed), len(failed)),
            "reprompt_pct": pct(sum(x["reprompts"] > 0 for x in failed), len(failed)),
            "empty_pct": pct(sum(x["empty"] > 0 for x in failed), len(failed)),
            "noaction_pct": pct(sum(x["noaction"] > 0 for x in failed), len(failed)),
            "steps_failed": med([x["steps"] for x in failed]), "steps_ok": med([x["steps"] for x in scored if (x["reward"] or 0) >= 1])}


def main(a) -> int:
    e = C.experiments().get(a.experiment)
    if not e:
        raise SystemExit(f"unknown experiment {a.experiment!r}; see `exp list experiments`")
    src = T.Source.of(a)
    print(f"# failure proxies for {e['title']}   (runs: {src.describe()})")
    print("# per cell: scored trials (infra errors excluded) · failed % · among failed trials: hit the time budget %, "
          "had re-prompts for unparseable output %, had empty replies %, had turns without a call % · median steps failed / solved\n")
    for s in e["settings"]:
        b = C.benchmark(s["benchmark"])
        split = s.get("split", "test")
        hs = b.harnesses(s.get("harnesses", "trained"))
        tier = f" ({b.tier(split)})" if b.tier(split) else ""
        print(f"## {b.cfg['display']}{tier}", flush=True)
        print("| setting | harness | trials | infra | failed % | timeout % | re-prompt % | empty % | no-action % | steps fail/ok |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for st in T._settings(b, s):
            arm = T._arm_label(b, st, s["model"])
            for h in hs:
                sig = None
                for p in src.passes:
                    sig = sig or cell_signals(R.eval_dir(b, s["model"], arm, h, split, seed=a.seed, n_attempts=src.n_attempts,
                                                         eval_pass=p, root=src.root))
                label = b.pool(st).get("paper", st) if st in b.pools() else st
                if not sig:
                    print(f"| {label} | {h} | — | | | | | | | |", flush=True)
                    continue
                print(f"| {label} | {h} | {sig['trials']} | {sig['infra']} | {sig['fail_pct']:.0f} | {sig['timeout_pct']:.0f} | "
                      f"{sig['reprompt_pct']:.0f} | {sig['empty_pct']:.0f} | {sig['noaction_pct']:.0f} | "
                      f"{sig['steps_failed']:.0f} / {sig['steps_ok']:.0f} |", flush=True)
        print(flush=True)
    return 0
