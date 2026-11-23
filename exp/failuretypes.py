"""`exp failure-types`: rule-based failure-type labels for failed trials, interface vs task knowledge.

Each failed trial (verifier reward 0; infrastructure exceptions excluded) gets one label from signals the harness
and the verifier record, without reading the model's reasoning:

  interface / invalid       the harness could not parse or run the model's actions
  interface / submission    the answer was never delivered the way the harness expects
  interface / termination   the run ended without finishing: the agent's time budget, or a stop in a re-prompt loop
  task                      the actions were well-formed and the answer was delivered, and it was wrong

The primary variant ("delivery") follows the human protocol's "first decisive error": a trial whose answer reached
the verifier and was judged wrong is a task failure, whatever happened in the turns after it; a trial whose answer
never arrived is an interface failure, typed by why (invalid calls if most turns were re-prompts or any call was in
another harness's syntax, termination if the agent timed out or ended on a re-prompt, otherwise submission). The
"trace" variant checks the trace before the delivery (invalid, then termination, then submission, else task), so a
trial that delivered a wrong answer and then looped counts as invalid calls. The signals per benchmark:

  Terminus-2 (MedAgentBench, Reasoning Gym): a re-prompt is a turn whose prompt is the harness asking for valid
    output again ("parsing errors"); a foreign call is a reply in another harness's tool-call syntax (<tool_call>).
    invalid      re-prompts on at least THRESHOLD of the turns, or any foreign call
    termination  the agent timeout, or the last turn is a re-prompt
    submission   the verifier reports no answer file, an empty one, or one it cannot parse
  BFCL multi-turn (fc): the checker's error type.
    invalid      an assistant turn whose text carries an unparsed tool call (the call never reached the API)
    termination  multi_turn:empty_turn_model_response (a user turn left without any action)
    task         a state or response mismatch after well-formed calls

THRESHOLD ("most turns invalid") is 0.5. Both variants are reported side by side; the human labels on the
annotation sample check them.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from pathlib import Path

from exp.failures import _tail

FOREIGN = re.compile(r"<tool_call>")
NO_ANSWER = re.compile(r"Missing answer file|answer\.txt not found|Answer file is not valid JSON|No answer file"
                       r"|^Found answer = *$", re.I | re.M)      # an answer file left empty (the command that should write it failed)


def _reward(trial: Path):
    try:
        return float((trial / "verifier" / "reward.txt").read_text().strip())
    except (OSError, ValueError):
        return None


def _verifier_text(trial: Path) -> str:
    for f in ("test-stdout.txt", "test-output.txt"):
        p = trial / "verifier" / f
        if p.is_file():
            return p.read_text(errors="replace")[:4000]
    return ""


def signals(trial: Path) -> dict | None:
    """Signals of one failed trial, or None (not failed, unscored, or an infrastructure exception)."""
    try:
        res = json.load(open(trial / "result.json"))
    except (OSError, ValueError):
        return None
    exc = (res.get("exception_info") or {}).get("exception_type")
    if exc and exc != "AgentTimeoutError":
        return None
    if _reward(trial) != 0.0:
        return None
    s = {"trial": trial.name, "task": trial.name.split("__")[0], "timeout": exc == "AgentTimeoutError"}
    episodes = sorted(glob(str(trial / "agent" / "episode-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    if episodes:                                          # Terminus-2
        rep, foreign, last_rep = 0, 0, False
        for i, e in enumerate(episodes):
            try:
                r = "parsing errors" in _tail(f"{e}/prompt.txt", 600)
                rep += r
                last_rep = r if i == len(episodes) - 1 else last_rep
                if FOREIGN.search(_tail(f"{e}/response.txt", 20000)):
                    foreign += 1
            except OSError:
                pass
        v = _verifier_text(trial)
        s.update(kind="t2", turns=len(episodes), reprompts=rep, foreign=foreign, last_reprompt=last_rep,
                 no_answer=bool(NO_ANSWER.search(v)))
        return s
    mt = trial / "agent" / "mt_transcript.json"
    if mt.is_file():                                      # BFCL multi-turn
        try:
            msgs = json.load(open(mt))["messages"]
        except (OSError, ValueError, KeyError):
            return None
        # a call written as text never reaches the API, whether the reply puts it after its reasoning or inside it
        unparsed = sum(1 for m in msgs if m.get("role") == "assistant" and not m.get("tool_calls")
                       and FOREIGN.search(str(m.get("content") or "")))
        try:
            err = json.load(open(trial / "verifier" / "checker_detail.json")).get("error_type", "")
        except (OSError, ValueError):
            err = ""
        s.update(kind="fc", turns=len(msgs), unparsed=unparsed, error_type=err)
        return s
    return None


def label(s: dict, variant: str = "delivery", threshold: float = 0.5) -> tuple[str, str]:
    """(primary, subtype) for one trial's signals."""
    if s["kind"] == "t2":
        frac = s["reprompts"] / s["turns"] if s["turns"] else 0.0
        if variant == "delivery" and not s["no_answer"]:
            return "task", ""
        if s["foreign"] or (s["reprompts"] and frac >= threshold):
            return "interface", "invalid"
        if s["timeout"] or s["last_reprompt"]:
            return "interface", "termination"
        if s["no_answer"]:
            return "interface", "submission"
        return "task", ""
    if s["unparsed"]:
        return "interface", "invalid"
    if s["timeout"] or s["error_type"] == "multi_turn:empty_turn_model_response":
        return "interface", "termination"
    return "task", ""


def job_signals(job_dir: Path, readers: int = 16) -> list[dict]:
    trials = [Path(t) for t in sorted(glob(str(job_dir / "*/"))) if os.path.isdir(t)]
    with ThreadPoolExecutor(readers) as ex:
        rows = list(ex.map(signals, trials))
    scored = sum(1 for t in trials if _reward(t) is not None)
    out = [r for r in rows if r]
    for r in out:
        r["scored_in_job"] = scored
    return out


VARIANTS = ("delivery", "trace")


def main(a) -> int:
    spec = json.load(open(a.spec)) if a.spec.endswith(".json") else __import__("yaml").safe_load(open(a.spec))
    root = Path(spec.get("evals_root", "runs/evals"))
    out_rows, summary = [], []
    for cell in spec["cells"]:
        for arm, job in cell["arms"].items():
            rows = job_signals(root / job)
            scored = rows[0]["scored_in_job"] if rows else 0
            for r in rows:
                r.update(benchmark=cell["benchmark"], tier=cell.get("tier", ""), arm=arm, job=job)
                for v in VARIANTS:
                    r[f"label_{v}"], r[f"sub_{v}"] = label(r, v)
            out_rows += rows
            row = {"benchmark": cell["benchmark"], "tier": cell.get("tier", ""), "arm": arm, "job": job,
                   "scored": scored, "failed": len(rows)}
            for v in VARIANTS:
                n_if = sum(r[f"label_{v}"] == "interface" for r in rows)
                row[f"{v}_interface_share"] = round(n_if / len(rows), 3) if rows else None
                row[f"{v}_interface_per100"] = round(100 * n_if / scored, 1) if scored else None
                for sub in ("invalid", "submission", "termination"):
                    row[f"{v}_{sub}_per100"] = round(100 * sum(r[f"sub_{v}"] == sub for r in rows) / scored, 1) if scored else None
            summary.append(row)
            print(f"{cell['benchmark']} {cell.get('tier', '') or '-'} {arm}: {len(rows)} failed of {scored}; "
                  f"interface per 100 trials " + " / ".join(f"{v} {row[f'{v}_interface_per100']}" for v in VARIANTS), flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in out_rows for k in r})
    with open(out / "trials.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(out_rows)
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    print(f"wrote {out}/trials.csv ({len(out_rows)} failed trials), {out}/summary.csv", file=sys.stderr)
    return 0
