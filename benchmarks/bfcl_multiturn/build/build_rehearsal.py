"""Build BFCL multi-turn rehearsal datasets.

Input: rehearsal traces (a demonstrator run on the rehearsal tasks that
`gen_rehearsal.py` generates and `qc_rehearsal.py` checks), plus teacher
datasets from `build_pools.py`. Each solved rehearsal trial becomes a row, with
at most two rows per task.

For `--target fc`, builds:
- `json+pycall`
- `reh-fc`
- `json+pycall+reh-fc`

For `--target xml`, builds `3h+reh-xml`; `3h` is the matched control because it
contains no xml rows.

For `--target json`, builds only `reh-json`: rehearsal trials run through the json harness
(a training harness) that passed the verifier, at most two per task, capped at the size of
`reh-fc`; the non-target rehearsal control's rows (paired with json+pycall by a derived pool).

With `--demonstrator 32b` (target fc), reads the `rehearsal-fc~32b` job (the Qwen3-32B teacher
on the same rehearsal tasks) and builds only `reh-fc~32b`, capped at the size of `reh-fc`: the
clean-demonstrator control's rows.

With `--failed-only` (target fc), builds only `reh-fc~fail`: the fc rehearsal trials that
did not pass the verifier, at most two per task, capped at the size of `reh-fc`; the
failed-rehearsal control's rows. The shipped pools are left untouched, so the control can
be built from a later collection of the same demonstrator on the same tasks.

All messages use the same four fields (`role`, `content`, `tool_calls`,
`tool_call_id`) so mixed datasets share a consistent schema.
"""
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PER_TASK = 2
SHIPPED = Path(__file__).resolve().parents[1] / "pools"   # the pools the paper trained on


def reference_pool(out, setting):
    """A pool another pool is sized against: the one being (re)built under `out` if present, else the
    shipped one. Lets `--failed-only`, `--demonstrator 32b` and `--target json` be built on their own,
    without first rebuilding `reh-fc` from the original collection."""
    for d in (out, SHIPPED):
        if (Path(d) / f"{setting}.jsonl").exists():
            return C.read_pool(Path(d), setting)
    raise SystemExit(f"{setting}.jsonl not found under {out} or {SHIPPED}")


def solved_rehearsal(dirs, harness, source="rehearsal", failed=False):
    """Rehearsal trials that passed the verifier (or, with `failed`, were scored and did not)."""
    rows, per_task = [], {}
    for trial in C.trials(dirs, "*/"):
        reward, transcript = os.path.join(trial, "verifier", "reward.txt"), os.path.join(trial, "agent", "mt_transcript.json")
        if not (os.path.exists(reward) and os.path.exists(transcript)):
            continue
        try:
            if (open(reward).read().strip() not in ("1", "1.0")) != failed:
                continue
            messages = json.load(open(transcript))["messages"]
        except Exception:
            continue
        task = os.path.basename(trial.rstrip("/")).split("__")[0]
        if per_task.get(task, 0) >= PER_TASK:
            continue
        per_task[task] = per_task.get(task, 0) + 1
        msgs = []
        for m in messages:
            row = {"role": m["role"], "content": m.get("content") or ""}
            if m.get("tool_call_id"):
                row["tool_call_id"] = m["tool_call_id"]
            if m.get("tool_calls"):
                row["tool_calls"] = [{"id": tc.get("id") or f"call_{i}", "type": "function",
                                      "function": {"name": (tc.get("function") or {}).get("name") or "",
                                                   "arguments": (tc.get("function") or {}).get("arguments") or "{}"}}
                                     for i, tc in enumerate(m["tool_calls"])]
            msgs.append(row)
        rows.append({"messages": msgs, "meta": {"task": task, "harness": harness, "source": source}})
    return rows


def uniform(rows):
    return [{"messages": [{"role": m["role"], "content": m.get("content") or "", "tool_calls": m.get("tool_calls") or [],
                           "tool_call_id": m.get("tool_call_id") or ""} for m in r["messages"]],
             "meta": r.get("meta", {})} for r in rows]


def main():
    p = C.parser(__doc__, {"rehearsal-fc": "rehearsal-fc", "rehearsal-xml": "rehearsal-xml", "rehearsal-fc~32b": "rehearsal-fc~32b",
                           "rehearsal-json": "rehearsal-json"})
    p.add_argument("--target", choices=["fc", "xml", "json"], default="fc")
    p.add_argument("--failed-only", action="store_true", help="write only reh-fc~fail (failed fc rehearsal trials)")
    p.add_argument("--min-failed-rows", type=int, default=40,
                   help="with --failed-only: refuse to write a pool smaller than this (the 3h demonstrator fails few "
                        "fc rehearsal attempts, so the failed pool is small by nature; its comparison arm is matched in size)")
    p.add_argument("--demonstrator", choices=["student", "32b"], default="student",
                   help="32b: write only reh-fc~32b from the Qwen3-32B teacher's rehearsal job")
    args = p.parse_args()
    out = args.out_dir
    if args.demonstrator == "32b":
        if args.target != "fc":
            raise SystemExit("--demonstrator 32b is defined for the fc target")
        jobs = C.job_dirs(args, {"rehearsal-fc~32b": "rehearsal-fc~32b"})
        rows = solved_rehearsal(jobs["rehearsal-fc~32b"], "fc", "rehearsal~32b")
        cap = len(reference_pool(out, "reh-fc"))
        if len(rows) > cap:
            rows = random.Random(0).sample(rows, cap)
        if len(rows) < 40:
            raise SystemExit("fewer than 40 rehearsal rows from the 32B teacher; check the traces")
        C.write_pool(out, "reh-fc~32b", uniform(rows))
        return
    role = f"rehearsal-{args.target}"
    jobs = C.job_dirs(args, {role: role})

    if args.target == "json":
        rows = solved_rehearsal(jobs[role], "json")
        cap = len(reference_pool(out, "reh-fc"))
        if len(rows) > cap:
            rows = random.Random(0).sample(rows, cap)
        if len(rows) < 40:
            raise SystemExit("fewer than 40 json rehearsal rows; check the traces")
        C.write_pool(out, "reh-json", uniform(rows))
        return

    if args.failed_only:
        if args.target != "fc":
            raise SystemExit("--failed-only is defined for the fc target")
        failed = solved_rehearsal(jobs[role], "fc", "rehearsal~fail", failed=True)
        cap = len(reference_pool(out, "reh-fc"))
        if len(failed) > cap:
            failed = random.Random(0).sample(failed, cap)
        if len(failed) < args.min_failed_rows:
            raise SystemExit(f"{len(failed)} failed rehearsal rows, fewer than --min-failed-rows {args.min_failed_rows}; check the traces")
        C.write_pool(out, "reh-fc~fail", uniform(failed))
        return

    if args.target == "fc":
        rehearsal = solved_rehearsal(jobs[role], "fc")
        cross = C.read_pool(out, "json-only") + C.read_pool(out, "pycall-only")
        C.write_pool(out, "json+pycall", uniform(cross))
        C.write_pool(out, "reh-fc", uniform(rehearsal))
        C.write_pool(out, "json+pycall+reh-fc", uniform(rehearsal + cross))
        return

    rehearsal = solved_rehearsal(jobs[role], "xml")
    # Drop over-length rehearsal rows (estimated at 3 characters per token; none were, as run).
    # 3h's own over-length rows stay, since its control arm was trained with them.
    rehearsal = [r for r in rehearsal if len(json.dumps(r["messages"])) // 3 <= 32768]
    # rehearsal rows go first (see the module docstring: populated tool_calls must come first)
    pool = sorted(uniform(rehearsal + C.read_pool(out, "3h")), key=lambda r: r["meta"]["source"] != "rehearsal")
    C.write_pool(out, "3h+reh-xml", pool)


if __name__ == "__main__":
    main()
