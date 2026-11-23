"""`exp build-pool`: build SFT pools from collected traces with the benchmark's build/ scripts.

configs/benchmarks/<b>.yaml `build:` lists, in order, which script writes which settings.
`--pool <setting>` runs the one step that writes it; without --pool every step runs in
order (later steps read pools written by earlier ones). Traces are read from
runs/collect/<benchmark>/ (where `exp collect` writes them) and pools are written to
runs/pools/<benchmark>/<setting>.jsonl; train on one with `exp sft --data <path>`.

Every row is {"messages": [...], "meta": {...}}. `meta` names the row's `task`, the `harness` it was
collected through and its `source`: `teacher` for a teacher trace, `rehearsal` for a rehearsal trace, with
`~32b`, `~gemma` or `~fail` for the rehearsal variants. Training reads `messages` only.

A pool declared with `derived:` in the config is not built from traces but from other pools of the
same benchmark (subsampled or repeated rows, seed 0, optionally with the reasoning removed from
the assistant turns), see `build_derived`; `exp sft` builds it on
first use, `exp build-pool --pool <name>` builds it explicitly.
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
from pathlib import Path

from exp import config as C
from exp.render import RUNS

ROOT = C.ROOT


def derived_file(b, name: str) -> Path:
    """Where a derived pool is written: runs/pools/<benchmark>/<name>.jsonl."""
    return RUNS / "pools" / b.name / f"{name}.jsonl"


_THINK = re.compile(r"<think>.*?</think>", re.S)


def strip_reasoning(row: dict) -> dict:
    """The row with every assistant turn's reasoning block emptied (`<think>\n\n</think>`, the
    form Qwen3 writes when it does not think); actions, tool calls and answers are unchanged."""
    msgs = [dict(m, content=_THINK.sub("<think>\n\n</think>", m.get("content") or ""))
            if m.get("role") == "assistant" else m for m in row.get("messages", [])]
    return dict(row, messages=msgs)


def _has_tool_calls(row: dict) -> bool:
    """Whether any message of the row carries a populated `tool_calls` list."""
    return any(m.get("tool_calls") for m in row.get("messages", []))


def build_derived(b, name: str, force: bool = False) -> Path:
    """Materialise a pool declared as `derived: {rows: [...], seed: 0}` in the benchmark config.

    Each entry of `rows` names a source pool of the same benchmark and takes its rows, optionally
    only those with `meta.harness == harness` and `meta.source == source`; `n` subsamples to n rows (seed `seed`, order kept),
    and `n_like: <pool>` sets n to that pool's row count, for an arm size-matched to a pool whose size
    is only known once it is built
    or, with `repeat: true`, grows the pool to n rows: whole copies of it, then a uniform random
    subsample for the remainder; `strip_reasoning: true` empties the reasoning block of every
    assistant turn of those rows (`strip_reasoning`), to separate what a trace's reasoning carries
    from what its actions and answers carry. The result is the concatenation with the rows that carry tool
    calls first (the training loader infers the column schema from the first rows, so a pool
    that starts with rows whose `tool_calls` are all empty cannot hold populated ones later),
    written once; `force` rewrites it. Prints the composition by harness and source.
    """
    out = derived_file(b, name)
    if out.exists() and not force:
        return out
    spec = b.pool(name)["derived"]
    rng = random.Random(spec.get("seed", 0))
    rows, parts = [], []
    for part in spec["rows"]:
        src = b.pool(part["pool"])
        path = build_derived(b, part["pool"]) if "derived" in src else ROOT / src["file"]
        chosen = [json.loads(line) for line in open(path)]
        for key in ("harness", "source"):
            if part.get(key):
                chosen = [r for r in chosen if (r.get("meta") or {}).get(key) == part[key]]
        n = part.get("n")
        if part.get("n_like"):
            like = b.pool(part["n_like"])
            like_path = build_derived(b, part["n_like"]) if "derived" in like else ROOT / like["file"]
            n = sum(1 for _ in open(like_path))
        if n is not None and n < len(chosen):
            keep = set(rng.sample(range(len(chosen)), n))
            chosen = [r for i, r in enumerate(chosen) if i in keep]
        elif n is not None and n > len(chosen):
            if not part.get("repeat"):
                raise SystemExit(f"{b.name}/{name}: {part['pool']} has {len(chosen)} rows, {n} asked; add repeat: true to cycle them")
            copies, extra = divmod(n, len(chosen))
            keep = set(rng.sample(range(len(chosen)), extra))
            chosen = chosen * copies + [r for i, r in enumerate(chosen) if i in keep]
        if part.get("strip_reasoning"):
            chosen = [strip_reasoning(r) for r in chosen]
        rows += chosen
        parts.append(f"{len(chosen)} from {part['pool']}" + "".join(f" ({part[k]})" for k in ("harness", "source") if part.get(k))
                     + (", reasoning removed" if part.get("strip_reasoning") else ""))
    rows.sort(key=lambda r: not _has_tool_calls(r))   # stable: source order kept within each group
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    mix = {}
    for r in rows:
        meta = r.get("meta") or {}
        key = f"{meta.get('harness')} {meta.get('source')}"
        mix[key] = mix.get(key, 0) + 1
    print(f"{b.name}/{name}: {len(rows)} rows = {' + '.join(parts)} -> {C.rel(out)}; by harness and source {mix}")
    return out


def steps_for(b, pool: str | None) -> list[dict]:
    steps = b.cfg.get("build") or []
    if not steps:
        raise SystemExit(f"{b.name}: no build steps configured")
    if pool is None:
        return steps
    hit = [s for s in steps if pool in s["writes"]]
    if not hit:
        known = [w for s in steps for w in s["writes"]]
        raise SystemExit(f"{b.name}: no build step writes {pool!r}; buildable: {', '.join(known)}")
    return hit[:1]


def main(a) -> int:
    b = C.benchmark(a.benchmark)
    if a.pool and "derived" in b.pool(a.pool):
        if a.dry_run:
            print(f"# {b.name}/{a.pool}: derived pool -> {C.rel(derived_file(b, a.pool))} (not built: --dry-run)")
            return 0
        build_derived(b, a.pool, force=a.force)
        return 0
    jobs = a.jobs_dir or str(RUNS / "collect" / b.name)
    out = a.out_dir or str(RUNS / "pools" / b.name)
    rc = 0
    for step in steps_for(b, a.pool):
        script = ROOT / "benchmarks" / b.name / step["script"]
        argv = [str(script), *(step.get("args") or []), "--out-dir", out]
        if step.get("reads", "jobs") == "jobs":  # `reads: pools` = a step that only reads earlier pools
            argv += ["--jobs-dir", jobs]
        if step.get("env") == "sft" and a.tokenizer:
            argv += ["--tokenizer", a.tokenizer]
        # --min-trajectories goes only to the builders whose step lists it under `options`
        if a.min_trajectories is not None and "min-trajectories" in (step.get("options") or []):
            argv += ["--min-trajectories", str(a.min_trajectories)]
        # builders that render with the student's tokenizer run in the SFT environment (transformers)
        py = ["uv", "run", "--project", str(ROOT / "setup" / "sft"), "python"] if step.get("env") == "sft" else [sys.executable]
        print(f"# {b.name}: {', '.join(step['writes'])}")
        print("+ " + " ".join(py + argv))
        if not a.dry_run:
            rc |= subprocess.run(py + argv, cwd=ROOT).returncode
            if rc:
                break
    if not a.dry_run and not rc:
        print(f"train on a pool with: uv run exp sft -b {b.name} --pool <setting> --data {out}/<setting>.jsonl")
    return rc
