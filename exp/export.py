"""`exp export-results`: write the result files that ship with the repository, from the evaluation jobs.

  results/rewards.csv   one row per task of every evaluation job under runs/evals and runs/evals-cluster2
                        that the configs describe (model, benchmark and setting):
                        the rewards of its scored attempts and the number of attempts found. It holds what
                        every mean and interval is computed from, without the trial folders.
  results/cells.csv     one row per run of an arm on a harness, per metric: mean, pass@n and their 95% CI.
  results/runs.csv      one row per run of an arm: the experiments that use the arm, where its checkpoint is,
                        what it was started from, and how many evaluation jobs it has.

`exp table`, `exp seed-tables` and `exp control-tables` read the first with `--shipped`.

Rows of results/cells.csv whose `source` is `published` are values without per-trial files here (evaluated
elsewhere, or metrics over several cells); they are kept as they are. Every other row is recomputed.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from exp import config as C, render as R, seedtables as S, table as T

ROOTS = ("evals", "evals-cluster2")
REWARD_COLS = ["root", "job", "task", "trials", "rewards"]
DEPTH = 6   # <task set>/<model>/<setting>/<run>/<harness>/<pass>


def _trial(t: Path):
    """(task, trial folder, reward or None) for one trial folder; the task is named as exp.table.read_job names it."""
    task = t.name.split("__")[0]
    try:
        task = os.path.basename(json.load(open(t / "config.json"))["task"]["path"].rstrip("/"))
    except Exception:
        pass
    try:
        reward = float((t / "verifier" / "reward.txt").read_text().strip())
    except (OSError, ValueError):
        reward = None
    return task, t.name, reward


def read_job(job_dir: Path) -> dict:
    """task -> [reward or None per attempt], in trial-folder order."""
    trials = sorted(t for t in job_dir.iterdir() if t.is_dir() and "__" in t.name)
    per = defaultdict(list)
    for task, _, reward in S._POOL.map(_trial, trials):
        per[task].append(reward)
    return dict(per)


def configured(job: str) -> bool:
    """Whether the configs describe the job: a model of configs/models.yaml on `base`, `teacher`, a GRPO run or a
    pool of the benchmark."""
    task_set, model, setting = job.split("/")[:3]
    bench = next((b for b in sorted(C.benchmark_names(), key=len, reverse=True) if task_set.startswith(b)), None)
    return (bench is not None and model in C.models()
            and (setting in R.UNTRAINED or setting.startswith("grpo-") or setting in C.benchmark(bench).pools()))


def jobs(root: Path) -> list[Path]:
    """The evaluation jobs under a root that the configs describe."""
    return sorted(p for p in root.glob("/".join(["*"] * DEPTH))
                  if p.is_dir() and configured(p.relative_to(root).as_posix()))


def export_rewards(out: Path) -> dict:
    """Write rewards.csv; returns (root, job) -> task -> [scored rewards]."""
    data, rows = {}, []
    for root in ROOTS:
        found = jobs(R.RUNS / root)
        for i, job_dir in enumerate(found, 1):
            job = job_dir.relative_to(R.RUNS / root).as_posix()
            per = read_job(job_dir)
            data[(root, job)] = {t: [r for r in rs if r is not None] for t, rs in per.items()}
            for task in sorted(per):
                scored = [r for r in per[task] if r is not None]
                rows.append({"root": root, "job": job, "task": task, "trials": len(per[task]),
                             "rewards": ";".join(f"{r:g}" for r in scored)})
            if i % 100 == 0 or i == len(found):
                print(f"  {root}: {i}/{len(found)} jobs", file=sys.stderr, flush=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=REWARD_COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {C.rel(out)}: {len(rows)} rows, {len(data)} jobs", file=sys.stderr)
    return data


def _reported_jobs() -> set:
    """Jobs a committed table reads: the manifests of the seed and control tables."""
    used = set()
    for name in ("seed-tables", "control-tables"):
        path = C.ROOT / "results" / "tables" / name / "manifest.csv"
        if path.is_file():
            for r in csv.DictReader(open(path)):
                for p in (r.get("paths") or "").split(";"):
                    parts = Path(p.strip()).parts
                    if len(parts) > 2 and parts[0] == "runs" and parts[1] in ROOTS:
                        used.add((parts[1], "/".join(parts[2:])))
    return used


def computed_cells(data: dict) -> list[dict]:
    """One row per run, harness and metric, from the rewards. Passes of one evaluation are pooled per task; the
    second evaluation of an untrained model is its run s0b."""
    groups = defaultdict(list)
    for (root, job) in data:
        task_set, model, setting, run, harness, pass_id = job.split("/")
        if "-train" in task_set:      # probes on training tasks are inputs of the RL bands, not results
            continue
        n, _, repeat = pass_id.partition("-p")
        if run == "none":
            run, repeat = ("s0b" if repeat else "s0"), ""
        groups[(root, task_set, model, setting, run, harness, n)].append((repeat, job))
    reported = _reported_jobs()
    rows = []
    for (root, task_set, model, setting, run, harness, n), members in sorted(groups.items()):
        bench = next(b for b in sorted(C.benchmark_names(), key=len, reverse=True) if task_set.startswith(b))
        b = C.benchmark(bench)
        per = defaultdict(list)
        for _, job in sorted(members):
            for task, rewards in data[(root, job)].items():
                if rewards:
                    per[task].extend(rewards)
        if not per:
            continue
        attempts = int(n[1:]) * len(members)
        layer = re.sub(r"^n\d+", f"n{attempts}", T._layer(b, setting, harness))
        used = any((root, job) in reported for _, job in members)
        base = dict(benchmark=bench, model=model, tier=task_set[len(bench) + 1:], layer=layer, arm=setting, run=run,
                    harness=harness, bootstrap_unit=b.eval.get("bootstrap_unit") or "task",
                    status="reported" if used else "extra", source="computed",
                    note="jobs=" + ",".join(f"{root}:{job}" for _, job in sorted(members)))
        m = T.stats(dict(per), b)
        rows.append({**base, "metric": "mean", "value": f"{m['mean']:.2f}", "ci_lo": f"{m['lo']:.2f}",
                     "ci_hi": f"{m['hi']:.2f}", "plusminus": f"{m['pm']:.2f}", "tasks": m["tasks"],
                     "trials": m["trials"], "bootstrap_n": m["tasks"]})
        if attempts > 1:
            pn = T.pass_at(dict(per), b)
            rows.append({**base, "metric": f"pass{attempts}", "value": f"{pn['value']:.2f}", "ci_lo": f"{pn['lo']:.2f}",
                         "ci_hi": f"{pn['hi']:.2f}", "plusminus": f"{pn['pm']:.2f}", "tasks": pn["tasks"],
                         "trials": pn["trials"], "bootstrap_n": pn["units"]})
    return rows


KEY = ("benchmark", "model", "tier", "layer", "arm", "run", "harness", "metric")


def export_cells(out: Path, data: dict) -> None:
    published = []
    if out.is_file():
        published = [r for r in csv.DictReader(open(out, newline="")) if r.get("source") == "published"]
    have = {tuple(r[k] for k in KEY) for r in published if r["status"] == "reported"}
    # a published row marked "superseded by run <run>" hands its place to that run's row
    promoted = set()
    for r in published:
        m = re.search(r"superseded by run (\w+)", r["note"]) if r["status"] == "superseded" else None
        if m:
            promoted.add(tuple(m.group(1) if k == "run" else r[k] for k in KEY))
    rows, seen = list(published), set(have)
    for r in computed_cells(data):
        k = tuple(r[c] for c in KEY)
        if k in promoted:
            r["status"] = "reported"
        if r["status"] == "reported" and k in seen:      # one reported row per cell: the published value, then the first cluster
            r["status"] = "extra"
        seen.add(k)
        rows.append(r)
    rows.sort(key=lambda r: tuple(r[k] for k in KEY) + (r["status"], r["note"]))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=T.CELL_COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {C.rel(out)}: {len(rows)} rows ({len(published)} published, {len(rows) - len(published)} computed)", file=sys.stderr)


def _arm_experiments() -> dict:
    """(benchmark, model, setting) -> the experiments whose settings include it."""
    uses = defaultdict(list)
    for name, e in C.experiments().items():
        for s in e["settings"]:
            b = C.benchmark(s["benchmark"])
            for st in T._settings(b, s):
                key = (b.name, s["model"], T._arm_label(b, st, s["model"]))
                if name not in uses[key]:
                    uses[key].append(name)
    return uses


RUN_COLS = ["benchmark", "model", "setting", "run", "kind", "groups", "experiments", "checkpoint", "started_from", "eval_jobs"]


def export_runs(out: Path, data: dict) -> None:
    """The index of runs: every (benchmark, model, setting, run) that has a checkpoint, a GRPO run or an evaluation."""
    evals = defaultdict(int)
    for (root, job) in data:
        task_set, model, setting, run, _, _ = job.split("/")
        bench = next(b for b in sorted(C.benchmark_names(), key=len, reverse=True) if task_set.startswith(b))
        evals[(bench, model, setting, run)] += 1
    found = {k: None for k in evals}
    for kind in ("checkpoints", "rl"):
        for d in sorted((R.RUNS / kind).glob("*/*/*/*")):
            if d.is_dir():
                found[tuple(d.relative_to(R.RUNS / kind).parts)] = d
    uses, groups = _arm_experiments(), {k: e.get("group", "") for k, e in C.experiments().items()}
    rows = []
    for key in sorted(found):
        bench, model, setting, run = key
        d = found[key]
        start = ""
        if d is not None and (d / "exp_grpo.json").is_file():
            start = json.load(open(d / "exp_grpo.json")).get("init", "")
            start = "the base model" if start == "base" else C.repo_path(start)
        names = uses.get((bench, model, setting), [])
        rows.append({"benchmark": bench, "model": model, "setting": setting, "run": run,
                     "kind": "untrained" if setting in R.UNTRAINED else "grpo" if setting.startswith("grpo-") else "sft",
                     "groups": ";".join(sorted({groups[n] for n in names})), "experiments": ";".join(names),
                     "checkpoint": C.rel(d) if d is not None else "", "started_from": start, "eval_jobs": evals.get(key, 0)})
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RUN_COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    unclaimed = sum(1 for r in rows if not r["experiments"])
    print(f"wrote {C.rel(out)}: {len(rows)} runs ({unclaimed} of arms no experiment lists)", file=sys.stderr)


def main(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = export_rewards(out / "rewards.csv")
    export_cells(out / "cells.csv", data)
    export_runs(out / "runs.csv", data)
    return 0
