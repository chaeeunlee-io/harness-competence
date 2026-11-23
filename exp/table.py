"""`exp table`: rebuild experiment results.

By default, reads trial rewards and task IDs from the evaluation jobs under
`runs/evals/` (another root with `--evals-dir`) and applies the benchmark's
configured aggregation and bootstrap procedure. AppWorld bootstraps over
scenarios.

With `--shipped`, reads cell values from `results/cells.csv`; per-trial outcomes
are not included in the public release. `--format cells` writes results in the
same schema as `results/cells.csv`, so a re-run can be compared with them.
"""
from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from exp import config as C, render as R

ROOT = C.ROOT


@dataclass(frozen=True)
class Source:
    """Which evaluation jobs a table reads: the root, the attempts per task (default: the reported
    protocol's) and the passes to pool per task (default: the first)."""
    root: Path | None = None
    n_attempts: int | None = None
    passes: tuple = (None,)

    @classmethod
    def of(cls, a) -> "Source":
        root = getattr(a, "evals_dir", None)
        return cls(Path(root) if root else None, getattr(a, "n_attempts", None),
                   tuple(getattr(a, "eval_pass", None) or (None,)))

    def describe(self) -> str:
        out = C.rel(self.root or R.evals_root())
        if self.n_attempts:
            out += f", {self.n_attempts} attempt(s) per task"
        if self.passes != (None,):
            out += ", pass " + "+".join(str(p) for p in self.passes)
        return out


def stats(per: dict, b, cluster_unit: str | None = None, estimator: str = "task_mean") -> dict | None:
    """per: task -> [reward]. Mirrors the as-run implementations exactly (seed, ordering, index).

    estimator: task_mean    = resample tasks, statistic = mean of per-task means
                              (the as-run per-task estimator)
               trial_pooled = resample tasks, statistic = pooled trial mean of the resample
                              (the 3H-SFT and 3H-GRPO rows of grpo-3h-vs-single)
    """
    if not per:
        return None
    bs = b.eval.get("bootstrap") or {"B": 1000, "seed": 20260923, "hi_index": 975, "binarize": True}
    if bs.get("binarize"):
        per = {t: [1 if float(r) in (1.0,) else 0 for r in v] for t, v in per.items()}
    tasks = sorted(per)
    trials = sum(len(per[t]) for t in tasks)
    pooled = 100 * sum(sum(per[t]) for t in tasks) / trials
    rates = {t: sum(per[t]) / len(per[t]) for t in tasks}
    task_level = 100 * sum(rates.values()) / len(rates)
    if (cluster_unit or b.eval.get("bootstrap_unit")) == "scenario":
        groups = defaultdict(list)
        for t, r in rates.items():
            groups[t.split("_")[0]].append(r)
        units = [sum(v) / len(v) for _, v in sorted(groups.items())]
    else:
        units = [rates[t] for t in tasks]
    rng = random.Random(bs["seed"])
    n, out = len(units), []
    if estimator == "trial_pooled":
        data = [per[t] for t in tasks]
        for _ in range(bs["B"]):
            idx = [rng.randrange(n) for _ in range(n)]
            out.append(100 * sum(sum(data[i]) for i in idx) / sum(len(data[i]) for i in idx))
    else:
        for _ in range(bs["B"]):
            out.append(100 * sum(units[rng.randrange(n)] for _ in range(n)) / n)
    out.sort()
    lo, hi = out[int(0.025 * bs["B"])], out[bs["hi_index"]]
    mean = pooled if b.eval.get("aggregate") == "pooled" else task_level
    return {"mean": mean, "pooled": pooled, "task_level": task_level, "lo": lo, "hi": hi,
            "pm": (hi - lo) / 2, "tasks": len(tasks), "trials": trials}


def job_finished(job_dir: Path) -> bool:
    """A Harbor job folder is complete once its result.json carries finished_at (it is written at start)."""
    rj = job_dir / "result.json"
    try:
        return bool(json.load(open(rj)).get("finished_at"))
    except (OSError, ValueError):
        return False


def read_job(job_dir: Path) -> dict:
    per = defaultdict(list)
    if not job_dir.is_dir():
        return {}
    for t in sorted(job_dir.iterdir()):
        rp = t / "verifier" / "reward.txt"
        if not rp.is_file():
            continue
        try:
            r = float(rp.read_text().strip())
        except ValueError:
            continue
        task = t.name.split("__")[0]
        cfg = t / "config.json"
        if cfg.is_file():
            try:
                task = os.path.basename(json.load(open(cfg))["task"]["path"].rstrip("/"))
            except Exception:
                pass
        per[task].append(r)
    return dict(per)


REWARDS_CSV = ROOT / "results" / "rewards.csv"
_REWARDS = None


def shipped_rewards() -> dict:
    """results/rewards.csv: (root, job) -> (task -> [reward], trials found). The file has one row per task of an
    evaluation job: the rewards of its scored attempts and the number of attempts found."""
    global _REWARDS
    if _REWARDS is None:
        _REWARDS = {}
        if REWARDS_CSV.is_file():
            with open(REWARDS_CSV, newline="") as fh:
                for r in csv.DictReader(fh):
                    per, found = _REWARDS.setdefault((r["root"], r["job"]), ({}, [0]))
                    found[0] += int(r["trials"])
                    if r["rewards"]:
                        per[r["task"]] = [float(x) for x in r["rewards"].split(";")]
            _REWARDS = {k: (per, found[0]) for k, (per, found) in _REWARDS.items()}
    return _REWARDS


def shipped_job(job_dir: Path) -> tuple[dict, int] | None:
    """A job's shipped rewards, by its folder (runs/<root>/<job>), or None if the file does not have it."""
    try:
        root, *job = Path(job_dir).resolve().relative_to(R.RUNS.resolve()).parts
    except ValueError:
        return None
    return shipped_rewards().get((root, "/".join(job)))


_CELLS = None


def _cells():
    """results/cells.csv (reported rows) indexed by (benchmark, model, tier, layer, arm, run, harness, metric)."""
    global _CELLS
    if _CELLS is None:
        _CELLS = {}
        path = ROOT / "results" / "cells.csv"
        if path.is_file():
            with open(path) as fh:
                for r in csv.DictReader(fh):
                    if r.get("status", "reported") != "reported":
                        continue
                    k = (r["benchmark"], r["model"], r["tier"], r["layer"], r["arm"], r["run"], r["harness"], r["metric"])
                    assert k not in _CELLS, f"results/cells.csv: duplicate reported cell {k}"
                    _CELLS[k] = r
    return _CELLS


def _shipped_cell(b, model: str, setting: str, harness: str, split: str, metric: str = "mean", seed: int | None = None):
    """A published cell, from the unified results/cells.csv. Untrained settings are filed under run s0."""
    arm = _arm_label(b, setting)
    run = "s0" if arm in R.UNTRAINED else R.run_id(seed)
    r = _cells().get((b.name, model, b.tier(split) or "", _layer(b, setting, harness), arm, run, harness, metric))
    if not r:
        return None
    f = lambda k: float(r[k]) if r[k] not in ("", None) else 0.0
    return {"mean": f("value"), "pooled": f("value"), "task_level": f("value"),
            "lo": f("ci_lo"), "hi": f("ci_hi"), "pm": f("plusminus"),
            "tasks": int(r["tasks"] or 0), "trials": int(r["trials"] or 0)}


def _per(b, model: str, setting: str, harness: str, split: str, shipped: bool, src: Source,
         seed: int | None = None):
    """task -> [reward] for one cell, plus the job ids it came from.

    With --shipped the rewards come from results/rewards.csv, which holds the scores of every attempt
    without the trial folders. (None, []) if it does not have the cell: the published value is then
    read from results/cells.csv.
    With `seed`, trained settings are read from their `s<seed>` run.
    Several passes of a cell are pooled per task.
    """
    arm = _arm_label(b, setting, model)
    dirs = [R.eval_dir(b, model, arm, harness, split, seed=seed, n_attempts=src.n_attempts, eval_pass=p, root=src.root)
            for p in src.passes]
    name = R.job_id(dirs[0], src.root)
    per = defaultdict(list)
    if shipped:
        found = [shipped_job(d) for d in dirs]
        if not any(found):
            return None, []
        for per_job, _ in filter(None, found):
            for task, rewards in per_job.items():
                per[task].extend(rewards)
        return dict(per), [name]
    running = False
    for job in dirs:
        rewards_by_task = read_job(job)
        running |= bool(rewards_by_task) and not job_finished(job)
        for task, rewards in rewards_by_task.items():
            per[task].extend(rewards)
    return dict(per), [name + ("*" if running else "")]


def pass_at(per: dict, b, thresh: float = 0.99) -> dict | None:
    """pass@n over the benchmark's resampling unit, same bootstrap constants as stats()."""
    if not per:
        return None
    bs = b.eval.get("bootstrap") or {"B": 1000, "seed": 20260923, "hi_index": 974}
    tasks = sorted(per)
    solved = {t: 1 if max(per[t]) >= thresh else 0 for t in tasks}
    if b.eval.get("bootstrap_unit") == "scenario":
        groups = defaultdict(list)
        for t, v in solved.items():
            groups[t.split("_")[0]].append(v)
        units = [sum(v) / len(v) for _, v in sorted(groups.items())]
    else:
        units = [solved[t] for t in tasks]
    n = len(units)
    point = 100 * sum(units) / n
    rng = random.Random(bs["seed"])
    out = [100 * sum(units[rng.randrange(n)] for _ in range(n)) / n for _ in range(bs["B"])]
    out.sort()
    lo, hi = out[int(0.025 * bs["B"])], out[bs["hi_index"]]
    return {"value": point, "lo": lo, "hi": hi, "pm": (hi - lo) / 2,
            "tasks": len(tasks), "trials": sum(len(v) for v in per.values()), "units": n}


def cell(b, model: str, setting: str, harness: str, split: str, shipped: bool, src: Source, estimator: str = "task_mean",
         seed: int | None = None):
    per, jobs = _per(b, model, setting, harness, split, shipped, src, seed)
    if per is None:
        return _shipped_cell(b, model, setting, harness, split, seed=seed)
    c = stats(per, b, estimator=estimator)
    return {**c, "running": _running(jobs)} if c else None


def seeds_cell(b, model: str, setting: str, harness: str, split: str, src: Source, seeds: list[int],
               estimator: str = "task_mean") -> dict | None:
    """One cell over several retraining seeds (`exp table --seeds 1,2,3`).

    The per-task trials of every seed that has been run are pooled and bootstrapped with the
    benchmark's procedure, exactly as repeated passes are; the per-seed means are reported next to
    it so the spread across retrainings is visible. Base and teacher rows are not retrained, so
    they are read once, unseeded.
    """
    if _arm_label(b, setting) in ("base", "teacher"):
        per, jobs = _per(b, model, setting, harness, split, False, src)
        return {**stats(per, b, estimator=estimator), "running": _running(jobs)} if per else None
    pooled, by_seed, running = defaultdict(list), {}, set()
    for seed in seeds:
        per, jobs = _per(b, model, setting, harness, split, False, src, seed)
        if not per:
            continue
        by_seed[seed] = stats(per, b, estimator=estimator)["mean"]
        if _running(jobs):
            running.add(seed)
        for task, rewards in per.items():
            pooled[task].extend(rewards)
    if not by_seed:
        return None
    c = stats(dict(pooled), b, estimator=estimator)
    means = list(by_seed.values())
    avg = sum(means) / len(means)
    c["seeds"], c["running"] = by_seed, running
    c["sd"] = (sum((m - avg) ** 2 for m in means) / (len(means) - 1)) ** 0.5 if len(means) > 1 else None
    return c


def _running(jobs: list[str]) -> bool:
    return any(j.endswith("*") for j in jobs)


def _fmt_seeds(c: dict | None, seeds: list[int]) -> str:
    if not c:
        return "—"
    out = f"{c['mean']:.1f} ± {c['pm']:.1f}"
    if "seeds" not in c:
        return out + ("*" if c.get("running") else "")
    per_seed = ", ".join((f"s{s}={c['seeds'][s]:.1f}" + ("*" if s in c["running"] else "")) if s in c["seeds"]
                         else f"s{s}=—" for s in seeds)
    sd = f"; SD {c['sd']:.1f}" if c["sd"] is not None else ""
    return f"{out} ({per_seed}{sd})"


CELL_COLS = ["benchmark", "model", "tier", "layer", "arm", "run", "harness", "metric", "value",
             "ci_lo", "ci_hi", "plusminus", "tasks", "trials", "bootstrap_unit",
             "bootstrap_n", "status", "source", "note"]


def _layer(b, setting: str, harness: str) -> str:
    """The sampling protocol of a cell: the teacher's own protocol, else the benchmark's per-harness layer."""
    if setting == "teacher":
        return R.teacher_protocol(b).get("layer") or b.layer(harness)
    return b.layer(harness)


def _settings(b, s) -> list[str]:
    """Rows of one experiment setting, in order, without repeats (rl rows bring their init pool)."""
    st = (["teacher"] if s.get("teacher") else []) + ([b.rl["init_pool"]] if s.get("rl") else []) + \
         list(s.get("pools", [])) + [f"rl:{r}" for r in s.get("rl", [])]
    return list(dict.fromkeys(st))


def _arm_label(b, setting: str, model: str | None = None) -> str:
    """A setting's arm name. With `model`, a GRPO arm is named as its runs and jobs are, with the serving protocol
    it was trained under; without, as published (results/cells.csv)."""
    if setting == "teacher":
        return "teacher"
    if setting.startswith("rl:"):
        rollout, _, init = setting[3:].partition("@")   # `rl:<rollout>[@<init>]`, see experiments.yaml
        return R.rl_name(b, rollout, init or None, model=model)
    return setting


def cells(a) -> int:
    """Emit this experiment's cells in the unified schema of results/cells.csv."""
    e = C.experiments().get(a.experiment)
    if not e:
        raise SystemExit(f"unknown experiment {a.experiment!r}; see `exp list experiments`")
    src = Source.of(a)
    seed = getattr(a, "seed", None)
    rows = []
    for s in e["settings"]:
        b = C.benchmark(s["benchmark"])
        split = s.get("split", "test")
        tier = b.tier(split) or ""
        unit = b.eval.get("bootstrap_unit") or "task"
        for st in _settings(b, s):
            for h in b.harnesses(s.get("harnesses", "trained")):
                per, jobs = _per(b, s["model"], st, h, split, a.shipped, src, seed)
                layer = _layer(b, st, h)
                arm = _arm_label(b, st, s["model"])
                base = dict(benchmark=b.name, model=s["model"], tier=tier, layer=layer, arm=arm,
                            run="s0" if arm in R.UNTRAINED else R.run_id(seed), harness=h, bootstrap_unit=unit,
                            status="reported", source="published" if per is None else "computed")
                note = ("shipped" if a.shipped else "runs") + (": " + ",".join(j.rstrip("*") for j in jobs) if jobs else "")
                if _running(jobs):
                    note += " (job still running: partial)"
                if per is None:            # no per-trial data shipped: echo results/cells.csv
                    for metric in ("mean", f"pass{b.eval.get('n_attempts', 1)}"):
                        c = _shipped_cell(b, s["model"], st, h, split, metric, seed)
                        if not c:
                            continue
                        rows.append({**base, "metric": metric, "value": f"{c['mean']:.2f}",
                                     "ci_lo": f"{c['lo']:.2f}", "ci_hi": f"{c['hi']:.2f}",
                                     "plusminus": f"{c['pm']:.2f}", "tasks": c["tasks"],
                                     "trials": c["trials"] or "", "bootstrap_n": "",
                                     "note": "published value from results/cells.csv"})
                    continue
                if not per:
                    continue
                m = stats(per, b, estimator=(s.get("ci") or {}).get(st, "task_mean"))
                rows.append({**base, "metric": "mean", "value": f"{m['mean']:.2f}",
                             "ci_lo": f"{m['lo']:.2f}", "ci_hi": f"{m['hi']:.2f}",
                             "plusminus": f"{m['pm']:.2f}", "tasks": m["tasks"],
                             "trials": m["trials"], "bootstrap_n": m["tasks"], "note": note})
                if b.eval.get("n_attempts", 1) > 1:
                    pn = pass_at(per, b)
                    rows.append({**base, "metric": f"pass{b.eval['n_attempts']}",
                                 "value": f"{pn['value']:.2f}", "ci_lo": f"{pn['lo']:.2f}",
                                 "ci_hi": f"{pn['hi']:.2f}", "plusminus": f"{pn['pm']:.2f}",
                                 "tasks": pn["tasks"], "trials": pn["trials"],
                                 "bootstrap_n": pn["units"], "note": note})
    out = open(a.out, "w", newline="") if a.out else __import__("sys").stdout
    w = csv.DictWriter(out, fieldnames=CELL_COLS)
    w.writeheader()
    w.writerows(rows)
    if a.out:
        out.close()
        print(f"wrote {a.out}: {len(rows)} rows")
    return 0


def main(a) -> int:
    if getattr(a, "format", "table") == "cells":
        return cells(a)
    e = C.experiments().get(a.experiment)
    if not e:
        raise SystemExit(f"unknown experiment {a.experiment!r}; see `exp list experiments`")
    src = Source.of(a)
    seeds = getattr(a, "seeds", None)
    print(f"# {e['title']}   ({'shipped results' if a.shipped else 'runs: ' + src.describe()})\n")
    if seeds:
        print(f"Trained rows pool the per-task trials of seeds {', '.join(map(str, seeds))}: "
              "mean ± 95% CI half-width of the pooled bootstrap, then each seed's own mean and their "
              "sample SD. Base rows are the unseeded run.\n")
    empties = partial = False
    for s in e["settings"]:
        b = C.benchmark(s["benchmark"])
        split = s.get("split", "test")
        hs = b.harnesses(s.get("harnesses", "trained"))
        settings = _settings(b, s)
        tier = f" ({b.tier(split)})" if b.tier(split) else ""
        n_att = src.n_attempts or R.protocol_attempts(b, "teacher" if s.get("teacher") else "")
        print(f"## {b.cfg['display']}{tier}   model={s['model']}   attempts per task={n_att}   mean ± 95% CI half-width")
        print("| setting | " + " | ".join(hs) + " |")
        print("|---|" + "---|" * len(hs))
        for st in settings:
            row = []
            for h in hs:
                est = (s.get("ci") or {}).get(st, "task_mean")
                if seeds:
                    row.append(_fmt_seeds(seeds_cell(b, s["model"], st, h, split, src, seeds, est), seeds))
                    continue
                c = cell(b, s["model"], st, h, split, a.shipped, src, est, seed=getattr(a, "seed", None))
                row.append("—" if not c else f"{c['mean']:.1f} ± {c['pm']:.1f}" + ("*" if c.get("running") else ""))
            label = {"teacher": "Qwen3-32B teacher"}.get(st, st)
            if st.startswith("rl:"):
                rollout, _, init = st[3:].partition("@")
                label = f"GRPO, rollouts on {rollout}" + (f", from {init}" if init else "")
            elif st in b.pools():
                label = b.pool(st).get("paper", st)
            print(f"| {label} | " + " | ".join(row) + " |")
            if "—" in row:
                empties = True
            partial |= any("*" in x for x in row)
        print()
    if partial:
        print("* : partial: the job is still running (its result.json has no finished_at); the value covers the trials done so far.")
    if empties:
        print("— : no result: " + ("the cell has not been run under runs/ yet." if not a.shipped else
              "no reported row in results/cells.csv for this cell (the paper does not report it)."))
    return 0
