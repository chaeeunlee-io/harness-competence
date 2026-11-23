"""`exp seed-tables`: tables pooled over independent training runs (seeds), from per-trial files.

A spec file (default `configs/tables/seed_tables.yaml`) lists, for every table cell, the job
folders of each run (seed). This module reads the trials, scores each run exactly as the paper
does (`exp.table.stats`, the benchmark's aggregation and bootstrap settings), pools runs with
equal weight, and writes:

  <out>/seed_tables.md      every table, once per variant in the spec (e.g. the retraining seeds
                            only, s0 and the retraining seeds)
  <out>/seed_tables.csv     one row per cell and variant: pooled mean, CI, per-run values
  <out>/manifest.csv        one row per cell and run: job paths, trials found, trials scored,
                            run mean, and a checksum of the (trial, reward) list
  <compact out>/<id>.md     each table once more as a title, a caption and `mean ± CI` cells only

Runs are named s0, s1, s2, ...: s0 is an arm's first training run (the trainers' default seed), s1 and
up are retrainings with that seed.

Pooling rule. Each run's mean is computed on its own (the paper's estimator for that
benchmark, `exp.table.stats`); the cell mean is the unweighted mean of the run means, so a run with 4 attempts
per task counts the same as a run with 1. The 95% interval is a task bootstrap: each
resample draws tasks with replacement (the same draw for every run, since all runs share
the test set), takes each run's mean of per-task rates on the drawn tasks (the paper's interval
statistic), and averages over runs. With one run this is exactly the paper's interval.
Training-seed variation is shown separately, as the per-run values.

A trial counts when its `verifier/reward.txt` exists (as in `exp table`). A run whose
unscored share is at or above `unscored_flag` (default 3%) is marked ‡. Runs with no local
trials can be given as external numbers (`external` csv: a value from another cluster's own
run of this command); they are marked †, and a cell that has one has no pooled interval.
Runs not available yet are marked ⏳; the cell then shows the mean of the available runs,
marked ⏳ too.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from exp import config as C
from exp.table import read_job, shipped_job, stats  # noqa: F401  (read_job: the reference reader)

SHIPPED = False   # --shipped: read jobs from results/rewards.csv instead of their folders under runs/

ROOT = C.ROOT
PENDING, EXTERNAL = "pending", "ext"


# --------------------------------------------------------------------------- spec
def _expand(s: str) -> str:
    """${VAR:-default} and ${VAR} from the environment."""
    def sub(m):
        name, _, default = m.group(1).partition(":-")
        return os.environ.get(name, default)
    return re.sub(r"\$\{([^}]+)\}", sub, s)


def resolve(ref: str, roots: dict) -> Path:
    """'root:relative/path' -> absolute Path (roots relative to the repo unless absolute). Links are not
    followed, so a job keeps the path the spec gives it."""
    root, sep, rel = ref.partition(":")
    if not sep or root not in roots:
        raise SystemExit(f"seed-tables: job reference {ref!r} must start with one of {sorted(roots)}:")
    base = Path(_expand(roots[root]))
    if not base.is_absolute():
        base = ROOT / base
    return Path(os.path.normpath(base / rel))


def manifest_path(path: Path) -> str:
    """A job folder as the manifest records it: relative to the repository when it is inside it."""
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def cell_refs(table: dict, row: dict, col: dict) -> dict:
    """run -> list of job references (or PENDING / EXTERNAL) for one cell."""
    vars_ = {**table.get("vars", {}), **col.get("vars", {}), **row.get("vars", {})}
    over = (row.get("cells") or {}).get(col["id"], {})
    runs = row.get("runs") or table["runs"]
    out = {}
    for run in runs:
        if run in over:
            v = over[run]
        elif run in (row.get("patterns") or {}):
            v = row["patterns"][run]
        elif run in (table.get("patterns") or {}):
            v = table["patterns"][run]
        else:
            continue
        if v is None:
            continue
        vs = [x.format(**vars_) for x in (v if isinstance(v, list) else [v])]
        out[run] = vs[0] if vs in ([PENDING], [EXTERNAL]) else vs
    return out


def load_external(path: Path) -> dict:
    """(table, row, col, run) -> {value, note}."""
    ext = {}
    if path and path.is_file():
        with open(path) as fh:
            for r in csv.DictReader(fh):
                ext[(r["table"], r["row"], r["col"], r["run"])] = {"value": float(r["value"]), "note": r.get("note", ""),
                                                                    "tasks": int(r["tasks"]) if r.get("tasks") else None}
    return ext


# --------------------------------------------------------------------------- one run
_POOL = ThreadPoolExecutor(max_workers=48)
_CACHE: dict = {}


def _trial(t: Path):
    """(task, reward) for one trial folder, or (None, None) if unscored. Same rules as exp.table.read_job."""
    try:
        r = float((t / "verifier" / "reward.txt").read_text().strip())
    except (OSError, ValueError):
        return None, None
    task = t.name.split("__")[0]
    try:
        task = os.path.basename(json.load(open(t / "config.json"))["task"]["path"].rstrip("/"))
    except Exception:
        pass
    return task, r


def read_folder(job_dir: Path) -> tuple[dict, int]:
    """task -> [reward] and the number of trial folders, read in parallel and cached per folder
    (job folders on a network filesystem are slow to read one file at a time)."""
    if job_dir not in _CACHE and SHIPPED:
        per, found = shipped_job(job_dir) or ({}, 0)
        _CACHE[job_dir] = ({task: list(rewards) for task, rewards in per.items()}, found)
    if job_dir not in _CACHE:
        trials = [t for t in job_dir.iterdir() if t.is_dir() and "__" in t.name] if job_dir.is_dir() else []
        per = {}
        for task, r in _POOL.map(_trial, trials):
            if task is not None:
                per.setdefault(task, []).append(r)
        _CACHE[job_dir] = (per, len(trials))
    return _CACHE[job_dir]


def read_run(paths: list[Path]) -> tuple[dict, int]:
    """Pooled task -> [reward] over a run's job folders (e.g. several passes), and trials found."""
    per, found = {}, 0
    for p in paths:
        fp, n = read_folder(p)
        found += n
        for task, rs in fp.items():
            per.setdefault(task, []).extend(rs)
    return per, found


def checksum(per: dict) -> str:
    h = hashlib.sha256()
    for task in sorted(per):
        h.update(f"{task}:{','.join(f'{r:g}' for r in sorted(per[task]))};".encode())
    return h.hexdigest()[:16]


def _prep(per: dict, b) -> dict:
    bs = b.eval.get("bootstrap") or {}
    if bs.get("binarize"):
        return {t: [1 if float(r) == 1.0 else 0 for r in v] for t, v in per.items()}
    return {t: [float(r) for r in v] for t, v in per.items()}


def _stat(per: dict, tasks: list, b) -> float | None:
    """The benchmark's point statistic on a list of tasks (repeats allowed)."""
    tasks = [t for t in tasks if t in per]
    if not tasks:
        return None
    if b.eval.get("aggregate") == "pooled":
        return 100 * sum(sum(per[t]) for t in tasks) / sum(len(per[t]) for t in tasks)
    return 100 * sum(sum(per[t]) / len(per[t]) for t in tasks) / len(tasks)


def pooled(runs: list[dict], b) -> dict:
    """Equal-weight mean over runs and the task-bootstrap interval of that mean."""
    pers = [_prep(r, b) for r in runs]
    tasks = sorted(set().union(*pers))
    means = [_stat(p, tasks, b) for p in pers]
    mean = sum(means) / len(means)
    bs = b.eval.get("bootstrap") or {"B": 1000, "seed": 20260923, "hi_index": 975}
    rng = random.Random(bs["seed"])
    rates = [{t: sum(v) / len(v) for t, v in p.items()} for p in pers]
    out = []
    for _ in range(bs["B"]):
        # Same draw order as exp.table.stats; like it, the resampled statistic is the mean of
        # per-task rates (the paper's as-run interval), averaged over runs.
        draw = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        vals = []
        for r in rates:
            got = [r[t] for t in draw if t in r]
            if got:
                vals.append(100 * sum(got) / len(got))
        out.append(sum(vals) / len(vals))
    out.sort()
    lo, hi = out[int(0.025 * bs["B"])], out[bs["hi_index"]]
    return {"mean": mean, "lo": lo, "hi": hi, "pm": (hi - lo) / 2, "tasks": len(tasks)}


# --------------------------------------------------------------------------- tables
def build(spec_path: Path, external_path: Path | None = None) -> tuple[list, list, list]:
    spec = yaml.safe_load(open(spec_path))
    roots = spec["roots"]
    flag = float(spec.get("unscored_flag", 0.03))
    texp = spec.get("tasks_expected", {})      # benchmark -> test tasks; a run scoring fewer is marked §
    ext = load_external(external_path or (spec_path.parent / spec.get("external", "external_cells.csv")))
    variants = spec["variants"]
    manifest, cells, rendered = [], [], []
    # Other specs whose local runs replace this spec's `ext` runs with the same ids (e.g. another
    # cluster's run of this command, with its per-trial files exported).
    merged = []
    for m in spec.get("merge") or []:
        mp = Path(m) if Path(m).is_absolute() else spec_path.parent / m
        ms = yaml.safe_load(open(mp))
        idx = {(t["id"], r["id"], c["id"]): (t, r, c) for t in ms["tables"] for r in t["rows"] for c in t["columns"]}
        merged.append((ms["roots"], idx, mp.name))

    t0 = time.time()
    for table in spec["tables"]:
        cols = table["columns"]
        sections, sizes = [], {}
        for vname, vruns in variants.items():
            lines = [f"### Table {table['id']} · {table['title']} · {spec.get('variant_labels', {}).get(vname, vname)}", ""]
            if table.get("note"):
                lines += [table["note"], ""]
            lines.append("| " + " | ".join([table.get("row_header", "")] + [c["label"] for c in cols]) + " |")
            lines.append("|" + "---|" * (len(cols) + 1))
            for row in table["rows"]:
                texts = []
                for col in cols:
                    b = C.benchmark(col.get("benchmark") or row.get("benchmark") or table["benchmark"])
                    refs = cell_refs(table, row, col)
                    use = [r for r in (row.get("variant_runs", {}).get(vname) or vruns) if r in refs]
                    vals, marks, run_data, missing_ext = [], [], [], False
                    for run in use:
                        ref = refs[run]
                        m = {"table": table["id"], "row": row["id"], "col": col["id"], "run": run}
                        run_roots = roots
                        if ref == EXTERNAL:
                            for mroots, idx, mname in merged:
                                hit = idx.get((table["id"], row["id"], col["id"]))
                                mref = cell_refs(*hit).get(run) if hit else None
                                if isinstance(mref, list):
                                    ref, run_roots = mref, mroots
                                    m["merged_from"] = mname
                                    break
                        if ref == PENDING:
                            vals.append(None); marks.append("⏳")
                            if vname == list(variants)[-1]:
                                manifest.append({**m, "source": "pending"})
                            continue
                        if ref == EXTERNAL:
                            e = ext.get((table["id"], row["id"], col["id"], run))
                            if e is None:
                                vals.append(None); marks.append("⏳")
                            else:
                                short = e.get("tasks") and e["tasks"] < texp.get(b.name, 0)
                                vals.append(e["value"]); marks.append("†" + ("§" if short else "")); missing_ext = True
                                if vname == list(variants)[-1]:
                                    sizes.setdefault((col["id"], run), []).append("ext")
                            if vname == list(variants)[-1]:
                                manifest.append({**m, "source": "external" if e else "external-missing",
                                                 "mean": e["value"] if e else "", "note": e["note"] if e else ""})
                            continue
                        paths = [resolve(x, run_roots) for x in ref]
                        per, found = read_run(paths)
                        scored = sum(len(v) for v in per.values())
                        if not per:
                            vals.append(None); marks.append("⏳")
                            if vname == list(variants)[-1]:
                                manifest.append({**m, "source": "missing", "paths": ";".join(manifest_path(p) for p in paths)})
                            continue
                        s = stats(per, b)
                        unscored = 1 - scored / found if found else 0.0
                        if vname == list(variants)[-1]:
                            sizes.setdefault((col["id"], run), []).append((len(per), found))
                        short = len(per) < texp.get(b.name, 0)
                        vals.append(s["mean"]); marks.append(("‡" if unscored >= flag else "") + ("§" if short else "")
                                                             + ("°" if "merged_from" in m else ""))
                        run_data.append(per)
                        if vname == list(variants)[-1]:
                            manifest.append({**m, "source": "local", "paths": ";".join(manifest_path(p) for p in paths),
                                             "trials_found": found, "trials_scored": scored,
                                             "unscored_pct": round(100 * unscored, 1), "tasks": len(per),
                                             "mean": round(s["mean"], 2), "checksum": checksum(per)})
                    texts.append(_render(vals, marks, run_data, missing_ext, b, row, cells, table, col, vname, use))
                label = f"**{row['label']}**" if row.get("bold") else row["label"]
                lines.append("| " + " | ".join([label] + texts) + " |")
            sections.append("\n".join(lines))
        rendered.append(_legend(table, spec, variants, sizes) + "\n\n" + "\n\n".join(sections))
        print(f"  table {table['id']}: done ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    return rendered, cells, manifest


def _run_size(table: dict, col: dict, run: str, entries: list):
    """(attempts, tasks, trials, external?) for one run of one column, or None.

    Local runs: the most common trial count across rows, over the most tasks seen (a job that
    lost every attempt of a task still has that task in its test set). External runs: the size
    declared in the spec (`external_size: {run: [attempts, tasks]}` on the column or table)."""
    local = [e for e in entries if e != "ext"]
    if local:
        found = max(set(f for _, f in local), key=[f for _, f in local].count)
        seen = max(t for t, _ in local)
        att = max(1, round(found / seen)) if seen else 0
        return att, (found // att if att else 0), found, False
    decl = (col.get("external_size") or {}).get(run) or (table.get("external_size") or {}).get(run)
    if decl and entries:
        a, t = decl
        return a, t, a * t, True
    return None


def _legend(table: dict, spec: dict, variants: dict, sizes: dict) -> str:
    """A 'how to read' block: what the parentheses hold, and the size of every run and pooled cell."""
    labels = spec.get("variant_labels", {})
    order = "; ".join(f"*{labels.get(v, v)}*: **{' / '.join(r)}**" for v, r in variants.items())
    runs = list(dict.fromkeys(r for rs in variants.values() for r in rs))
    out = [f"**How to read Table {table['id']}.** Each cell is the pooled mean ± the 95% CI half-width, then the "
           f"per-run means in parentheses, in this order: {order}. s0 is the first training run (the trainers' default "
           "seed); s1 and up are retrainings with that seed. Runs are averaged with equal weight, whatever their size. "
           "A row with its own order (e.g. base: s0 / s0b) says so in its label. Run size = attempts per task × test tasks = trials "
           "(† = a number from elsewhere, size from its protocol):", ""]
    out.append("| Column | " + " | ".join(runs) + " | " + " | ".join(f"Trials in a pooled cell, {labels.get(v, v)}" for v in variants) + " |")
    out.append("|" + "---|" * (1 + len(runs) + len(variants)))
    for col in table["columns"]:
        cells, size = [], {}
        for r in runs:
            entries = sizes.get((col["id"], r), [])
            z = _run_size(table, col, r, entries)
            size[r] = z
            if z is None:
                cells.append("—")
            else:
                mixed = z[3] is False and "ext" in entries
                cells.append(f"{z[0]} × {z[1]} = {z[2]}" + ("†" if z[3] else "") + (" (some rows †)" if mixed else ""))
        tot = []
        for v, rs in variants.items():
            zs = [size[r] for r in rs if size.get(r)]
            n = sum(z[2] for z in zs)
            tot.append(f"{n:,}" + ("†" if any(z[3] for z in zs) else ""))
        out.append(f"| {col['label']} | " + " | ".join(cells) + " | " + " | ".join(tot) + " |")
    return "\n".join(out)


def _fmt(v):
    return "⏳" if v is None else f"{v:.1f}"


def _render(vals, marks, run_data, has_ext, b, row, cells, table, col, vname, use) -> str:
    per_run = " / ".join(_fmt(v) + mk if v is not None else "⏳" for v, mk in zip(vals, marks))
    avail = [v for v in vals if v is not None]
    rec = {"table": table["id"], "row": row["id"], "col": col["id"], "variant": vname,
           "runs": ";".join(use), "per_run": per_run}
    if not avail:
        cells.append({**rec, "mean": ""})
        return "⏳"
    mean = sum(avail) / len(avail)
    partial = len(avail) < len(vals)
    if len(vals) == 1:
        p = pooled(run_data, b) if run_data else None
        txt = f"{mean:.1f}" + (f" ± {p['pm']:.1f}" if p else "") + (marks[0] or "")
        cells.append({**rec, "mean": round(mean, 2), "pm": round(p["pm"], 2) if p else ""})
        return txt
    if partial or has_ext:
        txt = f"{mean:.1f}{'⏳' if partial else ''} ({per_run})"
        cells.append({**rec, "mean": round(mean, 2), "pm": ""})
        return txt
    p = pooled(run_data, b)
    txt = f"{mean:.1f} ± {p['pm']:.1f} ({per_run})"
    cells.append({**rec, "mean": round(mean, 2), "pm": round(p["pm"], 2), "lo": round(p["lo"], 2), "hi": round(p["hi"], 2)})
    return txt


def _write_csv(path: Path, rows: list):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def compact_table(md_table: str) -> str:
    """A table with each cell `mean ± CI` only (per-run values and run flags removed)."""
    t = re.sub(r" \([^()|]*(?: / [^()|]*)+\)", "", md_table)
    return re.sub(r"[‡§°†]", "", t)


def write_compact(folder: Path, table: dict, md_table: str):
    """<folder>/<id>.md: the table's short title (`post_title`), its caption and the compact table.
    Rows with `post: false` stay in the full tables but are left out here."""
    folder.mkdir(parents=True, exist_ok=True)
    for row in table.get("rows", []):
        if row.get("post") is False:
            md_table = re.sub(rf"(?m)^\| (?:\*\*)?{re.escape(row['label'])}(?:\*\*)? \|.*\n?", "", md_table)
    body = f"**{table.get('post_title', table['title'])}**\n\n"
    if table.get("caption"):
        body += f"*{table['caption']}*\n\n"
    (folder / f"{table['id']}.md").write_text(body + compact_table(md_table).strip() + "\n")


def first_table(md: str, after: str) -> str:
    i = md.index(after)
    return re.search(r"(?m)^\|.*\n(?:\|.*\n?)+", md[i:]).group(0)


def compare(spec: dict, cells: list, threshold: float) -> tuple[str, dict]:
    """Where the spec's first and last variants differ (e.g. the retraining seeds alone, and with s0):
    a summary block for the top of the document, and one line per table."""
    vfirst, vlast = list(spec["variants"])[0], list(spec["variants"])[-1]
    lab = spec.get("variant_labels", {})
    first, last = lab.get(vfirst, vfirst), lab.get(vlast, vlast)
    by = {(c["table"], c["row"], c["col"], c["variant"]): c for c in cells}
    rows, per_table, flips = [], {}, []
    for t in spec["tables"]:
        rl = {r["id"]: r["label"] for r in t["rows"]}
        cl = {c["id"]: c["label"] for c in t["columns"]}
        diffs = []
        for r in t["rows"]:
            for c in t["columns"]:
                x, y = by.get((t["id"], r["id"], c["id"], vfirst)), by.get((t["id"], r["id"], c["id"], vlast))
                if not x or not y or x["mean"] == "" or y["mean"] == "":
                    continue
                d = y["mean"] - x["mean"]
                diffs.append((abs(d), d, r["id"], c["id"], x["mean"], y["mean"]))
                if abs(d) >= threshold:
                    rows.append((abs(d), f"| {t['id']} | {rl[r['id']]} | {cl[c['id']]} | {x['mean']:.1f} | {y['mean']:.1f} | {d:+.1f} |"))
        # does the best row in a column change? (base rows are references, not compared)
        for c in t["columns"]:
            best = {}
            for v in (vfirst, vlast):
                cand = [(by[(t["id"], r["id"], c["id"], v)]["mean"], r["id"]) for r in t["rows"]
                        if r["id"] != "base" and (t["id"], r["id"], c["id"], v) in by and by[(t["id"], r["id"], c["id"], v)]["mean"] != ""]
                if cand:
                    cand.sort(reverse=True)
                    best[v] = cand
            if len(best) == 2 and best[vfirst][0][1] != best[vlast][0][1]:
                f, l = best[vfirst], best[vlast]
                flips.append(f"- **Table {t['id']}, {cl[c['id']]}:** best row is {rl[f[0][1]]} ({f[0][0]:.1f} vs {f[1][0]:.1f} "
                             f"{rl[f[1][1]]}) with {first}, but {rl[l[0][1]]} ({l[0][0]:.1f} vs {l[1][0]:.1f} "
                             f"{rl[l[1][1]]}) with {last}.")
        if diffs:
            diffs.sort(reverse=True)
            n = sum(1 for d in diffs if d[0] >= threshold)
            top = diffs[0]
            per_table[t["id"]] = (f"**Variants compared (Table {t['id']}):** {n} of {len(diffs)} cells move by ≥{threshold:g} points "
                                  f"between {first} and {last}; largest: {rl[top[2]]} × {cl[top[3]]}, "
                                  f"{top[4]:.1f} → {top[5]:.1f} ({top[1]:+.1f}).")
    rows.sort(reverse=True)
    out = ["## Summary: where the two variants differ", "",
           f"Difference = ({last}) − ({first}), in points. Cells whose pooled mean moves "
           f"by **≥{threshold:g} points**, largest first. All other cells move by less.", ""]
    if rows:
        out += [f"| Table | Row | Column | {first} | {last} | Difference |", "|---|---|---|---|---|---|"] + [r for _, r in rows]
    else:
        out.append("No cell moves by that much.")
    out += ["", "**Columns where the best row changes:**", ""]
    out += flips or ["- None: in every column the best row is the same in both variants."]
    return "\n".join(out), per_table


def main(a) -> int:
    spec = Path(a.spec)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rendered, cells, manifest = build(spec, Path(a.external) if a.external else None)
    head = (f"# Seed tables\n\nGenerated by `exp seed-tables --spec {C.rel(spec)}`. Do not edit by hand.\n\n"
            "Cell: pooled mean ± 95% task-bootstrap half-width (per-run values). Runs are pooled with equal weight. "
            "‡ ≥3% of the run's trials unscored · § the run scored fewer tasks than the test set (trials never created) · "
            "° run scored here from another cluster's per-trial export (merged spec) · "
            "† number from another cluster (no pooled interval) · "
            "⏳ not available yet (cell mean over the available runs). Paths and checksums: manifest.csv.\n")
    spec_d = yaml.safe_load(open(spec))
    summary, per_table = compare(spec_d, cells, float(spec_d.get("diff_flag", 2.0)))
    rendered = [(per_table.get(t["id"], "") + "\n\n" if per_table.get(t["id"]) else "") + r
                for t, r in zip(spec_d["tables"], rendered)]
    (out / "seed_tables.md").write_text(head + "\n" + summary + "\n\n---\n\n" + "\n\n---\n\n".join(rendered) + "\n")
    post = spec_d.get("compact_variant", list(spec_d["variants"])[-1])
    label = spec_d.get("variant_labels", {}).get(post, post)
    for t, r in zip(spec_d["tables"], rendered):
        write_compact(Path(a.compact_out), t, first_table(r, f"### Table {t['id']} · {t['title']} · {label}"))
    _write_csv(out / "seed_tables.csv", cells)
    _write_csv(out / "manifest.csv", manifest)
    print(f"wrote {out / 'seed_tables.md'}, seed_tables.csv, manifest.csv "
          f"({len(cells)} cells, {sum(1 for m in manifest if m['source'] == 'local')} local runs, "
          f"{sum(1 for m in manifest if m['source'] != 'local')} external/pending)")
    return 0
