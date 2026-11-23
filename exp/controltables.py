"""`exp control-tables`: tables of single runs and small run sets (controls, GRPO), from per-trial files.

The sibling of `exp seed-tables` for tables whose cells do not share one run layout: a cell may be
one run (a control trained once), several runs (seeds), the s0 run, or a number from
elsewhere. It reuses `exp.seedtables` for reading, scoring (`exp.table.stats`, the paper's
estimator), equal-weight pooling and the task-bootstrap interval.

Spec (default `configs/tables/control_tables.yaml`): tables of rows x columns. A cell is a map
`run label -> job reference(s)`, `ext` (external_cells.csv) or `pending`, built from patterns:

  - row `runs: {label: pattern}` (patterns use `{var}` from table, column and row `vars`), or
  - an explicit `cells: {col_id: {label: pattern}}` on the row; `cells: {col_id: none}` blanks it.

Extras:
  - `tasks_re` on a column or row keeps only tasks whose id matches (e.g. MAB templates).
  - a row with `derived: "<expr>"` computes each column from other rows' means
    (row ids as variables, e.g. "(fail - task) / (comb - task)"); `format: pct` shows it as %.
  - a column with `derived: "<expr>"` computes each row from other columns' means.
  - `optional: true` on a table: a run whose folder is missing is left out silently
    (for seeds that exist for some cells only).

Outputs in --out: control_tables.md, control_tables.csv (one row per cell), manifest.csv (one row
per cell and run: absolute paths, trials found / scored, mean, checksum).
"""
from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path

import yaml

from exp import config as C
from exp import seedtables as S
from exp.table import stats

PENDING, EXTERNAL, NONE = "pending", "ext", "none"
RUN_NAMES = {"s0": "the arm's first training run (default seed)",
             "s0b": "a second training run at the default seed (for the base model, its second evaluation)",
             "s1": "retraining seed 1", "s2": "retraining seed 2", "s3": "retraining seed 3", "s4": "retraining seed 4",
             "s5": "retraining seed 5"}


def _refs(table: dict, row: dict, col: dict) -> dict | None:
    """run label -> [job refs] | 'ext' | 'pending', or None for a blank cell."""
    vars_ = {**table.get("vars", {}), **col.get("vars", {}), **row.get("vars", {})}
    over = (row.get("cells") or {}).get(col["id"])
    if over == NONE:
        return None
    src = over if isinstance(over, dict) else (row.get("runs") or table.get("runs") or {})
    out = {}
    for label, v in src.items():
        vs = [_fill(x, vars_) for x in (v if isinstance(v, list) else [v])]
        out[_fill(label, vars_)] = vs[0] if vs in ([PENDING], [EXTERNAL]) else vs   # a label may name a variable too
    return out


def _fill(pattern: str, vars_: dict) -> str:
    """str.format, repeated so a variable may name another one (e.g. arm: "{reh}")."""
    for _ in range(3):
        new = pattern.format(**vars_)
        if new == pattern:
            break
        pattern = new
    return pattern


def _filter(per: dict, pattern: str | None) -> dict:
    if not pattern:
        return per
    rx = re.compile(pattern)
    return {t: v for t, v in per.items() if rx.search(t)}


def _eval(expr: str, env: dict):
    try:
        return eval(expr, {"__builtins__": {}, "max": max, "min": min, "abs": abs}, env)   # spec expressions only
    except (ZeroDivisionError, TypeError, NameError):
        return None


def build(spec_path: Path, external_path: Path | None = None):
    spec = yaml.safe_load(open(spec_path))
    roots = spec["roots"]
    flag = float(spec.get("unscored_flag", 0.03))
    ext = S.load_external(external_path or (spec_path.parent / spec.get("external", "external_cells.csv")))
    rendered, cells, manifest = [], [], []
    t0 = time.time()
    for table in spec["tables"]:
        if table.get("kind") == "pools":
            from exp import poolstats
            md, c2, m2 = poolstats.render(table, lambda ref: S.resolve(ref, roots))
            rendered.append(md); cells += c2; manifest += m2
            print(f"  table {table['id']}: done ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
            continue
        cols, rows = table["columns"], table["rows"]
        means, texts, runs_lines = {}, {}, []
        plain_cols = [c for c in cols if not c.get("derived")]
        for row in rows:
            if row.get("derived"):
                continue
            for col in plain_cols:
                refs = _refs(table, row, col)
                if refs is None:
                    texts[(row["id"], col["id"])] = "—"
                    continue
                b = C.benchmark(col.get("benchmark") or row.get("benchmark") or table["benchmark"])
                tre = col.get("tasks_re") or row.get("tasks_re") or table.get("tasks_re")
                vals, marks, data, sizes, has_ext = [], [], [], [], False
                for label, ref in refs.items():
                    m = {"table": table["id"], "row": row["id"], "col": col["id"], "run": label}
                    if ref == PENDING:
                        vals.append((label, None, "⏳")); sizes.append(f"{label}: pending")
                        manifest.append({**m, "source": "pending"}); continue
                    if ref == EXTERNAL:
                        e = ext.get((table["id"], row["id"], col["id"], label))
                        vals.append((label, e["value"] if e else None, "†" if e else "⏳")); has_ext = has_ext or bool(e)
                        sizes.append(f"{label}: † {e['note'] if e else 'missing'}")
                        manifest.append({**m, "source": "external" if e else "external-missing",
                                         "mean": e["value"] if e else "", "note": e["note"] if e else ""}); continue
                    paths = [S.resolve(x, roots) for x in ref]
                    per, found = S.read_run(paths)
                    if not per and (table.get("optional") or row.get("optional")):
                        continue
                    if not per:
                        vals.append((label, None, "⏳")); sizes.append(f"{label}: missing")
                        manifest.append({**m, "source": "missing", "paths": ";".join(S.manifest_path(p) for p in paths)}); continue
                    sub = _filter(per, tre)
                    scored = sum(len(v) for v in per.values())
                    unscored = 1 - scored / found if found else 0.0
                    s = stats(sub, b)
                    n_sub = sum(len(v) for v in sub.values())
                    vals.append((label, s["mean"], "‡" if unscored >= flag else ""))
                    data.append(sub)
                    att = max(1, round(found / max(len(per), 1)))
                    sizes.append(f"{label}: {att} × {len(sub)} tasks = {n_sub} trials" + (f" (of {scored} in the job)" if tre else "")
                                 + (f", {100 * unscored:.0f}% unscored" if unscored >= flag else ""))
                    manifest.append({**m, "source": "local", "paths": ";".join(S.manifest_path(p) for p in paths), "tasks_re": tre or "",
                                     "trials_found": found, "trials_scored": scored, "trials_in_cell": n_sub,
                                     "unscored_pct": round(100 * unscored, 1), "tasks": len(sub),
                                     "mean": round(s["mean"], 2), "checksum": S.checksum(sub)})
                avail = [v for _, v, _ in vals if v is not None]
                if not vals:
                    texts[(row["id"], col["id"])] = "—"; continue
                if not avail:
                    texts[(row["id"], col["id"])] = "⏳"; means[(row["id"], col["id"])] = None
                    cells.append({"table": table["id"], "row": row["id"], "col": col["id"], "mean": ""}); continue
                mean = sum(avail) / len(avail)
                means[(row["id"], col["id"])] = mean
                partial = len(avail) < len(vals)
                per_run = " / ".join(f"{l} " + (f"{v:.1f}" if v is not None else "⏳") + mk for l, v, mk in vals)
                rec = {"table": table["id"], "row": row["id"], "col": col["id"], "mean": round(mean, 2),
                       "runs": ";".join(l for l, _, _ in vals), "per_run": per_run}
                if not has_ext and not partial and data:
                    p = S.pooled(data, b)
                    rec.update(pm=round(p["pm"], 2), lo=round(p["lo"], 2), hi=round(p["hi"], 2))
                    txt = f"{mean:.1f} ± {p['pm']:.1f}"
                else:
                    txt = f"{mean:.1f}" + ("⏳" if partial else "")
                if len(vals) > 1:
                    txt += f" ({per_run})"
                elif vals[0][2]:
                    txt += vals[0][2]
                texts[(row["id"], col["id"])] = txt
                cells.append(rec)
                runs_lines.append(f"| {row['label']} | {col['label']} | {' · '.join(sizes)} |")
        # derived columns, then derived rows
        for row in rows:
            if row.get("derived"):
                continue
            for col in cols:
                if col.get("derived"):
                    env = {c["id"].replace("-", "_"): means.get((row["id"], c["id"])) for c in plain_cols}
                    v = _eval(col["derived"], env)
                    means[(row["id"], col["id"])] = v
                    texts[(row["id"], col["id"])] = _fmt_derived(v, col.get("format"))
        for row in rows:
            if not row.get("derived"):
                continue
            for col in cols:
                env = {r["id"].replace("-", "_"): means.get((r["id"], col["id"])) for r in rows if not r.get("derived")}
                v = _eval(row["derived"], env)
                texts[(row["id"], col["id"])] = _fmt_derived(v, row.get("format"))
        lines = [f"### Table {table['id']} · {table['title']}", ""]
        if table.get("note"):
            lines += [table["note"].strip(), ""]
        used = sorted({l for c in cells if c["table"] == table["id"] for l in (c.get("runs") or "").split(";") if l},
                      key=lambda l: list(RUN_NAMES).index(l) if l in RUN_NAMES else 99)
        multi = any(";" in (c.get("runs") or "") for c in cells if c["table"] == table["id"])
        if multi:
            lines += ["**In parentheses:** each run's own mean, labelled by run. The number before the parentheses is "
                      "their equal-weight mean ± the pooled 95% CI. Run labels: "
                      + "; ".join(f"**{l}** = {RUN_NAMES.get(l, l)}" for l in used) + ".", ""]
        else:
            lines += ["**Each cell:** one run, mean ± 95% CI.", ""]
        lines.append("| " + " | ".join([table.get("row_header", "")] + [c["label"] for c in cols]) + " |")
        lines.append("|" + "---|" * (len(cols) + 1))
        for row in rows:
            label = f"**{row['label']}**" if row.get("bold") else row["label"]
            lines.append("| " + " | ".join([label] + [texts.get((row["id"], c["id"]), "") for c in cols]) + " |")
        if runs_lines:
            lines += ["", f"<details><summary>Runs behind each cell of Table {table['id']}</summary>", "",
                      "| Row | Column | Runs (attempts per task × tasks = trials) |", "|---|---|---|"] + runs_lines + ["", "</details>"]
        rendered.append("\n".join(lines))
        print(f"  table {table['id']}: done ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    return rendered, cells, manifest


def _fmt_derived(v, fmt):
    if v is None:
        return "—"
    if fmt == "pct":
        return f"{100 * v:.0f}%"
    if fmt == "pts":
        return f"{v:+.1f}"
    return f"{v:.1f}"


def main(a) -> int:
    spec, out = Path(a.spec), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rendered, cells, manifest = build(spec, Path(a.external) if a.external else None)
    head = (f"# Control and GRPO tables\n\nGenerated by `exp control-tables --spec {C.rel(spec)}`. Do not edit by hand.\n\n"
            "**How to read.** A cell with one run: mean ± 95% task-bootstrap half-width (the paper's interval). A cell with "
            "several runs: their equal-weight mean ± the pooled interval, then each run's mean in parentheses, in the "
            "order given under the table (\"Runs behind each cell\", which also gives every run's size). "
            "‡ ≥3% of the run's trials unscored · † a number from elsewhere (no interval) · ⏳ not available yet · "
            "— not run. Rows or columns marked as derived are computed from the cells above them. "
            "Paths and checksums: manifest.csv.\n\n"
            "The rehearsal 2×2 and the distillation ladders over training runs are in the seed tables "
            "(`exp seed-tables`).\n")
    (out / "control_tables.md").write_text(head + "\n" + "\n\n---\n\n".join(rendered) + "\n")
    spec_d = yaml.safe_load(open(spec))
    for t, r in zip(spec_d["tables"], rendered):
        S.write_compact(Path(a.compact_out), t, S.first_table(r, f"### Table {t['id']} · {t['title']}"))
    S._write_csv(out / "control_tables.csv", cells)
    S._write_csv(out / "manifest.csv", manifest)
    print(f"wrote {out / 'control_tables.md'}, control_tables.csv, manifest.csv ({len(cells)} cells, "
          f"{sum(1 for m in manifest if m['source'] == 'local')} local runs, "
          f"{sum(1 for m in manifest if m['source'] != 'local')} external/pending)")
    return 0
