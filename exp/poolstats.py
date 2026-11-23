"""Per-trace statistics of SFT pools (rehearsal pools): how many traces interact validly, and how long they are.

Used by `exp control-tables` for tables with `kind: pools` (e.g. failed vs successful rehearsal).

A trace is one pool row (`messages`). An assistant turn is **invalid** when the harness rejected it:
  - Terminus-2 (`format: terminus`): the next message starts with "Previous response had parsing errors"
    (invalid JSON or missing fields). Warnings such as "AUTO-CORRECTED" count as valid, because the
    harness recovered and ran the action. A final turn with no next message counts as valid.
  - native function calling (`format: fc`): the turn has no structured `tool_calls` but its visible text
    (after the reasoning block) contains a `<tool_call>` that did not parse, or it is empty, including a
    reasoning block that never closes (the reasoning parser then reads the whole reply as reasoning). A
    plain-text reply counts as valid.
Per pool: traces with every turn valid, traces with at least one invalid turn, traces with no valid turn,
and the median number of assistant turns.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from exp.seedtables import manifest_path

_THINK = re.compile(r"(?s)^\s*<think>.*?</think>\s*")


def _visible(text: str) -> str:
    """The reply after the reasoning block. A reasoning block that never closes swallows the whole reply
    (the served reasoning parser reads it all as reasoning, so the turn has no action)."""
    text = text or ""
    if text.lstrip().startswith("<think>") and "</think>" not in text:
        return ""
    return _THINK.sub("", text).strip()


def turn_validity(messages: list, fmt: str) -> list[bool]:
    """One bool per assistant turn."""
    out = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        if fmt == "terminus":
            nxt = next((x for x in messages[i + 1:] if x.get("role") != "assistant"), None)
            out.append(not (nxt and str(nxt.get("content") or "").startswith("Previous response had parsing errors")))
        elif fmt == "fc":
            if m.get("tool_calls"):
                out.append(True)
            else:
                vis = _visible(str(m.get("content") or ""))
                out.append(bool(vis) and "<tool_call>" not in vis)
        else:
            raise ValueError(f"unknown trace format {fmt!r}")
    return out


def pool_stats(path: Path, fmt: str) -> dict:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    per = [turn_validity(r["messages"], fmt) for r in rows]
    n = len(per)
    all_valid = sum(1 for v in per if v and all(v))
    some_invalid = sum(1 for v in per if v and not all(v))
    none_valid = sum(1 for v in per if v and not any(v))
    turns = [len(v) for v in per]
    return {"traces": n, "all_valid": all_valid, "some_invalid": some_invalid, "none_valid": none_valid,
            "median_turns": statistics.median(turns) if turns else 0, "path": str(path)}


METRICS = [
    ("all_valid", "Traces with every turn valid", lambda s: f"{100 * s['all_valid'] / s['traces']:.0f}%"),
    ("none_valid", "Traces with no valid turn", lambda s: f"{100 * s['none_valid'] / s['traces']:.0f}%"),
    ("median_turns", "Median assistant turns", lambda s: f"{s['median_turns']:g}"),
    ("traces", "Traces in the pool", lambda s: f"{s['traces']}"),
]


def render(table: dict, resolve) -> tuple[str, list, list]:
    """Markdown for a `kind: pools` table: rows = metrics, columns = benchmarks, cells 'failed / successful'."""
    cols = table["columns"]
    order = table.get("pool_order", ["failed", "successful"])
    stats, manifest, cells = {}, [], []
    for c in cols:
        for which in order:
            p = resolve(c["pools"][which])
            s = pool_stats(p, c["format"])
            stats[(c["id"], which)] = s
            manifest.append({"table": table["id"], "row": "pool", "col": c["id"], "run": which, "source": "local",
                             "paths": manifest_path(p), "trials_found": s["traces"], "trials_scored": s["traces"],
                             "mean": "", "note": f"format={c['format']}"})
    lines = [f"### Table {table['id']} · {table['title']}", ""]
    if table.get("note"):
        lines += [table["note"].strip(), ""]
    lines.append("| | " + " | ".join(c["label"] for c in cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for key, label, fmt in METRICS:
        vals = [" / ".join(fmt(stats[(c["id"], w)]) for w in order) for c in cols]
        lines.append(f"| {label} ({' / '.join(order)}) | " + " | ".join(vals) + " |")
        for c, v in zip(cols, vals):
            cells.append({"table": table["id"], "row": key, "col": c["id"], "per_run": v})
    return "\n".join(lines), cells, manifest
