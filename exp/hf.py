"""`exp hf-export`: write the dataset that goes on the Hugging Face Hub as one folder, ready to upload.

  <out>/harness-competence/rehearsal/tasks/<benchmark>.jsonl      the rehearsal tasks: the instruction each one gives the agent
  <out>/harness-competence/rehearsal/traces/<benchmark>.jsonl     the trajectories collected on those tasks, as chat messages
  <out>/harness-competence/distillation-traces/<benchmark>.jsonl  the teacher's trajectories on the training tasks, one row
                                                                  per trajectory
  <out>/harness-competence/failure-annotation/traces.jsonl        the failed trials three annotators labelled, as the annotators
                                                                  read them, with each annotator's label and the rule-based
                                                                  label of `exp failure-types`

The folder has its card as README.md, from docs/dataset_cards/harness-competence.md. Nothing is uploaded: upload
the folder with `hf upload <owner>/harness-competence <out>/harness-competence . --repo-type dataset`. `DATASETS`
says which part goes into which dataset, so the parts can be split into datasets of their own (each then needs a
card of its name).

Messages are written in one shape for every benchmark (role, content, tool_calls, tool_call_id, all text), so
the files load without a per-benchmark schema. The trajectories are the pool rows as trained on, with two kinds
of row left out: a row without an assistant turn, and a row whose terminal output lists the environment of
the cluster it ran on.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from exp import config as C, render as R

# The benchmarks whose trajectories are exported by default. BFCL single-turn is left out, and AppWorld,
# whose authors ask that data derived from it is not posted in plain text;
# `exp hf-export --benchmarks all` (or a list of names) includes them.
BENCHMARKS = ("medagentbench", "reasoning_gym", "bfcl_multiturn")
# dataset -> {part: its folder in the dataset}
DATASETS = {"harness-competence": {"rehearsal_tasks": "rehearsal/tasks", "rehearsal_traces": "rehearsal/traces",
                                   "teacher_traces": "distillation-traces", "failure_annotation": "failure-annotation"}}
# card text that only applies when the benchmark is part of the export
CARD_NOTES = {"bfcl_singleturn": "In `bfcl_singleturn`, the Terminus-2 rows carry no leading `<think>` block: the reply is the "
                                 "harness's JSON\nobject, with the reasoning inside its `analysis` field where there is any."}

REHEARSAL_POOL = re.compile(r"reh-([a-z0-9]+)(?:~(32b|fail|gemma))?")
DEMONSTRATOR = {None: "the student's 3h checkpoint", "fail": "the student's 3h checkpoint",
                "32b": "the Qwen3-32B teacher", "gemma": "Gemma's 3h checkpoint"}


def _text(value) -> str:
    return "" if value is None else value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def message(m: dict) -> dict:
    return {"role": m.get("role", ""), "content": _text(m.get("content")),
            "tool_calls": _text(m.get("tool_calls") or None), "tool_call_id": _text(m.get("tool_call_id"))}


_CLUSTER_ENV = re.compile(r"^KUBERNETES_SERVICE_HOST=", re.M)


def publishable(messages: list) -> bool:
    return any(m.get("role") == "assistant" for m in messages) and \
        not any(_CLUSTER_ENV.search(_text(m.get("content"))) for m in messages)


def _write(path: Path, rows: list) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def rehearsal_tasks(b) -> list:
    harnesses = defaultdict(list)
    for e in b.cfg.get("rehearsal") or []:
        if e["harness"] not in harnesses[e["split"]]:
            harnesses[e["split"]].append(e["harness"])
    rows = []
    for split in sorted(harnesses):
        folder = R.tasks_root() / b.name / split
        for task in sorted(p for p in folder.iterdir() if (p / "instruction.md").is_file()) if folder.is_dir() else []:
            rows.append({"benchmark": b.name, "split": split, "harnesses": harnesses[split], "task": task.name,
                         "instruction": (task / "instruction.md").read_text()})
    return rows


def rehearsal_traces(b) -> list:
    rows = []
    files = sorted((C.ROOT / "benchmarks" / b.name / "pools").glob("reh-*.jsonl")) + sorted((R.RUNS / "pools" / b.name).glob("reh-*.jsonl"))
    seen = set()
    for path in files:
        m = REHEARSAL_POOL.fullmatch(path.stem)
        if not m or path.stem in seen:      # resampled pools (~rows..., ~likefail, ~noreason) repeat rows of these
            continue
        seen.add(path.stem)
        harness, variant = m.groups()
        for line in open(path):
            r = json.loads(line)
            meta = r.get("meta") or {}
            if not publishable(r["messages"]):
                continue
            rows.append({"benchmark": b.name, "pool": path.stem, "harness": harness, "demonstrator": DEMONSTRATOR[variant],
                         "passed_verifier": variant != "fail", "task": _text(meta.get("task")),
                         "messages": [message(x) for x in r["messages"]]})
    return rows


def teacher_traces(b) -> list:
    """The teacher rows of the benchmark's shipped pools, each once."""
    rows = {}
    for path in sorted((C.ROOT / "benchmarks" / b.name / "pools").glob("*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            meta = r.get("meta") or {}
            if meta.get("source") != "teacher" or not publishable(r["messages"]):
                continue
            messages = [message(x) for x in r["messages"]]   # one shape, so a row is the same in every pool
            key = json.dumps(messages)
            if key not in rows:
                rows[key] = {"benchmark": b.name, "harness": meta["harness"], "difficulty": _text(meta.get("tier")),
                             "task": _text(meta.get("task")), "messages": messages}
    return sorted(rows.values(), key=lambda r: (r["harness"], r["task"]))


def annotated_failures(sample: Path, labels: Path, auto: Path) -> list:
    by_trial = defaultdict(dict)
    for r in csv.DictReader(open(labels, newline="")):
        by_trial[(r["job"], r["trial"])][r["annotator"]] = (r["primary"], r["sub"])
    annotators = sorted({a for v in by_trial.values() for a in v})
    automatic = {(r["job"], r["trial"]): (r["label_delivery"], r["sub_delivery"])
                 for r in csv.DictReader(open(auto, newline=""))} if auto.is_file() else {}

    def steps(transcript: list) -> list:
        out = []
        for s in transcript:
            if "role" in s:
                out.append({"role": s["role"], "text": s.get("text", "")})
            else:
                out += [{"role": k, "text": s[k]} for k in ("prompt", "reasoning", "response") if s.get(k)]
        return out

    rows = []
    for t in json.load(open(sample))["traces"]:
        row = {"benchmark": t["benchmark"], "difficulty": t.get("tier", ""), "harness": t["harness"], "arm": t["arm"],
               "arm_label": t["arm_label"], "job": t["job"], "trial": t["trial"], "task": t["task"],
               "worked_example": bool(t.get("example")), "timed_out": bool(t.get("timeout")), "steps": t["steps"],
               "reference": _text(t.get("reference")), "verifier_output": _text(t.get("verifier")),
               "transcript": steps(t["transcript"])}
        correct = t.get("correct") or {}
        row["passing_trial"] = {"job": correct.get("job", ""), "trial": correct.get("trial", ""),
                                "same_task": bool(correct.get("same_task")), "transcript": steps(correct.get("transcript") or [])}
        got = by_trial.get((t["job"], t["trial"]), {})
        for a in annotators:
            row[f"{a}_primary"], row[f"{a}_sub"] = got.get(a, ("", ""))
        row["automatic_primary"], row["automatic_sub"] = automatic.get((t["job"], t["trial"]), ("", ""))
        rows.append(row)
    return rows


def main(a) -> int:
    out = Path(a.out)
    names = C.benchmark_names() if a.benchmarks == ["all"] else a.benchmarks or list(BENCHMARKS)
    folder = {subset: out / dataset / sub for dataset, subsets in DATASETS.items() for subset, sub in subsets.items()}
    for d in folder.values():   # a benchmark left out of this export leaves no file from an earlier one
        for old in d.glob("*.jsonl"):
            old.unlink()
    counts = {}
    for name in names:
        b = C.benchmark(name)
        for subset, rows in (("rehearsal_tasks", rehearsal_tasks(b)), ("rehearsal_traces", rehearsal_traces(b)),
                             ("teacher_traces", teacher_traces(b))):
            if rows:
                counts[(subset, name)] = _write(folder[subset] / f"{name}.jsonl", rows)
    sample = Path(a.sample)
    if sample.is_file():
        counts[("failure_annotation", "traces")] = _write(folder["failure_annotation"] / "traces.jsonl",
                                                            annotated_failures(sample, Path(a.labels), Path(a.auto)))
    teachers = [C.benchmark(n) for n in names if ("teacher_traces", n) in counts]
    for dataset, subsets in DATASETS.items():
        files = sorted((f"{sub}/{name}.jsonl", counts[(subset, name)]) for subset, sub in subsets.items()
                       for (s, name) in counts if s == subset)
        if not files:
            continue
        card = (C.ROOT / "docs" / "dataset_cards" / f"{dataset}.md").read_text()
        card = card.replace("{{files}}", "| file | rows |\n|---|---|\n" + "\n".join(f"| `{f}` | {n} |" for f, n in files))
        card = card.replace("{{teacher_benchmarks}}", ", ".join(f"{b.cfg.get('display', b.name)} (`{b.name}`)" for b in teachers))
        card = card.replace("{{notes}}\n\n", "".join(CARD_NOTES[n] + "\n\n" for n in names if n in CARD_NOTES))
        (out / dataset / "README.md").write_text(card)
        print(f"{C.rel(out / dataset)}: " + ", ".join(f"{f} {n} rows" for f, n in files), file=sys.stderr)
    return 0
