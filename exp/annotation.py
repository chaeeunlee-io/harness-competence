"""`exp annotation-sample` and `exp annotation-tables`: the human check of the automatic failure types.

`exp failure-types` labels every failed trial by rule. Three annotators labelled a sample of the same
failed trials by reading the traces. This module draws that sample and compares the two sets of labels.

annotation-sample
    For each cell of the spec (a benchmark, a tier, and the task-only and task + rehearsal arms on the
    target harness) it takes `annotate` failed trials per arm. Trials that ended in an infrastructure
    exception are left out, as in `exp failure-types`. Within an arm the failed trials are shuffled with
    a fixed seed and taken one per task first, so the sample covers as many tasks as it can. Writes

      <out>/sample.csv            which trials were drawn, in annotation order
      <traces out>/sample.json    the same trials as an annotator reads them: a trimmed transcript, the
                                  verifier's verdict, the reference answer where the task has one, and a
                                  passing run of the same task for comparison

    The spec's `examples` are worked examples shown to the annotators. They are part of the sample and
    are not labelled.

annotation-tables
    Reads the human labels (one row per annotator and trial) and the automatic labels (`trials.csv`
    of `exp failure-types`) and writes <out>/annotation_tables.md:

      - interface failures per arm, automatic against human, as a share of failed trials and of all
        trials. The automatic share uses every failed trial. The human share uses the sampled ones,
        scaled by the arm's failure rate for the all-trials column.
      - the same for each interface failure mode (invalid calls, submission, termination)
      - annotator agreement: Fleiss' kappa and the share of trials on which all annotators agree

    A trial's human label is the mean over its annotators, so a 2-to-1 split counts as two thirds.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

import yaml

from exp import config as C
from exp.failuretypes import _reward, label

ARMS = {"task": "Task supervision only", "comb": "Task supervision + rehearsal"}
MODES = ("invalid", "submission", "termination")
HEAD, TAIL = 8, 6          # turns kept at each end of a long transcript
CAP = 3500                 # characters kept per message


# --------------------------------------------------------------------------- reading a trial

def _clip(text: str, cap: int = CAP) -> str:
    text = text or ""
    if len(text) <= cap:
        return text
    return text[: cap // 3] + f"\n… [{len(text) - cap:,} characters omitted] …\n" + text[-2 * cap // 3:]


def _json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _exception(trial: Path) -> str | None:
    result = _json(trial / "result.json")
    if result is None:
        return "unreadable"
    return (result.get("exception_info") or {}).get("exception_type")


def _is_infrastructure_failure(trial: Path) -> bool:
    """A trial that ended in any exception but the agent's own timeout says nothing about the model."""
    exc = _exception(trial)
    return bool(exc) and exc != "AgentTimeoutError"


def _terminus_transcript(trial: Path) -> tuple[list[dict], int]:
    """(turns shown, turns in total). Each turn is what the harness sent and what the model replied, with
    the model's reasoning, which the server splits off and the trajectory file keeps."""
    episodes = sorted((trial / "agent").glob("episode-*"), key=lambda p: int(p.name.split("-")[1]))
    steps = (_json(trial / "agent" / "trajectory.json") or {}).get("steps", [])
    reasoning = [s.get("reasoning_content") or "" for s in steps if s.get("source") == "agent"]
    turns = []
    for e in episodes:
        i = int(e.name.split("-")[1])
        prompt = (e / "prompt.txt").read_text(errors="replace") if (e / "prompt.txt").exists() else ""
        response = (e / "response.txt").read_text(errors="replace") if (e / "response.txt").exists() else ""
        turns.append({"i": i, "prompt": prompt, "response": response})
    shown = turns if len(turns) <= HEAD + TAIL else turns[:HEAD] + turns[-TAIL:]
    out = []
    for t in shown:
        prompt = t["prompt"]
        if t["i"] == 0:        # the harness instructions are the same in every trial: keep the task part
            k = prompt.find("Task Description")
            if k > 0:
                prompt = prompt[:600] + "\n… [harness instructions omitted] …\n" + prompt[k:]
        turn = {"i": t["i"], "prompt": _clip(prompt, 6000 if t["i"] == 0 else CAP), "response": _clip(t["response"])}
        if t["i"] < len(reasoning):
            turn["reasoning"] = _clip(reasoning[t["i"]])
        out.append(turn)
    return out, len(turns)


def _fc_transcript(trial: Path) -> tuple[list[dict], int]:
    """(messages shown, messages in total) of a BFCL multi-turn fc conversation."""
    messages = json.loads((trial / "agent" / "mt_transcript.json").read_text()).get("messages", [])
    rows = []
    for m in messages:
        text = m.get("content") or ""
        if isinstance(text, list):
            text = "\n".join(part.get("text", "") for part in text if isinstance(part, dict))
        if m.get("tool_calls"):
            calls = [f"{c.get('function', {}).get('name')}({c.get('function', {}).get('arguments')})" for c in m["tool_calls"]]
            text = (text + "\n" if text else "") + "TOOL CALLS: " + "; ".join(calls)
        rows.append({"role": m.get("role", "?"), "text": _clip(text)})
    shown = rows if len(rows) <= 2 * (HEAD + TAIL) else rows[: 2 * HEAD] + rows[-2 * TAIL:]
    return shown, len(rows)


def _transcript(trial: Path, harness: str) -> tuple[list[dict], int]:
    return _terminus_transcript(trial) if harness == "t2" else _fc_transcript(trial)


def _task_dir(trial: Path) -> Path | None:
    """The task folder a trial ran, wherever the repository was checked out when it ran."""
    recorded = ((_json(trial / "config.json") or {}).get("task") or {}).get("path")
    if not recorded:
        return None
    if Path(recorded).is_dir():
        return Path(recorded)
    _, found, tail = recorded.partition("/runs/tasks/")
    return C.ROOT / "runs" / "tasks" / tail if found else None


def _reference(benchmark: str, trial: Path) -> str:
    """What the verifier compared against, where the task records it."""
    if benchmark == "reasoning_gym":
        task = _task_dir(trial)
        entry = _json(task / "tests" / "entry_data.json") if task else None
        return f"Expected answer: {entry.get('answer')}" if entry else ""
    detail = _json(trial / "verifier" / "checker_detail.json")
    if benchmark == "bfcl_multiturn" and detail:
        return f"Checker: {detail.get('error_type', '')}. {detail.get('error_message', '')}"
    return ""


def _verifier_output(trial: Path) -> str:
    for name in ("test-stdout.txt", "test-output.txt"):
        path = trial / "verifier" / name
        if path.exists():
            return _clip(path.read_text(errors="replace").strip(), 600)
    return ""


# --------------------------------------------------------------------------- drawing the sample

def _failed_trials(job: Path) -> tuple[list[Path], int, int]:
    """(failed trials that are not infrastructure failures, trials scored, trials failed)."""
    scored = [t for t in job.iterdir() if t.is_dir() and _reward(t) is not None]
    failed = [t for t in scored if _reward(t) == 0.0]
    return sorted(t for t in failed if not _is_infrastructure_failure(t)), len(scored), len(failed)


def _one_per_task_first(trials: list[Path]) -> list[Path]:
    first, seen = [], set()
    for t in trials:
        task = t.name.split("__")[0]
        if task not in seen:
            first.append(t)
            seen.add(task)
    return first + [t for t in trials if t not in first]


def _passing_run(evals_root: Path, cell: dict, task: str) -> tuple[Path, bool] | None:
    """A trial with reward 1 on the same task and harness: from the 3H SFT model of the same run first, then the
    cell's own arms, then any other Qwen3-4B evaluation on that task set and harness under the same protocol. If
    the task was never solved, a passing trial of another task. Returns (trial, whether it is the same task)."""
    arms = [evals_root / job for job in cell["arms"].values()]
    task_set, _, _, run, harness, pass_id = arms[0].relative_to(evals_root).parts
    first = [evals_root / task_set / "qwen3-4b" / "3h" / run / harness / pass_id] + arms
    others = sorted(p for p in (evals_root / task_set / "qwen3-4b").glob(f"*/*/{harness}/{pass_id}") if p not in first)
    jobs = [p for p in first + others if p.is_dir()]
    for pattern, same_task in ((f"{task}__*", True), ("*__*", False)):
        for job in jobs:
            for trial in sorted(job.glob(pattern)):
                if trial.is_dir() and _reward(trial) == 1.0:
                    return trial, same_task
    return None


def _trace(cell: dict, arm: str, trial: Path, evals_root: Path) -> dict | None:
    """One sampled trial as an annotator reads it, or None if its transcript cannot be read."""
    bench, harness, job = cell["benchmark"], cell["harness"], cell["arms"][arm]
    try:
        transcript, steps = _transcript(trial, harness)
    except (OSError, ValueError):
        return None
    if not transcript:
        return None
    exc = _exception(trial)
    task = trial.name.split("__")[0]
    trace = {"id": f"{bench}:{job}:{trial.name}", "benchmark": bench, "tier": cell.get("tier", ""), "harness": harness,
             "arm": arm, "arm_label": ARMS[arm], "job": job, "trial": trial.name, "task": task,
             "timeout": exc == "AgentTimeoutError", "steps": steps, "reference": _reference(bench, trial),
             "verifier": _verifier_output(trial), "transcript": transcript}
    passing = _passing_run(evals_root, cell, task)
    if passing:
        run, same_task = passing
        shown, n = _transcript(run, harness)
        trace["correct"] = {"job": run.parent.relative_to(evals_root).as_posix(), "trial": run.name, "same_task": same_task, "steps": n, "transcript": shown}
    return trace


def draw_sample(spec: dict) -> tuple[list[dict], list[dict]]:
    """(traces in annotation order, per-arm counts)."""
    evals_root = C.ROOT / spec.get("evals_root", "runs/evals")
    sampling = spec.get("annotation", {})
    examples = {(e["job"], e["task"]) for e in sampling.get("examples", [])}
    rng = random.Random(sampling.get("seed", 0))
    traces, counts = [], []
    for cell in spec["cells"]:
        for arm, job in cell["arms"].items():
            failed, n_scored, n_failed = _failed_trials(evals_root / job)
            counts.append({"benchmark": cell["benchmark"], "tier": cell.get("tier", ""), "arm": arm, "job": job,
                           "scored": n_scored, "failed": n_failed, "failed_not_infrastructure": len(failed)})
            rng.shuffle(failed)
            drawn = 0
            for trial in _one_per_task_first(failed):
                if drawn == cell["annotate"]:
                    break
                trace = _trace(cell, arm, trial, evals_root)
                if trace:
                    trace["example"] = (job, trace["task"]) in examples
                    traces.append(trace)
                    drawn += 1
            print(f"{cell['benchmark']} {cell.get('tier') or '-'} {arm}: {drawn} of {len(failed)} failed trials", flush=True)
    random.Random(sampling.get("order_seed", 1)).shuffle(traces)     # annotation order mixes arms and benchmarks
    return traces, counts


def cmd_sample(a) -> int:
    spec = yaml.safe_load(open(a.spec))
    traces, counts = draw_sample(spec)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sample.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["benchmark", "tier", "harness", "arm", "job", "trial", "task", "example"])
        for t in traces:
            w.writerow([t["benchmark"], t["tier"], t["harness"], t["arm"], t["job"], t["trial"], t["task"],
                        "yes" if t["example"] else ""])
    traces_out = Path(a.traces_out)
    traces_out.mkdir(parents=True, exist_ok=True)
    (traces_out / "sample.json").write_text(json.dumps({"traces": traces, "arm_counts": counts}, ensure_ascii=False))
    n_examples = sum(t["example"] for t in traces)
    print(f"wrote {out}/sample.csv and {traces_out}/sample.json: {len(traces)} trials, {n_examples} of them worked examples")
    return 0


# --------------------------------------------------------------------------- comparing the labels

def fleiss_kappa(ratings: list[list[str]]) -> float:
    """Fleiss' kappa for items that each have the same number of ratings."""
    n = len(ratings[0])
    agreement = [(sum(c * c for c in Counter(item).values()) - n) / (n * (n - 1)) for item in ratings]
    totals = Counter(r for item in ratings for r in item)
    expected = sum((c / (len(ratings) * n)) ** 2 for c in totals.values())
    return (sum(agreement) / len(ratings) - expected) / (1 - expected)


def _human_mode(row: dict) -> str:
    """An annotator's label on the four-way scale of the automatic labels."""
    return row["sub"] if row["primary"] == "interface" else "task"


def _arm_totals(trials_csv: Path) -> dict:
    """(benchmark, arm) -> trials scored, trials failed, and automatic counts per label, over the arm's tiers."""
    jobs, totals = set(), {}
    for r in csv.DictReader(open(trials_csv)):
        t = totals.setdefault((r["benchmark"], r["arm"]), {"scored": 0, "failed": 0, "interface": 0, **{m: 0 for m in MODES}})
        if r["job"] not in jobs:
            jobs.add(r["job"])
            t["scored"] += int(r["scored_in_job"])
        t["failed"] += 1
        if r["label_delivery"] == "interface":
            t["interface"] += 1
            t[r["sub_delivery"]] += 1
    return totals


def compare(labels: list[dict], trials_csv: Path) -> dict:
    """Per (benchmark, arm): failure rate, and automatic against human interface shares and modes."""
    totals = _arm_totals(trials_csv)
    by_trial: dict = {}
    for row in labels:
        by_trial.setdefault((row["benchmark"], row["arm"], row["job"], row["trial"]), []).append(row)
    rows = {}
    for key, t in totals.items():
        trials = [v for k, v in by_trial.items() if k[:2] == key]
        if not trials:
            continue
        human_share = sum(sum(r["primary"] == "interface" for r in v) / len(v) for v in trials) / len(trials)
        arm_labels = [r for v in trials for r in v]
        modes = Counter(_human_mode(r) for r in arm_labels)
        failure_rate = t["failed"] / t["scored"]
        rows[key] = {
            "sampled": len(trials), "failed_pct": 100 * failure_rate,
            "auto_of_failed": 100 * t["interface"] / t["failed"], "human_of_failed": 100 * human_share,
            "auto_of_all": 100 * t["interface"] / t["scored"], "human_of_all": 100 * failure_rate * human_share,
            "auto_modes": {m: 100 * t[m] / t["failed"] for m in MODES},
            "human_modes": {m: 100 * modes[m] / len(arm_labels) for m in MODES},
        }
    return rows


def agreement(labels: list[dict]) -> dict:
    """Fleiss' kappa and unanimity on interface vs task, over the trials every annotator labelled."""
    annotators = {r["annotator"] for r in labels}
    by_trial: dict = {}
    for row in labels:
        by_trial.setdefault((row["job"], row["trial"]), []).append(row["primary"])
    complete = [v for v in by_trial.values() if len(v) == len(annotators)]
    return {"annotators": len(annotators), "trials": len(by_trial), "complete": len(complete),
            "kappa": fleiss_kappa(complete), "unanimous": sum(len(set(v)) == 1 for v in complete) / len(complete)}


def render(rows: dict, agree: dict, benchmarks: list[tuple[str, str]]) -> str:
    out = ["## Interface failures, automatic vs human", "",
           f"Annotator agreement (interface vs task knowledge): Fleiss κ = {agree['kappa']:.2f}; all "
           f"{agree['annotators']} annotators agree on {agree['unanimous']:.0%} of the {agree['complete']} trials "
           f"each of them labelled ({agree['trials']} trials in total).", "",
           "| | | Failed trials (% of all) | Interface failures, % of all trials (auto · human) | Interface failures, % of failed trials (auto · human) | Sampled |",
           "|---|---|---|---|---|---|"]
    for bench, name in benchmarks:
        for arm in ARMS:
            r = rows[(bench, arm)]
            out.append(f"| {name} | {ARMS[arm]} | {r['failed_pct']:.1f} | {r['auto_of_all']:.1f} · {r['human_of_all']:.1f} "
                       f"| {r['auto_of_failed']:.0f} · {r['human_of_failed']:.0f} | {r['sampled']} |")
    out += ["", "## Interface failure modes, % of failed trials (automatic · human)", "",
            "Automatic percentages use every failed trial, human percentages the sampled ones. The remainder of "
            "each row is task-knowledge failures. The BFCL checker has no submission signal, so that automatic "
            "cell is n/a.", "",
            "| | | Invalid calls | Incorrect submissions | Termination failures |", "|---|---|---|---|---|"]
    for bench, name in benchmarks:
        for arm in ARMS:
            r = rows[(bench, arm)]
            cells = []
            for m in MODES:
                auto = "n/a" if (bench, m) == ("bfcl_multiturn", "submission") else f"{r['auto_modes'][m]:.0f}"
                cells.append(f"{auto} · {r['human_modes'][m]:.0f}")
            out.append(f"| {name} | {ARMS[arm]} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def cmd_tables(a) -> int:
    spec = yaml.safe_load(open(a.spec))
    labels = list(csv.DictReader(open(a.labels)))
    names = {"bfcl_multiturn": "BFCL multi-turn (fc)", "reasoning_gym": "Reasoning Gym (Terminus-2)",
             "medagentbench": "MedAgentBench (Terminus-2)"}
    order = spec.get("annotation", {}).get("table_order") or list(dict.fromkeys(c["benchmark"] for c in spec["cells"]))
    text = render(compare(labels, Path(a.auto)), agreement(labels), [(b, names.get(b, b)) for b in order])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "annotation_tables.md").write_text(text)
    print(text)
    print(f"wrote {out}/annotation_tables.md")
    return 0
