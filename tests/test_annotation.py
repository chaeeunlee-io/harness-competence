"""Tests for exp/annotation.py: annotator agreement and the automatic-vs-human comparison.

Run: python -m pytest tests/test_annotation.py   (or: python tests/test_annotation.py)
"""
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp import annotation as A  # noqa: E402


def test_fleiss_kappa_bounds():
    unanimous = [["interface"] * 3, ["task"] * 3, ["interface"] * 3, ["task"] * 3]
    assert abs(A.fleiss_kappa(unanimous) - 1.0) < 1e-9
    # every item split 2 to 1 the same way: observed agreement equals chance agreement
    split = [["interface", "interface", "task"]] * 6
    assert abs(A.fleiss_kappa(split) - (1 / 3 - 5 / 9) / (1 - 5 / 9)) < 1e-9


def test_one_per_task_first():
    trials = [Path(f"{task}__{i}") for task, i in [("a", 1), ("a", 2), ("b", 1), ("a", 3), ("c", 1)]]
    order = [t.name for t in A._one_per_task_first(trials)]
    assert order == ["a__1", "b__1", "c__1", "a__2", "a__3"]


def _write(path: Path, header: list, rows: list):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def test_compare_scales_the_human_share_by_the_failure_rate():
    with tempfile.TemporaryDirectory() as d:
        trials = Path(d) / "trials.csv"
        # one arm: 10 trials scored, 4 failed, 3 of them automatic interface failures (2 invalid, 1 termination)
        auto = [("interface", "invalid"), ("interface", "invalid"), ("interface", "termination"), ("task", "")]
        _write(trials, ["benchmark", "arm", "job", "trial", "scored_in_job", "label_delivery", "sub_delivery"],
               [["bfcl_multiturn", "task", "job", f"t{i}", 10, p, s] for i, (p, s) in enumerate(auto)])
        # two sampled trials, three annotators each: one unanimous interface, one 2-to-1 for task
        labels = [dict(annotator=a, benchmark="bfcl_multiturn", arm="task", job="job", trial="t0",
                       primary="interface", sub="invalid") for a in "ABC"]
        labels += [dict(annotator=a, benchmark="bfcl_multiturn", arm="task", job="job", trial="t3",
                        primary=p, sub=s) for a, (p, s) in zip("ABC", [("task", "reasoning"), ("task", "approach"), ("interface", "submission")])]
        row = A.compare(labels, trials)[("bfcl_multiturn", "task")]
        assert row["sampled"] == 2
        assert abs(row["failed_pct"] - 40.0) < 1e-9
        assert abs(row["auto_of_failed"] - 75.0) < 1e-9 and abs(row["auto_of_all"] - 30.0) < 1e-9
        human_share = (1.0 + 1 / 3) / 2
        assert abs(row["human_of_failed"] - 100 * human_share) < 1e-9
        assert abs(row["human_of_all"] - 40.0 * human_share) < 1e-9
        assert abs(row["auto_modes"]["invalid"] - 50.0) < 1e-9
        assert abs(row["human_modes"]["invalid"] - 50.0) < 1e-9 and abs(row["human_modes"]["submission"] - 100 / 6) < 1e-9
        agree = A.agreement(labels)
        assert agree["complete"] == 2 and agree["unanimous"] == 0.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
