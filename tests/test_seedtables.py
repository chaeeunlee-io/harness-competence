"""Tests for exp/seedtables.py on small synthetic job folders.

Run: python -m pytest tests/test_seedtables.py   (or: python tests/test_seedtables.py)
"""
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp import config as C  # noqa: E402
from exp import seedtables as S  # noqa: E402


def make_job(root: Path, name: str, rewards: dict, unscored: int = 0) -> Path:
    """rewards: task -> list of rewards; `unscored` extra trial folders without reward.txt."""
    job = root / name
    for task, rs in rewards.items():
        for i, r in enumerate(rs):
            v = job / f"{task}__{name}{i}" / "verifier"
            v.mkdir(parents=True)
            (v / "reward.txt").write_text(str(r))
    for i in range(unscored):
        (job / f"lost__{name}{i}").mkdir(parents=True)
    return job


def test_run_mean_matches_table_stats():
    b = C.benchmark("medagentbench")          # aggregate: pooled
    with tempfile.TemporaryDirectory() as d:
        job = make_job(Path(d), "j", {"t1": [1, 0, 0, 0], "t2": [1, 1, 1, 1]})
        per, found = S.read_run([job])
        assert found == 8
        assert abs(S.stats(per, b)["mean"] - 62.5) < 1e-9


def test_equal_weight_pooling_ignores_attempt_count():
    b = C.benchmark("medagentbench")
    run_4 = {"t1": [1, 1, 1, 1], "t2": [0, 0, 0, 0]}      # 50.0
    run_1 = {"t1": [0], "t2": [0]}                         # 0.0
    p = S.pooled([run_4, run_1], b)
    assert abs(p["mean"] - 25.0) < 1e-9                    # not 40.0 (pooling trials)
    assert p["lo"] <= p["mean"] <= p["hi"]


def test_bootstrap_is_deterministic():
    b = C.benchmark("reasoning_gym")
    runs = [{f"t{i}": [i % 2, (i + 1) % 2] for i in range(20)}, {f"t{i}": [i % 3 == 0] for i in range(20)}]
    assert S.pooled(runs, b) == S.pooled(runs, b)


def test_build_flags_unscored_external_and_pending():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        make_job(d / "jobs", "a-s1", {"t1": [1] * 4, "t2": [0] * 4})
        make_job(d / "jobs", "a-s2", {"t1": [1] * 4, "t2": [1] * 3}, unscored=1)    # 1/8 unscored -> ‡
        spec = d / "spec.yaml"
        spec.write_text(f"""
roots: {{jobs: {d / 'jobs'}}}
external: ext.csv
variants: {{v: [s1, s2, s3, s4]}}
tables:
  - id: T
    title: test
    benchmark: medagentbench
    runs: [s1, s2, s3, s4]
    patterns: {{s1: "jobs:a-s1", s2: "jobs:a-s2", s3: ext, s4: pending}}
    columns: [{{id: c, label: c}}]
    rows: [{{id: r, label: r}}]
""")
        with open(d / "ext.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["table", "row", "col", "run", "value", "note"])
            w.writerow(["T", "r", "c", "s3", "10.0", "x"])
        rendered, cells, manifest = S.build(spec)
        text = rendered[0]
        assert "50.0 / 100.0‡ / 10.0† / ⏳" in text
        assert "53.3⏳" in text                     # mean of the available runs, marked partial
        src = {m["run"]: m["source"] for m in manifest}
        assert src == {"s1": "local", "s2": "local", "s3": "external", "s4": "pending"}
        assert all(Path(m["paths"]).is_absolute() for m in manifest if m["source"] == "local")


def test_merge_replaces_ext_with_other_cluster_trials():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        make_job(d / "here", "a-s1", {"t1": [1] * 4, "t2": [0] * 4})
        make_job(d / "there", "a-s2", {"t1": [1] * 4, "t2": [1] * 4})
        (d / "other.yaml").write_text(f"""
roots: {{x: {d / 'there'}}}
tables:
  - {{id: T, title: t, benchmark: medagentbench, runs: [s2], patterns: {{s2: "x:a-s2"}},
     columns: [{{id: c, label: c}}], rows: [{{id: r, label: r}}]}}
""")
        (d / "spec.yaml").write_text(f"""
roots: {{jobs: {d / 'here'}}}
merge: [other.yaml]
variants: {{v: [s1, s2]}}
tables:
  - {{id: T, title: t, benchmark: medagentbench, runs: [s1, s2], patterns: {{s1: "jobs:a-s1", s2: ext}},
     columns: [{{id: c, label: c}}], rows: [{{id: r, label: r}}]}}
""")
        rendered, cells, manifest = S.build(d / "spec.yaml")
        assert "75.0 ± " in rendered[0] and "(50.0 / 100.0°)" in rendered[0], rendered[0]
        assert {m["run"]: m.get("merged_from") for m in manifest} == {"s1": None, "s2": "other.yaml"}


def test_parallel_reader_matches_read_job():
    with tempfile.TemporaryDirectory() as d:
        job = make_job(Path(d), "j", {"t1": [1, 0.5], "t2": [0]}, unscored=2)
        per, found = S.read_folder(job)
        ref = S.read_job(job)
        assert found == 5
        assert {k: sorted(v) for k, v in per.items()} == {k: sorted(v) for k, v in ref.items()}


def test_compact_drops_post_false_rows():
    md = "|  | a |\n|---|---|\n| keep | 1.0 ± 0.1 |\n| **bold keep** | 2.0 |\n| gone | 3.0 (1.0 / 5.0) |\n"
    table = {"id": "X", "title": "t", "rows": [{"label": "keep"}, {"label": "bold keep", "bold": True},
                                               {"label": "gone", "post": False}]}
    with tempfile.TemporaryDirectory() as d:
        S.write_compact(Path(d) / "compact", table, md)
        out = (Path(d) / "compact" / "X.md").read_text()
    assert "| keep |" in out and "| **bold keep** |" in out and "gone" not in out


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
