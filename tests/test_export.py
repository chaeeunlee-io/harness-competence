"""Tests for exp/export.py: the shipped result files hold what the job folders hold.

Run: python tests/test_export.py
"""
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp import export as X, render as R, seedtables as S, table as T  # noqa: E402


def _job(runs: Path, job: str, trials: dict):
    """trials: trial folder -> reward, or None for an attempt the verifier never scored."""
    for name, reward in trials.items():
        d = runs / "evals" / job / name
        (d / "verifier").mkdir(parents=True)
        if reward is not None:
            (d / "verifier" / "reward.txt").write_text(f"{reward}\n")


def _tree(runs: Path):
    _job(runs, "medagentbench/qwen3-4b/3h/s1/t2/n1", {"task1_1__a": 1, "task1_2__a": 0, "task2_1__a": None})
    _job(runs, "medagentbench/qwen3-4b/3h/s1/t2/n1-p2", {"task1_1__b": 0, "task1_2__b": 0, "task2_1__b": 1})
    _job(runs, "medagentbench/qwen3-4b/base/none/t2/n4", {f"task1_1__{i}": 1 for i in "abcd"})
    _job(runs, "medagentbench/qwen3-4b/base/none/t2/n4-p2", {f"task1_1__{i}": 0 for i in "abcd"})
    _job(runs, "medagentbench-train/qwen3-4b/3h/s0/t2/n2", {"task9_1__a": 1, "task9_1__b": 0})


def _with_tree(fn):
    def run():
        with tempfile.TemporaryDirectory() as d:
            runs, saved = Path(d) / "runs", (R.RUNS, T.REWARDS_CSV, T._REWARDS, S.SHIPPED)
            _tree(runs)
            R.RUNS, T.REWARDS_CSV, T._REWARDS = runs, Path(d) / "rewards.csv", None
            try:
                fn(Path(d), X.export_rewards(Path(d) / "rewards.csv"))
            finally:
                R.RUNS, T.REWARDS_CSV, T._REWARDS, S.SHIPPED = saved
                S._CACHE.clear()
    run.__name__ = fn.__name__
    return run


@_with_tree
def test_rewards_file_has_one_row_per_task_with_scored_rewards_and_attempts_found(d, data):
    rows = {(r["job"], r["task"]): r for r in csv.DictReader(open(d / "rewards.csv"))}
    assert rows[("medagentbench/qwen3-4b/3h/s1/t2/n1", "task1_1")]["rewards"] == "1"
    unscored = rows[("medagentbench/qwen3-4b/3h/s1/t2/n1", "task2_1")]
    assert unscored["trials"] == "1" and unscored["rewards"] == ""
    assert rows[("medagentbench/qwen3-4b/base/none/t2/n4", "task1_1")]["rewards"] == "1;1;1;1"


@_with_tree
def test_shipped_reader_returns_what_the_folder_reader_returns(d, data):
    for job in ("medagentbench/qwen3-4b/3h/s1/t2/n1", "medagentbench/qwen3-4b/base/none/t2/n4"):
        folder = R.RUNS / "evals" / job
        S.SHIPPED = False
        S._CACHE.clear()
        live = S.read_folder(folder)
        S.SHIPPED = True
        S._CACHE.clear()
        shipped = S.read_folder(folder)
        assert shipped == live and S.checksum(shipped[0]) == S.checksum(live[0])
    assert T.shipped_job(R.RUNS / "evals" / "medagentbench/qwen3-4b/3h/s9/t2/n1") is None


@_with_tree
def test_cells_pool_passes_and_file_a_second_base_evaluation_as_s0b(d, data):
    cells = {(c["arm"], c["run"], c["layer"], c["metric"]): c for c in X.computed_cells(data)}
    assert all(c["benchmark"] == "medagentbench" for c in cells.values())      # the training-split probe is not a result
    pooled = cells[("3h", "s1", "n2-T0.7", "mean")]                            # two single-attempt passes = two attempts
    assert pooled["trials"] == 5 and float(pooled["value"]) == 40.0            # MedAgentBench: the mean over all trials
    assert ("3h", "s1", "n2-T0.7", "pass2") in cells
    assert float(cells[("base", "s0", "n4-T0.7", "mean")]["value"]) == 100.0
    assert float(cells[("base", "s0b", "n4-T0.7", "mean")]["value"]) == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
