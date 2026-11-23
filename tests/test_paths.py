"""Tests for the run addresses in exp/render.py: where checkpoints, GRPO runs and evaluation jobs are filed.

Run: python tests/test_paths.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp import config as C, render as R  # noqa: E402

MAB = C.benchmark("medagentbench")
RG = C.benchmark("reasoning_gym")


def rel(path: Path) -> str:
    return path.relative_to(R.RUNS).as_posix()


def test_run_id_names_the_training_run():
    assert R.run_id() == "s0" and R.run_id(3) == "s3"
    assert R.run_id(None, "base") == "none" and R.run_id(2, "teacher") == "none"


def test_pass_id_names_attempts_and_repeats():
    assert R.pass_id(4) == "n4" and R.pass_id(1, 1) == "n1" and R.pass_id(1, 3) == "n1-p3"


def test_task_set_is_the_benchmark_with_its_tier_or_split():
    assert R.task_set(MAB) == "medagentbench" and R.task_set(MAB, "train") == "medagentbench-train"
    assert R.task_set(RG) == "reasoning_gym-easy" and R.task_set(RG, "test-hard") == "reasoning_gym-hard"
    assert R.task_set(RG, "train-easy") == "reasoning_gym-train-easy"


def test_a_checkpoint_and_its_evaluations_share_one_address():
    assert rel(R.checkpoint_dir("medagentbench", "qwen3-4b", "parity+fc+reh-t2", 2)) == \
        "checkpoints/medagentbench/qwen3-4b/parity+fc+reh-t2/s2"
    assert rel(R.eval_dir(MAB, "qwen3-4b", "parity+fc+reh-t2", "t2", seed=2)) == \
        "evals/medagentbench/qwen3-4b/parity+fc+reh-t2/s2/t2/n4"
    assert rel(R.eval_dir(RG, "qwen3-4b", "3h", "goose", "test-hard", n_attempts=1, eval_pass=2)) == \
        "evals/reasoning_gym-hard/qwen3-4b/3h/s0/goose/n1-p2"


def test_untrained_models_have_no_run():
    assert rel(R.eval_dir(MAB, "qwen3-4b", "base", "parity", seed=3)) == "evals/medagentbench/qwen3-4b/base/none/parity/n4"
    n = R.protocol_attempts(MAB, "teacher")
    assert rel(R.eval_dir(MAB, "qwen3-32b", "teacher", "fc")) == f"evals/medagentbench/qwen3-32b/teacher/none/fc/n{n}"


def test_grpo_runs_are_settings_with_runs():
    assert R.rl_name(MAB, "parity") == "grpo-parity" and R.rl_name(MAB, "parity", "base") == "grpo-parity-from-base"
    assert rel(R.rl_dir(MAB, "qwen3-4b", "3h", 1)) == "rl/medagentbench/qwen3-4b/grpo-3h/s1"
    assert rel(R.eval_dir(MAB, "qwen3-4b", R.rl_name(MAB, "parity", "base", model="qwen3-4b"), "xml")) == \
        "evals/medagentbench/qwen3-4b/grpo-parity-from-base/s0/xml/n4"


def test_job_config_names_the_folder_harbor_writes():
    job_dir = R.eval_dir(MAB, "qwen3-4b", "3h", "t2", seed=1)
    job = R.eval_job(MAB, "t2", job_dir=job_dir, placeholders=True)
    assert Path(job["jobs_dir"]) / job["job_name"] == job_dir
    assert R.job_id(job_dir) == "medagentbench/qwen3-4b/3h/s1/t2/n4"


def test_records_store_paths_relative_to_the_repo():
    pool = "benchmarks/medagentbench/pools/3h.jsonl"
    assert C.repo_path(C.ROOT / pool) == pool
    assert C.repo_path(pool) == pool                       # already relative: unchanged
    assert C.repo_path("/models/Qwen/Qwen3-4B") == "/models/Qwen/Qwen3-4B"   # outside the repo: unchanged


def test_a_record_is_reused_whether_it_stores_the_full_path_or_the_relative_one():
    import json
    import tempfile
    from exp import cli
    pool = "benchmarks/medagentbench/pools/3h.jsonl"
    wanted = {"model": "qwen3-4b", "data": pool, "seed": None}
    with tempfile.TemporaryDirectory() as d:
        for stored in (pool, str(C.ROOT / pool)):
            record = Path(d) / "exp_sft.json"
            record.write_text(json.dumps({**wanted, "data": stored, "max_steps": None}))
            assert cli._reusable(record, wanted) == (True, "")
        record.write_text(json.dumps({**wanted, "data": "benchmarks/medagentbench/pools/fc-only.jsonl", "max_steps": None}))
        assert cli._reusable(record, wanted)[0] is False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
