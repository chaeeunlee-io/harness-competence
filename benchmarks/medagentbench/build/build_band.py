"""Build the MedAgentBench RL learning band (`splits/band.json`).

GRPO needs tasks with mixed rollout outcomes. Tasks where every rollout receives
the same reward give zero group-relative advantage.

The band is built from probe runs of the 3H SFT checkpoint on training tasks.
A task is kept if at least one harness has both successful and failed probe
rollouts.

The probe runs are evaluations of that checkpoint on the training split, two attempts per task:
    exp eval -b medagentbench --pool 3h --split train --harness trained --n-attempts 2
    python build_band.py

Writes `benchmarks/medagentbench/splits/band.json` by default (96 tasks as run).

GRPO from another start (`exp grpo --rl-init base`) reuses this band by default, so only the start
differs from the paper's run. A start-specific band, if wanted, is built the same way from that
checkpoint's probe runs and registered as `rl.bands.<start>` in the benchmark config, e.g.
    exp eval -b medagentbench --pool base --split train --harness parity,fc --n-attempts 2
    python build_band.py --setting base --run none --harnesses parity,fc \
        --out benchmarks/medagentbench/splits/band-base.json
`--harnesses` restricts the check to the probed harnesses.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

HARNESSES = ["parity", "t2", "fc"]
REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "splits" / "band.json"


def read_rewards(probe_dir: Path, harnesses=HARNESSES) -> dict:
    """(task, harness) -> list of rewards from the probe runs in <probe_dir>/<harness>/<pass>/."""
    rewards = defaultdict(list)
    for harness in harnesses:
        job_dir = probe_dir / harness
        result_files = list(job_dir.glob("*/task*/result.json"))
        if not result_files:
            raise SystemExit(f"no probe results in {job_dir}; check --evals-dir / --model / --setting / --run")
        for result_file in result_files:
            result = json.loads(result_file.read_text())
            reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            if reward is None:  # trial crashed before the verifier ran
                continue
            rewards[(result.get("task_name", "?"), harness)].append(reward)
    return rewards


def is_mixed(rewards: list) -> bool:
    """True if the task was neither always solved nor always failed."""
    mean = sum(rewards) / len(rewards)
    solved = sum(1 for r in rewards if r >= 1)
    return 0 < mean < 1 or 0 < solved < len(rewards)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evals-dir", type=Path, default=REPO / "runs" / "evals", help="evaluations root")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--setting", default="3h", help="the probed checkpoint's setting")
    parser.add_argument("--run", default="s0", help="its training run (none for the base model)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--harnesses", default=",".join(HARNESSES),
                        help="probed harnesses to read, comma-separated (default: all three)")
    args = parser.parse_args()

    harnesses = [h.strip() for h in args.harnesses.split(",") if h.strip()]
    rewards = read_rewards(args.evals_dir / "medagentbench-train" / args.model / args.setting / args.run, harnesses)
    band = sorted({task for (task, _), rs in rewards.items() if is_mixed(rs)})

    with open(args.out, "w") as f:
        json.dump({"train": band, "test": []}, f, indent=1)

    in_band = {h: sum(1 for (t, hh) in rewards if hh == h and t in band) for h in harnesses}
    print(f"band: {len(band)} tasks (of these, probed on each harness: {in_band}) -> {args.out}")


if __name__ == "__main__":
    main()
