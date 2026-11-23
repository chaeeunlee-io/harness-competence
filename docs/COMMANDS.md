# Run experiments

Run these commands from the repository root. `--backend eidf` uses the Kubernetes sandbox backend used for the original runs; use `--backend docker` on a single machine. Cluster settings come from `.env` (see `env.example`). See [PATHS.md](PATHS.md) for directory locations.

[What each experiment runs](#what-each-experiment-runs) summarises each experiment. Sections 0–8 show the individual commands. To print the full command list for an experiment:

```bash
uv run exp plan <experiment>
```

## What each experiment runs

`uv run exp run <experiment>` writes a self-contained script to `runs/plans/<experiment>.sh` and runs it. For each benchmark, the script runs the required task generation, SFT, GRPO, serving, evaluation, teardown, and finally `exp table <experiment>`. `base` and `teacher` require no training.

### `specialists-vs-3h`

Single-harness specialists vs. 3H on the training harnesses.

```bash
uv run exp run specialists-vs-3h --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| medagentbench | test | parity-only, t2-only, fc-only, 3h | - | base, parity-only, t2-only, fc-only, 3h on parity, t2, fc, `test` |
| appworld | test | code-only, fc-only, miniswe-only, 3h | - | base, code-only, fc-only, miniswe-only, 3h on code, fc, miniswe, `test` |
| bfcl_multiturn | test | fc-only, json-only, pycall-only, 3h | - | base, fc-only, json-only, pycall-only, 3h on fc, json, pycall, `test` |
| reasoning_gym | test-hard | t2-only, miniswe-only, goose-only, 3h | - | base, t2-only, miniswe-only, goose-only, 3h on t2, miniswe, goose, `test-hard` |

### `heldout-transfer`

The same policies evaluated on the held-out harness.

```bash
uv run exp run heldout-transfer --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| medagentbench | test | parity-only, t2-only, fc-only, 3h | - | base, parity-only, t2-only, fc-only, 3h on xml, `test` |
| appworld | test | code-only, fc-only, miniswe-only, 3h | - | base, code-only, fc-only, miniswe-only, 3h on xml, `test` |
| bfcl_multiturn | test | fc-only, json-only, pycall-only, 3h | - | base, fc-only, json-only, pycall-only, 3h on xml, `test` |
| bfcl_singleturn | test | fc-only, json-only, t2-only, 3h | - | base, fc-only, json-only, t2-only, 3h on xml, `test` |
| reasoning_gym | test, test-hard | t2-only, miniswe-only, goose-only, 3h | - | base, t2-only, miniswe-only, goose-only, 3h on aider, `test`; same on `test-hard` |

### `cross-harness-vs-rehearsal`

Cross-harness task supervision vs. rehearsal on the target harness.

```bash
uv run exp run cross-harness-vs-rehearsal --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| medagentbench | test | parity+fc, reh-t2, parity+fc+reh-t2, t2-only, 3h | - | base, parity+fc, reh-t2, parity+fc+reh-t2, t2-only, 3h on parity, t2, fc, `test` |
| reasoning_gym | test | miniswe+goose, reh-t2, miniswe+goose+reh-t2, t2-only, 3h | - | base, miniswe+goose, reh-t2, miniswe+goose+reh-t2, t2-only, 3h on t2, miniswe, goose, `test` |
| bfcl_multiturn | test | json+pycall, reh-fc, json+pycall+reh-fc, fc-only, 3h | - | base, json+pycall, reh-fc, json+pycall+reh-fc, fc-only, 3h on fc, json, pycall, `test` |

### `seed-variance-rehearsal`

The three trained arms of the rehearsal 2×2, retrained with another seed and evaluated on the target harness. Without `--seed` an arm is trained with the trainers' default seed (42); run the experiment once per extra seed. Base rows are not retrained.

```bash
uv run exp run seed-variance-rehearsal --seed 1 --backend <backend> [--sandbox-api-base <url>]
uv run exp table seed-variance-rehearsal --seed 1          # one seed
uv run exp table seed-variance-rehearsal --seeds 1,2,3     # all seeds: pooled per-task bootstrap, per-seed means, SD
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| medagentbench | test | parity+fc, reh-t2, parity+fc+reh-t2 (seed N) | - | the three on t2, `test` |
| reasoning_gym | test, test-hard | miniswe+goose, reh-t2, miniswe+goose+reh-t2 (seed N) | - | the three on t2, `test` and `test-hard` |
| bfcl_multiturn | test | json+pycall, reh-fc, json+pycall+reh-fc (seed N) | - | the three on fc, `test` |

`seed-variance-rehearsal-mab`, `-rg` and `-bfclmt` are the same arms split one benchmark per experiment, for running one benchmark's evaluation on its own GPU or pod. On EIDF, run at most one evaluation whose agent runs on the host (Terminus-2, the MedAgentBench dispatch agent) per 4-CPU pod at a time: two starve each other's agents into timeouts.

### `size-matched-rehearsal-mab`, `-rg`, `-bfclmt`

The rehearsal 2×2 with arms of equal size: the combined pool shrunk to the task-only size (two ways: every rehearsal row kept and task rows subsampled, or the pool's own mixture kept and both subsampled), the task-only and rehearsal-only pools repeated up to the combined size, and task supervision with the same amount of rehearsal through a non-target harness (MedAgentBench: fc rehearsal; BFCL multi-turn: the 186 xml rehearsal rows). The extra pools are declared with `derived:` in the benchmark config and built from the shipped pools on first use (`exp sft`) or with `exp build-pool -b <benchmark> --pool <name>`, into `runs/pools/<benchmark>/`.

Row order: a derived pool is written with the rows that carry tool calls first. Ten of these pools were trained on, for the reported runs, in files built before that ordering: the same rows in another order. They are, on MedAgentBench, `parity+fc+reh-t2~rows258`, `parity+fc+reh-t2~rows258mix`, `parity+fc~rows365rep`, `parity+fc+reh-fc~rows365`, `parity+fc+reh-t2~fail` and `parity+reh-fc`, and on Reasoning Gym, `miniswe+goose+reh-t2~rows269`, `miniswe+goose+reh-t2~rows269mix`, `miniswe+goose~rows389rep` and `miniswe+goose+reh-t2~fail`. A rebuild gives the same training set, and a retrain on it sees the rows in a different order.

```bash
uv run exp run size-matched-rehearsal-mab --backend <backend> [--sandbox-api-base <url>] [--concurrency N]
uv run exp table size-matched-rehearsal-mab
```

| experiment | tasks | trains (SFT) | evaluates |
|---|---|---|---|
| size-matched-rehearsal-mab | test | parity+fc, reh-t2, parity+fc+reh-t2, parity+fc+reh-t2~rows258, …~rows258mix, parity+fc~rows365rep, reh-t2~rows365rep, parity+fc+reh-fc~rows365 | all, with base, on t2 |
| size-matched-rehearsal-rg | test, test-hard | miniswe+goose, reh-t2, miniswe+goose+reh-t2, …~rows269, …~rows269mix, …~rows389rep (×2) | all, with base, on t2 |
| size-matched-rehearsal-bfclmt | test | json+pycall, reh-fc, json+pycall+reh-fc, …~rows802, …~rows802mix, …~rows1070rep (×2), json+pycall+reh-xml~rows988 | all, with base, on fc |

### `failed-rehearsal-control-mab`, `-rg`, `-bfclmt`

Does the rehearsal gain need rehearsal trajectories that passed the verifier? Task supervision plus the same number of *failed* rehearsal trajectories, against the combined arm, with the failed rows alone as a reference. MedAgentBench ships its failed rows (`reh-t2~fail`, the 107 failed trials of the original collection). Reasoning Gym and BFCL multi-turn need one fresh rehearsal collection each (the original collections' failed trials were not kept): serve the `3h` checkpoint, collect, build the failed pool, then run the experiment.

```bash
# MedAgentBench: everything shipped
uv run exp run failed-rehearsal-control-mab --backend <backend> [--sandbox-api-base <url>]

# Reasoning Gym: one collection first (about 1 h), same for bfcl_multiturn with --job rehearsal-fc and --pool reh-fc~fail
uv run exp tasks -b reasoning_gym --split rehearsal-t2
setsid uv run exp serve -b reasoning_gym --model qwen3-4b --pool 3h --port 8000 & SERVER=$!
uv run exp wait --api-base http://127.0.0.1:8000/v1 --pid $SERVER
uv run exp collect -b reasoning_gym --rehearsal --job rehearsal-t2 --backend <backend> --api-base http://127.0.0.1:8000/v1 [--sandbox-api-base <url>]
uv run exp stop --pid $SERVER --port 8000
uv run exp build-pool -b reasoning_gym --pool reh-t2~fail
uv run exp run failed-rehearsal-control-rg --backend <backend> [--sandbox-api-base <url>]
uv run exp table failed-rehearsal-control-rg
```

| experiment | tasks | trains (SFT) | evaluates |
|---|---|---|---|
| failed-rehearsal-control-mab | test | parity+fc, parity+fc+reh-t2, reh-t2~fail, parity+fc+reh-t2~fail (derived) | all, with base, on t2 |
| failed-rehearsal-control-rg | test, test-hard | miniswe+goose, miniswe+goose+reh-t2, reh-t2~fail (new collection), miniswe+goose+reh-t2~fail (derived) | all, with base, on t2 |
| failed-rehearsal-control-bfclmt | test | json+pycall, json+pycall+reh-fc, reh-fc~fail (new collection), json+pycall+reh-fc~fail, reh-fc~likefail, json+pycall+reh-fc~likefail (derived) | all, with base, on fc |

### `grpo-from-base-mab`, `grpo-from-base-rg`

Does the cross-harness transfer of single-harness RL depend on the interface competence acquired by multi-harness SFT? Single-harness GRPO from the **untuned** Qwen3-4B instead of the 3H SFT checkpoint (MedAgentBench: parity rollouts; Reasoning Gym: Terminus-2 rollouts), evaluated on every harness next to base and 3H. The run rolls out on the same learning band as the runs that start from 3H SFT (see §5), so the starting checkpoint is the only difference.

```bash
uv run exp run grpo-from-base-mab --backend <backend> [--sandbox-api-base <url>]
uv run exp run grpo-from-base-mab --seed 2 --only rl:parity@base --backend <backend> [--sandbox-api-base <url>]   # a second GRPO seed
uv run exp run grpo-from-base-rg  --backend <backend> [--sandbox-api-base <url>]
uv run exp run grpo-from-base-rg  --seed 2 --only rl:t2@base --backend <backend> [--sandbox-api-base <url>]
uv run exp table grpo-from-base-mab
```

| experiment | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| grpo-from-base-mab | test, train | 3h | parity, from base | base, 3h, grpo-parity-from-base on parity, t2, fc, xml |
| grpo-from-base-rg | test, test-hard, train | 3h | t2, from base | base, 3h, grpo-t2-from-base on t2, miniswe, goose, aider, both tiers |

With `--seed`, `--only rl:<rollout>@base` keeps the plan to the GRPO run: the 3H row is an SFT setting, and `--seed` would otherwise retrain it too.

### `fc-target-mab`

Why is the rehearsal interaction flat when fc is the MedAgentBench target (base 24.3, parity+t2 25.0, reh-fc 39.6, both 36.6)? One hypothesis is interference from the Terminus-2 traces in that block's task supervision. `parity+reh-fc` (derived: the 108 parity rows plus the 150 fc rehearsal rows, the model's own chat template like the other fc-target arms) removes them. The four arms of the fc-target 2×2 are listed too, so they can be retrained next to it.

```bash
uv run exp run fc-target-mab --only parity+reh-fc --backend <backend> [--sandbox-api-base <url>]   # the new arm only
uv run exp table fc-target-mab
```

### `non-target-rehearsal-bfclmt`

Task supervision plus the same amount of rehearsal through a harness that is not the target. Primary arm: rehearsal through the **json** harness, a training harness (`reh-json`, a new collection with the `3h` checkpoint on the BFCL rehearsal tasks, capped at `reh-fc`'s 268 rows, so `json+pycall+reh-json` has up to 1,070 rows, as `json+pycall+reh-fc` does). Secondary arm: rehearsal through the held-out **xml** harness (`json+pycall+reh-xml~rows988`, the 186 xml rows that exist). On MedAgentBench the same control is `parity+fc+reh-fc~rows365` (size-matched-rehearsal-mab); Reasoning Gym would need a Goose rehearsal collection.

```bash
uv run exp tasks -b bfcl_multiturn --split rehearsal
uv run exp sft -b bfcl_multiturn --pool 3h                       # the demonstrator, if not on this machine
setsid uv run exp serve -b bfcl_multiturn --pool 3h --port 8000 & SERVER=$!
uv run exp wait --api-base http://127.0.0.1:8000/v1 --pid $SERVER
uv run exp collect -b bfcl_multiturn --rehearsal --job rehearsal-json --backend <backend> --api-base http://127.0.0.1:8000/v1 --sandbox-api-base <url>
uv run exp stop --pid $SERVER --port 8000
uv run exp build-pool -b bfcl_multiturn --pool reh-json
uv run exp run non-target-rehearsal-bfclmt --backend <backend> --sandbox-api-base <url>
uv run exp table non-target-rehearsal-bfclmt
```

### `clean-demonstrator-rehearsal-mab`, `-rg`, `-bfclmt`

The rehearsal traces are made by the `3h` checkpoint, which was fine-tuned on the teacher's target-harness solutions of the same training tasks. This repeats the rehearsal 2×2 with rehearsal traces from the Qwen3-32B teacher, which was never fine-tuned on them. MedAgentBench's 32B rehearsal rows ship inside that pool (`reh-t2~32b` is derived from it); Reasoning Gym and BFCL multi-turn need one collection each with the teacher served (same split, attempts and temperature 1.0 as the 3H rehearsal jobs):

```bash
# Reasoning Gym (same for bfcl_multiturn with --split rehearsal, --job rehearsal-fc~32b, --pool reh-fc~32b)
uv run exp tasks -b reasoning_gym --split rehearsal-t2
setsid uv run exp serve -b reasoning_gym --teacher --port 8000 & SERVER=$!      # Qwen3-32B; --tensor-parallel-size 1 fits one H200
uv run exp wait --api-base http://127.0.0.1:8000/v1 --pid $SERVER
uv run exp collect -b reasoning_gym --rehearsal --job rehearsal-t2~32b --backend <backend> --api-base http://127.0.0.1:8000/v1 [--sandbox-api-base <url>]
uv run exp stop --pid $SERVER --port 8000
uv run exp build-pool -b reasoning_gym --pool reh-t2~32b
uv run exp run clean-demonstrator-rehearsal-rg --only reh-t2~32b,miniswe+goose+reh-t2~32b --backend <backend> [--sandbox-api-base <url>]
uv run exp table clean-demonstrator-rehearsal-rg     # the 3h-demonstrator rows next to the 32B ones
```

Watch the verified-solve rate of the collection (`reward.txt` per trial): the pools need up to 107 / 120 / 268 rows at two per task. If the teacher solves too few rehearsal tasks on a benchmark, a different clean demonstrator (`--model`) can be served instead.

| experiment | tasks | trains (SFT) | evaluates |
|---|---|---|---|
| clean-demonstrator-rehearsal-mab | test | reh-t2~32b (derived), parity+fc+reh-t2~32b, and the 2×2 arms | all, with base, on t2 |
| clean-demonstrator-rehearsal-rg | test, test-hard | reh-t2~32b (new collection), miniswe+goose+reh-t2~32b (derived), and the 2×2 arms | all, with base, on t2 |
| clean-demonstrator-rehearsal-bfclmt | test | reh-fc~32b (new collection), json+pycall+reh-fc~32b (derived), and the 2×2 arms | all, with base, on fc |

### `reasoning-stripped-rehearsal-rg`

The rehearsal 2×2 on Reasoning Gym with the reasoning removed from the rehearsal traces, for both demonstrators (the `3h` checkpoint and the Qwen3-32B teacher). The `~noreason` pools are derived from `reh-t2` and `reh-t2~32b`, so the 32B rows need the collection above first.

```bash
uv run exp run reasoning-stripped-rehearsal-rg --backend <backend> [--sandbox-api-base <url>]
uv run exp table reasoning-stripped-rehearsal-rg
```

| experiment | tasks | trains (SFT) | evaluates |
|---|---|---|---|
| reasoning-stripped-rehearsal-rg | test, test-hard | reh-t2~noreason, miniswe+goose+reh-t2~noreason, reh-t2~32b~noreason, miniswe+goose+reh-t2~32b~noreason (all derived), and the arms of clean-demonstrator-rehearsal-rg | all, with base, on t2 |

### `seed-variance-ladder-mab`, `seed-variance-ladder-rg`, `seed-variance-ladder-bfclmt`, `seed-variance-ladder-appworld`

The specialist-vs-3H ladder of one benchmark, retrained with another seed and evaluated on the training harnesses (MedAgentBench: also xml; Reasoning Gym: easy and hard). One benchmark per experiment so a seed of one benchmark can run on its own machine. Base rows are not retrained.

```bash
uv run exp run seed-variance-ladder-bfclmt --seed 1 --backend <backend> [--sandbox-api-base <url>] [--concurrency N]
uv run exp table seed-variance-ladder-bfclmt --seed 1
```

| experiment | tasks | trains (SFT, seed N) | evaluates |
|---|---|---|---|
| seed-variance-ladder-mab | test | parity-only, t2-only, fc-only, 3h | the four on parity, t2, fc, xml |
| seed-variance-ladder-rg | test, test-hard | t2-only, miniswe-only, goose-only, 3h | the four on t2, miniswe, goose, both tiers |
| seed-variance-ladder-bfclmt | test | fc-only, json-only, pycall-only, 3h | the four on fc, json, pycall |
| seed-variance-ladder-appworld | test | code-only, fc-only, miniswe-only, 3h | the four on code, fc, miniswe, xml (every agent runs in the sandbox: pass `--sandbox-api-base`) |
| seed-variance-ladder-rg-t2 / -rg-agents | test, test-hard | (same RG pools) | the RG ladder split by harness: Terminus-2 only (agent on the host, any backend) / mini-swe and Goose (agent in the sandbox) |

### `grpo-3h-vs-single`

3H GRPO vs. single-harness GRPO, both starting from the 3H SFT checkpoint. The reported rows average two GRPO seeds: run the experiment once per seed. Every seed starts from the same 3H SFT checkpoint (the default-seed one); `--seed` sets the trainer's seed.

```bash
uv run exp run grpo-3h-vs-single --seed 1 --backend <backend> [--sandbox-api-base <url>]
uv run exp run grpo-3h-vs-single --seed 2 --backend <backend> [--sandbox-api-base <url>]
uv run exp run grpo-3h-vs-single --seed 1 --only rl:t2 --backend <backend> [--sandbox-api-base <url>]   # one rollout setting only
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| medagentbench | test, train | 3h | 3h, parity, t2 | grpo-3h, grpo-parity, grpo-t2 on parity, t2, fc, xml, `test` |
| reasoning_gym | test, train, test-hard | 3h | 3h, miniswe, t2 | grpo-3h, grpo-miniswe, grpo-t2 on t2, miniswe, goose, aider, `test`; same on `test-hard` |

GRPO needs two GPUs and a pod with 128 GiB of memory, and rolls out only on the backends in `configs/rl.yaml` `trainer_backends` (see §5). The table of GRPO rows next to base and 3H SFT is one of the control tables: `uv run exp control-tables` (see [Tables over training runs](#tables-over-training-runs-and-controls)).

### `base-and-teacher`

Base model and teacher, evaluated per harness.

```bash
uv run exp run base-and-teacher --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| medagentbench | test | - | - | base, teacher on parity, t2, fc, `test` |
| appworld | test | - | - | base, teacher on code, fc, miniswe, `test` |
| bfcl_multiturn | test | - | - | base, teacher on fc, json, pycall, `test` |
| bfcl_singleturn | test | - | - | base, teacher on fc, json, t2, `test` |
| reasoning_gym | test, test-hard | - | - | base, teacher on t2, miniswe, goose, `test`; teacher on the same harnesses for `test-hard` |

### `matched-volume-mab`

MedAgentBench 3H at matched data volume.

```bash
uv run exp run matched-volume-mab --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| medagentbench | test | parity-only, fc-only, 3h~n108, 3h~n108~sub2, 3h~n108~sub3, 3h~n108~cov, 3h~n150, 3h~n150~cov, 3h | - | all listed settings on parity, t2, fc, `test` |

### `specialists-vs-3h-bfclst-rgeasy`

Single-harness specialists vs. 3H on BFCL single-turn and Reasoning Gym easy.

```bash
uv run exp run specialists-vs-3h-bfclst-rgeasy --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| bfcl_singleturn | test | fc-only, json-only, t2-only, 3h | - | base, fc-only, json-only, t2-only, 3h on fc, json, t2, `test` |
| reasoning_gym | test | t2-only, miniswe-only, goose-only, 3h | - | base, t2-only, miniswe-only, goose-only, 3h on t2, miniswe, goose, `test` |

### `all-conditions-mab`

All MedAgentBench SFT and GRPO conditions.

```bash
uv run exp run all-conditions-mab --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| medagentbench | test, train | parity-only, t2-only, fc-only, 3h, parity+fc, reh-t2, parity+fc+reh-t2, parity+fc+reh-t2~32b, reh-t2~fail, parity+t2, reh-fc, parity+t2+reh-fc | 3h, parity | base, all listed SFT settings, grpo-3h and grpo-parity on parity, t2, fc, xml, `test` |

### `gemma-mab`

Gemma-3-4B-it replication with SFT and GRPO.

```bash
uv run exp run gemma-mab --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| medagentbench | test, train | parity-only, t2-only, fc-only, reh-t2, reh-t2~fail, parity+reh-t2, parity+t2, reh-fc~gemma, parity+t2+reh-fc~gemma, 3h | 3h, parity | base, all listed SFT settings, grpo-3h and grpo-parity, each served per harness group (parity and xml; t2; fc), `test` |

All settings in this experiment use `gemma-3-4b-it`. The reported GRPO rows average two seeds: `uv run exp run gemma-mab --seed 1 --only rl:3h,rl:parity --backend <backend>`, then the same with `--seed 2`.

### `gemma-rg-easy`

Gemma-3-4B-it SFT replication on Reasoning Gym easy.

```bash
uv run exp run gemma-rg-easy --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| reasoning_gym | test | t2-only, miniswe-only, goose-only, 3h | - | base, t2-only, miniswe-only, goose-only, 3h on t2, miniswe, goose, aider, `test` |

All settings in this experiment use `gemma-3-4b-it`.

### `rehearsal-all-harnesses-bfclmt`

Rehearsal on each BFCL multi-turn training harness.

```bash
uv run exp run rehearsal-all-harnesses-bfclmt --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| bfcl_multiturn | test | json+pycall, reh-fc, json+pycall+reh-fc, fc-only, 3h | - | base, json+pycall, reh-fc, json+pycall+reh-fc, fc-only, 3h on fc, json, pycall, `test` |

### `heldout-rehearsal-bfclmt`

Rehearsal on the held-out harness, which is absent from the 3H dataset.

```bash
uv run exp run heldout-rehearsal-bfclmt --backend <backend> [--sandbox-api-base <url>]
```

| benchmark | tasks | trains (SFT) | trains (GRPO) | evaluates |
|---|---|---|---|---|
| bfcl_multiturn | test | 3h, 3h+reh-xml | - | 3h, 3h+reh-xml on fc, json, pycall, `test` |

### Tables over training runs and controls

`exp table <experiment>` prints one experiment from one training run. Three more commands build the tables that combine runs. Each reads a spec under `configs/tables/` that names, for every cell, the job folders it is computed from, and writes Markdown and CSV tables plus a manifest (job paths, trials found and scored, a checksum per run) to `results/tables/<command>/`. The outputs of the reported runs are in the repository.

Training runs are named `s0`, `s1`, `s2`, ...: `s0` is an arm's first training run, with the trainers' default seed, and `s1` and up are its retrainings with `--seed 1` and up. `s0b` is a second training run at the default seed (see [PATHS.md](PATHS.md#run-names)). The reported multi-run cells average `s0` and seeds 1 to 3 (AppWorld: 1 to 4).

Both table commands take `--shipped` to read the jobs from `results/rewards.csv` instead of `runs/`; the failure-type commands read the trial transcripts and need the runs.

```bash
uv run exp seed-tables       # the rehearsal 2×2 and the distillation ladders, pooled over training runs and per run
uv run exp control-tables    # size- and budget-matched arms, failed and non-target rehearsal, clean demonstrator, per-template leakage, GRPO
uv run exp failure-types     # rule-based failure types of the failed trials, task supervision with and without rehearsal
uv run exp annotation-sample # the failed trials drawn for human annotation, as an annotator reads them
uv run exp annotation-tables # the rule-based types against the human labels, and annotator agreement
```

| command | spec | needs the jobs of |
|---|---|---|
| `exp seed-tables` | `configs/tables/seed_tables.yaml` | `cross-harness-vs-rehearsal`, `specialists-vs-3h`, `heldout-transfer` (the `s0` runs), then `seed-variance-rehearsal` and the `seed-variance-ladder-*` experiments with `--seed 1`, `2`, `3` (AppWorld: also `4`) |
| `exp control-tables` | `configs/tables/control_tables.yaml` | the `size-matched-rehearsal-*`, `failed-rehearsal-control-*`, `non-target-rehearsal-bfclmt`, `clean-demonstrator-rehearsal-*`, `fc-target-mab`, `grpo-3h-vs-single` (`--seed 1`, `2`) and `grpo-from-base-*` experiments |
| `exp failure-types` | `configs/tables/failure_types.yaml` | `cross-harness-vs-rehearsal` (the task-only and task + rehearsal arms on each target harness) |
| `exp annotation-sample`, `exp annotation-tables` | the same spec | the same jobs; for the tables, also the output of `exp failure-types` and the human labels in `results/failure_annotation_labels.csv` |

A spec refers to a job as `<root>:<path>`, with the roots listed at its top, all inside the repository: `runs/evals` (what `exp eval` writes) and `runs/evals-cluster2` (per-trial rewards of the runs evaluated on a second cluster, in the same layout). The path of a job is `<task set>/<model>/<setting>/<run>/<harness>/<pass>`, see [PATHS.md](PATHS.md). A run with no per-trial files is read from `configs/tables/external_cells.csv` and marked †; a cell that has one has no pooled interval.

Failure annotation: `exp annotation-sample` draws a fixed number of failed trials per arm (seeded, one per task first) and writes each with a trimmed transcript, the verifier's verdict and a passing run of the same task. Three annotators labelled each drawn trial as an interface failure (invalid calls, incorrect submission, termination) or a task-knowledge failure by its first decisive error; their labels are in `results/failure_annotation_labels.csv`, one row per annotator and trial.

Pooling: each run is scored on its own with the benchmark's estimator, the cell is the unweighted mean of the run means, and the 95% interval is a task bootstrap that draws the same tasks for every run (`exp/seedtables.py`).

### Checkpoints shared between experiments

Each experiment writes SFT checkpoints to `runs/checkpoints/<benchmark>/<model>/<pool>/<run>/` and GRPO runs to `runs/rl/<benchmark>/<model>/grpo-<rollout>/<run>/`.

If another experiment needs the same completed checkpoint, `exp sft` or `exp grpo` reuses it. Pass `--force` to retrain.

| benchmark | model | checkpoint | trained by |
|---|---|---|---|
| appworld | qwen3-4b | sft 3h | specialists-vs-3h, heldout-transfer |
| appworld | qwen3-4b | sft code-only | specialists-vs-3h, heldout-transfer |
| appworld | qwen3-4b | sft fc-only | specialists-vs-3h, heldout-transfer |
| appworld | qwen3-4b | sft miniswe-only | specialists-vs-3h, heldout-transfer |
| bfcl_multiturn | qwen3-4b | sft 3h | specialists-vs-3h, heldout-transfer, cross-harness-vs-rehearsal, rehearsal-all-harnesses-bfclmt, heldout-rehearsal-bfclmt |
| bfcl_multiturn | qwen3-4b | sft fc-only | specialists-vs-3h, heldout-transfer, cross-harness-vs-rehearsal, rehearsal-all-harnesses-bfclmt |
| bfcl_multiturn | qwen3-4b | sft json+pycall | cross-harness-vs-rehearsal, size-matched-rehearsal-bfclmt, failed-rehearsal-control-bfclmt, non-target-rehearsal-bfclmt, clean-demonstrator-rehearsal-bfclmt, rehearsal-all-harnesses-bfclmt |
| bfcl_multiturn | qwen3-4b | sft json+pycall+reh-fc | cross-harness-vs-rehearsal, size-matched-rehearsal-bfclmt, failed-rehearsal-control-bfclmt, non-target-rehearsal-bfclmt, clean-demonstrator-rehearsal-bfclmt, rehearsal-all-harnesses-bfclmt |
| bfcl_multiturn | qwen3-4b | sft json+pycall+reh-xml~rows988 | size-matched-rehearsal-bfclmt, non-target-rehearsal-bfclmt |
| bfcl_multiturn | qwen3-4b | sft json-only | specialists-vs-3h, heldout-transfer |
| bfcl_multiturn | qwen3-4b | sft pycall-only | specialists-vs-3h, heldout-transfer |
| bfcl_multiturn | qwen3-4b | sft reh-fc | cross-harness-vs-rehearsal, size-matched-rehearsal-bfclmt, clean-demonstrator-rehearsal-bfclmt, rehearsal-all-harnesses-bfclmt |
| bfcl_singleturn | qwen3-4b | sft 3h | heldout-transfer, specialists-vs-3h-bfclst-rgeasy |
| bfcl_singleturn | qwen3-4b | sft fc-only | heldout-transfer, specialists-vs-3h-bfclst-rgeasy |
| bfcl_singleturn | qwen3-4b | sft json-only | heldout-transfer, specialists-vs-3h-bfclst-rgeasy |
| bfcl_singleturn | qwen3-4b | sft t2-only | heldout-transfer, specialists-vs-3h-bfclst-rgeasy |
| medagentbench | qwen3-4b | grpo 3h | grpo-3h-vs-single, all-conditions-mab |
| medagentbench | qwen3-4b | grpo parity | grpo-3h-vs-single, all-conditions-mab |
| medagentbench | qwen3-4b | grpo t2 | grpo-3h-vs-single |
| medagentbench | qwen3-4b | sft 3h | specialists-vs-3h, heldout-transfer, cross-harness-vs-rehearsal, grpo-3h-vs-single, matched-volume-mab, grpo-from-base-mab, all-conditions-mab |
| medagentbench | qwen3-4b | sft fc-only | specialists-vs-3h, heldout-transfer, matched-volume-mab, all-conditions-mab |
| medagentbench | qwen3-4b | sft parity+fc | cross-harness-vs-rehearsal, size-matched-rehearsal-mab, failed-rehearsal-control-mab, clean-demonstrator-rehearsal-mab, all-conditions-mab |
| medagentbench | qwen3-4b | sft parity+fc+reh-t2 | cross-harness-vs-rehearsal, size-matched-rehearsal-mab, failed-rehearsal-control-mab, clean-demonstrator-rehearsal-mab, all-conditions-mab |
| medagentbench | qwen3-4b | sft parity+fc+reh-t2~32b | clean-demonstrator-rehearsal-mab, all-conditions-mab |
| medagentbench | qwen3-4b | sft parity+t2 | fc-target-mab, all-conditions-mab |
| medagentbench | qwen3-4b | sft parity+t2+reh-fc | fc-target-mab, all-conditions-mab |
| medagentbench | qwen3-4b | sft parity-only | specialists-vs-3h, heldout-transfer, matched-volume-mab, all-conditions-mab |
| medagentbench | qwen3-4b | sft reh-fc | fc-target-mab, all-conditions-mab |
| medagentbench | qwen3-4b | sft reh-t2 | cross-harness-vs-rehearsal, size-matched-rehearsal-mab, clean-demonstrator-rehearsal-mab, all-conditions-mab |
| medagentbench | qwen3-4b | sft reh-t2~fail | failed-rehearsal-control-mab, all-conditions-mab |
| medagentbench | qwen3-4b | sft t2-only | specialists-vs-3h, heldout-transfer, cross-harness-vs-rehearsal, all-conditions-mab |
| reasoning_gym | qwen3-4b | sft 3h | specialists-vs-3h, heldout-transfer, cross-harness-vs-rehearsal, grpo-3h-vs-single, specialists-vs-3h-bfclst-rgeasy, grpo-from-base-rg |
| reasoning_gym | qwen3-4b | sft goose-only | specialists-vs-3h, heldout-transfer, specialists-vs-3h-bfclst-rgeasy |
| reasoning_gym | qwen3-4b | sft miniswe+goose | cross-harness-vs-rehearsal, size-matched-rehearsal-rg, failed-rehearsal-control-rg, clean-demonstrator-rehearsal-rg, reasoning-stripped-rehearsal-rg |
| reasoning_gym | qwen3-4b | sft miniswe+goose+reh-t2 | cross-harness-vs-rehearsal, size-matched-rehearsal-rg, failed-rehearsal-control-rg, clean-demonstrator-rehearsal-rg, reasoning-stripped-rehearsal-rg |
| reasoning_gym | qwen3-4b | sft miniswe+goose+reh-t2~32b | clean-demonstrator-rehearsal-rg, reasoning-stripped-rehearsal-rg |
| reasoning_gym | qwen3-4b | sft miniswe-only | specialists-vs-3h, heldout-transfer, specialists-vs-3h-bfclst-rgeasy |
| reasoning_gym | qwen3-4b | sft reh-t2 | cross-harness-vs-rehearsal, size-matched-rehearsal-rg, clean-demonstrator-rehearsal-rg, reasoning-stripped-rehearsal-rg |
| reasoning_gym | qwen3-4b | sft reh-t2~32b | clean-demonstrator-rehearsal-rg, reasoning-stripped-rehearsal-rg |
| reasoning_gym | qwen3-4b | sft t2-only | specialists-vs-3h, heldout-transfer, cross-harness-vs-rehearsal, specialists-vs-3h-bfclst-rgeasy |

The seed-variance experiments train the same settings again with `--seed`, each seed in its own run folder.

## 0. Session setup

For each new shell or tmux window, start tmux first and then source the session setup inside it:

```bash
cd ${REPO}
. setup/eidf/session.sh
```

On EIDF this loads `.env` and sets `PATH`, the uv cache, `KUBECONFIG`, and `POD_IP`. See `setup/eidf/README.md`.

`exp` runs the installed environments with `uv run --no-sync`, so no command touches the network at run time (a lost package fetch mid-experiment once stopped a chain). After changing a lock file, run `setup/install.sh` again to sync.

The serve/stop pattern used below is:

```bash
serve () {
  setsid uv run exp serve -b "$1" --model qwen3-4b "${@:2}" --port 8000 & SERVER=$!
  uv run exp wait --api-base http://127.0.0.1:8000/v1 --pid $SERVER
}

down () {
  uv run exp stop --pid $SERVER --port 8000
}

EV="--backend eidf --api-base http://127.0.0.1:8000/v1 --sandbox-api-base http://$POD_IP:8000/v1 --concurrency 16"
```

`setsid` gives the server its own process group so `exp stop` can clean it up. `exp wait --pid` exits if the server crashes, and fails if the process answering on the port is not that server (one left over from an earlier run would otherwise be evaluated in its place). `exp stop --port` stops whatever listens on the port, which covers a server whose recorded pid is wrong: in an interactive shell `setsid … &` forks, so `$!` is a parent that exits at once (in a script it is the server itself). `exp stop` then waits for the GPUs to become free. The GPU smoke test uses the same pattern. `exp stop` touches only the server it names (`--pid`, `--port`): it waits for that server's processes to end and force-stops its leftover vLLM engines, so a server on the other GPU of the same pod keeps running; called with neither flag it instead waits for the visible GPUs to be free and stops every leftover engine on the machine.

A finished job whose trials failed on infrastructure (the model server not answering yet, a sandbox pod that disappeared) keeps those trials as errored, unscored, and a resume does not redo them. `exp retry-errors <job folder>` removes them (agent timeouts are outcomes and stay); the next `exp eval` of the same cell runs them again. Harbor resumes a job only under the config it was started with, so use the same `--concurrency` as the original run:

```bash
uv run exp retry-errors runs/evals/medagentbench/qwen3-4b/3h/s0/t2/n4 --dry-run   # list
uv run exp retry-errors runs/evals/medagentbench/qwen3-4b/3h/s0/t2/n4             # remove, then exp eval again
uv run exp retry-errors --down 2026-11-09T00:50:40 2026-11-09T00:56:37 runs/evals/<job>   # also the trials that ran while the server was down
```

**Daytona sandboxes left behind.** Harbor deletes a trial's Daytona sandbox when the trial ends, but not when Harbor itself is killed; such sandboxes keep their disk, and at the organisation's limit (300 GiB on the tier these runs used; a MedAgentBench sandbox takes 9 GiB, a Reasoning Gym one 1 GiB) every new trial fails with `DaytonaValidationError: Total disk limit exceeded`. After stopping a run on Daytona, and before a large one:

```bash
uv run exp daytona-sweep                    # list sandboxes with no activity for 30 minutes
uv run exp daytona-sweep --delete           # delete them
uv run exp daytona-wait --free-gib 90       # block until 90 GiB are free (an evaluation: 10 MAB sandboxes)
```

Plan concurrent Daytona runs within the limit. GRPO does not use Daytona: the trainer rolls out only on the backends in `configs/rl.yaml` `trainer_backends` (on EIDF: `eidf`), and `exp grpo` refuses any other. After training it audits the rollouts per harness (see [Reinforcement learning](#5-reinforcement-learning)); a run that fails the audit is neither exported nor recorded.

A trial that runs while the model server is down may end as an ordinary failure with reward 0 and no exception (the agent swallows the connection error), which nothing can tell apart from a real failure afterwards. `--down FROM TO` (UTC, as written in `result.json`) therefore removes every trial that overlapped that window too; take FROM from the server's last log line and TO from the replacement server's "startup complete" line.

These examples use `--concurrency 16`. Without it, each benchmark uses its configured value: 10 for MedAgentBench, Reasoning Gym, and BFCL single-turn; 18 for BFCL multi-turn; and 8 for AppWorld.

Run long jobs inside tmux. Avoid SIGKILL, which can leave sandbox jobs behind.

## 1. Task directories

Generate each required benchmark split once:

```bash
uv run exp tasks -b medagentbench   --split test
uv run exp tasks -b reasoning_gym   --split test
uv run exp tasks -b reasoning_gym   --split test-hard
uv run exp tasks -b bfcl_singleturn --split test
uv run exp tasks -b bfcl_multiturn  --split test
uv run exp tasks -b appworld        --split test
```

## 2. No training

Base policy on all harnesses, including held out:

```bash
serve medagentbench --pool base
uv run exp eval -b medagentbench --model qwen3-4b --pool base --harness all --split test $EV
down
```

Teacher evaluation uses Qwen3-32B with tensor parallelism 2, as configured in `configs/models.yaml`, so it needs two GPUs:

```bash
setsid uv run exp serve -b medagentbench --model qwen3-4b --teacher --port 8000 & SERVER=$!
uv run exp wait --api-base http://127.0.0.1:8000/v1 --pid $SERVER
uv run exp eval -b medagentbench --model qwen3-4b --teacher --harness all --split test $EV
down
```

## 3. Supervised training

Available datasets by benchmark (`uv run exp list pools` is authoritative):

| benchmark | single-harness | multi | rehearsal: cross-task, rehearsal-only, both |
|---|---|---|---|
| medagentbench | `parity-only` `t2-only` `fc-only` | `3h` | `parity+fc` `reh-t2` `parity+fc+reh-t2` |
| reasoning_gym | `t2-only` `miniswe-only` `goose-only` | `3h` | `miniswe+goose` `reh-t2` `miniswe+goose+reh-t2` |
| bfcl_singleturn | `fc-only` `json-only` `t2-only` | `3h` | — |
| bfcl_multiturn | `fc-only` `json-only` `pycall-only` | `3h` | `json+pycall` `reh-fc` `json+pycall+reh-fc` |
| appworld | `code-only` `fc-only` `miniswe-only` | `3h` | — |

Additional settings, including volume-matched datasets, MedAgentBench fc-target rehearsal, rehearsal variants, BFCL multi-turn `3h+reh-xml`, and AppWorld `3h~matched`, are documented in each benchmark README.

Example:

```bash
B=medagentbench
for P in parity-only t2-only fc-only 3h parity+fc reh-t2 parity+fc+reh-t2; do
  uv run exp sft -b $B --model qwen3-4b --pool $P --wandb-run ${B}-$P
done
```

Checkpoints are written to:

```text
runs/checkpoints/<benchmark>/qwen3-4b/<pool>/
```

`final/` is the model that serving and evaluation use. The trainer's intermediate `checkpoint-<N>` folders (model plus optimizer state, about 27 GB each for Qwen3-4B) are removed once a run finishes; pass `--keep-checkpoints` to keep them.

**Evaluating what is finished.** `exp run <experiment> --eval-only` writes and runs the same script without the training lines; a setting whose checkpoint is not finished is left out with a `# skipped` note, so the command can be run again as checkpoints land (finished evaluation jobs resume instantly). `--concurrency N` sets the concurrent trials of every evaluation job. Both combine with `--seed`.

**One run of a plan at a time.** `exp run` writes `runs/plans/<plan>.sh` as the record, executes a copy from memory (so rewriting the file cannot alter a run in progress), and holds `runs/plans/<plan>.lock` (host, pid, start time) until it exits; a second `exp run` of the same plan refuses to start while the lock exists, since two runs would resume each other's evaluation jobs. After a crash, `--ignore-lock` (or deleting the file) clears it. Plans that differ in seed, attempts, pass or `--only` are different plans and can run side by side.

**Part of an experiment.** `exp run <experiment> --only <setting>,<setting>` runs only those settings (pool names, `base`, `rl:<rollout>`), so one experiment can be split across machines; each subset gets its own script (`runs/plans/<experiment>-only-<hash>.sh`) and `exp table` reads the finished jobs of every part together.

**Fewer attempts, in passes.** `exp run <experiment> --n-attempts N` evaluates every setting with N attempts per task instead of the protocol's, each into its own `n<N>` folder; `--pass P` (P ≥ 2) repeats the same evaluations into `n<N>-p<P>` folders (harbor cannot add attempts to a finished job, so each pass is its own folder). `exp table <experiment> --n-attempts 1 --pass 1 --pass 2` pools the passes per task. Useful for seeing every setting once before spending the full protocol on any of them.

**Seeds.** Without `--seed`, training uses the trainers' default seed (42). `--seed N` retrains a setting with another seed (data order, dropout) into its own run folder, `<setting>/s<N>` (the default-seed run is `<setting>/s0`), and the same flag on `exp serve`, `exp eval`, `exp grpo`, `exp plan`/`exp run` and `exp table` points them at that retraining. Nothing without `--seed` is touched.

```bash
uv run exp sft  -b medagentbench --pool parity+fc+reh-t2 --seed 1     # -> runs/checkpoints/medagentbench/qwen3-4b/parity+fc+reh-t2/s1/
uv run exp serve -b medagentbench --pool parity+fc+reh-t2 --seed 1 &
uv run exp eval -b medagentbench --pool parity+fc+reh-t2 --seed 1 --harness t2 ...
uv run exp grpo -b medagentbench --rollout parity --seed 1            # -> runs/rl/medagentbench/qwen3-4b/grpo-parity/s1/ (trainer.seed=1)
```

## 4. Supervised evaluation

**Serving protocol (reasoning parser).** MedAgentBench and Reasoning Gym serve Qwen3 with vLLM's `deepseek_r1` reasoning parser (BFCL and AppWorld use none). The chat template does not open `<think>`, and `deepseek_r1` reads text up to `</think>` as reasoning, so a reply that does not close a reasoning block is read as containing no action, and the harness asks the model to act; all reported results were served this way. An alternative parser that reads reasoning only inside a `<think>` block the reply opened, `explicit_think` (`training/templates/explicit_think_reasoning_parser.py`, a vLLM plugin), can be configured per model in the benchmark's `serve` settings (`reasoning_parser`, `reasoning_parser_plugin`, overridable per harness). Give such a setting a `protocol` name: every evaluation job and GRPO run served under it then carries the name (`...-fc-<protocol>`, `grpo-3h-<protocol>-s1`), so its results never share a name with results served otherwise.

Evaluate one trained setting at a time:

```bash
B=medagentbench
for P in base parity-only t2-only fc-only 3h parity+fc reh-t2 parity+fc+reh-t2; do
  serve $B --pool $P
  uv run exp eval -b $B --model qwen3-4b --pool $P --harness all --split test $EV
  down
done
```

`--harness all` evaluates the three training harnesses and the held-out harness. Use `heldout`, `trained`, or a specific harness name to select a subset. Add `--checkpoint <path>` to `serve` to serve a checkpoint from another location. `--evals-dir <path>` writes the jobs under another root than `runs/evals`; `exp table` takes the same option to read them.

Reasoning Gym is evaluated on both tiers:

```bash
serve reasoning_gym --pool 3h
for S in test test-hard; do
  uv run exp eval -b reasoning_gym --model qwen3-4b --pool 3h --harness all --split $S $EV
done
down
```

Volume-matched settings are ordinary SFT datasets:

```bash
for P in 3h~n108 3h~n150 3h~n108~cov 3h~n150~cov 3h~n108~sub2 3h~n108~sub3; do
  uv run exp sft -b medagentbench --model qwen3-4b --pool $P
  serve medagentbench --pool $P
  uv run exp eval -b medagentbench --model qwen3-4b --pool $P --harness all --split test $EV
  down
done
```

## 5. Reinforcement learning

By default, `exp grpo` starts from the model's 3H SFT checkpoint at:

```text
runs/checkpoints/<benchmark>/<model>/3h/final
```

Pass `--init` to use another checkpoint.

Gemma 3 is multimodal, and GRPO's weight sync into vLLM fails on its vision tower, which nothing here uses. For `--model gemma-3-4b-it`, `exp grpo` therefore starts from a text-only export of the 3H SFT checkpoint, `runs/checkpoints/<benchmark>/gemma-3-4b-it/3h/final-text-only`, and writes it the first time with `training/export_text_only.py`. The reported Gemma GRPO run started from the same kind of export.

```bash
uv run exp grpo -b medagentbench --model qwen3-4b --rollout 3h      --backend eidf --run-name mab-grpo-3h
uv run exp grpo -b medagentbench --model qwen3-4b --rollout parity  --backend eidf --run-name mab-grpo-parity
uv run exp grpo -b reasoning_gym --model qwen3-4b --rollout 3h      --backend eidf --run-name rg-grpo-3h
uv run exp grpo -b reasoning_gym --model qwen3-4b --rollout miniswe --backend eidf --run-name rg-grpo-miniswe
```

GRPO uses two GPUs with TP=2 and colocated vLLM, while also running the sandbox backend. `exp grpo` adds the repository root to `PYTHONPATH`.

**Which agent rolls out, and how it is served.** Rollouts go through the benchmark's dispatch agent (`rl.agent_import_path`), which runs each task under the harness its dataset row names. Harbor selects an agent by `name` before `import_path`, and the trainer's default trial config names `terminus-2`, so `exp grpo` clears the name (`harbor_trial_config.agent.name=null`) and refuses to train if the rendered command does not. The trainer's vLLM serves the rollouts with the benchmark's evaluation parsers (`serve.reasoning_parser`, `serve.tool_call_parser` in `configs/benchmarks/<benchmark>.yaml`): the MedAgentBench parity agent reads its action from the response content, and fc makes native tool calls.

**What `exp grpo` checks after training.** From the rollouts' `result.json` files: every rollout ran its harness's own agent (its `agent_info.name` is the one `rl.rollout_agents` in `configs/benchmarks/<benchmark>.yaml` lists for its harness; under the old Terminus-2 default every harness recorded `terminus-2`), the harnesses rolled out are exactly the run's, and each harness has rollouts with token-level details (what the trainer learns from) and rollouts that reached the verifier. It prints one line per harness (rollouts, with token details, scored, mean reward) and writes the first prompt of one rollout per harness to `<run>/harness-prompts/<harness>.txt` for a human check. A run that fails a check is neither exported nor recorded. A one-step test run shows the whole chain before a full run:

```bash
uv run exp grpo -b medagentbench --model qwen3-4b --rollout 3h --backend eidf --max-steps 1 --batch-size 6 \
    --out runs/rl/medagentbench/qwen3-4b/grpo-3h-smoke --run-name mab-grpo-3h-smoke
```

**Starting elsewhere than the 3H checkpoint.** `exp grpo --rl-init base` starts from the untuned model (`--rl-init <pool>` from another pool's checkpoint); the run is named `grpo-<rollout>-from-<start>`. It rolls out on the paper's learning band (`rl.band`, the tasks probed from 3H), so the comparison with the paper's run changes only the starting checkpoint. How much of that band the start can solve at all is measured by probing it on the train split (optional; it does not change the run):

```bash
uv run exp tasks -b medagentbench --split train
serve medagentbench --pool base
uv run exp eval -b medagentbench --pool base --harness parity --split train --n-attempts 2 $EV   # optional probe
down
uv run exp grpo -b medagentbench --model qwen3-4b --rollout parity --rl-init base --backend eidf --run-name mab-grpo-parity-from-base
uv run exp eval -b medagentbench --model qwen3-4b --rl parity --rl-init base --harness all --split test $EV
```

A start-specific band, built with `build/build_band.py` from that checkpoint's probe runs and registered as `rl.bands.<start>` in the benchmark config, replaces the paper's band for that start; `--band` overrides both for one run.

In an experiment, such a run is listed as `rl: [parity@base]` (see `grpo-from-base-mab`); `exp table` labels it "GRPO, rollouts on parity, from base".

Evaluate RL checkpoints with:

```bash
for RL in 3h parity; do
  serve medagentbench --rl $RL
  uv run exp eval -b medagentbench --model qwen3-4b --rl $RL --harness all --split test $EV
  down
done
```

After training, `exp grpo` writes a Hugging Face export of the checkpoints the paper evaluates, step 5 and the last one (`export_steps` in `configs/rl.yaml`), to:

```text
runs/rl/<benchmark>/<model>/grpo-<rollout>/exports/global_step_<n>/policy/
```

The exports are built on the CPU from the trainer's native checkpoints with `training/merge_fsdp2_to_hf.py`; the trainer's own export is disabled because it gathers the whole model in CPU memory and ran out on a 128 GiB pod. If a run crashes, `exp grpo ... --resume` continues it from the latest checkpoint, and `exp grpo ... --export-only` writes any exports that are missing.

Reported `grpo-<rollout>` results use the final checkpoint. `grpo-<rollout>-step5` uses `global_step_5`, served with `--rl-step 5`.

## 6. Building datasets from scratch

Only needed when rebuilding the distillation datasets.

Collect teacher traces first, then rehearsal traces, then build the datasets. Trace jobs are stored under `runs/collect/<benchmark>/<job>` and datasets under `runs/pools/<benchmark>/<setting>.jsonl`. Each benchmark README gives the full sequence.

Example for MedAgentBench:

```bash
uv run exp tasks -b medagentbench --split train

setsid uv run exp serve -b medagentbench --model qwen3-4b --teacher --port 8000 & SERVER=$!
uv run exp wait --api-base http://127.0.0.1:8000/v1 --pid $SERVER

uv run exp collect -b medagentbench --backend eidf \
  --api-base http://127.0.0.1:8000/v1 \
  --sandbox-api-base http://$POD_IP:8000/v1

down

uv run exp tasks -b medagentbench --split rehearsal-t2

serve medagentbench --pool 3h
uv run exp collect -b medagentbench --rehearsal --job rehearsal-t2 $EV
down

uv run exp build-pool -b medagentbench

uv run exp sft -b medagentbench --model qwen3-4b --pool 3h \
  --data runs/pools/medagentbench/3h.jsonl
```

The dataset for the Hugging Face Hub is written from the pools, the rehearsal tasks and the annotation sample. Nothing is uploaded.

```bash
uv run exp hf-export                      # to runs/hf/harness-competence/
uv run exp hf-export --benchmarks all     # with BFCL single-turn and AppWorld, which are left out by default
```

## 7. Scoring and checks

```bash
uv run exp table <experiment>
uv run exp table <experiment> --shipped
uv run exp table <experiment> --format cells --out mine.csv
```

The first reads local runs, the second reads the shipped results, and the third writes local results using the same schema as `results/cells.csv`.

The shipped results are three files, written by `exp export-results` from the evaluation jobs:

| file | one row per | used by |
|---|---|---|
| `results/rewards.csv` | task of an evaluation job: the rewards of its attempts | `exp table --shipped`, `exp seed-tables --shipped`, `exp control-tables --shipped`, which compute every mean and interval from it exactly as from the job folders |
| `results/cells.csv` | run of an arm on a harness, per metric: mean, pass@n, 95% CI | `exp table --shipped`, for cells with no per-task rewards |
| `results/runs.csv` | training run of an arm: the experiments that use it, its checkpoint folder, its evaluation jobs | no command; it is the index of runs |

```bash
uv run exp export-results                 # rewrite the three files from runs/evals and runs/evals-cluster2
uv run exp seed-tables --shipped          # the tables over training runs, from results/rewards.csv
uv run exp control-tables --shipped
```

`exp failures` counts automatic signs of failure per cell of an experiment, from the trial folders:

```bash
uv run exp failures <experiment>          # timeouts, re-prompts, empty replies, turns without a call
```

## 8. Gemma replication

Use the same commands with:

```bash
--model gemma-3-4b-it
```

Gemma experiments cover MedAgentBench and Reasoning Gym. Training and serving use the Gemma-specific chat templates defined in the benchmark configs. `exp grpo --model gemma-3-4b-it` starts from a text-only export of Gemma's 3H SFT checkpoint (see section 5).

For MedAgentBench, Gemma is served three ways: `fc` with the Hermes template and tool parser, `t2` with the merge-tolerant template (Gemma's own, with consecutive same-role messages merged, which Terminus-2's summarisation hand-off sends), and `parity` and `xml` with Gemma's own template. Start separate servers with `exp serve --harness fc`, `exp serve --harness t2` and `exp serve --harness parity`. `exp run gemma-mab` handles this automatically.

GRPO serves all three rollout harnesses from one server, so Gemma's MedAgentBench GRPO uses its own serving (`rl.rollout_serve` in the benchmark config): `training/templates/gemma3_rl_unified.jinja2` with the Hermes tool parser on every harness. That template matches Gemma's own on alternating conversations and `gemma3_hermes_fc.jinja2` on tool calls, and merges the consecutive same-role messages Terminus-2 sends. Evaluation serving is unchanged. `tests/test_grpo_render.py` checks both.

The `serve` helper above fixes `--model qwen3-4b`, so call `exp serve` directly for Gemma.

## Notes

- `--concurrency 16` uses roughly 32 CPUs of sandbox quota; leave at least 20 CPUs free in the namespace.
- One MedAgentBench harness at `n=4` is about 244 trials and takes roughly 40 minutes at concurrency 10.
- Sandboxes reach the model server through `$POD_IP`, not `127.0.0.1`.
- Run `down` between settings. A leftover vLLM process can hold 72 GB of GPU memory and prevent the next server from starting.
- `--dry-run` prints the underlying command for `tasks`, `serve`, `sft`, `eval`, `collect`, `build-pool`, `oracle`, and `grpo`. `exp run --dry-run` only writes the generated script.
