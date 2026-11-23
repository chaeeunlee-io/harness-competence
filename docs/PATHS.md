# Paths and dependencies

Where files live, what each command reads, and what it writes.

## Paths

`exp` finds the repo root from `exp/config.py` and builds all paths from there. There are no machine-specific paths in the code.

Cluster-specific values go in `.env` (copy `env.example`).

Most outputs go under `runs/`, which is gitignored. The few exceptions are listed below.

## Files in the repo

| Path | Contents |
|---|---|
| `configs/experiments.yaml` | experiment definitions |
| `configs/benchmarks/<benchmark>.yaml` | benchmark splits, harnesses, serving settings, evaluation settings, datasets, build steps, and GRPO settings |
| `configs/models.yaml`, `configs/rl.yaml` | model settings, chat templates, and GRPO config |
| `configs/tables/` | which job folders each cell of the pooled-run, control and failure-type tables is computed from |
| `benchmarks/<benchmark>/pools/<setting>.jsonl` | SFT datasets used for training |
| `benchmarks/<benchmark>/splits/` | train and test task lists |
| `benchmarks/<benchmark>/tasks/` | task generators |
| `benchmarks/<benchmark>/harnesses/` | harness implementations |
| `benchmarks/<benchmark>/build/` | scripts for rebuilding datasets from traces |
| `training/templates/` | chat templates |
| `results/rewards.csv` | the reward of every attempt, per task and evaluation job, written by `exp export-results`; what `--shipped` tables are computed from |
| `results/runs.csv` | the index of runs: the experiments that use each arm, its checkpoint folder, what a GRPO run started from, its evaluation jobs |
| `results/cells.csv` | one row per run of an arm on a harness and metric (mean, pass@n, 95% CI); read by `exp table --shipped` for cells without per-task rewards |
| `results/failure_annotation_labels.csv` | human failure-type labels of the annotated failed trials, read by `exp annotation-tables` |
| `results/tables/<command>/` | the tables over training runs, the control tables and the failure-type tables, as written by `exp seed-tables`, `exp control-tables`, `exp failure-types` and `exp annotation-tables`: Markdown, CSV, and a manifest of the job folders behind every cell |
| `third_party/harbor/` | vendored Harbor |

## Files written under `runs/`

| Path | Written by | Contents |
|---|---|---|
| `runs/tasks/<benchmark>/<split>/` | `exp tasks` | generated task folders and `.hc-tasks.json` |
| `runs/staging/` | `exp tasks` | temporary clones and generator output before tasks are copied into `runs/tasks/` |
| `runs/checkpoints/<benchmark>/<model>/<setting>/<run>/final/` | `exp sft` | SFT checkpoint and its training record |
| `runs/checkpoints/<benchmark>/<model>/<setting>/<run>/final-text-only/` | `exp grpo` | text-only export of a Gemma 3 SFT checkpoint, which its GRPO starts from |
| `runs/rl/<benchmark>/<model>/<setting>/<run>/` | `exp grpo` | native trainer checkpoints (`ckpts/`), their Hugging Face exports written after training (`exports/`), and rollout trials |
| `runs/serve/` | `exp serve` | a copy of the base model's chat template, written when a setting is served with it |
| `runs/evals/<task set>/<model>/<setting>/<run>/<harness>/<pass>/` | `exp eval` | one evaluation job: trial outputs, trajectories, and rewards; its Harbor config is `<pass>.yaml` beside it |
| `runs/evals-cluster2/` | - | per-trial rewards (`reward.txt`, `config.json`) of the runs evaluated on a second cluster, laid out like `runs/evals/` |
| `runs/collect/<benchmark>/<job>/` | `exp collect` | trace-collection jobs (teacher and rehearsal traces), under the job names the benchmark config gives them |
| `runs/oracle/<benchmark>-<split>/` | `exp oracle` | reference-solution runs |
| `runs/hf/harness-competence/` | `exp hf-export` | the dataset for the Hugging Face Hub as one folder with its card, ready to upload: rehearsal tasks and trajectories, the teacher's distillation trajectories, and the annotated failures |
| `runs/pools/<benchmark>/<setting>.jsonl` | `exp build-pool`, `exp sft` | datasets that are built, not shipped: rebuilt from local traces, or declared with `derived:` in the config and built from other pools of the benchmark (subsampled or repeated rows) |
| `runs/plans/<experiment>[-s<seed>][-n<N>][-p<P>][-only-<hash>].sh` | `exp run` | the script it ran (what `exp plan` prints), one per experiment, seed, attempts and pass |
| `runs/wandb/` | `exp sft --wandb-run` | local W&B files |
| `runs/smoke/` | `setup/smoke_gpu.sh` | smoke-test outputs |
| `runs/plans-check/` | `setup/check_plans.sh` | checked experiment scripts |

`runs/staging/` can be deleted after task generation.

### How a run is addressed

A trained model and everything made from it share one address, one folder level per fact:

| Level | Values | Meaning |
|---|---|---|
| `<task set>` | `medagentbench`, `reasoning_gym-easy`, `medagentbench-train`, ... | the tasks evaluated: the benchmark's test split, with its tier where the test split has tiers, or `<benchmark>-<split>` for another split |
| `<model>` | `qwen3-4b`, `gemma-3-4b-it`, `qwen3-32b` | the base model |
| `<setting>` | `3h`, `parity+fc+reh-t2`, `parity+fc+reh-t2~rows258`, `grpo-parity`, `base`, `teacher` | what it was trained on, as named in the configs. `+` joins the datasets of an arm and `~` sets off a variant of it (`~32b`, `~fail`, `~rows258`, ...), see [results/README.md](../results/README.md#setting-names) |
| `<run>` | `s0`, `s0b`, `s1`, `s2`, ...; `none` | which training run, see below; `none` for the base model and the teacher, which are not trained |
| `<harness>` | `t2`, `parity`, `fc`, ... | the harness evaluated through |
| `<pass>` | `n4`, `n1`, `n1-p2`, ... | `n<N>`: N attempts per task; `n<N>-p<P>`: the P-th repeat of that evaluation (`--pass P`) |

So `runs/evals/medagentbench/qwen3-4b/parity+fc+reh-t2/s2/t2/n4` is the four-attempt Terminus-2 evaluation of the seed-2 model trained on `parity+fc+reh-t2`, whose checkpoint is `runs/checkpoints/medagentbench/qwen3-4b/parity+fc+reh-t2/s2/final`. Tables and result files name a job by its path below `runs/evals/`.

### Run names

| Run | What it is |
|---|---|
| `s0` | an arm's first training run, at the trainers' default seed (42). For the arms of the main tables this is the run the paper reports. What `exp sft` and `exp grpo` write without `--seed`. |
| `s1`, `s2`, ... | a retraining of the arm with `--seed 1`, `--seed 2`, ... |
| `s0b` | a second training run of the arm at the default seed: same data, recipe and seed as `s0`, trained again, so its weights differ. It exists only for arms that were trained twice at that seed. |

The control experiments were trained later than the main tables. So that a control is compared with reference arms trained alongside it, the reference arms were trained again at the default seed, and those runs are `s0b`. A control arm was trained once at that seed, so its run is `s0`. In a control table a reference row labelled `s0b` and a control row labelled `s0` are therefore the same kind of run. The GRPO runs start from the `s0b` run of the 3H checkpoint.

For the base model, which is not trained, the tables use the same two labels for its two evaluations: `s0` is the first (`n4`) and `s0b` the repeat (`n4-p2`).

## `.env`

| Variable | Used for |
|---|---|
| `SHARED_MODEL_STORE` | shared model weights under `<store>/<Hugging Face id>`; if unset, models are downloaded from Hugging Face |
| `SANDBOX_API_BASE` | model-server URL reachable from inside sandboxes |
| `RG_TASK_IMAGE`, `APPWORLD_IMAGE` | prebuilt task images for EIDF; AppWorld always needs one |
| `K8S_NAMESPACE`, `KUEUE_QUEUE`, `EIDF_USER`, `EIDF_GROUPS`, `EIDF_HOME`, `KUBECONFIG` | EIDF setup |
| `WANDB_API_KEY` | optional W&B logging |
| `KANIKO_*` | image builds through `setup/eidf/kaniko_build.sh` |

Build the AppWorld image with:

```bash
benchmarks/appworld/tasks/build_appworld_image.sh <image>
```

## Outside the repo

| What | Where | Why |
|---|---|---|
| Model weights | `SHARED_MODEL_STORE` or `~/.cache/huggingface` | large and often shared between users |
| Python installs and uv cache | `~/.local/share/uv/`, `/tmp/uv_cache` on EIDF, otherwise `~/.cache/uv` | managed by uv |
| vLLM, PyTorch, Ray scratch | `~/.cache/vllm`, `/tmp/ray`, etc. | library caches and temporary files |
| Sandboxes | Docker containers or Kubernetes jobs | where agents run |
| Container images | image registry | task images |
| AppWorld data | location managed by the `appworld` CLI | licensed and not redistributed |

The virtual environments themselves stay inside the clone: `.venv`, `third_party/*/.venv`, and `setup/sft/.venv`.

## Command dependencies

| Step | Needs | Writes |
|---|---|---|
| `exp tasks -b B --split S` | upstream task source; AppWorld also needs its data and `APPWORLD_IMAGE` | `runs/tasks/B/S/` |
| `exp sft -b B --pool P` | `benchmarks/B/pools/P.jsonl` or `--data`, plus the base model | `runs/checkpoints/B/<model>/P/<run>/final/` |
| `exp serve -b B --pool P` | the matching checkpoint | running model server |
| `exp eval -b B --pool P --split S` | model server, task directory, sandbox backend | `runs/evals/B/<model>/P/<run>/<harness>/<pass>/` |
| `exp grpo -b B --rollout R` | `runs/checkpoints/B/<model>/3h/s0/final/` (Gemma: its `final-text-only/` export, made on first use) and `runs/tasks/B/train/` | `runs/rl/B/<model>/grpo-R/<run>/` |
| `exp serve` / `exp eval --rl R` | GRPO export, optionally selected with `--rl-step N` | evaluation job |
| `exp collect -b B` | teacher server and training tasks | `runs/collect/B/teacher-*/` |
| `exp collect -b B --rehearsal` | 3H SFT server and rehearsal tasks | `runs/collect/B/rehearsal-*/` |
| `exp build-pool -b B` | teacher traces, plus rehearsal traces where required | `runs/pools/B/` |
| `exp table E` | evaluation jobs; `--shipped` reads `results/rewards.csv` and `results/cells.csv` instead | table on stdout |
| `exp seed-tables`, `exp control-tables`, `exp failure-types`, `exp annotation-sample` | the evaluation jobs their spec in `configs/tables/` names | `results/tables/<command>/` (the sampled transcripts: `runs/annotation/sample.json`) |
| `exp annotation-tables` | `results/tables/failure-types/trials.csv` and `results/failure_annotation_labels.csv` | `results/tables/failure-annotation/annotation_tables.md` |

`collect` and `build-pool` are only needed if you want to rebuild the training datasets. Normal experiment runs use the shipped datasets.

## Experiments

Experiments can run in any order. Each script produced by

```bash
uv run exp plan <experiment>
```

contains everything that experiment needs: task generation, training, serving, evaluation, and result tables.

Outputs are shared between experiments. SFT checkpoints live at

```text
runs/checkpoints/<benchmark>/<model>/<pool>/
```

and GRPO runs at

```text
runs/rl/<benchmark>/<model>/grpo-<rollout>/
```

If a completed run with the same inputs already exists, `exp sft` and `exp grpo` reuse it. Successful runs record this in `final/exp_sft.json` or `exp_grpo.json`. Short test runs using options such as `--max-steps` or `--batch-size` are not reused. `--force` retrains.

Do not run two experiments at the same time if they may write the same checkpoint; the shared checkpoints are listed in [COMMANDS.md](COMMANDS.md#checkpoints-shared-between-experiments).

SFT, serving, and evaluation need one GPU. GRPO and the Qwen3-32B teacher need two.
