# Result data

Three files hold the results, all written by `exp export-results`:

- `rewards.csv`: the reward of every attempt, one row per task of an evaluation job. Every mean and interval in the repository can be recomputed from it.
- `cells.csv`: one row per training run of an arm on a harness, per metric.
- `runs.csv`: the index of runs. One row per training run of an arm, with the experiments that use the arm (`exp list experiments`) and their groups, the checkpoint folder, what a GRPO run was started from, and the number of evaluation jobs.

## `rewards.csv`

```text
root, job, task, trials, rewards
```

| column | meaning |
|---|---|
| `root` | `evals`, or `evals-cluster2` for jobs evaluated on a second cluster |
| `job` | the evaluation job: `<task set>/<model>/<setting>/<run>/<harness>/<pass>`, as described in [docs/PATHS.md](../docs/PATHS.md#how-a-run-is-addressed) |
| `task` | task id |
| `trials` | attempts found for the task, scored or not |
| `rewards` | the verifier reward of each scored attempt, separated by `;` |

Task transcripts are not included.

## `cells.csv`

```text
benchmark, model, tier, layer, arm, run, harness, metric, value,
ci_lo, ci_hi, plusminus, tasks, trials, bootstrap_unit, bootstrap_n, status, source, note
```

| column | meaning |
|---|---|
| `benchmark` | Benchmark name accepted by `--benchmark`: `medagentbench`, `reasoning_gym`, `appworld`, `bfcl_singleturn`, or `bfcl_multiturn` |
| `model` | Evaluated policy: a key from `configs/models.yaml` (`qwen3-4b`, `gemma-3-4b-it`, `qwen3-32b`) |
| `tier` | Reasoning Gym tier (`easy` or `hard`); empty for other benchmarks |
| `layer` | Sampling setup: `n4-T0.7`, `n4-T0.6` (4 attempts at that temperature), `n4-gencfg-default` (4 attempts using the released agent's generation config), `pass1-greedy` (1 greedy attempt), or `n4-T0.7-SUBSET61` (AppWorld, first 61 test tasks). `n1-...` is the same setup with one attempt per task |
| `arm` | Training setting. For SFT settings, names match `benchmarks/<benchmark>/pools/<arm>.jsonl` and the corresponding key in `configs/benchmarks/<benchmark>.yaml`; see below |
| `run` | Training run of the arm: `s0`, `s0b`, `s1`, ... (see [docs/PATHS.md](../docs/PATHS.md#run-names)). The base model and the teacher are filed under `s0`, and under `s0b` for a second evaluation |
| `harness` | Harness used for evaluation, including held-out harnesses; empty for set-level counts |
| `metric` | `mean` (mean verifier reward, percent), `pass4` (percent of tasks solved in at least one of 4 attempts), or solve-set counts: `solved`, `unique-<set>`, `union-<set>`, and `common-<set>`. `<set>` is `trained` for the three training harnesses or `all` when the held-out harness is included |
| `value` | Percentage for `mean` and `pass4`; task count for solve-set metrics |
| `ci_lo`, `ci_hi`, `plusminus` | Percentile bootstrap interval and half-width. Some rows only report `plusminus` |
| `tasks`, `trials` | Number of tasks and trials in the cell |
| `bootstrap_unit`, `bootstrap_n` | Resampling unit and number of units: tasks, or AppWorld's 56 scenarios |
| `status` | `reported` (reported result), `superseded` (replaced by another result; see `note`), or `extra` (computed but not reported) |
| `source` | `computed`: written by `exp export-results` from `rewards.csv` with the benchmark's statistic and task bootstrap. `published`: a value kept as published, for cells evaluated elsewhere or metrics over several cells |
| `note` | The evaluation jobs behind the row (`jobs=<root>:<job>`), or how the row was produced |

## `runs.csv`

```text
benchmark, model, setting, run, kind, groups, experiments, checkpoint, started_from, eval_jobs
```

| column | meaning |
|---|---|
| `benchmark`, `model` | As in `cells.csv` |
| `setting` | Training setting: the `arm` of `cells.csv` |
| `run` | Training run of the setting, as in `cells.csv` |
| `kind` | `sft`, `grpo`, or `untrained` for the base model and the teacher |
| `groups` | Groups of the experiments that use the setting (`main`, `seeds`, `controls`, `other`), separated by `;` |
| `experiments` | Experiments that use the setting, separated by `;`; empty when no experiment lists it |
| `checkpoint` | Checkpoint folder under `runs/checkpoints/` or `runs/rl/`; empty when the run has evaluations only |
| `started_from` | For a GRPO run, the checkpoint it started from, or `the base model` |
| `eval_jobs` | Number of evaluation jobs of the run |

## Setting names

The same naming scheme is used across benchmarks. Harness names come from `configs/benchmarks/<b>.yaml`.

| name | trained on |
|---|---|
| `base` | No training; the base model |
| `teacher` | The teacher model itself; `model` identifies which teacher |
| `<h>-only` | Teacher demonstrations from harness `h` |
| `3h` | Demonstrations from all three training harnesses |
| `<h1>+<h2>` | Demonstrations from two harnesses |
| `reh-<h>` | Rehearsal traces from harness `h` only |
| `<...>+reh-<h>` | The preceding dataset plus rehearsal traces from `h` |
| `<arm>~<variant>` | A variant of an arm. Everything before the first `~` is the arm; each `~` adds one variant, as below |
| `~32b`, `~gemma`, `~fail` | Rehearsal traces from the 32B teacher, Gemma, or failed attempts; otherwise rehearsal uses the 4B student |
| `~likefail`, `~noreason` | Successful rehearsal traces subsampled to the number of failed ones; rehearsal traces with their reasoning removed |
| `~rows<N>`, `~rows<N>mix`, `~rows<N>rep` | The dataset brought to N rows: shrunk keeping every rehearsal row, shrunk keeping its mixture, or repeated |
| `3h~n108`, `3h~n150` (`~cov`, `~sub2`, `~sub3`) | `3h` reduced to 108 or 150 rows using random sampling, coverage-maximising selection, or another random subsample |
| `3h~matched` | AppWorld `3h` reduced to 74 rows to match `code-only` |
| `grpo-<x>` | GRPO starting from the `3h` SFT checkpoint, with rollouts on `x`; `grpo-3h` uses all three training harnesses. `grpo-<x>-from-base` starts from the base model |

## Rebuilding and comparing

```bash
uv run exp table <experiment> --shipped
uv run exp table <experiment> --format cells --out mine.csv
```

The first command reads the shipped results: `rewards.csv`, and `cells.csv` for the cells it does not have. The second writes results from a local run in the same schema, so they can be compared directly with `cells.csv`.

## Reading the results

- Reward meaning differs by benchmark. MedAgentBench, BFCL, and AppWorld use binary verifier rewards. Reasoning Gym uses scores in `[0, 1]`, so its means include partial credit and `solved` means near-full credit.
- Means use completed trials. Cells with substantial infrastructure failures were rerun rather than reported.
- Confidence intervals use percentile bootstrap with `B=1000` and a fixed seed, resampling the unit named in `bootstrap_unit`. Rows of `cells.csv` use the benchmark's point statistic and a bootstrap over tasks, as `exp table` does by default. `exp table grpo-3h-vs-single` shows its `3h` and `grpo-3h` rows with the trial-pooled statistic, as in the paper, computed from `rewards.csv`.

## Tables over training runs

`tables/` holds the tables that combine training runs, as written by the commands in [docs/COMMANDS.md](../docs/COMMANDS.md#tables-over-training-runs-and-controls). `cells.csv` has one training run per row; these average several. With `--shipped`, `exp seed-tables` and `exp control-tables` rebuild them from `rewards.csv`.

| folder | written by | contents |
|---|---|---|
| `tables/seed-tables/` | `exp seed-tables` | the rehearsal 2×2 and the distillation ladders averaged over training runs, with each run's own value |
| `tables/control-tables/` | `exp control-tables` | size- and budget-matched arms, failed and non-target rehearsal, the clean demonstrator, per-template leakage, and GRPO |
| `tables/failure-types/` | `exp failure-types` | rule-based failure types: one row per failed trial (`trials.csv`) and per arm (`summary.csv`) |
| `tables/failure-annotation/` | `exp annotation-sample`, `exp annotation-tables` | the failed trials drawn for human annotation (`sample.csv`) and the rule-based types against the human labels |

Each table folder has the tables in Markdown and CSV and a `manifest.csv`: for every cell and run, the job folder it was read from, the trials found and scored, and a checksum of the rewards.

Runs are named `s0`, `s1`, `s2`, ...: `s0` is an arm's first training run, with the trainers' default seed, and `s1` and up are retrainings with `--seed 1` and up. `s0b` is a second training run at the default seed, made for arms whose `s0` run predates the controls it is compared with; for the base model, which is not trained, `s0` and `s0b` are its first and second evaluation. The `s0` values of a cell are the ones in `cells.csv`.

## Failure annotation

`failure_annotation_labels.csv` holds the human failure-type labels behind the automatic-versus-human comparison (`exp annotation-tables`): one row per annotator and failed trial, 100 trials by three annotators.

| column | meaning |
|---|---|
| `annotator` | `A1`, `A2`, `A3` |
| `benchmark`, `tier`, `harness`, `arm`, `job`, `trial`, `task` | the failed trial: `arm` is `task` (task supervision only) or `comb` (task supervision + rehearsal); `job` is its evaluation job, named by its folder below `runs/evals/` (`<task set>/<model>/<setting>/<run>/<harness>/<pass>`), and `trial` its folder inside it |
| `primary` | `interface` or `task`: whether the first decisive error was in operating the harness or in solving the task |
| `sub` | interface: `invalid`, `submission`, `termination`, `other`; task: `reasoning`, `approach`, `requirement` |
