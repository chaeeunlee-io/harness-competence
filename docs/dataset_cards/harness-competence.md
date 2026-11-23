---
configs:
  - config_name: rehearsal-tasks
    data_files: "rehearsal/tasks/*.jsonl"
  - config_name: rehearsal-traces
    data_files: "rehearsal/traces/*.jsonl"
  - config_name: distillation-traces
    data_files: "distillation-traces/*.jsonl"
  - config_name: failure-annotation
    data_files: "failure-annotation/*.jsonl"
language:
  - en
license: cc-by-4.0
---

# Harness competence

Data from *Learning What vs. Learning How: Decomposing Task and Interface Competence in Agentic Training*.

Chaeeun Lee, Hossein A. Rahmani, Aryo Pradipta Gema, Kexin Xu, Emine Yilmaz, Pasquale Minervini, T. Ian Simpson,
Chen Jin, Amrutha Saseendran

[Paper](https://arxiv.org/abs/XXXX.XXXXX) · [Code](https://github.com/chaeeunlee-io/harness-competence)

LLM agents act through a harness: the interface through which a model interacts with its environment. The
harness shapes what the model sees, how it can act, and the constraints under which it operates. The paper
distinguishes task competence, an agent's knowledge of a task, from interface competence, its ability to act
through a given harness, and studies how training with one or multiple harnesses affects both. It introduces
*harness rehearsal*: training on traces that carry little task-specific information but still require correct
use of the target harness. This dataset contains the trajectories used in those experiments, along with the
failures analysed, organised into four parts:

| part | what it holds |
|---|---|
| `rehearsal-tasks` | rehearsal tasks: instructions that spell out the steps to take, so solving one only needs the interface |
| `rehearsal-traces` | the trajectories collected on the rehearsal tasks |
| `distillation-traces` | the teacher's successful trajectories on the training tasks, through several harnesses per benchmark |
| `failure-annotation` | failed evaluation trials, each labelled by three annotators as an interface failure or a task failure |

{{files}}

Benchmarks: MedAgentBench (`medagentbench`), Reasoning Gym (`reasoning_gym`), BFCL multi-turn (`bfcl_multiturn`).
Harnesses: Terminus-2 (`t2`), function calling (`fc`), and others per benchmark, named in the `harness` column.

## Loading

Each part is a configuration:

```python
from datasets import load_dataset

traces = load_dataset("chaeeunlee/harness-competence", "distillation-traces", split="train")
row = traces[0]
print(row["benchmark"], row["harness"], row["task"])
for message in row["messages"]:
    print(message["role"], message["content"][:80])
```

The other parts are `rehearsal-tasks`, `rehearsal-traces` and `failure-annotation`.

## `rehearsal-tasks`

One row per rehearsal task.

| column | meaning |
|---|---|
| `benchmark` | benchmark the task is built on |
| `split` | name of the rehearsal task set in the benchmark's config |
| `harnesses` | harnesses rehearsal trajectories were collected through on this task set |
| `task` | task id |
| `instruction` | the instruction the agent receives, in full |

## `rehearsal-traces`

One row per trajectory collected on a rehearsal task.

| column | meaning |
|---|---|
| `benchmark` | benchmark |
| `pool` | training set the row belongs to: `reh-<harness>`, with `~32b`, `~gemma` or `~fail` for the variants below |
| `harness` | harness the trajectory was collected through |
| `demonstrator` | model that produced it: the student's `3h` checkpoint, the Qwen3-32B teacher, or Gemma's `3h` checkpoint |
| `passed_verifier` | whether the task's verifier accepted the trajectory; false only for the `~fail` pools |
| `task` | task id, as in `rehearsal-tasks`. Four Reasoning Gym rows of `reh-t2` carry an id cut at 32 characters: two tasks share the cut id and the instruction, so the full id cannot be told |
| `messages` | the conversation, as a list of messages |

## `distillation-traces`

One row per trajectory of the Qwen3-32B teacher on a training task that passed the verifier. These are the task
demonstrations the student models were distilled on. Benchmarks: {{teacher_benchmarks}}.

| column | meaning |
|---|---|
| `benchmark` | benchmark |
| `harness` | harness the trajectory was collected through |
| `difficulty` | Reasoning Gym's difficulty setting, `easy` or `hard`; empty for the other benchmarks |
| `task` | task id |
| `messages` | the conversation, as a list of messages |

{{notes}}

## Messages

In `rehearsal-traces` and `distillation-traces`, `messages` is the conversation as a list of messages.
Each message has `role`, `content`, `tool_calls` and `tool_call_id`, all strings. `tool_calls` holds the
message's tool calls as JSON, and is empty for a message without any. Reasoning the model wrote is part of
`content`, as it was trained on.

The trajectories are the rows the models were trained on, as the demonstrator wrote them. A trajectory that
passed its verifier can still contain a turn that went wrong, such as a reply the harness could not read,
after which the harness asked again and the model recovered.

One training row is left out: it has no assistant turn.

## `failure-annotation`

Failed evaluation trials, each labelled by three annotators as an interface failure or a task failure. The
trials come from two arms per benchmark: task supervision only (`task`), and task supervision plus rehearsal
on the target harness (`comb`). 100 trials are labelled; 6 more are worked examples that carry no labels.

| column | meaning |
|---|---|
| `benchmark`, `difficulty`, `harness` | where the trial ran; `difficulty` is Reasoning Gym's setting, `easy` or `hard`, and empty for the other benchmarks |
| `arm`, `arm_label` | `task` or `comb`, and the arm in words |
| `job`, `trial`, `task` | the evaluation job (`<task set>/<model>/<setting>/<run>/<harness>/<pass>`), the trial and its task |
| `worked_example` | true for the six unlabelled examples |
| `timed_out` | whether the trial hit its time limit |
| `steps` | number of model turns |
| `reference` | the expected answer, or the checker's verdict |
| `verifier_output` | what the verifier printed |
| `transcript` | the trial as the annotators read it: a list of `{role, text}`. Long harness instructions and long outputs are shortened, and say so where they are |
| `passing_trial` | a trial that passed, shown for comparison: its `job`, `trial`, whether it is the `same_task`, and its `transcript` |
| `annotator_<n>_primary` | annotator n's label: `interface` or `task`; empty if not labelled |
| `annotator_<n>_sub` | the finer label. Interface: `invalid` (malformed action), `submission` (answer not delivered), `termination` (stopped or ran out without acting), `other`. Task: `reasoning`, `approach`, `requirement` |
| `automatic_primary` | the rule-based label of the same trial, computed from the trial's logs: `interface` or `task` |
| `automatic_sub` | the rule-based interface failure mode: `invalid`, `submission` or `termination`; empty for a task failure |

## Citation

To be added.

## Licence

CC BY 4.0. The task text comes from BFCL, MedAgentBench and Reasoning Gym and stays under the licences of those
benchmarks: Apache-2.0, MIT and Apache-2.0.
