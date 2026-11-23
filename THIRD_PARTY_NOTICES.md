# Third-party material

The MIT licence in `LICENSE` covers the code written for this project. The material below comes from other projects and stays under their licences.

| Where | From | Licence |
|---|---|---|
| `third_party/harbor/` | [harbor](https://github.com/harbor-framework/harbor), a fork (see `third_party/harbor/VENDORED.md`) | Apache-2.0, `third_party/harbor/LICENSE` |
| `training/rl/` | [SkyRL](https://github.com/NovaSky-AI/SkyRL): its harbor integration, `pyproject.toml` and `uv.lock`, modified | Apache-2.0 |
| `benchmarks/bfcl_singleturn/tasks/`, `benchmarks/bfcl_multiturn/tasks/` | [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard): question files, function docs, the simulated APIs and the evaluation code, with harbor's adapter | Apache-2.0 |
| `benchmarks/medagentbench/tasks/` | [MedAgentBench](https://github.com/stanfordmlgroup/MedAgentBench): task data and reference solutions, with harbor's adapter | MIT, Copyright (c) 2025 Stanford Machine Learning Group |
| `benchmarks/reasoning_gym/tasks/` | harbor's adapter for [Reasoning Gym](https://github.com/open-thought/reasoning-gym) | Apache-2.0 |
| `benchmarks/appworld/tasks/` | An adapter for [AppWorld](https://github.com/StonyBrookNLP/appworld). AppWorld's data is not included: install it with the `appworld` CLI | Apache-2.0 |
| `benchmarks/reasoning_gym/build/goose_system_prompt.json` | The system prompt of [goose](https://github.com/block/goose) | Apache-2.0, Copyright 2024 Block, Inc. |

The agents `mini-swe-agent` (MIT), `goose`, `aider` and Terminus-2 are installed at run time and are not included.

The trace files under `benchmarks/*/pools/` contain task text from the benchmark they were collected on, and model outputs from Qwen3-32B, Qwen3-4B and Gemma-3-4B-it.
