## Interface failures, automatic vs human

Annotator agreement (interface vs task knowledge): Fleiss κ = 0.61; all 3 annotators agree on 80% of the 99 trials each of them labelled (100 trials in total).

| | | Failed trials (% of all) | Interface failures, % of all trials (auto · human) | Interface failures, % of failed trials (auto · human) | Sampled |
|---|---|---|---|---|---|
| BFCL multi-turn (fc) | Task supervision only | 76.2 | 65.8 · 71.5 | 86 · 94 | 16 |
| BFCL multi-turn (fc) | Task supervision + rehearsal | 42.1 | 7.0 · 10.5 | 17 · 25 | 16 |
| Reasoning Gym (Terminus-2) | Task supervision only | 78.1 | 69.7 · 70.5 | 89 · 90 | 17 |
| Reasoning Gym (Terminus-2) | Task supervision + rehearsal | 55.1 | 38.2 · 38.9 | 69 · 71 | 17 |
| MedAgentBench (Terminus-2) | Task supervision only | 91.4 | 73.8 · 78.8 | 81 · 86 | 17 |
| MedAgentBench (Terminus-2) | Task supervision + rehearsal | 64.8 | 25.4 · 59.7 | 39 · 92 | 17 |

## Interface failure modes, % of failed trials (automatic · human)

Automatic percentages use every failed trial, human percentages the sampled ones. The remainder of each row is task-knowledge failures. The BFCL checker has no submission signal, so that automatic cell is n/a.

| | | Invalid calls | Incorrect submissions | Termination failures |
|---|---|---|---|---|
| BFCL multi-turn (fc) | Task supervision only | 85 · 88 | n/a · 4 | 2 · 2 |
| BFCL multi-turn (fc) | Task supervision + rehearsal | 4 · 15 | n/a · 0 | 12 · 10 |
| Reasoning Gym (Terminus-2) | Task supervision only | 74 · 69 | 6 · 14 | 9 · 8 |
| Reasoning Gym (Terminus-2) | Task supervision + rehearsal | 5 · 8 | 52 · 60 | 12 · 2 |
| MedAgentBench (Terminus-2) | Task supervision only | 3 · 4 | 76 · 82 | 2 · 0 |
| MedAgentBench (Terminus-2) | Task supervision + rehearsal | 38 · 41 | 1 · 41 | 1 · 8 |
