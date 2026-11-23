"""Render serve / SFT / eval / GRPO invocations from configs/.

Pure functions: nothing here runs anything. The CLI decides whether to print or execute.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from exp import backends
from exp.config import ROOT, Benchmark, model as model_cfg, model_source

RUNS = ROOT / "runs"
SERVED = "policy"  # served model name used by `exp serve`; any name works


def fmt(obj, ctx: dict):
    """Fill {placeholders} in strings, recursively, leaving unknown braces alone."""
    if isinstance(obj, str):
        out = obj
        for k, v in ctx.items():
            if v is not None:
                out = out.replace("{" + k + "}", str(v))
        return out
    if isinstance(obj, dict):
        return {k: fmt(v, ctx) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fmt(v, ctx) for v in obj]
    return obj


# --------------------------------------------------------------------------- paths
# Everything a trained model produces is filed under one address, one folder level per fact:
#
#   runs/checkpoints/<benchmark>/<model>/<setting>/<run>/              an SFT checkpoint
#   runs/rl/<benchmark>/<model>/<setting>/<run>/                       a GRPO run
#   runs/evals/<task set>/<model>/<setting>/<run>/<harness>/<pass>/    one evaluation job
#
# <setting> is the name the configs use (a pool, `grpo-<rollout>`, `base`, `teacher`). <run> says which
# training run of that setting. <pass> says how many attempts each task got and which repeat it is.
UNTRAINED = ("base", "teacher")


def run_id(seed: int | None = None, setting: str | None = None) -> str:
    """Which training run of a setting: `s0` at the trainers' default seed (42), `s<seed>` for a retraining
    with another seed, `none` for the settings that are not trained (the base model, the teacher)."""
    if setting in UNTRAINED:
        return "none"
    return "s0" if seed is None else f"s{seed}"


def pass_id(n_attempts: int, eval_pass: int | None = None) -> str:
    """How an evaluation job was run: `n<N>` for N attempts per task, `n<N>-p<P>` for the P-th repeat of
    that evaluation (P >= 2). Repeats of one cell are pooled per task by `exp table --pass`."""
    return f"n{n_attempts}" + (f"-p{eval_pass}" if eval_pass and eval_pass > 1 else "")


def task_set(b: "Benchmark", split: str = "test") -> str:
    """The folder of the tasks an evaluation ran on: the benchmark for its test split, the benchmark and tier
    where the test split has tiers (`reasoning_gym-easy`), the benchmark and split otherwise (`<benchmark>-train`)."""
    if split.startswith("test"):
        tier = b.tier(split)
        return f"{b.name}-{tier}" if tier else b.name
    return f"{b.name}-{split}"


def checkpoint_dir(bench: str, model: str, pool: str, seed: int | None = None) -> Path:
    return RUNS / "checkpoints" / bench / model / pool / run_id(seed)


def rl_name(b: "Benchmark", rollout: str, init: str | None = None, model: str | None = None) -> str:
    """The setting name of a GRPO run: `grpo-<rollout>`, with `-from-<init>` when it did not start from the
    benchmark's default init pool (`base` or another pool), and the serving protocol it was trained under when
    `model` is given and that protocol is named (`-explicit-think`, see `serving_protocol`)."""
    tag = "" if init in (None, b.rl.get("init_pool")) else f"-from-{init}"
    name = f"grpo-{rollout}{tag}"
    return with_protocol(name, b, model) if model is not None else name


def serving_protocol(b: "Benchmark", model: str, harness: str | None = None) -> str | None:
    """The name of the serving protocol the model is served under on this benchmark (`serve.protocol`, overridable
    per model and per harness), or None for the protocol the paper's results were served with. Everything served
    under a named protocol carries it in its name (the harness folder of an evaluation, the setting of a GRPO run),
    so results of two protocols never share a folder; SFT checkpoints are not served and keep theirs."""
    serve = b.cfg.get("serve") or {}
    pm = (serve.get("per_model") or {}).get(model) or {}
    tag = pm.get("protocol", serve.get("protocol"))
    if harness is not None:
        tag = ((pm.get("per_harness") or {}).get(harness) or {}).get("protocol", tag)
    return tag


def with_protocol(name: str, b: "Benchmark", model: str, harness: str | None = None) -> str:
    tag = serving_protocol(b, model, harness)
    return f"{name}-{tag}" if tag else name


def rl_dir(b: "Benchmark", model: str, rollout: str, seed: int | None = None, init: str | None = None) -> Path:
    return RUNS / "rl" / b.name / model / rl_name(b, rollout, init, model) / run_id(seed)


def tasks_root() -> Path:
    return RUNS / "tasks"


def evals_root() -> Path:
    return RUNS / "evals"


def collect_root() -> Path:
    """Trace-collection jobs (teacher and rehearsal traces), by benchmark and the job name the config gives them."""
    return RUNS / "collect"


def protocol_attempts(b: "Benchmark", setting: str) -> int:
    """Attempts per task under the reported protocol: the benchmark's, or the teacher protocol's for the teacher."""
    return int((teacher_protocol(b).get("n_attempts") if setting == "teacher" else None) or b.eval["n_attempts"])


def eval_dir(b: "Benchmark", model: str, setting: str, harness: str, split: str = "test", *, seed: int | None = None,
             n_attempts: int | None = None, eval_pass: int | None = None, root: Path | None = None) -> Path:
    """The folder of one evaluation job (see the address scheme above). `n_attempts` defaults to the
    reported protocol's."""
    n = int(n_attempts or protocol_attempts(b, setting))
    return (root or evals_root()) / task_set(b, split) / model / setting / run_id(seed, setting) / \
        with_protocol(harness, b, model, harness) / pass_id(n, eval_pass)


def job_id(job_dir: Path, root: Path | None = None) -> str:
    """How tables and result files name an evaluation job: its folder below the evaluations root."""
    try:
        return Path(job_dir).resolve().relative_to((root or evals_root()).resolve()).as_posix()
    except ValueError:
        return Path(job_dir).as_posix()


def grpo_init(b: Benchmark, model: str, init: str | None = None) -> Path:
    """The checkpoint GRPO starts from. By default the model's SFT checkpoint on the benchmark's
    `init_pool` (or, for a model whose `grpo_init` is `text-only` (Gemma 3), the text-only export
    beside it); `init` names another start: `base` for the untuned model, or another pool's checkpoint."""
    if init == "base":
        return Path(model_source(model))
    if init not in (None, b.rl["init_pool"]):
        return checkpoint_dir(b.name, model, init) / "final"
    final = checkpoint_dir(b.name, model, b.rl["init_pool"]) / "final"
    if model_cfg(model).get("grpo_init") == "text-only":
        return final.parent / "final-text-only"
    return final


def export_text_only_cmd(src: Path, out: Path) -> list[str]:
    """Run `training/export_text_only.py` in the SFT environment (the trainer's transformers)."""
    return ["uv", "run", "--project", str(ROOT / "setup" / "sft"), "--no-sync", "python",
            str(ROOT / "training" / "export_text_only.py"), "--src", str(src), "--out", str(out)]


def model_path(b: Benchmark, model: str, pool: str | None, rl: str | None = None, rl_step: str = "final",
               seed: int | None = None, rl_init: str | None = None) -> str:
    """What `exp serve` serves for a setting: the base model, an SFT checkpoint, or a GRPO export
    (the retraining with `seed`, when given; the run started from `rl_init` when not the default)."""
    if rl:
        d = rl_dir(b, model, rl, seed, rl_init) / "exports"
        if rl_step != "final":
            return str(d / f"global_step_{rl_step}" / "policy")
        steps = sorted((int(p.name.split("_")[-1]), p) for p in d.glob("global_step_*") if p.name.split("_")[-1].isdigit()) \
            if d.is_dir() else []
        # the paper evaluates the final export (step 12 on MAB, ~14 on RG) and step 5 (--rl-step 5)
        return str(steps[-1][1] / "policy") if steps else str(d / "global_step_<final>" / "policy")
    if pool in (None, "base"):
        return model_source(model)
    return str(checkpoint_dir(b.name, model, pool, seed) / "final")


# --------------------------------------------------------------------------- serve
def serve_settings(b: Benchmark, model: str, harness: str | None = None) -> dict:
    """The benchmark's serve settings, overridden by `serve.per_model.<model>` and, within that, by
    `per_harness.<harness>` (Gemma was served differently per harness on MedAgentBench)."""
    s = {k: v for k, v in b.cfg["serve"].items() if k != "per_model"}
    pm = dict((b.cfg["serve"].get("per_model") or {}).get(model) or {})
    ph = pm.pop("per_harness", None) or {}
    s.update(pm)
    if ph:
        if harness is None:
            raise SystemExit(f"{b.name}: {model} was served differently per harness ({', '.join(sorted(ph))} vs the rest); "
                             "pass --harness to serve it for one harness")
        s.update(ph.get(harness) or {})
    return s


# Per-harness overrides that one server can carry for several harnesses: they change how the server parses replies,
# not what the model is given, so a multi-harness GRPO run serves every rollout harness with the model's own setting.
PARSER_ONLY_OVERRIDES = {"reasoning_parser", "reasoning_parser_plugin", "protocol"}


def rollout_serve_override(b: Benchmark, model: str) -> dict:
    """The benchmark's `rl.rollout_serve.<model>`: serve settings for the model's GRPO rollouts that replace its
    evaluation serving (Gemma on MedAgentBench), or {}."""
    return ((b.rl or {}).get("rollout_serve") or {}).get(model) or {}


def rollout_serve_settings(b: Benchmark, model: str, harnesses: list[str]) -> dict:
    """The serve settings of a GRPO run's one inference server. A model with `rl.rollout_serve` settings gets
    those on top of its own, for every rollout harness. Otherwise a single rollout harness gets that harness's
    settings; several get the model's (per_harness overrides that only change reply parsing are left out, e.g.
    MedAgentBench Qwen3's t2 entry); any other per-harness difference cannot be served by one server and is refused."""
    pm = (b.cfg["serve"].get("per_model") or {}).get(model) or {}
    if rollout_serve_override(b, model):
        s = {k: v for k, v in b.cfg["serve"].items() if k != "per_model"}
        s.update({k: v for k, v in pm.items() if k != "per_harness"})
        s.update(rollout_serve_override(b, model))
        return s
    if len(harnesses) == 1:
        return serve_settings(b, model, harnesses[0])
    ph = pm.get("per_harness") or {}
    other = {h: sorted(set(o) - PARSER_ONLY_OVERRIDES) for h, o in ph.items() if h in harnesses and set(o) - PARSER_ONLY_OVERRIDES}
    if other:
        raise SystemExit(f"{b.name}: {model} is served differently per harness ({other}); one GRPO server cannot serve "
                         f"{', '.join(harnesses)} together")
    s = {k: v for k, v in b.cfg["serve"].items() if k != "per_model"}
    s.update({k: v for k, v in pm.items() if k != "per_harness"})
    return s


def _base_template(model: str) -> str:
    """The base model's own chat template, written to a file vLLM can load (`chat_template: base`)."""
    src = Path(model_source(model))
    tc = src / "tokenizer_config.json"
    if not tc.exists():
        raise SystemExit(f"chat_template: base needs the base model on disk ({src}); set SHARED_MODEL_STORE")
    out = RUNS / "serve" / f"{model}-base-template.jinja"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.load(open(tc))["chat_template"])
    return str(out)


def serve_cmd(b: Benchmark, model: str, path: str, port: int, served: str = SERVED,
              tensor_parallel_size: int | None = None, harness: str | None = None) -> list[str]:
    s = serve_settings(b, model, harness)
    m = model_cfg(model)
    tp = tensor_parallel_size or (m.get("serve") or {}).get("tensor_parallel_size")
    cmd = ["uv", "run", "--project", str(ROOT / "third_party" / "SkyRL"), "--extra", "fsdp", "--extra", "harbor", "--no-sync",
           "python", "-m", "vllm.entrypoints.openai.api_server",
           "--model", path, "--served-model-name", served, "--host", "0.0.0.0", "--port", str(port),
           "--gpu-memory-utilization", "0.9", "--max-model-len", str(s["max_model_len"])]
    if tp and int(tp) > 1:
        cmd += ["--tensor-parallel-size", str(tp)]
    if s.get("reasoning_parser_plugin"):
        cmd += ["--reasoning-parser-plugin", str(ROOT / s["reasoning_parser_plugin"])]
    if s.get("reasoning_parser"):
        cmd += ["--reasoning-parser", s["reasoning_parser"]]
    if s.get("tool_call_parser"):   # null: served plain, tool calls stay in the text
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", s["tool_call_parser"]]
    tmpl = s.get("chat_template") or (m.get("serve") or {}).get("chat_template")
    if tmpl == "base":
        cmd += ["--chat-template", _base_template(model)]
    elif tmpl:
        cmd += ["--chat-template", str(ROOT / tmpl)]
    return cmd


# --------------------------------------------------------------------------- SFT
def chat_template(b: Benchmark, model: str, pool: str) -> str | None:
    """The chat template an arm trained with: configs/models.yaml, overridden per model by the
    benchmark's sft.chat_template, then by the pool's chat_template (null = the model's own)."""
    tmpl = model_cfg(model).get("chat_template")
    for over in ((b.cfg.get("sft") or {}).get("chat_template") or {}, b.pool(pool).get("chat_template") or {}):
        if model in over:
            tmpl = over[model]
    return tmpl


def sft_cmd(b: Benchmark, model: str, pool: str, out: Path | None = None, wandb_run: str | None = None,
            data: str | None = None, max_steps: int | None = None, seed: int | None = None) -> list[str]:
    p = b.pool(pool)
    if not p.get("file"):
        raise SystemExit(f"{b.name}/{pool}: nothing to train (base model)")
    m = model_cfg(model)
    sft = b.cfg["sft"]
    a = sft["args"]
    out = out or checkpoint_dir(b.name, model, pool, seed)
    tmpl = chat_template(b, model, pool)
    cmd = ["python", "-m", "exp.sft_launch"]
    if not sft.get("cudnn_sdp", True):
        cmd.append("--disable-cudnn-sdp")
    cmd += ["--", "--data", str(data or (ROOT / p["file"])), "--model", model_source(model), "--out", str(out),
            "--epochs", str(a["epochs"]), "--max-len", str(a["max_len"])]
    if a.get("liger"):
        cmd.append("--liger")
    if a.get("optim"):
        cmd += ["--optim", a["optim"]]
    cmd += ["--chat-template", str(ROOT / tmpl) if tmpl else ""]
    if wandb_run:
        cmd += ["--wandb-run", wandb_run]
    if max_steps:  # small test run
        cmd += ["--max-steps", str(max_steps)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    return cmd


# --------------------------------------------------------------------------- eval
def eval_job(b: Benchmark, harness: str, *, job_dir: Path, split: str = "test", backend: str = "docker",
             api_base: str = "http://127.0.0.1:8000/v1", sandbox_api_base: str | None = None,
             served: str = SERVED, concurrency: int | None = None,
             n_attempts: int | None = None, temperature: float | None = None, placeholders: bool = False,
             tasks: Path | None = None, job_extra: dict | None = None, limit: int | None = None) -> dict:
    spec = b.agent_spec(harness)
    sandbox = backends.sandbox_endpoint(backend, api_base, sandbox_api_base)
    if spec.get("runs_in") == "sandbox" and not sandbox:
        if placeholders:
            sandbox = "${SANDBOX_API_BASE}"
        else:
            raise SystemExit(f"{b.name}/{harness} runs inside the sandbox: pass --sandbox-api-base "
                             f"(a URL the {backend} sandboxes can reach the model server at)")
    ctx = {"harness": harness, "api_base": api_base,
           "sandbox_api_base": sandbox, "sandbox_host": backends.host_of(sandbox), "served_model": served}
    agent: dict = {}
    if spec.get("name"):
        agent["name"] = spec["name"]
    if spec.get("import_path"):
        agent["import_path"] = spec["import_path"]
    agent["model_name"] = f"{spec['provider']}/{served}"
    kwargs = fmt(spec.get("kwargs") or {}, ctx)
    if temperature is not None and "temperature" in kwargs:
        kwargs["temperature"] = temperature
    if kwargs:
        agent["kwargs"] = kwargs
    env = fmt(spec.get("env") or {}, ctx)
    if env:
        agent["env"] = env
    sp = b.split(split)
    ds: dict = {"path": str((tasks or tasks_root()) / b.name / split)}
    names = b.split_tasks(split)
    if limit:  # small test run: `limit` tasks spread evenly over the split (sorted names group task families)
        if names is None:
            split_dir = Path(ds["path"])
            names = sorted(p.name for p in split_dir.iterdir() if p.is_dir()) if split_dir.is_dir() else []
        names = names[::max(1, len(names) // limit)][:limit]
    if names:
        ds["task_names"] = names
    job = {
        "job_name": job_dir.name,          # harbor writes the job to <jobs_dir>/<job_name>
        "jobs_dir": str(job_dir.parent),
        "n_attempts": int(n_attempts or b.eval["n_attempts"]),
        "n_concurrent_trials": int(concurrency or b.eval["concurrency"]),
        "environment": backends.environment(b, backend, placeholders=placeholders),
        "agents": [agent],
        "datasets": [ds],
    }
    job.update(spec.get("job") or {})
    job.update(job_extra or {})
    return job


def collect_job(b: Benchmark, entry: dict, *, backend: str = "docker", api_base: str = "http://127.0.0.1:8000/v1",
                sandbox_api_base: str | None = None, served: str = SERVED, jobs_dir: Path | None = None,
                placeholders: bool = False, tasks: Path | None = None, limit: int | None = None,
                n_attempts: int | None = None, concurrency: int | None = None) -> dict:
    """One trace-collection job: the eval job for (harness, split) with the collection protocol, written to
    runs/collect/<benchmark>/ under the config's `job` name (the pool builders in benchmarks/<b>/build/ find
    their inputs by that name).
    Used for teacher traces (`collect.jobs`) and rehearsal traces (`rehearsal`)."""
    hv = b.cfg["collect"]
    h = entry["harness"]
    split = entry.get("split", hv.get("split", "train"))
    job = eval_job(b, h, job_dir=(jobs_dir or collect_root() / b.name) / entry["job"], split=split, backend=backend,
                   api_base=api_base, sandbox_api_base=sandbox_api_base, served=served,
                   n_attempts=n_attempts or entry.get("n_attempts", hv.get("n_attempts")), placeholders=placeholders,
                   tasks=tasks, limit=limit, concurrency=concurrency)
    over = (hv.get("per_harness") or {}).get(h) or {}
    agent = job["agents"][0]
    kw = dict(agent.get("kwargs") or {})
    if kw or hv.get("kwargs"):
        for k, v in {**(hv.get("kwargs") or {}), **(over.get("kwargs") or {}), **(entry.get("kwargs") or {})}.items():
            if k == "temperature" and "temperature" not in kw:
                continue  # agents that take sampling from their own config (e.g. mini-swe) stay that way
            kw[k] = v
        for k in over.get("drop_kwargs") or []:
            kw.pop(k, None)
        if kw:
            agent["kwargs"] = kw
    for k in over.get("job_drop") or []:
        job.pop(k, None)
    return job


def teacher_protocol(b: Benchmark) -> dict:
    return dict(((b.cfg.get("teacher") or {}).get("protocol")) or {})


def dump_job(job: dict) -> str:
    return yaml.safe_dump(job, sort_keys=False, width=1000)


# --------------------------------------------------------------------------- GRPO
def rollout_harnesses(b: Benchmark, rollout: str) -> list[str]:
    """The harnesses a GRPO run rolls out on: every trained harness for `3h`, else the one named."""
    if rollout != "3h" and rollout not in b.harnesses("trained"):
        raise SystemExit(f"{b.name}: rollout harness must be 3h or one of {b.harnesses('trained')}")
    return b.harnesses("trained") if rollout == "3h" else [rollout]


def grpo_total_steps(b: Benchmark, rollout: str, init_pool: str | None = None, band: str | None = None,
                     batch_size: int | None = None) -> int:
    """The number of training steps a full GRPO run takes under the recipe: epochs x the band's
    rollout items (tasks x rollout harnesses) / the train batch, rounded down (MAB: 12 and 12; RG: 14 for 3H,
    12 for mini-swe-only)."""
    rl = b.rl
    recipe = yaml.safe_load(open(ROOT / "configs" / "rl.yaml"))
    harnesses = rollout_harnesses(b, rollout)
    band = band or (rl.get("bands") or {}).get(init_pool) or rl["band"]
    tasks = len(json.load(open(ROOT / band))["train"])
    rm = rl["rollouts_matched"]
    epochs = rm["multi_harness_epochs"] if len(harnesses) > 1 else rm["single_harness_epochs"]
    batch = batch_size or int(recipe["overrides"]["trainer.train_batch_size"])
    # The trainer's dataloaders drop the last partial batch of each epoch (drop_last=True)
    return epochs * (tasks * len(harnesses) // batch)


def grpo(b: Benchmark, model: str, rollout: str, *, init: str | None = None, backend: str = "docker",
         run_name: str | None = None, placeholders: bool = False, max_steps: int | None = None,
         out: str | None = None, batch_size: int | None = None, resume: bool = False,
         seed: int | None = None, init_pool: str | None = None, band: str | None = None,
         concurrency: int | None = None) -> tuple[list[str], dict, Path]:
    """Return (argv, env, cwd) for one GRPO run. rollout = '3h' or one harness name.
    `resume` continues from the latest checkpoint under <out>/ckpts instead of starting over.
    `seed` retrains the run with another trainer seed, in its own folder (see `run_id`).
    `init_pool` names the start of the run (`base` or a pool; default the benchmark's init_pool) and
    picks the run's folder. The learning band is `band`, else the benchmark's `rl.bands[init_pool]` when the
    config registers one, else its default `rl.band`: by default every start rolls out on the same tasks
    as the paper's runs, so a comparison across starts changes only the starting checkpoint."""
    rl = b.rl
    if not rl:
        raise SystemExit(f"{b.name}: RL is reported for medagentbench and reasoning_gym only")
    recipe = yaml.safe_load(open(ROOT / "configs" / "rl.yaml"))
    harnesses = rollout_harnesses(b, rollout)
    m = model_cfg(model)
    out = Path(out).resolve() if out else rl_dir(b, model, rollout, seed, init_pool)
    init = init or str(grpo_init(b, model, init_pool))
    band = band or (rl.get("bands") or {}).get(init_pool) or rl["band"]
    init = str(Path(init).resolve())   # the trainer runs from third_party/SkyRL; a relative path would not resolve
    if not placeholders and not Path(init).is_dir():
        raise SystemExit(f"GRPO init checkpoint not found: {init}")
    train = str(tasks_root() / b.name / "train")
    rm = rl["rollouts_matched"]
    ov: dict[str, str] = {
        "data.train_data": f"['{train}']",
        "trainer.policy.model.path": init,
        "generator.inference_engine.served_model_name": m["hf"].split("/")[-1],
        "harbor_trial_config.trials_dir": str(out / "trials"),
        "trainer.export_path": str(out / "exports"),
        "trainer.ckpt_path": str(out / "ckpts"),
        "trainer.log_path": str(out / "logs"),
        # the template the model trained with, unless rl.rollout_serve gives the model another for its rollouts
        "generator.inference_engine.engine_init_kwargs.chat_template":
            str(ROOT / (rollout_serve_override(b, model).get("chat_template") or m["chat_template"])),
        "trainer.epochs": str(rm["multi_harness_epochs"] if len(harnesses) > 1 else rm["single_harness_epochs"]),
        "trainer.run_name": run_name or f"{b.name}-{model}-" + out.name,
        # Harbor builds an agent by `name` before `import_path` (harbor/agents/factory.py), and the trainer's
        # default trial config names terminus-2: the name must be cleared, or every rollout runs Terminus-2
        # whatever `xh.harnesses` says. `exp grpo` checks the rollouts afterwards (`_audit_rollouts`).
        "harbor_trial_config.agent.name": "null",
        "harbor_trial_config.agent.import_path": rl["agent_import_path"],
        "xh.dataset_root": train,
        "xh.split_file": str(ROOT / band),
        "xh.harnesses": "[" + ",".join(f"'{h}'" for h in harnesses) + "]",
    }
    ov.update({k: str(v) for k, v in recipe["overrides"].items()})
    ov.update({k: str(v) for k, v in (rl.get("extra_overrides") or {}).items()})
    # The trainer's vLLM serves the rollouts with the parsers evaluation serving uses (the benchmark's `serve`
    # settings): the parity agent reads its action from `content`, which needs the reasoning parser, and fc
    # makes native tool calls, which need the tool-call parser. A model served differently per harness needs
    # `rl.rollout_serve` settings for a multi-harness rollout (Gemma on MedAgentBench).
    s = rollout_serve_settings(b, model, harnesses)
    if s.get("reasoning_parser_plugin"):
        # vLLM's own server imports a parser plugin in setup_server; SkyRL's inference server builds the app without
        # it (skyrl/backends/skyrl_train/inference_servers/vllm_server_actor.py), so the parser would not be found
        raise SystemExit(f"{b.name}: {model} is served with a reasoning-parser plugin ({s['reasoning_parser_plugin']}), "
                         "which the GRPO trainer's inference server does not load")
    if s.get("reasoning_parser"):
        ov["generator.inference_engine.engine_init_kwargs.reasoning_parser"] = s["reasoning_parser"]
    if s.get("tool_call_parser"):
        ov["generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice"] = "true"
        ov["generator.inference_engine.engine_init_kwargs.tool_call_parser"] = s["tool_call_parser"]
    if ov.get("trainer.logger") == "wandb" and not os.environ.get("WANDB_API_KEY"):
        print("note: WANDB_API_KEY not set; logging GRPO metrics to the console instead of wandb")
        ov["trainer.logger"] = "console"
    if max_steps:  # small test run: stop after max_steps updates and save then
        ov.update({"trainer.max_training_steps": str(max_steps), "trainer.ckpt_interval": str(max_steps)})
    if resume:
        ov["trainer.resume_mode"] = "latest"
    if seed is not None:
        ov["trainer.seed"] = str(seed)
    if concurrency:  # fewer rollouts (sandboxes) at a time than the recipe's 24: throughput only, same updates
        ov["generator.rate_limit.max_concurrency"] = str(concurrency)
    if batch_size:  # small test run: fewer prompts per update (the recipe's batch is 24)
        # Each task is rolled out once per harness, so the batch must split evenly between them
        if batch_size % len(harnesses):
            raise SystemExit(f"--batch-size {batch_size}: must be a multiple of the {len(harnesses)} rollout harnesses")
        ov.update({"trainer.train_batch_size": str(batch_size), "trainer.policy_mini_batch_size": str(batch_size)})
    for h, kw in (rl.get("harness_kwargs") or {}).items():
        if h not in harnesses:      # kwargs for harnesses that are not rolled out are inert; pass only the used ones
            continue
        for k, v in _flatten(kw).items():
            ov[f"xh.harness_kwargs.{h}.{k}"] = str(v)
    env_block = backends.environment(b, backend, placeholders=placeholders)
    ov["harbor_trial_config.environment.type"] = env_block["type"]
    for k, v in (env_block.get("kwargs") or {}).items():
        if k != "image":
            ov[f"harbor_trial_config.environment.kwargs.{k}"] = str(v)
    extra_pkg = (recipe.get("trainer_backend_packages") or {}).get(backend)
    argv = ["uv", "run", "--isolated"] + sum((["--extra", e] for e in recipe["uv_extras"]), []) + \
           (["--with", extra_pkg] if extra_pkg else []) + \
           ["-m", recipe["entrypoint"]] + [f"{k}={v}" for k, v in ov.items()]
    env = {"PYTHONPATH": str(ROOT), **recipe.get("env", {})}
    return argv, env, ROOT / "third_party" / "SkyRL"


def export_steps(ckpts: list[Path]) -> list[Path]:
    """The checkpoints to export, by the recipe's `export_steps` (step numbers, or `last`)."""
    recipe = yaml.safe_load(open(ROOT / "configs" / "rl.yaml"))
    ckpts = sorted(ckpts, key=lambda p: int(p.name.split("_")[-1]))
    wanted = []
    for step in recipe.get("export_steps") or ["last"]:
        if step == "last":
            wanted.append(ckpts[-1])
        else:
            wanted += [c for c in ckpts if c.name == f"global_step_{step}"]
    return sorted(set(wanted), key=ckpts.index)


def export_cmd(ckpt_policy: Path, out: Path) -> tuple[list[str], dict, Path]:
    """(argv, env, cwd) to write the Hugging Face export of one native GRPO checkpoint
    (`ckpts/global_step_N/policy`, one shard per trainer rank) with training/merge_fsdp2_to_hf.py.
    Runs on the CPU in the SkyRL project environment that training uses (`--no-sync`: nothing is
    installed); one process per shard. An `--isolated` environment, built afresh for each export,
    twice ended the export silently right after its install (exit 141, SIGPIPE)."""
    shards = len(list(ckpt_policy.glob("model_world_size_*_rank_*.pt")))
    if not shards:
        raise SystemExit(f"no checkpoint shards in {ckpt_policy}")
    argv = ["uv", "run", "--no-sync", "--extra", "fsdp", "--extra", "harbor", "torchrun", f"--nproc_per_node={shards}",
            str(ROOT / "training" / "merge_fsdp2_to_hf.py"), "--ckpt-policy-dir", str(ckpt_policy), "--out", str(out)]
    return argv, {"CUDA_VISIBLE_DEVICES": ""}, ROOT / "third_party" / "SkyRL"


def _flatten(d: dict, p: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{p}.{k}" if p else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out
