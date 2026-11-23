"""exp command line. Run `uv run exp --help` or `uv run exp <command> --help`."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

from exp import config as C, render as R
from exp.backends import BACKENDS


# --------------------------------------------------------------------------- helpers
def _show(argv, env=None, cwd=None):
    pre = "".join(f"{k}={shlex.quote(str(v))} " for k, v in (env or {}).items())
    where = f"cd {shlex.quote(str(cwd))} && " if cwd else ""
    print(where + pre + " ".join(shlex.quote(str(a)) for a in argv), flush=True)


def _run(argv, env=None, cwd=None, dry=False) -> int:
    _show(argv, env, cwd)
    if dry:
        return 0
    full = {**os.environ, **{k: str(v) for k, v in (env or {}).items()}}
    return subprocess.call([str(a) for a in argv], env=full, cwd=cwd)


def _experiment_options(p):
    """Options shared by `exp plan` and `exp run`."""
    p.add_argument("experiment", help="see `exp list experiments`")
    p.add_argument("--backend", default="docker", choices=BACKENDS)
    p.add_argument("--sandbox-api-base")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--n-attempts", type=int,
                   help="attempts per task for every evaluation (default: each benchmark's protocol, as in the "
                        "paper); each evaluation is written to its own n<N> folder")
    p.add_argument("--pass", dest="eval_pass", type=int,
                   help="repeat the same evaluations as pass P (P >= 2), in n<N>-p<P> folders; "
                        "exp table --pass 1 --pass 2 pools the passes per task")
    p.add_argument("--seed", type=int, help="retrain every trained setting with this seed and evaluate those "
                                            "checkpoints (base and teacher rows are left out)")
    p.add_argument("--eval-only", action="store_true",
                   help="no training: evaluate the settings whose checkpoint is finished, note the rest; "
                        "run again as more finish (finished evaluation jobs resume instantly)")
    p.add_argument("--only", help="comma-separated settings of the experiment to run (pool names, base, rl:<rollout>); "
                                  "the rest are left out, so an experiment can be split across machines")
    p.add_argument("--ignore-lock", action="store_true",
                   help="start even if runs/plans/<plan>.lock says this plan is already running (a crashed run leaves it)")
    p.add_argument("--concurrency", type=int, help="concurrent trials per evaluation job (default: the benchmark's)")


def _setting(a, b):
    """The setting a command serves or evaluates: base, a pool, a GRPO run, or the teacher. Which training
    run of it (--seed) is a separate folder level, see `render.run_id`."""
    if getattr(a, "rl", None):
        return R.rl_name(b, a.rl, getattr(a, "rl_init", None), model=a.model)
    if getattr(a, "teacher", False):
        return "teacher"
    return a.pool or "base"


# --------------------------------------------------------------------------- commands
def _benchmarks_of(e: dict) -> str:
    """The benchmarks an experiment runs on, e.g. "MAB, RG (easy, hard)", from its settings."""
    splits: dict[str, list[str]] = {}
    for s in e["settings"]:
        b = C.benchmark(s["benchmark"])
        name = b.cfg.get("short", s["benchmark"])
        split = s.get("split", "test")
        splits.setdefault(name, [])
        if split not in splits[name]:
            splits[name].append(split)
    parts = []
    for name, sp in splits.items():
        if sp == ["test"]:
            parts.append(name)
        else:   # a benchmark with more than one test split: name them (test -> easy on Reasoning Gym)
            parts.append(f"{name} ({', '.join('easy' if x == 'test' else x.removeprefix('test-') for x in sp)})")
    return ", ".join(parts)


def cmd_list(a):
    if a.what == "benchmarks":
        for n in C.benchmark_names():
            b = C.benchmark(n)
            print(f"{n:16s} {b.cfg['display']:18s} trained={b.harnesses('trained')} heldout={b.harnesses('heldout')} splits={list(b.cfg['splits'])}")
    elif a.what == "pools":
        for n in ([a.benchmark] if a.benchmark else C.benchmark_names()):
            b = C.benchmark(n)
            for pn, p in b.pools().items():
                print(f"{n:16s} {pn:22s} {p.get('paper',''):30s} {p.get('file') or '(base model)'}")
    elif a.what == "models":
        for k, m in C.models().items():
            print(f"{k:16s} {m['hf']:24s} {m.get('role','')}")
    elif a.what == "experiments":
        rows = [(k, e.get("group", ""), _benchmarks_of(e), e["title"]) for k, e in C.experiments().items()
                if not a.group or e.get("group") == a.group]
        width = max((len(b) for _, _, b, _ in rows), default=0)
        for k, group, b, title in rows:
            print(f"{k:32s} {group:9s} {b:{width}s}  {title}")
    return 0


def cmd_serve(a):
    b = C.benchmark(a.benchmark)
    model = "qwen3-32b" if a.teacher else a.model
    path = a.checkpoint or R.model_path(b, model, None if a.teacher else a.pool, a.rl, a.rl_step, seed=a.seed,
                                        rl_init=getattr(a, "rl_init", None))
    return _run(R.serve_cmd(b, model, path, a.port, a.served_model, a.tensor_parallel_size, a.harness), dry=a.dry_run)


def cmd_wait(a):
    """Wait until the model server answers. With --pid, give up as soon as that process exits
    (a server that crashed at startup), instead of waiting out the whole timeout, and check that
    the server answering on the port is that process (or one of its children): a server left
    over from an earlier run would otherwise answer in its place, and the evaluation would be
    run against the wrong model without any error."""
    url = a.api_base.rstrip("/") + "/models"
    t0 = time.time()
    while time.time() - t0 < a.timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    served = ", ".join(m.get("root") or m.get("id", "?") for m in json.load(r).get("data", []))
                    break
        except Exception:
            pass
        if a.pid and not _process_running(a.pid):
            print(f"server process {a.pid} exited before answering: see the server's log", file=sys.stderr)
            return 1
        time.sleep(5)
    else:
        print(f"timed out waiting for {url}", file=sys.stderr)
        return 1
    if a.pid:
        listener = _port_listener(_api_port(a.api_base))
        if listener is not None and listener != a.pid and listener not in _descendants(a.pid):
            print(f"another server answers on {url}: process {listener} (serving {served}), not server {a.pid}. "
                  f"Stop it first: exp stop --port {_api_port(a.api_base)}", file=sys.stderr)
            return 1
    print(f"server up: {url} (serving {served})")
    return 0


def _api_port(api_base: str) -> int:
    return urllib.parse.urlsplit(api_base).port or 80


def _port_listener(port: int) -> int | None:
    """The process listening on a local TCP port, found through /proc (Linux; None if not found
    or not readable). Used to tell the server we started from one left over on the same port."""
    inodes = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            f = line.split()
            if len(f) > 9 and f[3] == "0A" and int(f[1].rsplit(":", 1)[1], 16) == port:   # 0A = LISTEN
                inodes.add(f[9])
    if not inodes:
        return None
    targets = {f"socket:[{i}]" for i in inodes}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for fd in (proc / "fd").iterdir():
                if os.readlink(fd) in targets:
                    return int(proc.name)
        except OSError:
            continue
    return None


def _process_running(pid: int) -> bool:
    """True while process `pid` is running."""
    try:
        os.kill(pid, 0)  # signal 0 sends nothing; it only checks that the process exists
    except ProcessLookupError:
        return False
    except PermissionError:  # it exists but belongs to another user
        return True
    # On Linux, a process that has exited but whose parent has not collected it yet still
    # exists as a "zombie" (state Z). Count that as exited.
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return True
    return state != "Z"


def _descendants(pid: int) -> list[int]:
    """Process ids below `pid`, deepest last (so parents can be signalled after children)."""
    out, queue = [], [pid]
    while queue:
        p = queue.pop(0)
        try:
            kids = [int(x) for x in subprocess.run(["pgrep", "-P", str(p)], capture_output=True, text=True).stdout.split()]
        except (OSError, ValueError):
            kids = []
        out += kids
        queue += kids
    return out


def _stop_server(pid: int) -> None:
    """SIGTERM a server started as `setsid … & SERVER=$!`: its whole process group. A server started
    without setsid (e.g. `… &` inside a pipeline, where the shell gives it no group of its own) is
    not a group leader; then its process tree is signalled instead, so vLLM's engine processes do
    not survive as orphans holding the GPU. A process found by port (`exp stop --port`) may sit deep
    in its group: the group is signalled when it is not our own, else the process tree."""
    try:
        pgid = os.getpgid(pid)
        if pgid == pid or pgid not in (os.getpgid(0), os.getpgid(os.getppid())):
            os.killpg(pgid, signal.SIGTERM)
            return
    except ProcessLookupError:
        return  # already gone
    for p in list(reversed(_descendants(pid))) + [pid]:
        try:
            os.kill(p, signal.SIGTERM)
        except ProcessLookupError:
            pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _group_alive(pgid: int) -> bool:
    """Whether any process of the group is still running."""
    return subprocess.call(["pgrep", "-g", str(pgid)], stdout=subprocess.DEVNULL) == 0


def cmd_stop(a):
    """Stop a model server and wait until its GPU memory is released.

    Start the server in its own process group, so it can be stopped as a whole:
        setsid uv run exp serve ... --port 8000 & SERVER=$!
        uv run exp stop --pid $SERVER --port 8000
    This sends SIGTERM to that group (or, for a server started without setsid, to its process
    tree). --port names the server by the port it listens on, which also finds a server whose
    recorded pid is wrong (in an interactive shell `setsid … &` forks, so $! is a parent that
    has already exited) or a server left over from an earlier run. Then it waits (up to 3 minutes)
    for that server's processes to end; vLLM's engine processes (VLLM::EngineCore) can outlive
    their parent and keep holding GPU memory, so the ones still in the server's process group or
    tree are then stopped too. Only the named server is ever touched: another server on the same
    machine (the other GPU of a pod) keeps running, and its memory does not fail this command.
    Without --pid and --port there is no server to name: the command waits for the GPUs (those in
    CUDA_VISIBLE_DEVICES, when set) to be free and stops every leftover engine process on the
    machine, which assumes the machine's GPUs are yours.
    """
    targets = [a.pid] if a.pid else []
    if a.port:
        listener = _port_listener(a.port)
        if listener is not None and listener != a.pid and listener not in (_descendants(a.pid) if a.pid else []):
            print(f"stopping the server listening on port {a.port} (process {listener})")
            targets.append(listener)
    own = {os.getpgid(0), os.getpgid(os.getppid())}
    groups, members = set(), set()      # process groups of the server; its processes when it shares our group
    for t in targets:
        try:
            pgid = os.getpgid(t)
        except ProcessLookupError:
            continue
        if pgid in own:
            members.update(_descendants(t) + [t])
        else:
            groups.add(pgid)
    for t in targets:
        _stop_server(t)

    if a.pid or a.port:
        def running():
            return any(_group_alive(g) for g in groups) or any(_alive(p) for p in members)
        for _ in range(60):
            if not running():
                break
            time.sleep(3)
        else:
            print("server processes still running: stopping its leftover vLLM engine processes")
            for sig in ("-TERM", "-KILL"):
                for g in groups:
                    subprocess.call(["pkill", sig, "-g", str(g), "-f", "VLLM::EngineCore"])
                for p in members:
                    if _alive(p):
                        os.kill(p, signal.SIGTERM if sig == "-TERM" else signal.SIGKILL)
                time.sleep(10 if sig == "-TERM" else 5)
        if running():
            print("server processes could not be stopped", file=sys.stderr)
            return 1
        used = _busiest_gpu_mb()
        if used >= 1000:
            print(f"server stopped; GPU memory still in use by other processes ({used} MiB on the busiest GPU)")
        else:
            print("server stopped; GPUs free")
        return 0

    for _ in range(60):
        if _busiest_gpu_mb() < 1000:
            break
        time.sleep(3)
    else:
        print("GPU memory still in use: stopping every leftover vLLM engine process on this machine")
        subprocess.call(["pkill", "-TERM", "-f", "VLLM::EngineCore"])
        time.sleep(10)
        subprocess.call(["pkill", "-KILL", "-f", "VLLM::EngineCore"])
        time.sleep(5)
    used = _busiest_gpu_mb()
    if used >= 1000:
        print(f"GPU memory still in use ({used} MiB on the busiest GPU)", file=sys.stderr)
        return 1
    print("server stopped; GPUs free")
    return 0


def _busiest_gpu_mb() -> int:
    """Memory in use on the busiest GPU, in MiB (0 if there is no nvidia-smi). Only the GPUs in
    CUDA_VISIBLE_DEVICES count when it is set, so two experiments pinned to different GPUs of one
    pod do not see each other's memory."""
    cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        cmd.append("--id=" + os.environ["CUDA_VISIBLE_DEVICES"])
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return 0
    return max((int(x) for x in out.split()), default=0)


# Trial errors that are outcomes of the run, kept on a resume. Everything else (the model server
# not answering, a sandbox that could not be created, ...) is infrastructure and is run again.
OUTCOME_ERRORS = {"AgentTimeoutError"}


def _overlaps(r: dict, window: tuple[str, str] | None) -> bool:
    """Whether a trial (its result.json) ran during the window [from, to) of UTC ISO timestamps."""
    if not window:
        return False
    started, finished = r.get("started_at"), r.get("finished_at")
    return bool(started and finished and started < window[1] and finished > window[0])


def cmd_retry_errors(a):
    """Harbor resumes a job folder by re-running only the trials without a result.json. A trial
    that errored on infrastructure (e.g. `Cannot connect to host` while the model server was
    still loading, or a sandbox pod that disappeared) has one, so it would stay errored and
    unscored forever. This removes such trials so the next `exp eval` of the job runs them again.
    Agent timeouts are outcomes of the run (the paper's runs have them too) and are kept.

    --down FROM TO also removes every trial that ran while the model server was down (started
    before TO and finished after FROM, UTC as in result.json): an agent that cannot reach the
    server often ends its trial as an ordinary failure with reward 0 and no exception, which
    nothing else can tell apart from a real failure."""
    window = tuple(a.down) if a.down else None
    removed = 0
    for job in a.job_dir:
        job = Path(job)
        for trial in sorted(p for p in job.iterdir() if p.is_dir()):
            rj = trial / "result.json"
            try:
                r = json.load(open(rj))
            except (OSError, ValueError):
                continue
            exc = r.get("exception_info")
            if exc and exc.get("exception_type") not in OUTCOME_ERRORS:
                msg = ((exc.get("exception_message") or "").splitlines() or [""])[0][:100]
                why = f"{exc.get('exception_type')}: {msg}"
            elif _overlaps(r, window):
                why = f"ran while the model server was down ({r['started_at'][11:19]}–{r['finished_at'][11:19]} UTC)"
            else:
                continue
            print(f"{'would remove' if a.dry_run else 'removed'} {C.rel(trial)}: {why}")
            if not a.dry_run:
                shutil.rmtree(trial)
            removed += 1
    print(f"{removed} trial(s) " + ("to re-run" if a.dry_run else
          "removed; re-run them with the same exp eval command, including the same --concurrency "
          "(Harbor resumes a job only under the config it was started with)"))
    return 0


# --- Reusing finished training ----------------------------------------------------------------
# Experiments share checkpoints (the same benchmark, model and pool), so `exp sft` and `exp grpo`
# skip a run that has already finished in full. Each successful run writes a record of how it was
# trained next to its output; only a full-scale run (no --max-steps / --batch-size) with the same
# inputs is reused. --force always retrains.

SFT_RECORD = "exp_sft.json"     # in <checkpoint>/final/
GRPO_RECORD = "exp_grpo.json"   # in the GRPO output folder
RECORD_PATHS = ("data", "init")  # stored relative to the repo root (config.repo_path)


def _write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")


def _reusable(record_path: Path, wanted: dict) -> tuple[bool, str]:
    """Whether a finished run can be reused: its record says it ran in full with these inputs."""
    if not record_path.exists():
        return False, "no record of a finished run"
    record = json.loads(record_path.read_text())
    if record.get("max_steps") or record.get("batch_size"):
        return False, "it was a short test run"
    for key, value in wanted.items():
        recorded = record.get(key)
        if key in RECORD_PATHS and recorded:
            recorded = C.repo_path(recorded)
        if recorded != value:
            return False, f"it was trained with a different {key}"
    return True, ""


def cmd_sft(a):
    b = C.benchmark(a.benchmark)
    out = Path(a.out) if a.out else R.checkpoint_dir(b.name, a.model, a.pool, a.seed)
    final = out / "final"
    pool_file = b.pool(a.pool).get("file")
    if "derived" in b.pool(a.pool) and not a.data and not a.dry_run:
        from exp import pools
        pools.build_derived(b, a.pool)   # written once, from the pools it is declared from
    data = C.repo_path(Path(a.data).resolve()) if a.data else (C.repo_path(C.ROOT / pool_file) if pool_file else None)
    wanted = {"model": a.model, "data": data, "seed": a.seed}
    complete = (final / "config.json").exists() and any(final.glob("*.safetensors"))
    if complete and not (a.force or a.max_steps or a.dry_run):
        record = final / SFT_RECORD
        if not record.exists():
            # Trained before records existed: reuse it, but say so
            print(f"{C.rel(final)} exists (no training record): reusing it; --force retrains it")
            return 0
        ok, why = _reusable(record, wanted)
        if ok:
            print(f"{C.rel(final)} exists: skipping training (--force retrains it)")
            return 0
        print(f"{C.rel(final)} exists but {why}: training again")
    argv = R.sft_cmd(b, a.model, a.pool, Path(a.out) if a.out else None, a.wandb_run, data=a.data, max_steps=a.max_steps,
                     seed=a.seed)
    argv = ["uv", "run", "--project", str(C.ROOT / "setup" / "sft"), "--no-sync"] + argv
    # wandb (only with --wandb-run) keeps its local files in runs/wandb/, not the repo root
    env = {"PYTHONPATH": str(C.ROOT), "WANDB_DIR": str(R.RUNS)}
    # SFT trains on one GPU, as the paper did. With several visible, the trainer would spread the
    # model over all of them, and the fused loss kernel fails on tensors from different GPUs.
    # Set CUDA_VISIBLE_DEVICES yourself to pick another GPU.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    rc = _run(argv, env=env, cwd=C.ROOT, dry=a.dry_run)
    if rc == 0 and not a.dry_run:
        _write_record(final / SFT_RECORD, {**wanted, "max_steps": a.max_steps,
                                           "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
        if not a.keep_checkpoints:
            _drop_intermediate_checkpoints(out, final)
    return rc


def _drop_intermediate_checkpoints(out: Path, final: Path) -> None:
    """Remove the trainer's intermediate checkpoints (out/checkpoint-<N>, model plus optimizer state,
    ~27 GB each for a 4B model) once the run has finished: serving and evaluation use final/, and
    the intermediates only serve to resume an interrupted run. Kept with --keep-checkpoints."""
    if not ((final / "config.json").exists() and any(final.glob("*.safetensors")) and (final / SFT_RECORD).exists()):
        return
    for ckpt in sorted(p for p in out.glob("checkpoint-*") if p.is_dir() and p.name.split("-", 1)[1].isdigit()):
        shutil.rmtree(ckpt)
        print(f"removed intermediate checkpoint {C.rel(ckpt)} (final model in {C.rel(final)})")


def cmd_eval(a):
    b = C.benchmark(a.benchmark)
    model = "qwen3-32b" if a.teacher else a.model
    setting = _setting(a, b)
    proto = R.teacher_protocol(b) if a.teacher else {}
    run = R.run_id(getattr(a, "seed", None), setting)
    rc = 0
    for h in b.harnesses(a.harness):
        job_dir = Path(a.job_dir) if a.job_dir else R.eval_dir(
            b, model, setting, h, a.split, seed=getattr(a, "seed", None), n_attempts=a.n_attempts or proto.get("n_attempts"), eval_pass=a.eval_pass,
            root=Path(a.evals_dir) if a.evals_dir else None)
        job = R.eval_job(b, h, job_dir=job_dir, split=a.split, backend=a.backend, api_base=a.api_base,
                         sandbox_api_base=a.sandbox_api_base, served=a.served_model, concurrency=a.concurrency,
                         n_attempts=a.n_attempts or proto.get("n_attempts"), temperature=proto.get("temperature"),
                         job_extra=proto.get("job"), placeholders=a.dry_run,
                         tasks=Path(a.tasks_dir) if a.tasks_dir else None, limit=a.limit)
        path = _write_job_config(job_dir, job)
        print(f"# {b.name} / {setting} / {run} / {h} / {a.split}  ->  {C.rel(path)}")
        done = _job_complete(job_dir)
        if done and not a.dry_run:
            print(f"# {C.rel(job_dir)}: finished, all {done} trials present; not re-run "
                  "(remove the folder, or its errored trials with exp retry-errors, to evaluate again)")
            continue
        argv = ["uv", "run", "--project", str(C.ROOT / "third_party" / "harbor"), "--no-sync", "harbor", "run", "-c", str(path), "-y"]
        env = {"PYTHONPATH": str(C.ROOT), "OPENAI_API_BASE": a.api_base, "OPENAI_API_KEY": "EMPTY"}
        rc |= _run(argv, env=env, cwd=C.ROOT, dry=a.dry_run)
    return rc


def _write_job_config(job_dir: Path, job: dict) -> Path:
    """The Harbor config of a job, written next to the job's folder as <job>.yaml."""
    path = job_dir.parent / f"{job_dir.name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(R.dump_job(job))
    return path


def _job_complete(job_dir: Path) -> int:
    """The number of trials of a finished Harbor job whose trial folders are all still present (0 if
    the job is unfinished, missing, or some trials were removed for a re-run). A finished job is
    skipped by `exp eval`: Harbor would otherwise refuse it whenever the job config differs from the
    one it was started with (e.g. another --concurrency), and a resume of a complete job does nothing."""
    try:
        r = json.load(open(job_dir / "result.json"))
    except (OSError, ValueError):
        return 0
    if not r.get("finished_at"):
        return 0
    n = r.get("n_total_trials") or 0
    present = sum(1 for p in job_dir.iterdir() if p.is_dir() and (p / "result.json").is_file())
    return n if n and present >= n else 0


def cmd_oracle(a):
    """Run harbor's oracle agent (each task's reference solution) on a few tasks: checks the
    task directories, the sandbox backend and the verifier without any model or GPU."""
    b = C.benchmark(a.benchmark)
    tasks = Path(a.tasks_dir) if a.tasks_dir else R.tasks_root()
    split_dir = tasks / b.name / a.split
    names = b.split_tasks(a.split) or (sorted(p.name for p in split_dir.iterdir() if p.is_dir()) if split_dir.is_dir() else [])
    job_dir = R.RUNS / "oracle" / f"{b.name}-{a.split}"
    job = {"job_name": job_dir.name, "jobs_dir": str(job_dir.parent), "n_attempts": 1,
           "n_concurrent_trials": a.n, "environment": __import__("exp.backends", fromlist=["x"]).environment(b, a.backend, placeholders=a.dry_run),
           "agents": [{"name": "oracle"}], "datasets": [{"path": str(split_dir), "task_names": names[: a.n]}]}
    path = _write_job_config(job_dir, job)
    print(f"# oracle on {min(a.n, len(names))} {b.name}/{a.split} tasks -> {C.rel(path)}; every trial should score 1")
    argv = ["uv", "run", "--project", str(C.ROOT / "third_party" / "harbor"), "--no-sync", "harbor", "run", "-c", str(path), "-y"]
    return _run(argv, env={"PYTHONPATH": str(C.ROOT)}, cwd=C.ROOT, dry=a.dry_run)


def cmd_collect(a):
    b = C.benchmark(a.benchmark)
    hv = b.cfg.get("collect")
    if not hv:
        raise SystemExit(f"{b.name}: no trace collection configured")
    entries = (b.cfg.get("rehearsal") or []) if a.rehearsal else hv["jobs"]
    if a.rehearsal and not entries:
        raise SystemExit(f"{b.name}: no rehearsal trace collection configured")
    rc = 0
    for e in entries:
        if a.harness and e["harness"] != a.harness:
            continue
        if a.job and e["job"] != a.job:
            continue
        if e.get("demonstrator"):
            print(f"# {e['job']}: serve {e['demonstrator']} first; tasks: exp tasks -b {b.name} --split {e['split']}")
        job = R.collect_job(b, e, backend=a.backend, api_base=a.api_base, sandbox_api_base=a.sandbox_api_base,
                            served=a.served_model, jobs_dir=Path(a.jobs_dir) if a.jobs_dir else None,
                            placeholders=a.dry_run, tasks=Path(a.tasks_dir) if a.tasks_dir else None,
                            limit=a.limit, n_attempts=a.n_attempts, concurrency=a.concurrency)
        path = _write_job_config(Path(job["jobs_dir"]) / job["job_name"], job)
        print(f"# collect {b.name} / {e['harness']} / {e.get('split', hv.get('split', 'train'))} as {e['job']} -> {C.rel(path)}")
        argv = ["uv", "run", "--project", str(C.ROOT / "third_party" / "harbor"), "--no-sync", "harbor", "run", "-c", str(path), "-y"]
        rc |= _run(argv, env={"PYTHONPATH": str(C.ROOT), "OPENAI_API_BASE": a.api_base, "OPENAI_API_KEY": "EMPTY"},
                   cwd=C.ROOT, dry=a.dry_run)
    return rc


def cmd_build_pool(a):
    from exp import pools
    return pools.main(a)


def cmd_grpo(a):
    b = C.benchmark(a.benchmark)
    out = Path(a.out).resolve() if a.out else R.rl_dir(b, a.model, a.rollout, a.seed, a.rl_init)
    init = Path(a.init).resolve() if a.init else R.grpo_init(b, a.model, a.rl_init)
    start = "base" if a.rl_init == "base" and not a.init else C.repo_path(init)
    wanted = {"model": a.model, "rollout": a.rollout, "init": start, "seed": a.seed}
    if a.export_only:
        rc = 0 if a.dry_run else _audit_rollouts(out, R.rollout_harnesses(b, a.rollout), b.rl["rollout_agents"])
        if rc == 0:
            rc = _export_grpo(out, dry=a.dry_run)
        # A run whose training finished but whose export failed has no record yet; record it once its
        # last checkpoint is the recipe's final step, so it is reused and evaluated like any other.
        if rc == 0 and not a.dry_run and not (out / GRPO_RECORD).exists():
            last = int((out / "ckpts" / "latest_ckpt_global_step.txt").read_text().strip())
            total = R.grpo_total_steps(b, a.rollout, a.rl_init, a.band, a.batch_size)
            if last == total:
                _write_record(out / GRPO_RECORD, {**wanted, "max_steps": None, "batch_size": a.batch_size,
                                                  "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                  "recorded_by": "export-only"})
                print(f"{C.rel(out)}: training finished ({last}/{total} steps); recorded")
            else:
                print(f"{C.rel(out)}: last checkpoint is step {last} of {total}; not recorded as finished")
        return rc
    if not (a.force or a.max_steps or a.batch_size or a.dry_run or a.resume):
        ok, _ = _reusable(out / GRPO_RECORD, wanted)
        if ok:
            print(f"{C.rel(out)} has a finished GRPO run: skipping training (--force retrains it)")
            return 0
    sft_final = R.checkpoint_dir(b.name, a.model, b.rl["init_pool"]) / "final"
    if init != sft_final and not a.init and not a.rl_init and not (init / "config.json").exists():
        # This model's GRPO starts from a text-only export of its SFT checkpoint (see
        # training/export_text_only.py); make it the first time it is needed.
        if not a.dry_run and not (sft_final / "config.json").exists():
            raise SystemExit(f"GRPO init checkpoint not found: {sft_final}")
        rc = _run(R.export_text_only_cmd(sft_final, init), cwd=C.ROOT, dry=a.dry_run)
        if rc:
            return rc
    allowed = yaml.safe_load(open(C.ROOT / "configs" / "rl.yaml")).get("trainer_backends") or []
    if allowed and a.backend not in allowed:
        raise SystemExit(f"exp grpo: the trainer cannot roll out on --backend {a.backend} (configs/rl.yaml trainer_backends: "
                         f"{', '.join(allowed)}); every rollout would fail and train on zero rewards")
    argv, env, cwd = R.grpo(b, a.model, a.rollout, init=str(init), backend=a.backend, run_name=a.run_name,
                            placeholders=a.dry_run, max_steps=a.max_steps, out=a.out, batch_size=a.batch_size,
                            resume=a.resume, seed=a.seed, init_pool=a.rl_init, band=a.band,
                            concurrency=a.concurrency)
    _check_agent_override(argv)
    rc = _run(argv, env=env, cwd=cwd, dry=a.dry_run)
    if rc == 0 and not a.dry_run:
        rc = _audit_rollouts(out, R.rollout_harnesses(b, a.rollout), b.rl["rollout_agents"])
    if rc == 0 and not a.dry_run:
        rc = _export_grpo(out)
    if rc == 0 and not a.dry_run:
        _write_record(out / GRPO_RECORD, {**wanted, "max_steps": a.max_steps, "batch_size": a.batch_size,
                                          "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return rc


def _check_agent_override(argv: list[str]) -> None:
    """Refuse to train unless the rendered trainer command selects the dispatch agent: Harbor builds an agent
    by `name` before `import_path`, and the trainer's default trial config names terminus-2, so without
    `agent.name=null` every rollout would run Terminus-2 whatever the harness list says."""
    if "harbor_trial_config.agent.name=null" not in argv or \
            not any(x.startswith("harbor_trial_config.agent.import_path=") for x in argv):
        raise SystemExit("exp grpo: the rendered trainer command does not select the dispatch agent "
                         "(harbor_trial_config.agent.name=null with agent.import_path); refusing to train")


def _audit_rollouts(out: Path, harnesses: list[str], agents_by_harness: dict[str, str]) -> int:
    """Check a GRPO run's rollouts (their result.json files) before the run is exported or recorded: every
    rollout that ran an agent ran its harness's own agent (`agent_info.name`, as the dispatch agent's backend
    records it, against the benchmark's `rl.rollout_agents`), the harnesses rolled out are exactly the run's, and
    each harness has rollouts with token-level details (what the trainer learns from) and rollouts that
    reached the verifier (trials/<trial>/verifier/reward.txt). Prints one line per harness and writes one
    prompt per harness for a human check. A run failing a check trained on something other than what its
    name says (or, with no rollout scored, on zero rewards), so it is neither exported nor recorded."""
    trials = sorted(p for p in (out / "trials").iterdir() if p.is_dir()) if (out / "trials").is_dir() else []
    if not trials:
        print(f"no rollouts under {C.rel(out / 'trials')}; not exporting or recording the run", file=sys.stderr)
        return 1
    wrong_agent: dict[tuple[str, str], int] = {}
    unexpected: dict[str, int] = {}
    per = {h: {"trials": 0, "details": 0, "scored": 0, "reward": 0.0, "example": None} for h in harnesses}
    for t in trials:
        try:
            r = json.load(open(t / "result.json"))
        except (OSError, ValueError):
            continue
        name = (r.get("agent_info") or {}).get("name")
        h = (((r.get("config") or {}).get("agent") or {}).get("kwargs") or {}).get("harness")
        if h not in per:
            unexpected[str(h)] = unexpected.get(str(h), 0) + 1
            continue
        if name and name != agents_by_harness.get(h):   # a trial that failed before its agent started names none
            wrong_agent[(h, name)] = wrong_agent.get((h, name), 0) + 1
        s = per[h]
        s["trials"] += 1
        if _has_token_details(r):
            s["details"] += 1
            s["example"] = s["example"] or t
        reward = t / "verifier" / "reward.txt"
        if reward.exists():
            s["scored"] += 1
            try:
                s["reward"] += float(reward.read_text().strip() or 0)
            except ValueError:
                pass
    problems = []
    for (h, name), c in sorted(wrong_agent.items()):
        problems.append(f"{h}: {c} rollouts ran {name} instead of {agents_by_harness.get(h)}")
    if unexpected:
        problems.append("rollouts on harnesses outside the run's: " +
                        ", ".join(f"{h} x{c}" for h, c in sorted(unexpected.items())))
    for h, s in per.items():
        mean = f"{100 * s['reward'] / s['scored']:.1f}" if s["scored"] else "-"
        print(f"{C.rel(out)}: {h}: {s['trials']} rollouts, {s['details']} with token details, "
              f"{s['scored']} reached the verifier, mean reward {mean}")
        if not s["details"]:
            problems.append(f"{h}: no rollout has token details (nothing to train on)")
        if not s["scored"]:
            problems.append(f"{h}: no rollout reached the verifier")
    _dump_harness_prompts(out, {h: s["example"] for h, s in per.items() if s["example"]})
    if problems:
        print(f"{C.rel(out)}: " + "; ".join(problems) + "; not exporting or recording the run", file=sys.stderr)
        return 1
    return 0


def _has_token_details(result: dict) -> bool:
    """Whether a rollout's result carries one linear segment of per-turn token IDs with matching logprobs."""
    d = (result.get("agent_result") or {}).get("rollout_details")
    if not isinstance(d, list) or len(d) != 1:
        return False
    comp, lp = d[0].get("completion_token_ids") or [], d[0].get("logprobs") or []
    return bool(comp) and len(lp) == len(comp) and all(c and len(c) == len(l) for c, l in zip(comp, lp))


FIRST_PROMPT_FILES = [   # where each agent of the dispatch registries writes its first user message
    ("agent/episode-0/prompt.txt", None),                                                   # Terminus-2
    ("agent/medagentbench-parity-agent.json", lambda j: j["transcript"][0]["content"]),    # MedAgentBench parity
    ("agent/fc_history.json", lambda j: j[0]["content"]),                                   # MedAgentBench fc
]


def _dump_harness_prompts(out: Path, examples: dict[str, Path]) -> None:
    """Write the first user message of one rollout per harness to <out>/harness-prompts/<harness>.txt, so a
    human can see that each harness's rollouts carried that harness's interface (a Terminus-2 prompt starts
    "You are an AI assistant tasked with solving command-line tasks"; MedAgentBench's parity and fc prompts
    start "You are an expert in using FHIR functions")."""
    dest = out / "harness-prompts"
    for h, t in examples.items():
        text = None
        for rel, pick in FIRST_PROMPT_FILES:
            p = t / rel
            if not p.exists():
                continue
            try:
                text = pick(json.load(open(p))) if pick else p.read_text()
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                continue
            break
        if text is None:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{h}.txt").write_text(text)
        first = text.strip().splitlines()[0][:90] if text.strip() else ""
        print(f"{C.rel(out)}: {h}: first prompt of {t.name} -> {C.rel(dest / f'{h}.txt')}: {first!r}")


def _export_grpo(out: Path, dry: bool = False) -> int:
    """Write the Hugging Face exports a GRPO run is missing: exports/global_step_N/policy, what
    `exp serve --rl` serves, for the checkpoints the recipe's `export_steps` names (configs/rl.yaml).
    The trainer's own export is off; this runs training/merge_fsdp2_to_hf.py on the CPU instead."""
    ckpts = list((out / "ckpts").glob("global_step_*"))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {C.rel(out / 'ckpts')}")
    for ckpt in R.export_steps(ckpts):
        export = out / "exports" / ckpt.name / "policy"
        if (export / "config.json").exists():
            continue
        print(f"# export {C.rel(ckpt)} -> {C.rel(export)}")
        argv, env, cwd = R.export_cmd(ckpt / "policy", export)
        rc = _run(argv, env=env, cwd=cwd, dry=dry)
        if rc:
            return rc
    return 0


def cmd_plan(a):
    """Print the bash script that reproduces one experiment from the shipped pools."""
    print(_plan_script(a))
    return 0


def _evaluates_in_sandbox(experiment: str) -> bool:
    """Whether any harness the experiment evaluates runs its agent inside the sandbox."""
    e = C.experiments().get(experiment) or {}
    for s in e.get("settings", []):
        b = C.benchmark(s["benchmark"])
        hs = s.get("harnesses", "trained")
        for h in b.harnesses(hs if isinstance(hs, str) else ",".join(hs)):
            if b.agent_spec(h).get("runs_in") == "sandbox":
                return True
    return False


def _check_backend(a):
    """Stop before a run starts if its sandboxes could not work, instead of hours later at the
    first evaluation (the plan scripts carry on past a failed evaluation)."""
    from urllib.parse import urlparse
    problems = []
    if a.backend == "eidf":
        from exp.backends import EIDF_VARS
        problems += [f"${v} is not set (see env.example)" for v in EIDF_VARS.values() if not os.environ.get(v)]
    if a.backend != "docker" and _evaluates_in_sandbox(a.experiment):
        # only agents that run inside the sandbox call the model server from there; a run whose
        # agents all run on the host (Terminus-2, the MedAgentBench agents) needs no such URL
        url = a.sandbox_api_base or os.environ.get("SANDBOX_API_BASE") or ""
        if not urlparse(url).hostname:
            problems.append(f"--sandbox-api-base {url!r} has no host: give a URL the sandboxes can reach, "
                            "e.g. http://<this machine\'s IP>:8000/v1")
    if problems:
        raise SystemExit("not starting:\n  " + "\n  ".join(problems) +
                         "\n(on EIDF: run . setup/eidf/session.sh in the shell you start the run from, inside tmux)")


def cmd_run(a):
    """Run one experiment: write its script (the one `exp plan` prints) to
    runs/plans/<plan>.sh, then run it. With --dry-run, only write it.

    The script on disk is the record of what ran; bash executes a copy held in memory, so a later
    `exp run` that rewrites the file cannot change a run in progress (bash reads a script file as
    it goes). One run of a plan at a time: a lock file next to the script names the host and
    process that hold it, and a second `exp run` of the same plan refuses to start while it exists
    (two runs would resume each other's evaluation jobs). --ignore-lock overrides it, e.g. after a
    crash that left the lock behind.
    """
    # One plan per (experiment, seed, attempts, pass, --only): these may run at the same time on
    # different machines of one checkout and must not share a script or a lock
    suffix = (f"-s{a.seed}" if getattr(a, "seed", None) is not None else "") + \
             (f"-n{a.n_attempts}" if getattr(a, "n_attempts", None) else "") + \
             (f"-p{a.eval_pass}" if getattr(a, "eval_pass", None) else "") + \
             (f"-only-{hashlib.sha1(','.join(sorted(a.only.split(','))).encode()).hexdigest()[:8]}" if getattr(a, "only", None) else "")
    script = C.ROOT / "runs" / "plans" / f"{a.experiment}{suffix}.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    text = _plan_script(a) + "\n"
    script.write_text(text)
    script.chmod(0o755)
    print(f"wrote {C.rel(script)}")
    if a.dry_run:
        return 0
    _check_backend(a)
    lock = script.with_suffix(".lock")
    if lock.exists() and not a.ignore_lock:
        raise SystemExit(f"not starting: {C.rel(lock)} says this plan is already running ({lock.read_text().strip()}); "
                         f"wait for it, or pass --ignore-lock if that run is gone")
    lock.write_text(f"host={os.uname().nodename} pid={os.getpid()} started={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    print(f"running: bash {C.rel(script)} (from memory; lock {C.rel(lock)})", flush=True)
    proc = subprocess.Popen(["bash", "-c", text], cwd=C.ROOT)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        # Ctrl-C reaches the plan's bash and the current `harbor run`, but harbor shuts down
        # slowly and a second Ctrl-C would leave it (and the model server, which runs in its own
        # process group) behind, still writing trials into the job folder. Stop the whole tree.
        print("\ninterrupted: stopping the plan's processes (harbor run, model server)", flush=True)
        _terminate_tree(proc.pid)
        return 130
    finally:
        lock.unlink(missing_ok=True)


def _terminate_tree(pid: int, grace: float = 15.0) -> None:
    """SIGTERM every process under `pid` (children first) and `pid` itself, wait up to `grace`
    seconds, then SIGKILL whatever is left."""
    procs = list(reversed(_descendants(pid))) + [pid]
    for p in procs:
        try:
            os.kill(p, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + grace
    while time.time() < deadline and any(_process_running(p) for p in procs):
        time.sleep(0.5)
    for p in procs:
        if _process_running(p):
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _only_settings(e: dict, only: str | None) -> set[str] | None:
    """The settings `--only` names, checked against the experiment (pool names, base, rl:<rollout>)."""
    if not only:
        return None
    wanted = {x.strip() for x in only.split(",") if x.strip()}
    known = {p for s in e["settings"] for p in s.get("pools", [])} | {f"rl:{r}" for s in e["settings"] for r in s.get("rl", [])}
    unknown = sorted(wanted - known)
    if unknown:
        raise SystemExit(f"--only: not settings of this experiment: {', '.join(unknown)}; it has: {', '.join(sorted(known))}")
    return wanted


def _finished_sft(bench: str, model: str, pool: str, seed: int | None) -> bool:
    """Whether a setting's SFT checkpoint is complete: final/ with its weights and training record."""
    final = R.checkpoint_dir(bench, model, pool, seed) / "final"
    return (final / "config.json").exists() and (final / SFT_RECORD).exists()


def _finished_grpo(b, model: str, rollout: str, seed: int | None, init: str | None = None) -> bool:
    """Whether a GRPO run has at least one Hugging Face export to serve."""
    exports = R.rl_dir(b, model, rollout, seed, init) / "exports"
    return exports.is_dir() and any(exports.glob("global_step_*/policy/config.json"))


def _rl_entry(r: str) -> tuple[str, str | None]:
    """An experiment's `rl:` entry: `<rollout>` or `<rollout>@<init>` (a GRPO run started from `base` or
    another pool instead of the benchmark's init pool)."""
    rollout, _, init = r.partition("@")
    return rollout, (init or None)


def _plan_script(a) -> str:
    """The bash script for one experiment: every command, in order."""
    e = C.experiments().get(a.experiment)
    if not e:
        raise SystemExit(f"unknown experiment {a.experiment!r}; see `exp list experiments`")
    exe = "uv run exp"
    be = f"--backend {a.backend}" + (f" --sandbox-api-base {a.sandbox_api_base}" if a.sandbox_api_base else "")
    # --n-attempts: every eval uses N attempts, in its own n<N> folder next to the full-protocol one;
    # --pass P repeats the same evaluations into n<N>-p<P>, pooled later by exp table
    n = getattr(a, "n_attempts", None)
    ep = getattr(a, "eval_pass", None)
    ev_n = (f" --n-attempts {n}" if n else "") + (f" --pass {ep}" if ep else "")
    # --seed: retrain every SFT / GRPO setting with that seed and evaluate the retrained checkpoints
    # (run s<seed> of each setting); the base model and the teacher are not trained, so they are left out
    sd = f" --seed {a.seed}" if a.seed is not None else ""
    # --eval-only: no training lines; a trained setting whose checkpoint is not finished yet is left
    # out with a note, so the script can be run again as more checkpoints finish (harbor resumes
    # finished evaluation jobs instantly)
    eval_only = getattr(a, "eval_only", False)
    conc = f" --concurrency {a.concurrency}" if getattr(a, "concurrency", None) else ""
    only = _only_settings(e, getattr(a, "only", None))
    out = ["#!/usr/bin/env bash", f"# {e['title']}",
           f"# generated by: uv run exp plan {a.experiment} (uv run exp run {a.experiment} writes this script and runs it)",
           *([f"# --only {','.join(sorted(only))}: the other settings of the experiment are left out"] if only else []),
           "# Needs: setup/install.sh done, GPUs, a sandbox backend (on EIDF: . setup/eidf/session.sh).",
           "# Edit PORT / GPU placement as needed.",
           "set -euo pipefail",
           f"cd {shlex.quote(str(C.ROOT))}   # the checkout this script was generated from",
           f"PORT={a.port}",
           f"API=http://127.0.0.1:$PORT/v1",
           "SERVER=",
           "# whatever ends this script (a failed step under set -e, Ctrl-C), the model server it started goes with it",
           'trap \'[ -n "$SERVER" ] && ' + exe + ' stop --pid $SERVER --port $PORT\' EXIT', ""]
    done_tasks, done_sft, done_rl = set(), set(), set()
    for s in e["settings"]:
        bn, model, split = s["benchmark"], s["model"], s.get("split", "test")
        hs = s.get("harnesses", "trained")
        hs = hs if isinstance(hs, str) else ",".join(hs)
        for sp in [split] + (["train"] if s.get("rl") else []):
            if (bn, sp) not in done_tasks:
                out.append(f"{exe} tasks --benchmark {bn} --split {sp}")
                done_tasks.add((bn, sp))
        runs = []
        if s.get("teacher"):
            runs.append(("--teacher", "teacher"))
        for p in s.get("pools", []):
            if only is not None and p not in only:
                continue
            if p == "base":
                if not sd:   # the base model is not retrained, so with --seed its row is left out
                    runs.append(("--pool base", "base"))
                continue
            if eval_only:
                if not _finished_sft(bn, model, p, a.seed):
                    if (bn, model, p) not in done_sft:
                        out.append(f"# skipped {bn} / {p}{sd}: checkpoint not finished "
                                   f"({C.rel(R.checkpoint_dir(bn, model, p, a.seed) / 'final')})")
                        done_sft.add((bn, model, p))
                    continue
            elif (bn, model, p) not in done_sft:
                out.append(f"{exe} sft --benchmark {bn} --model {model} --pool {p}{sd}")
                done_sft.add((bn, model, p))
            runs.append((f"--pool {p}{sd}", p))
        for r in s.get("rl", []):
            if only is not None and f"rl:{r}" not in only:
                continue
            rollout, init = _rl_entry(r)
            b_ = C.benchmark(bn)
            init_flag = f" --rl-init {init}" if init else ""
            label = R.rl_name(b_, rollout, init, model=model)
            if eval_only:
                if not _finished_grpo(b_, model, rollout, a.seed, init):
                    out.append(f"# skipped {bn} / {label}{sd}: no export yet "
                               f"({C.rel(R.rl_dir(b_, model, rollout, a.seed, init) / 'exports')})")
                    continue
            else:
                init_pool = init or b_.rl["init_pool"]
                if init_pool != "base" and (bn, model, init_pool) not in done_sft:
                    out.append(f"{exe} sft --benchmark {bn} --model {model} --pool {init_pool}")
                    done_sft.add((bn, model, init_pool))
                # one GRPO run per rollout, even when several settings evaluate it (e.g. two test splits)
                if (bn, model, r) not in done_rl:
                    out.append(f"{exe} grpo --benchmark {bn} --model {model} --rollout {rollout}{init_flag} --backend {a.backend}{sd}")
                    done_rl.add((bn, model, r))
            runs.append((f"--rl {rollout}{init_flag}{sd}", label))
        for flag, label in runs:
            out.append(f"# --- {bn} / {label} ---")
            served_model = "qwen3-32b" if flag == "--teacher" else model
            for harnesses, serve_harness in _serving_groups(C.benchmark(bn), served_model, hs):
                out += [f"setsid {exe} serve --benchmark {bn} --model {model} {flag}{serve_harness} --port $PORT & SERVER=$!",
                        f"{exe} wait --api-base $API --pid $SERVER",
                        f"{exe} eval --benchmark {bn} --model {model} {flag} --harness {harnesses} --split {split}{ev_n}{conc} --api-base $API {be} || true",
                        f"{exe} stop --pid $SERVER --port $PORT"]
            out.append("")
    out.append(f"{exe} table {a.experiment}{ev_n}{sd}")
    return "\n".join(out)


def _serving_groups(b, model: str, harness_spec: str) -> list[tuple[str, str]]:
    """Split a harness set into groups that one model server can serve.

    Returns (harnesses to evaluate, extra `exp serve` option) pairs. Normally that is one pair:
    the whole set, served once. A model served differently per harness (a `per_harness` entry
    under serve.per_model in the benchmark config, e.g. Gemma on MedAgentBench: fc with a
    tool parser, the rest without) gets one server per distinct serving setup, started with
    --harness so `exp serve` picks that setup.
    """
    per_model = (b.cfg["serve"].get("per_model") or {}).get(model) or {}
    if not per_model.get("per_harness"):
        return [(harness_spec, "")]
    groups: dict[str, list[str]] = {}
    for h in b.harnesses(harness_spec):
        setup = json.dumps(R.serve_settings(b, model, h), sort_keys=True)
        groups.setdefault(setup, []).append(h)
    return [(",".join(hs), f" --harness {hs[0]}") for hs in groups.values()]


def cmd_tasks(a):
    from exp import tasks
    return tasks.main(a)


def cmd_failures(a):
    from exp import failures
    return failures.main(a)


def cmd_failure_types(a):
    from exp import failuretypes
    return failuretypes.main(a)


def cmd_annotation_sample(a):
    from exp import annotation
    return annotation.cmd_sample(a)


def cmd_annotation_tables(a):
    from exp import annotation
    return annotation.cmd_tables(a)


def cmd_daytona_sweep(a):
    from exp import daytona
    return daytona.main(a)


def cmd_daytona_wait(a):
    from exp import daytona
    return daytona.wait_free(a)


def cmd_hf_export(a):
    from exp import hf
    return hf.main(a)


def cmd_export_results(a):
    from exp import export
    return export.main(a)


def cmd_control_tables(a):
    from exp import controltables, seedtables
    seedtables.SHIPPED = a.shipped
    return controltables.main(a)


def cmd_seed_tables(a):
    from exp import seedtables
    seedtables.SHIPPED = a.shipped
    return seedtables.main(a)


def cmd_table(a):
    from exp import table
    if getattr(a, "seeds", None) and a.seed is not None:
        raise SystemExit("exp table: use --seed N for one seed or --seeds A,B,... for a pooled view, not both")
    if getattr(a, "seeds", None) and (a.shipped or a.format == "cells"):
        raise SystemExit("exp table --seeds reads per-trial outcomes under runs/; it has no --shipped or --format cells form")
    return table.main(a)


# --------------------------------------------------------------------------- parser
def _source_options(p):
    """Which evaluation jobs `exp table` and `exp failures` read."""
    p.add_argument("--evals-dir", help="evaluations root, default runs/evals")
    p.add_argument("--n-attempts", type=int, help="read the evaluations run with N attempts per task "
                                                  "(default: the reported protocol's)")
    p.add_argument("--pass", dest="eval_pass", type=int, action="append",
                   help="read pass P of each evaluation; repeat to pool passes per task (default: the first)")


def _common_setting(p, teacher=True):
    p.add_argument("--benchmark", "-b", required=True, choices=C.benchmark_names())
    p.add_argument("--model", "-m", default="qwen3-4b")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--pool", help="SFT pool name (see `exp list pools`); default: base model")
    g.add_argument("--rl", help="GRPO run: 3h or the single rollout harness")
    if teacher:
        g.add_argument("--teacher", action="store_true", help="the Qwen3-32B teacher")
    p.add_argument("--rl-init", metavar="POOL", help="with --rl: the run started from this checkpoint instead of the "
                                                      "benchmark's init pool: base, or a pool name (run folder grpo-<rollout>-from-<POOL>)")
    p.add_argument("--seed", type=int, help="the retraining of this setting with that seed (exp sft/grpo --seed): "
                                            "its s<seed> run")


def build_parser():
    ap = argparse.ArgumentParser(prog="exp", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="list benchmarks, pools, models or experiments")
    p.add_argument("what", choices=["benchmarks", "pools", "models", "experiments"])
    p.add_argument("--benchmark", "-b")
    p.add_argument("--group", choices=["main", "seeds", "controls", "other"], help="experiments: only this group")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("tasks", help="generate harbor task directories for a split")
    p.add_argument("--benchmark", "-b", required=True, choices=C.benchmark_names())
    p.add_argument("--split", default="test")
    p.add_argument("--out", help="default runs/tasks/<benchmark>/<split>")
    p.add_argument("--from", dest="from_dir", help="copy the split from an existing task directory instead of generating")
    p.add_argument("--source", choices=["adapter", "registry"], help="where tasks come from (default: what the clusters used)")
    p.add_argument("--source-commit", help="upstream commit to generate from (MedAgentBench, BFCL single-turn)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_tasks)

    p = sub.add_parser("serve", help="serve a model with vLLM, with the benchmark's as-run flags")
    _common_setting(p)
    p.add_argument("--checkpoint", help="serve this path instead of the default for the setting")
    p.add_argument("--rl-step", default="final")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--served-model", default=R.SERVED)
    p.add_argument("--tensor-parallel-size", type=int)
    p.add_argument("--harness", help="serve for this harness, where a model's serving differs per harness (Gemma on MedAgentBench)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("wait", help="wait until a model server answers")
    p.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--pid", type=int, help="the server's process id: stop waiting if it exits, and fail if "
                                           "another process answers on the port")
    p.set_defaults(fn=cmd_wait)

    p = sub.add_parser("stop", help="stop a model server and wait until the GPUs are free")
    p.add_argument("--pid", type=int, help="the server's process id, started with setsid (see `exp stop --help`)")
    p.add_argument("--port", type=int, help="also stop whatever listens on this port (a server whose pid is "
                                            "unknown or wrong, or one left over from an earlier run)")
    p.set_defaults(fn=cmd_stop)

    p = sub.add_parser("retry-errors", help="remove a job's trials that failed on infrastructure, so the next "
                                            "`exp eval` of the job (a Harbor resume) runs them again")
    p.add_argument("job_dir", nargs="+", help="evaluation job folder(s) under runs/evals/")
    p.add_argument("--down", nargs=2, metavar=("FROM", "TO"),
                   help="also remove the trials that ran while the model server was down, between these two "
                        "UTC ISO timestamps (as written in result.json, e.g. 2026-11-08T22:50:40Z)")
    p.add_argument("--dry-run", action="store_true", help="only list them")
    p.set_defaults(fn=cmd_retry_errors)

    p = sub.add_parser("sft", help="supervised fine-tuning on one pool")
    p.add_argument("--max-steps", type=int, help="small test run: stop after N optimizer steps")
    p.add_argument("--benchmark", "-b", required=True, choices=C.benchmark_names())
    p.add_argument("--model", "-m", default="qwen3-4b")
    p.add_argument("--pool", required=True)
    p.add_argument("--data", help="train on this JSONL instead of the shipped pool (e.g. one rebuilt by build-pool)")
    p.add_argument("--out", help="default runs/checkpoints/<benchmark>/<model>/<pool>")
    p.add_argument("--wandb-run")
    p.add_argument("--seed", type=int, help="train with this seed (data order, dropout) into the pool's s<seed> run; "
                                            "default: the trainer's own (42), as the paper ran")
    p.add_argument("--force", action="store_true", help="retrain even if the checkpoint already exists")
    p.add_argument("--keep-checkpoints", action="store_true",
                   help="keep the trainer's intermediate checkpoint-<N> folders (by default they are removed once "
                        "the final model is written: they only resume an interrupted run, ~27 GB each)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_sft)

    p = sub.add_parser("eval", help="evaluate a served model on one or more harnesses")
    _common_setting(p)
    p.add_argument("--limit", type=int, help="small test run: N tasks spread evenly over the split")
    p.add_argument("--n-attempts", type=int, help="small test run: attempts per task (default: the protocol's)")
    p.add_argument("--harness", default="all", help="a harness name, or trained / heldout / all")
    p.add_argument("--split", default="test")
    p.add_argument("--backend", default="docker", choices=BACKENDS)
    p.add_argument("--api-base", default="http://127.0.0.1:8000/v1", help="model server URL, as seen from this machine")
    p.add_argument("--sandbox-api-base", help="model server URL as seen from inside a sandbox (default: derived for docker)")
    p.add_argument("--served-model", default=R.SERVED)
    p.add_argument("--concurrency", type=int)
    p.add_argument("--pass", dest="eval_pass", type=int, help="write this evaluation as pass P (P >= 2) of the cell")
    p.add_argument("--evals-dir", help="evaluations root, default runs/evals; the job is written to "
                                       "<root>/<task set>/<model>/<setting>/<run>/<harness>/<pass>")
    p.add_argument("--tasks-dir", help="default runs/tasks")
    p.add_argument("--job-dir", help="write the job to this folder instead")
    p.add_argument("--dry-run", action="store_true", help="write the job config and print the command only")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("collect", help="collect teacher traces on the training split (serve the teacher first)")
    p.add_argument("--benchmark", "-b", required=True, choices=C.benchmark_names())
    p.add_argument("--harness", help="only this harness (default: every job)")
    p.add_argument("--rehearsal", action="store_true",
                   help="collect the rehearsal traces instead (config `rehearsal:`; serve each one's demonstrator)")
    p.add_argument("--job", help="only the job with this name")
    p.add_argument("--limit", type=int, help="small test run: N tasks spread evenly over the split")
    p.add_argument("--n-attempts", type=int, help="small test run: attempts per task (default: the protocol's)")
    p.add_argument("--backend", default="docker", choices=BACKENDS)
    p.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--sandbox-api-base")
    p.add_argument("--served-model", default=R.SERVED)
    p.add_argument("--concurrency", type=int, help="concurrent trials (default: the benchmark's eval concurrency)")
    p.add_argument("--jobs-dir", help="default runs/collect/<benchmark>")
    p.add_argument("--tasks-dir")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_collect)

    p = sub.add_parser("build-pool", help="build SFT pools from collected traces (config `build:`)")
    p.add_argument("--benchmark", "-b", required=True, choices=C.benchmark_names())
    p.add_argument("--pool", help="build only the step that writes this setting (default: every step, in order)")
    p.add_argument("--jobs-dir", help="the collected traces, default runs/collect/<benchmark>")
    p.add_argument("--force", action="store_true", help="rebuild a derived pool that is already there")
    p.add_argument("--out-dir", help="default runs/pools/<benchmark>")
    p.add_argument("--tokenizer", help="student tokenizer for the builders' render check (default Qwen/Qwen3-4B; a local path works)")
    p.add_argument("--min-trajectories", type=int,
                   help="small test run: accept fewer solved trajectories per harness than the builder's minimum (Reasoning Gym: 30)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_build_pool)

    p = sub.add_parser("oracle", help="run reference solutions on a few tasks (no model, no GPU)")
    p.add_argument("--benchmark", "-b", required=True, choices=C.benchmark_names())
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--backend", default="docker", choices=BACKENDS)
    p.add_argument("--tasks-dir")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_oracle)

    p = sub.add_parser("grpo", help="GRPO from the 3H SFT checkpoint (or another start, --rl-init)")
    p.add_argument("--max-steps", type=int, help="small test run: stop (and save) after N training steps")
    p.add_argument("--batch-size", type=int, help="small test run: prompts per update, a multiple of the rollout harnesses (default: the recipe's 24)")
    p.add_argument("--out", help="output folder (default runs/rl/<benchmark>/<model>/grpo-<rollout>)")
    p.add_argument("--benchmark", "-b", required=True, choices=["medagentbench", "reasoning_gym"])
    p.add_argument("--model", "-m", default="qwen3-4b")
    p.add_argument("--rollout", required=True, help="3h, or the single harness to roll out on")
    p.add_argument("--init", help="initial checkpoint path (default: this model's checkpoint on the benchmark's init pool)")
    p.add_argument("--rl-init", metavar="POOL", help="start from `base` (the untuned model) or another pool's checkpoint "
                                                      "instead of the init pool; the run is named grpo-<rollout>-from-<POOL> and "
                                                      "rolls out on the paper's learning band (rl.band), unless the benchmark "
                                                      "config registers rl.bands.<POOL>")
    p.add_argument("--band", help="learning band (a split JSON) to roll out on, overriding the configured one")
    p.add_argument("--backend", default="docker", choices=BACKENDS)
    p.add_argument("--run-name")
    p.add_argument("--seed", type=int, help="train with this trainer seed into the GRPO setting's s<seed> run")
    p.add_argument("--concurrency", type=int, help="rollouts (sandboxes) at a time, below the recipe's 24 when the "
                   "sandbox backend is short of capacity; throughput only, the updates are the same")
    p.add_argument("--force", action="store_true", help="retrain even if a finished run is already there")
    p.add_argument("--resume", action="store_true", help="continue a crashed run from its latest checkpoint")
    p.add_argument("--export-only", action="store_true",
                   help="only write the Hugging Face exports of the checkpoints already there (no training); a run "
                        "whose last checkpoint is the recipe's final step is then recorded as finished")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_grpo)

    p = sub.add_parser("plan", help="print the bash script that reproduces one experiment")
    _experiment_options(p)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("run", help="run one experiment: write its script to runs/plans/<experiment>.sh and run it")
    _experiment_options(p)
    p.add_argument("--dry-run", action="store_true", help="only write the script")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("failures", help="automatic failure-type proxies per cell of an experiment (timeouts, re-prompts, "
                                        "empty replies, turns without a call), from the trial folders")
    p.add_argument("experiment")
    _source_options(p)
    p.add_argument("--seed", type=int, help="the rows retrained with this seed (their s<seed> run)")
    p.set_defaults(fn=cmd_failures)

    p = sub.add_parser("failure-types", help="rule-based failure-type labels (interface: invalid / submission / "
                                             "termination, or task) for every failed trial of the cells in a spec")
    p.add_argument("--spec", default=str(C.CONFIGS / "tables" / "failure_types.yaml"))
    p.add_argument("--out", default="results/tables/failure-types", help="output folder")
    p.set_defaults(fn=cmd_failure_types)

    ft_spec, ft_trials = str(C.CONFIGS / "tables" / "failure_types.yaml"), "results/tables/failure-types/trials.csv"
    p = sub.add_parser("annotation-sample", help="draw the failed trials that human annotators label, with the "
                                                 "material they read (transcript, verifier verdict, a passing run)")
    p.add_argument("--spec", default=ft_spec)
    p.add_argument("--out", default="results/tables/failure-annotation", help="folder for sample.csv (which trials were drawn)")
    p.add_argument("--traces-out", default="runs/annotation", help="folder for sample.json (the drawn trials' transcripts)")
    p.set_defaults(fn=cmd_annotation_sample)

    p = sub.add_parser("annotation-tables", help="automatic failure types against the human labels, and annotator agreement")
    p.add_argument("--spec", default=ft_spec)
    p.add_argument("--labels", default="results/failure_annotation_labels.csv", help="one row per annotator and trial")
    p.add_argument("--auto", default=ft_trials, help="trials.csv of exp failure-types")
    p.add_argument("--out", default="results/tables/failure-annotation", help="output folder")
    p.set_defaults(fn=cmd_annotation_tables)

    p = sub.add_parser("daytona-sweep", help="list (or, with --delete, delete) Daytona sandboxes left behind by stopped "
                                             "runs: no activity for --idle-min minutes")
    p.add_argument("--idle-min", type=float, default=30, help="minutes without activity before a sandbox counts as idle (default 30)")
    p.add_argument("--delete", action="store_true", help="delete the idle sandboxes (default: only list them)")
    p.set_defaults(fn=cmd_daytona_sweep)

    p = sub.add_parser("daytona-wait", help="wait until the Daytona organisation has enough sandbox disk free for a run")
    p.add_argument("--free-gib", type=int, required=True, help="GiB that must be free (a MedAgentBench sandbox takes 9, a Reasoning Gym one 1)")
    p.add_argument("--limit-gib", type=int, default=300, help="the organisation's disk limit in GiB (default 300)")
    p.add_argument("--poll", type=int, default=120, help="seconds between checks (default 120)")
    p.set_defaults(fn=cmd_daytona_wait)

    p = sub.add_parser("control-tables", help="tables of controls and GRPO runs (single runs or small run sets), "
                                              "from per-trial files; writes markdown, a cell csv and a path manifest")
    p.add_argument("--spec", default=str(C.CONFIGS / "tables" / "control_tables.yaml"))
    p.add_argument("--external", help="csv of numbers from elsewhere (default: the spec's `external`)")
    p.add_argument("--out", default="results/tables/control-tables", help="output folder")
    p.add_argument("--compact-out", default="rebuttal/tables/control-tables",
                   help="folder for the compact tables: title, caption and mean ± CI cells, one file per table")
    p.add_argument("--shipped", action="store_true",
                   help="read every job from results/rewards.csv instead of its folder under runs/")
    p.set_defaults(fn=cmd_control_tables)

    p = sub.add_parser("seed-tables", help="tables pooled over training runs (s0 and the retraining seeds), "
                                           "from per-trial files; writes markdown, a cell csv and a path manifest")
    p.add_argument("--spec", default=str(C.CONFIGS / "tables" / "seed_tables.yaml"))
    p.add_argument("--external", help="csv of numbers from another cluster (default: the spec's `external`)")
    p.add_argument("--out", default="results/tables/seed-tables", help="output folder")
    p.add_argument("--compact-out", default="rebuttal/tables/seed-tables",
                   help="folder for the compact tables: title, caption and mean ± CI cells, one file per table")
    p.add_argument("--shipped", action="store_true",
                   help="read every job from results/rewards.csv instead of its folder under runs/")
    p.set_defaults(fn=cmd_seed_tables)

    p = sub.add_parser("export-results", help="write results/rewards.csv (every attempt's reward, per task and job) and "
                                              "results/cells.csv (mean, pass@n and CI per run) from the evaluation jobs")
    p.add_argument("--out", default="results", help="output folder")
    p.set_defaults(fn=cmd_export_results)

    p = sub.add_parser("hf-export", help="write the rehearsal tasks, the rehearsal and teacher trajectories and the annotated failures "
                                         "as dataset folders with their cards, ready to upload (nothing is uploaded)")
    p.add_argument("--out", default="runs/hf", help="output folder; one folder per dataset is written in it")
    p.add_argument("--benchmarks", nargs="+", metavar="B",
                   help="benchmarks to export, or `all` (default: every benchmark but bfcl_singleturn and appworld)")
    p.add_argument("--sample", default="runs/annotation/sample.json", help="the annotation sample (exp annotation-sample)")
    p.add_argument("--labels", default="results/failure_annotation_labels.csv")
    p.add_argument("--auto", default="results/tables/failure-types/trials.csv", help="trials.csv of exp failure-types")
    p.set_defaults(fn=cmd_hf_export)

    p = sub.add_parser("table", help="rebuild a paper table from finished runs (or --shipped results)")
    p.add_argument("--format", choices=["table", "cells"], default="table",
                   help="table = markdown (default); cells = the unified schema of results/cells.csv")
    p.add_argument("--out", help="write cells to this file instead of stdout")
    p.add_argument("experiment")
    p.add_argument("--shipped", action="store_true",
                   help="use the shipped results/ instead of runs/: results/rewards.csv, and results/cells.csv "
                        "for cells it does not have")
    _source_options(p)
    p.add_argument("--seed", type=int, help="the rows retrained with this seed (their s<seed> run); "
                                            "base and teacher rows stay the unseeded ones")
    p.add_argument("--seeds", type=lambda v: [int(x) for x in v.split(",")],
                   help="several retraining seeds, e.g. 1,2,3: trained rows pool the seeds' trials per task "
                        "(one bootstrap over the pool) and list each seed's mean and their SD; not with --seed")
    p.set_defaults(fn=cmd_table)

    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a) or 0
