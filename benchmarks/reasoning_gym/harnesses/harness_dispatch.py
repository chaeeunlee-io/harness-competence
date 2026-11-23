"""Dispatch Reasoning Gym rows to the selected harness.

Used for GRPO rollouts; evaluation runs Harbor's agents directly (see the
benchmark config). Supported harnesses: `t2`, `miniswe`, and `goose`. The
harness is selected per row through `kwargs["harness"]`.

Terminus-2 calls the model from the Harbor process through Harbor's LiteLLM and
returns its token-level rollout details itself. mini-swe-agent and Goose are
installed agents: CLIs that run inside the sandbox and call the model over HTTP,
configured as in evaluation (an `openai/<served model>` model name and the
`OPENAI_*` endpoint variables), and they return no token details. When the
trainer collects rollout details, each of their trials therefore gets a
`RecordingProxy`: the CLI talks to the proxy, the proxy forwards every
chat-completion request to the trainer's server asking for token IDs and
logprobs, keeps them, and answers with the server's response minus those fields,
so the agent sees what it would see without the proxy. After the run the kept
turns become the trial's rollout details (one segment, one entry per model call).
Settings that belong to Terminus-2 in the trainer's default agent kwargs
(`max_turns`, `temperature`, ...) are dropped, so the installed agents keep the
loop, budgets and context handling they have in evaluation; sampling is done at
the rollout temperature, which the trainer's logprob correction assumes.
"""
import asyncio
import json
import time
from typing import Any
from urllib.parse import urlsplit

from harbor.models.agent.context import AgentContext
from harbor.models.agent.rollout_detail import RolloutDetail

HARNESS_REGISTRY: dict[str, tuple[type, dict[str, Any]]] = {}

from harbor.agents.terminus_2.terminus_2 import Terminus2  # noqa: E402
HARNESS_REGISTRY["t2"] = (Terminus2, {"parser": "json"})

# Kwargs of the trainer's default trial config (harbor_trial_config/default.yaml) that configure Terminus-2 or the
# trainer, not the installed agents. Goose would turn `max_turns` into `--max-turns`.
TRAINER_ONLY_KWARGS = ("max_turns", "suppress_max_turns_warning", "enable_summarize", "temperature",
                       "model_info", "llm_kwargs", "session_id", "collect_rollout_details", "store_all_messages")
ROLLOUT_TEMPERATURE = 1.0   # generator.sampling_params.temperature of the GRPO recipe (the trainer's default)


class RecordingProxy:
    """A local HTTP server between an installed agent and the model server, for one trial.

    Each chat-completion request is forwarded upstream with `logprobs` and `return_token_ids` on, the trainer's
    `session_id` (router affinity) and the rollout temperature; the response goes back without those extra fields.
    A streaming request is answered as a stream of one content chunk and one finish chunk built from the full
    response (token details need the complete response). Every call's prompt token IDs, completion token IDs and
    logprobs are kept in order."""

    def __init__(self, upstream: str, session_id: str | None, temperature: float = ROLLOUT_TEMPERATURE):
        self.upstream = upstream.rstrip("/")            # .../v1
        self.session_id = session_id
        self.temperature = temperature
        self.prompt_token_ids: list[list[int]] = []
        self.completion_token_ids: list[list[int]] = []
        self.logprobs: list[list[float]] = []
        self.errors: list[str] = []
        self._runner = None
        self.port: int | None = None

    async def start(self) -> None:
        from aiohttp import web
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_post("/v1/chat/completions", self._chat)
        app.router.add_post("/chat/completions", self._chat)
        app.router.add_get("/v1/models", self._models)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _models(self, request):
        import aiohttp
        from aiohttp import web
        async with aiohttp.ClientSession() as s, s.get(f"{self.upstream}/models") as r:
            return web.Response(body=await r.read(), status=r.status, content_type="application/json")

    async def _chat(self, request):
        import aiohttp
        from aiohttp import web
        body = await request.json()
        stream = bool(body.pop("stream", False))
        body.pop("stream_options", None)
        body["logprobs"] = True
        body["return_token_ids"] = True
        body["temperature"] = self.temperature
        if self.session_id is not None:
            body["session_id"] = self.session_id
        headers = {"Content-Type": "application/json", "Authorization": request.headers.get("Authorization", "Bearer EMPTY")}
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
                async with s.post(f"{self.upstream}/chat/completions", json=body, headers=headers) as r:
                    status, raw = r.status, await r.read()
        except Exception as e:  # upstream unreachable: the agent sees a server error, as it would without the proxy
            self.errors.append(f"{type(e).__name__}: {e}")
            return web.json_response({"error": {"message": str(e)}}, status=502)
        if status != 200:
            self.errors.append(f"upstream {status}: {raw[:200]!r}")
            return web.Response(body=raw, status=status, content_type="application/json")
        resp = json.loads(raw)
        self._keep(resp)
        if not stream:
            return web.json_response(resp)
        return await self._as_stream(request, resp)

    def _keep(self, resp: dict) -> None:
        """Keep this call's token details and remove the fields the agent did not ask for."""
        prompt_ids = resp.pop("prompt_token_ids", None)
        choice = (resp.get("choices") or [{}])[0]
        comp_ids = choice.pop("token_ids", None)
        lp = choice.pop("logprobs", None)
        lps = [t["logprob"] for t in (lp or {}).get("content") or []]
        if prompt_ids and comp_ids and len(lps) == len(comp_ids):
            self.prompt_token_ids.append(prompt_ids)
            self.completion_token_ids.append(comp_ids)
            self.logprobs.append(lps)
        else:
            self.errors.append(f"call without token details (prompt {bool(prompt_ids)}, completion "
                               f"{len(comp_ids or [])}, logprobs {len(lps)})")

    async def _as_stream(self, request, resp: dict):
        from aiohttp import web
        out = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await out.prepare(request)
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        delta: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
        for k in ("reasoning_content", "reasoning"):
            if msg.get(k):
                delta[k] = msg[k]
        if msg.get("tool_calls"):
            delta["tool_calls"] = [{**tc, "index": i} for i, tc in enumerate(msg["tool_calls"])]
        base = {k: resp.get(k) for k in ("id", "created", "model")}
        base["object"] = "chat.completion.chunk"
        chunks = [
            {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason")}],
             "usage": resp.get("usage")},
        ]
        for c in chunks:
            await out.write(f"data: {json.dumps(c)}\n\n".encode())
        await out.write(b"data: [DONE]\n\n")
        await out.write_eof()
        return out

    def rollout_detail(self) -> RolloutDetail | None:
        if not self.completion_token_ids:
            return None
        return {"prompt_token_ids": self.prompt_token_ids, "completion_token_ids": self.completion_token_ids,
                "logprobs": self.logprobs}


class _RecordedInstalledAgent:
    """Mixin for an installed agent run in GRPO: the evaluation configuration, plus a RecordingProxy when the
    trainer collects rollout details (see the module docstring)."""

    def _setup_recording(self, kwargs: dict) -> str | None:
        collect = bool(kwargs.get("collect_rollout_details"))
        session_id = kwargs.get("session_id")
        for k in TRAINER_ONLY_KWARGS:
            kwargs.pop(k, None)
        kwargs.pop("harness", None)
        api_base = kwargs.pop("api_base", None)
        # The trainer names the model hosted_vllm/<served>; the installed agents were evaluated as openai/<served>
        if kwargs.get("model_name") and "/" in kwargs["model_name"]:
            kwargs["model_name"] = "openai/" + kwargs["model_name"].split("/", 1)[1]
        self._proxy = RecordingProxy(api_base, session_id) if (collect and api_base) else None
        return api_base

    def _endpoint_for_agent(self, api_base: str) -> str:
        """The URL the CLI in the sandbox calls: the proxy on this host's address, else the server itself."""
        if self._proxy is None:
            return api_base
        host = urlsplit(api_base).hostname
        return f"http://{host}:{self._proxy.port}/v1"

    async def run(self, instruction: str, environment, context: AgentContext) -> None:
        if self._proxy is None:
            return await super().run(instruction, environment, context)
        await self._proxy.start()
        self._point_agent_at(self._endpoint_for_agent(self._proxy.upstream))
        try:
            await super().run(instruction, environment, context)
        finally:
            await self._proxy.stop()
            detail = self._proxy.rollout_detail()
            context.metadata = context.metadata or {}
            context.metadata["n_episodes"] = len(self._proxy.completion_token_ids)
            context.metadata["summarization_count"] = 0
            if self._proxy.errors:
                context.metadata["proxy_errors"] = self._proxy.errors[:20]
            if detail is not None:
                context.rollout_details = [detail]


try:
    from benchmarks.mini_swe_agent import PinnedMiniSweAgent as _Mini   # mini-swe-agent 2.4.6, litellm pinned

    class MiniSweDispatch(_RecordedInstalledAgent, _Mini):
        def __init__(self, *args, **kwargs):
            api_base = self._setup_recording(kwargs)
            extra = kwargs.setdefault("extra_env", {})
            if api_base and self._proxy is None:
                extra.setdefault("OPENAI_API_BASE", api_base)
            super().__init__(*args, **kwargs)

        def _point_agent_at(self, url: str) -> None:
            self._extra_env["OPENAI_API_BASE"] = url

    HARNESS_REGISTRY["miniswe"] = (MiniSweDispatch, {})
except ImportError:
    pass

try:
    from harbor.agents.installed.goose import Goose as _Goose

    class GooseDispatch(_RecordedInstalledAgent, _Goose):
        def __init__(self, *args, **kwargs):
            api_base = self._setup_recording(kwargs)
            extra = kwargs.setdefault("extra_env", {})
            if api_base and self._proxy is None:
                self._set_goose_endpoint(extra, api_base)
            super().__init__(*args, **kwargs)

        @staticmethod
        def _set_goose_endpoint(env: dict, url: str) -> None:
            env["OPENAI_HOST"] = url.rsplit("/v1", 1)[0]
            env["OPENAI_BASE_PATH"] = "v1/chat/completions"

        def _point_agent_at(self, url: str) -> None:
            self._set_goose_endpoint(self._extra_env, url)

    HARNESS_REGISTRY["goose"] = (GooseDispatch, {})
except ImportError:
    pass


class DispatchAgent:
    """Instantiates the registered backend for kwargs['harness'] and proxies to it."""

    def __init__(self, logs_dir=None, model_name: str | None = None, *, harness: str, **kwargs):
        if harness not in HARNESS_REGISTRY:
            raise KeyError(f"Unknown rg harness {harness!r}. Known: {sorted(HARNESS_REGISTRY)}.")
        backend_cls, presets = HARNESS_REGISTRY[harness]
        merged = {**presets, **kwargs}
        self._backend = backend_cls(logs_dir=logs_dir, model_name=model_name, **merged)
        self._harness = harness

    def __getattr__(self, item: str) -> Any:
        return getattr(self._backend, item)

    @staticmethod
    def name() -> str:
        return "rg-dispatch"

    def version(self) -> str | None:
        return getattr(self._backend, "version", lambda: None)()

    async def run(self, *args, **kwargs):
        return await self._backend.run(*args, **kwargs)

    async def setup(self, *args, **kwargs):
        return await self._backend.setup(*args, **kwargs)
