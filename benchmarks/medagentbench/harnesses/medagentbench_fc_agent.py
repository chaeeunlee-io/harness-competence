"""MedAgentBench harness using native OpenAI tool calls.

Uses native `get`, `post`, and `finish` tool calls while keeping the same task
semantics and verifier behavior as the parity harness.

Runs entirely on the harness side. Registry key: `"fc"`.

Model calls go through Harbor's instrumented `LiteLLM`, as the trainable parity
agent's do, so the harness is trainable: with `collect_rollout_details` (set by
the SkyRL trainer) every call's prompt and completion token IDs and logprobs are
kept and returned as one linear rollout segment in `context.rollout_details`.
Evaluation uses the same call path.
"""

from __future__ import annotations

import json
from typing import Any

from harbor.environments.base import BaseEnvironment
from harbor.llms.base import OutputLengthExceededError
from harbor.llms.lite_llm import LiteLLM
from harbor.models.agent.context import AgentContext
from harbor.models.agent.rollout_detail import RolloutDetail

try:
    from benchmarks.medagentbench.harnesses.medagentbench_parity_agent import (
        DEFAULT_POST_ACK,
        MEDAGENTBENCH_PROMPT,
        MedAgentBenchParityAgent,
    )
except ImportError:  # SkyRL cwd fallback (same pattern as dispatch/trainable)
    import importlib.util
    import sys
    from pathlib import Path as _Path

    _spec = importlib.util.spec_from_file_location(
        "medagentbench_parity_agent",
        _Path(__file__).parent / "medagentbench_parity_agent.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules.setdefault("medagentbench_parity_agent", _mod)
    _spec.loader.exec_module(_mod)
    DEFAULT_POST_ACK = _mod.DEFAULT_POST_ACK
    MEDAGENTBENCH_PROMPT = _mod.MEDAGENTBENCH_PROMPT
    MedAgentBenchParityAgent = _mod.MedAgentBenchParityAgent

# Tool-native prompt (2026-09-07). The parity template was reused here originally,
# which instructs text-format GET/POST/FINISH — contradicting the tool-call modality.
# With tool_choice=required the contradiction was masked (guided decoding forced tools);
# with tool_choice=auto (needed so <think> reasoning can flow), near-identical prompts
# with conflicting target formats caused cross-harness format interference in SFT
# (v3 pool: parity 66->46). fc now has a DISTINCT, self-consistent interface signature.
FC_PROMPT = """You are an expert in using FHIR functions to assist medical professionals. \
You are given a question and access to tools for querying and updating a FHIR server. \
Use the provided tools to accomplish the task:

- Call the `get` tool to retrieve FHIR resources (pass the full request URL).
- Call the `post` tool to create or update FHIR resources (pass the URL and JSON payload).
- When you have completed all requested tasks, call the `finish` tool with your final \
answer(s) as a JSON-loadable list.

Think through what the question requires before each tool call. Make exactly one tool \
call per turn. Note that you should use {api_base} as the api_base.

Reference catalog of the underlying FHIR operations available through the tools:
{functions}

Context: {context}
Question: {question}"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get",
            "description": "Send a GET request to the FHIR API. Returns the response body.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full FHIR URL including query parameters."}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "post",
            "description": "Send a POST request to the FHIR API with a JSON payload.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full FHIR URL to POST to."},
                    "payload": {"type": "object", "description": "JSON body of the POST."},
                },
                "required": ["url", "payload"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Submit the final answer and end the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "string",
                        "description": "Final answer, JSON-encoded when the task asks for structured output.",
                    }
                },
                "required": ["result"],
            },
        },
    },
]


class MedAgentBenchFCAgent(MedAgentBenchParityAgent):
    """Native function-calling MedAgentBench harness (dialect axis)."""

    def __init__(
        self,
        *args,
        session_id: str | None = None,
        collect_rollout_details: bool = False,
        model_info: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._session_id = session_id
        self._collect_rollout_details = collect_rollout_details
        self._model_info = model_info
        self._extra_llm_kwargs = llm_kwargs or {}
        self._llm: LiteLLM | None = None
        # Token details of every model call, one entry per call: the trainer consumes them turn by turn.
        self._prompt_token_ids: list[list[int]] = []
        self._completion_token_ids: list[list[int]] = []
        self._logprobs: list[list[float]] = []

    @staticmethod
    def name() -> str:
        return "medagentbench-fc"

    def version(self) -> str | None:
        return "fc-v2"   # fc-v1 called the model over raw HTTP; same prompt, tools and loop

    # ------------------------------------------------------------------
    def _ensure_llm(self) -> LiteLLM:
        if self._llm is None:
            self._llm = LiteLLM(
                model_name=self.model_name,
                temperature=self._temperature,
                api_base=self._openai_base,
                session_id=self._session_id,
                collect_rollout_details=self._collect_rollout_details,
                model_info=self._model_info,
                max_tokens=self._max_tokens,
                **self._extra_llm_kwargs,
            )
        return self._llm

    async def _call_fc(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        """One model call on the complete tool-calling history. Returns the assistant message
        (`content`, `tool_calls`, and `reasoning` when the server separates it) and keeps the
        call's token details when collecting them."""
        try:
            r = await self._ensure_llm().call(
                None, message_history=history, tools=TOOLS, tool_choice="auto"
            )
        except OutputLengthExceededError as exc:
            # Reply cut off at max_tokens: keep the truncated turn (tokens included) instead of
            # failing the trial; a cut-off or missing tool call is handled by the loop like any
            # reply without one.
            if exc.llm_response is None:
                raise
            r = exc.llm_response
        if self._collect_rollout_details:
            if r.prompt_token_ids:
                self._prompt_token_ids.append(r.prompt_token_ids)
            if r.completion_token_ids:
                self._completion_token_ids.append(r.completion_token_ids)
                if r.logprobs:
                    self._logprobs.append(r.logprobs)
        msg: dict[str, Any] = {"content": r.content or "", "tool_calls": r.tool_calls or []}
        if r.reasoning_content:
            msg["reasoning"] = r.reasoning_content
        return msg

    # ------------------------------------------------------------------
    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        try:
            await self._run_loop(instruction, environment, context)
        finally:
            # Also when the loop raised (e.g. ContextLengthExceededError): the trainer trains such a rollout with
            # reward 0 and reads its token details and metadata
            self._record_rollout(context)

    def _record_rollout(self, context: AgentContext) -> None:
        if not self._collect_rollout_details:
            return
        context.metadata = context.metadata or {}
        # One linear segment, as the trainer expects (see the parity trainable agent)
        context.metadata["n_episodes"] = 1
        context.metadata["rollout_turns"] = len(self._completion_token_ids)
        if self._completion_token_ids:
            detail: RolloutDetail = {
                "prompt_token_ids": self._prompt_token_ids,
                "completion_token_ids": self._completion_token_ids,
                "logprobs": self._logprobs,
            }
            context.rollout_details = [detail]

    async def _run_loop(self, instruction: str, environment: BaseEnvironment,
                        context: AgentContext) -> None:
        if not self.model_name:
            raise RuntimeError("model_name is required for MedAgentBenchFCAgent")

        metadata = await self._load_task_metadata(environment)
        verifier_fhir_base = self._normalize_fhir_base(metadata.get("fhir_api_base"))
        original_instruction, task_context = self._parse_harbor_instruction(instruction)
        prompt = FC_PROMPT.format(
            api_base="http://localhost:8080/fhir/",
            functions=json.dumps(self._funcs),
            context=task_context,
            question=original_instruction,
        )

        history: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        # parity-shaped transcript so _build_answer_payload works unchanged
        transcript: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        finish_result: str | None = None

        for _ in range(self._max_rounds):
            msg = await self._call_fc(history)
            tool_calls = msg.get("tool_calls") or []
            assistant_entry: dict[str, Any] = {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": tool_calls,
            }
            if msg.get("reasoning") or msg.get("reasoning_content"):   # a reasoning parser's separate field
                assistant_entry["reasoning"] = msg.get("reasoning") or msg.get("reasoning_content")
            history.append(assistant_entry)

            if not tool_calls:
                history.append({"role": "user",
                                "content": "You must call one of the tools: get, post, finish."})
                continue

            done = False
            for tc in tool_calls:
                fn = (tc.get("function") or {}).get("name")
                try:
                    args = json.loads((tc.get("function") or {}).get("arguments") or "{}")
                except Exception:
                    args = {}
                tc_id = tc.get("id") or "call_0"

                if fn == "get":
                    url_arg = str(args.get("url", ""))
                    resp = await self._send_get_request(environment, url_arg)
                    content = json.dumps(resp)
                    transcript.append({"role": "agent", "content": f"GET {url_arg}"})
                    transcript.append({"role": "user", "content": content})
                elif fn == "post":
                    url_arg = str(args.get("url", ""))
                    payload = args.get("payload")
                    # mirror parity semantics: accumulate + ACK (verifier replays)
                    transcript.append({
                        "role": "agent",
                        "content": "POST " + url_arg + "\n" + json.dumps(payload),
                    })
                    transcript.append({"role": "user", "content": DEFAULT_POST_ACK})
                    content = DEFAULT_POST_ACK
                elif fn == "finish":
                    finish_result = str(args.get("result", ""))
                    content = "Task finished."
                    done = True
                else:
                    content = f"Unknown tool: {fn}"
                history.append({"role": "tool", "tool_call_id": tc_id,
                                "content": content})
                if done:
                    break
            if done:
                break

        payload = self._build_answer_payload(
            finish_result=finish_result,
            transcript=transcript,
            verifier_fhir_base=verifier_fhir_base,
        )
        await self._write_answer_json(environment, payload)

        context.metadata = context.metadata or {}
        context.metadata["harness"] = "fc"
        # raw FC history for trace conversion / privilege bank
        try:
            if self.logs_dir is not None:
                from pathlib import Path
                p = Path(self.logs_dir) / "fc_history.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(history))
        except Exception:  # noqa: BLE001 — logging must never kill the trial
            pass
