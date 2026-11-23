"""vLLM reasoning parser `explicit_think`: a reply is reasoning only inside a `<think>` block the model opened itself.

Our Qwen3 chat template (training/templates/qwen3_acc_thinking.jinja2) ends the prompt with `<|im_start|>assistant\n`,
like Qwen3's own: the model decides whether to open `<think>`. vLLM's `deepseek_r1` parser assumes the prompt opened
it (DeepSeek-R1's convention), so it files a reply without `</think>` entirely as reasoning: a MedAgentBench fc tool
call or a parity `GET ...` made without thinking reached the agent as an empty reply (10 Nov 2026: 43% of fc turns of
the 3H checkpoint's evaluation). vLLM's `qwen3` parser rescues a reply that starts with a tool call, but not a text
action. This parser follows the template:
  - the reply opens `<think>`: reasoning up to `</think>`, content after it (none if it never closes: cut off);
  - the reply does not open `<think>`: all of it is content.
Streaming follows the same rule (vLLM's base thinking parser already treats a reply without the start token as
content). Load with `--reasoning-parser-plugin <this file> --reasoning-parser explicit_think`.
"""
from collections.abc import Sequence

from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser


@ReasoningParserManager.register_module("explicit_think")
class ExplicitThinkReasoningParser(BaseThinkingReasoningParser):
    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def extract_reasoning(self, model_output: str, request) -> tuple[str | None, str | None]:
        head, start, rest = model_output.partition(self.start_token)
        if not start or head.strip():          # no <think> opened at the start of the reply: all content
            return None, model_output
        reasoning, end, content = rest.partition(self.end_token)
        if not end:                             # opened, never closed (cut off): all reasoning
            return reasoning, None
        return reasoning, content or None

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # The last think marker decides; with none, no reasoning was opened, so it is not pending
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == self.start_token_id:
                return False
            if input_ids[i] == self.end_token_id:
                return True
        return True

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self.start_token_id not in input_ids:
            return input_ids
        if self.end_token_id not in input_ids:
            return []
        return input_ids[input_ids.index(self.end_token_id) + 1:]
