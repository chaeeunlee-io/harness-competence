"""Tests for how the MedAgentBench parity agent reads a reply that opens with a reasoning block.

Run in the trainer environment (the agent imports harbor):
  uv run --project third_party/SkyRL --extra harbor --no-sync python tests/test_parity_reasoning.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.medagentbench.harnesses.medagentbench_parity_agent import split_leading_reasoning  # noqa: E402

GET = "GET http://localhost:8080/fhir/Patient?identifier=S0611496"


def test_reply_with_reasoning_is_read_after_the_block():
    # a Gemma SFT reply as served without a reasoning parser (from a one-step test run)
    reasoning, rest = split_leading_reasoning(f"<think>\nI need the birthdate first.\n</think>\n\n{GET}")
    assert reasoning == "I need the birthdate first."
    assert rest.strip() == GET


def test_reply_without_reasoning_is_unchanged():
    # Qwen3 under deepseek_r1, and base Gemma: content has no block
    assert split_leading_reasoning(GET) == (None, GET)


def test_unclosed_block_is_left_alone():
    # cut off mid-thought: no action, as under the reasoning parser
    text = "<think>\nStill thinking about GET http://localhost:8080/fhir/Patient"
    assert split_leading_reasoning(text) == (None, text)


def test_only_a_leading_block_is_removed():
    text = f"{GET}\n<think>later</think>"
    assert split_leading_reasoning(text) == (None, text)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
