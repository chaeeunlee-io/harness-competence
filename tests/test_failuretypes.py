"""Tests for exp/failuretypes.py: the rule-based failure-type labels.

Run: python -m pytest tests/test_failuretypes.py   (or: python tests/test_failuretypes.py)
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp import failuretypes as F  # noqa: E402


def _bfcl_trial(root: Path, messages: list, error_type: str = "multi_turn:empty_turn_model_response") -> Path:
    trial = root / "task__abc"
    (trial / "agent").mkdir(parents=True)
    (trial / "verifier").mkdir()
    (trial / "result.json").write_text(json.dumps({"exception_info": None}))
    (trial / "verifier" / "reward.txt").write_text("0")
    (trial / "verifier" / "checker_detail.json").write_text(json.dumps({"error_type": error_type}))
    (trial / "agent" / "mt_transcript.json").write_text(json.dumps({"messages": messages}))
    return trial


def _label(messages: list, **kw) -> tuple[str, str]:
    with tempfile.TemporaryDirectory() as d:
        return F.label(F.signals(_bfcl_trial(Path(d), messages, **kw)))


def test_call_written_as_text_is_an_invalid_call():
    after = [{"role": "user", "content": "book it"},
             {"role": "assistant", "content": "<think>plan</think><tool_call>book()</tool_call>"}]
    assert _label(after) == ("interface", "invalid")


def test_call_written_inside_the_reasoning_is_an_invalid_call_too():
    inside = [{"role": "user", "content": "book it"},
              {"role": "assistant", "content": "<think>I will call <tool_call>book()</tool_call></think>"}]
    assert _label(inside) == ("interface", "invalid")


def test_turn_left_without_any_action_is_a_termination_failure():
    silent = [{"role": "user", "content": "book it"}, {"role": "assistant", "content": "<think>hm</think>"}]
    assert _label(silent) == ("interface", "termination")


def test_well_formed_calls_with_a_wrong_result_are_a_task_failure():
    native = [{"role": "user", "content": "book it"},
              {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "book", "arguments": "{}"}}]}]
    assert _label(native, error_type="multi_turn:instance_state_mismatch") == ("task", "")


def test_terminus_delivered_wrong_answer_is_a_task_failure_under_the_delivery_rule():
    looped = {"kind": "t2", "turns": 10, "reprompts": 8, "foreign": 0, "last_reprompt": True, "timeout": True, "no_answer": False}
    assert F.label(looped, "delivery") == ("task", "")
    assert F.label(looped, "trace") == ("interface", "invalid")
    assert F.label({**looped, "no_answer": True}, "delivery") == ("interface", "invalid")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
