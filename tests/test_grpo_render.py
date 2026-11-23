"""Tests for the GRPO command exp/render.py writes, and for the chat template Gemma's GRPO rollouts use.

Run: python -m pytest tests/test_grpo_render.py   (or: python tests/test_grpo_render.py)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jinja2  # noqa: E402
from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: E402

from exp import config as C  # noqa: E402
from exp import render as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "training" / "templates"
GEMMA_STOCK = ROOT / "runs" / "serve" / "gemma-3-4b-it-base-template.jinja"   # written by `exp serve` (chat_template: base)
ENGINE = "generator.inference_engine.engine_init_kwargs."


def overrides(model: str, rollout: str = "3h") -> dict:
    """The trainer overrides (key=value arguments) of a MedAgentBench GRPO command."""
    argv, _, _ = R.grpo(C.benchmark("medagentbench"), model, rollout, placeholders=True, backend="eidf")
    return dict(a.split("=", 1) for a in argv if "=" in a and not a.startswith("-"))


def test_gemma_3h_rolls_out_all_three_harnesses_with_the_unified_template():
    ov = overrides("gemma-3-4b-it")
    assert ov["xh.harnesses"] == "['parity','t2','fc']"
    assert ov[ENGINE + "chat_template"] == str(TEMPLATES / "gemma3_rl_unified.jinja2")
    assert ov[ENGINE + "tool_call_parser"] == "hermes"
    assert ov[ENGINE + "enable_auto_tool_choice"] == "true"
    assert ENGINE + "reasoning_parser" not in ov          # Gemma has no reasoning block
    assert ov["harbor_trial_config.agent.name"] == "null"  # else every rollout runs Terminus-2


def test_qwen_3h_is_unchanged():
    ov = overrides("qwen3-4b")
    assert ov[ENGINE + "chat_template"] == str(TEMPLATES / "qwen3_acc_thinking.jinja2")
    assert ov[ENGINE + "reasoning_parser"] == "deepseek_r1"
    assert ov[ENGINE + "tool_call_parser"] == "hermes"


def test_gemma_evaluation_serving_is_unchanged():
    b = C.benchmark("medagentbench")
    assert R.serve_settings(b, "gemma-3-4b-it", "fc")["chat_template"] == "training/templates/gemma3_hermes_fc.jinja2"
    assert R.serve_settings(b, "gemma-3-4b-it", "t2")["chat_template"] == "training/templates/gemma3_merge_tolerant.jinja2"
    assert R.serve_settings(b, "gemma-3-4b-it", "t2")["tool_call_parser"] is None
    assert R.serve_settings(b, "gemma-3-4b-it", "parity")["chat_template"] == "base"


# --------------------------------------------------------------------------- the unified template
def render(template: Path, messages: list[dict]) -> str:
    """Render a chat template the way transformers does (sandboxed, with raise_exception)."""
    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
    return env.from_string(template.read_text()).render(messages=messages, bos_token="<bos>", add_generation_prompt=True)


ALTERNATING = [{"role": "system", "content": "You are a clinical assistant."},
               {"role": "user", "content": "Find the patient's latest glucose."},
               {"role": "assistant", "content": "GET /Observation?code=GLU"},
               {"role": "user", "content": "{\"value\": 108}"}]
# Terminus-2's summarisation hand-off sends consecutive assistant (and user) messages
NON_ALTERNATING = ALTERNATING + [{"role": "user", "content": "Context summary follows."},
                                 {"role": "assistant", "content": "Summary: glucose 108."},
                                 {"role": "assistant", "content": "Continuing."},
                                 {"role": "user", "content": "Go on."}]
TOOL_CALLS = [{"role": "user", "content": "Find the patient's latest glucose."},
              {"role": "assistant", "content": "", "tool_calls": [
                  {"type": "function", "function": {"name": "fhir_get", "arguments": {"path": "Observation?code=GLU"}}}]},
              {"role": "tool", "content": "{\"value\": 108}"},
              {"role": "assistant", "content": "", "tool_calls": [
                  {"type": "function", "function": {"name": "finish", "arguments": "{\"answer\": [108]}"}}]},
              {"role": "tool", "content": "ok"}]


def test_unified_matches_gemma_on_alternating_conversations():
    if not GEMMA_STOCK.exists():
        print(f"skipped: {GEMMA_STOCK} not written yet (exp serve with chat_template: base writes it)")
        return
    assert render(TEMPLATES / "gemma3_rl_unified.jinja2", ALTERNATING) == render(GEMMA_STOCK, ALTERNATING)


def test_unified_accepts_terminus2_handoffs():
    out = render(TEMPLATES / "gemma3_rl_unified.jinja2", NON_ALTERNATING)
    assert "Summary: glucose 108." in out and "Continuing." in out
    if GEMMA_STOCK.exists():   # the stock template rejects the same conversation: the reason for the unified one
        try:
            render(GEMMA_STOCK, NON_ALTERNATING)
        except jinja2.exceptions.TemplateError:
            pass
        else:
            raise AssertionError("expected Gemma's stock template to reject non-alternating roles")


def test_unified_matches_hermes_fc_on_tool_calls():
    unified = render(TEMPLATES / "gemma3_rl_unified.jinja2", TOOL_CALLS)
    assert unified == render(TEMPLATES / "gemma3_hermes_fc.jinja2", TOOL_CALLS)
    assert '<tool_call>\n{"name": "fhir_get", "arguments": {"path": "Observation?code=GLU"}}\n</tool_call>' in unified


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
