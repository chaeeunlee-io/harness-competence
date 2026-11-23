"""Tests for exp/controltables.py on synthetic job folders. Run: python tests/test_controltables.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp import controltables as CT  # noqa: E402
from test_seedtables import make_job  # noqa: E402


def test_filter_derived_optional():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        make_job(d, "task", {"task2_1": [0] * 4, "task5_1": [1] * 4})            # rehearsed 0, never 100
        make_job(d, "comb", {"task2_1": [1] * 4, "task5_1": [1, 1, 0, 0]})       # rehearsed 100, never 50
        spec = d / "spec.yaml"
        spec.write_text(f"""
roots: {{j: {d}}}
tables:
  - id: H
    title: t
    benchmark: medagentbench
    optional: true
    columns:
      - {{id: reh, label: r, tasks_re: '^task(2|3|4|8)_'}}
      - {{id: nev, label: n, tasks_re: '^task(1|5|6|7|9|10)_'}}
      - {{id: gap, label: g, derived: "nev - reh", format: pts}}
    rows:
      - {{id: task, label: task, runs: {{a: "j:task", b: "j:missing-seed"}}}}
      - {{id: comb, label: comb, runs: {{a: "j:comb"}}}}
      - {{id: gain, label: gain, derived: "comb - task", format: pts}}
""")
        rendered, cells, manifest = CT.build(spec)
        t = rendered[0]
        assert "| task | 0.0 ± 0.0 | 100.0 ± 0.0 | +100.0 |" in t, t
        assert "| comb | 100.0 ± 0.0 | 50.0 ± 0.0 | -50.0 |" in t, t
        assert "| gain | +100.0 | -50.0 | -150.0 |" in t, t
        assert all(m["source"] == "local" for m in manifest)          # the missing optional seed is left out
        assert len(manifest) == 4




def test_pool_validity_rules():
    from exp import poolstats as P
    t = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "a"},
         {"role": "user", "content": "Previous response had parsing errors:\nERROR"}, {"role": "assistant", "content": "b"},
         {"role": "user", "content": "Previous response had warnings: AUTO-CORRECTED"}, {"role": "assistant", "content": "c"}]
    assert P.turn_validity(t, "terminus") == [False, True, True]
    f = [{"role": "assistant", "content": "<think>x</think>", "tool_calls": [{"name": "ls"}]},     # structured call
         {"role": "assistant", "content": "<think>x</think>\nDone."},                              # plain text
         {"role": "assistant", "content": "<think>x</think>\n<tool_call>{bad</tool_call>"},        # unparsed call
         {"role": "assistant", "content": "<think>never closed <tool_call>{}</tool_call>"},        # unclosed reasoning
         {"role": "assistant", "content": "<think>x</think>\n"}]                                   # empty
    assert P.turn_validity(f, "fc") == [True, True, False, False, False]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
