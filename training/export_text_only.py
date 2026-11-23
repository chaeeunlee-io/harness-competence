"""Export the language model of a Gemma 3 checkpoint as a text-only model.

Gemma 3 4B is multimodal: its checkpoint (`Gemma3ForConditionalGeneration`) holds a
SigLIP vision tower and a projector next to the language model. Nothing in this
repository uses images, and GRPO cannot start from the multimodal checkpoint: SkyRL
syncs the trainer's weights into vLLM by parameter name, and the two name the vision
tower differently, so the first sync fails. The text-only model (`Gemma3ForCausalLM`)
has only the language model, which both sides name the same way.

The language-model weights are copied unchanged, so text behaviour is identical; only
the unused modules are dropped. The tokenizer, chat template and generation settings
are copied from the source. The generation settings must be the source's own: the
ones `save_pretrained` would write for the text-only config drop `<end_of_turn>` from
the stop tokens, and the model then never stops at a turn boundary.

The reported Gemma GRPO run started from such an export of the 3H SFT checkpoint.
`exp grpo` makes it the first time it is needed, for models whose `grpo_init` is
`text-only` in `configs/models.yaml`. Directly:

    python training/export_text_only.py \
        --src runs/checkpoints/medagentbench/gemma-3-4b-it/3h/final \
        --out runs/checkpoints/medagentbench/gemma-3-4b-it/3h/final-text-only
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM, Gemma3ForConditionalGeneration

SOURCE_ARCHITECTURE = "Gemma3ForConditionalGeneration"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="a Gemma 3 checkpoint folder (the SFT output's final/)")
    ap.add_argument("--out", required=True, help="where to write the text-only model; must not exist yet")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)

    if out.exists():
        raise SystemExit(f"{out} already exists; remove it to export again")
    print(f"loading {src}")
    multimodal = Gemma3ForConditionalGeneration.from_pretrained(src, dtype=torch.bfloat16, device_map="cpu")
    if multimodal.config.architectures != [SOURCE_ARCHITECTURE]:
        raise SystemExit(f"{src} is {multimodal.config.architectures}, not {SOURCE_ARCHITECTURE}")

    config = multimodal.config.text_config
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        setattr(config, key, getattr(multimodal.config, key))
    text_only = Gemma3ForCausalLM(config).to(torch.bfloat16)
    text_only.model.load_state_dict(multimodal.model.language_model.state_dict(), strict=True)
    if multimodal.config.tie_word_embeddings:
        text_only.tie_weights()
    else:
        text_only.lm_head.load_state_dict(multimodal.lm_head.state_dict(), strict=True)

    tokenizer = AutoTokenizer.from_pretrained(src)
    check_same_predictions(multimodal, text_only, tokenizer)

    print(f"writing {out}")
    text_only.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    shutil.copy2(src / "generation_config.json", out / "generation_config.json")
    print(f"done: {out} ({text_only.config.architectures[0]})")
    return 0


@torch.no_grad()
def check_same_predictions(multimodal, text_only, tokenizer) -> None:
    """Both models must predict the same next tokens for a short prompt."""
    messages = [{"role": "user", "content": "List three prime numbers."}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    a = multimodal(input_ids=ids).logits[0]
    b = text_only(input_ids=ids).logits[0]
    if not torch.equal(a.argmax(-1), b.argmax(-1)):
        raise SystemExit("the text-only model predicts different tokens from the source; not writing it")
    print(f"check: same next-token predictions at every position (max |logit diff| {(a - b).abs().max():.3g})")


if __name__ == "__main__":
    raise SystemExit(main())
