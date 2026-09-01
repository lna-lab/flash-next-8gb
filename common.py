"""Shared loading and prompt helpers for the 8 GB scripts. MIT — Lna-Lab, 2026."""
import argparse, os


def add_common_args(parser):
    """The flags every script here takes. `-mcl` is the one that makes it fit."""
    from exllamav3 import model_init
    model_init.add_args(parser, add_draft_model_args=True)
    return parser


def load(args):
    """Load the model with routed experts on the CPU. Returns (model, config, cache, tokenizer)."""
    from exllamav3 import model_init
    if not getattr(args, "moe_cpu_offload", 0):
        args.moe_cpu_offload = 64          # every MoE layer's experts to system RAM
    res = model_init.init(args)
    return res[0], res[1], res[2], res[3]


def make_generator(model, cache, tokenizer):
    from exllamav3.generator import Generator
    return Generator(model=model, cache=cache, tokenizer=tokenizer)


def _fallback_prompt(messages):
    out = []
    for m in messages:
        out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    out.append("<|im_start|>assistant\n")
    return "\n".join(out)


def render_chat(model_dir, messages):
    """Render messages with the model's own chat template when we can, else ChatML."""
    path = os.path.join(model_dir, "chat_template.jinja")
    if not os.path.exists(path):
        return _fallback_prompt(messages)
    try:
        from jinja2 import Template
        tpl = Template(open(path, encoding="utf-8").read())
        return tpl.render(messages=messages, add_generation_prompt=True)
    except Exception:
        return _fallback_prompt(messages)


def stop_conditions(tokenizer):
    """Where a reply ends. Without these the model runs on into <|endoftext|> and noise."""
    stops = ["<|im_end|>", "<|endoftext|>"]
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        stops.append(eos)
    return stops


def split_reasoning(text):
    """Flash-Next thinks before it answers. Return (reasoning, answer)."""
    for marker in ("</think>", "</thinking>"):
        if marker in text:
            head, _, tail = text.partition(marker)
            return head.strip(), tail.strip()
    return "", text.strip()
