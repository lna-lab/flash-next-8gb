#!/usr/bin/env python3
"""Interactive chat, streaming to the terminal. MIT — Lna-Lab, 2026.

    python chat.py -m ./Qwen3.8-Flash-Next-exl3-3.05bpw

Ctrl-C to interrupt a reply, Ctrl-D to leave. `/reset` clears the conversation.
"""
import argparse, sys
import common


def main():
    p = argparse.ArgumentParser(description="Flash-Next on 8 GB — chat")
    common.add_common_args(p)
    p.add_argument("--system", default="You are a helpful assistant.")
    p.add_argument("--max-new", type=int, default=1024)
    args = p.parse_args()

    model, config, cache, tokenizer = common.load(args)
    from exllamav3.generator import Job
    gen = common.make_generator(model, cache, tokenizer)
    messages = [{"role": "system", "content": args.system}]
    print("ready — Ctrl-D to quit, /reset to start over\n", flush=True)

    while True:
        try:
            line = input("you > ").strip()
        except EOFError:
            print()
            break
        if not line:
            continue
        if line == "/reset":
            messages = [{"role": "system", "content": args.system}]
            print("(cleared)\n")
            continue

        messages.append({"role": "user", "content": line})
        prompt = common.render_chat(args.model_dir, messages)
        ids = tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True)
        job = Job(input_ids=ids, max_new_tokens=args.max_new,
                  stop_conditions=common.stop_conditions(tokenizer))
        gen.enqueue(job)

        print("yuki> " if args.system.startswith("You are Yuki") else "bot > ", end="", flush=True)
        reply = []
        try:
            while gen.num_remaining_jobs():
                for r in gen.iterate():
                    chunk = r.get("text", "")
                    if chunk:
                        reply.append(chunk)
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
        except KeyboardInterrupt:
            gen.clear_queue()
            print("\n(interrupted)")
        print("\n", flush=True)
        _, answer = common.split_reasoning("".join(reply))
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
