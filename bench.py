#!/usr/bin/env python3
"""Reproduce our numbers: peak VRAM, decode speed in three languages. MIT — Lna-Lab, 2026.

    python bench.py -m ./Qwen3.8-Flash-Next-exl3-3.05bpw

RAM is deliberately not reported here: ExLlamaV3's CPU offload runs in worker processes, so this
process's RSS does not see it. Measure it from outside, e.g.
    systemd-run --user --scope -p MemoryMax=100G python bench.py -m ...
then read memory.peak from the scope's cgroup while it runs.
"""
import argparse, json, subprocess, time
import common


def gpu_mem():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return {r.split(",")[0].strip(): int(r.split(",")[1]) for r in out.strip().splitlines()}


PROMPTS = {
    "code": "Write a Python function that merges two sorted lists into one sorted list.",
    "en":   "Explain in three sentences why tidal pools host unusual biodiversity.",
    "ja":   "秋の海辺の朝を三文で描写してください。",
}


def main():
    p = argparse.ArgumentParser(description="Flash-Next on 8 GB — benchmark")
    common.add_common_args(p)
    p.add_argument("--ntok", type=int, default=128, help="tokens to generate per prompt")
    args = p.parse_args()

    before = gpu_mem()
    t0 = time.time()
    model, config, cache, tokenizer = common.load(args)
    print(f"load: {time.time()-t0:.1f} s", flush=True)

    from exllamav3.generator import Job
    gen = common.make_generator(model, cache, tokenizer)
    gen.generate("Hello", max_new_tokens=8)          # warm the kernels

    after = gpu_mem()
    used = {k: after[k] - before.get(k, 0) for k in after if after[k] - before.get(k, 0) > 100}
    print(f"VRAM used by this process: {json.dumps(used)} MiB", flush=True)

    for name, prompt in PROMPTS.items():
        ids = tokenizer.encode(prompt, add_bos=False)
        job = Job(input_ids=ids, max_new_tokens=args.ntok, stop_conditions=[])
        gen.enqueue(job)
        t0 = time.time()
        while gen.num_remaining_jobs():
            gen.iterate()
        dt = time.time() - t0
        n = getattr(job, "new_tokens", args.ntok)
        print(f"{name:5} {n:4d} tokens  {n/dt:6.2f} tok/s   ITL {1000*dt/n:5.1f} ms", flush=True)

    print(f"VRAM at end: {json.dumps(gpu_mem())} MiB", flush=True)


if __name__ == "__main__":
    main()
