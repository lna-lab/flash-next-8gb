# A 177B model on an 8 GB laptop GPU

Qwen3.8-Flash-Next is a 177-billion-parameter model. This repository is the short path to running it
on a machine with **8 GB of VRAM and 64 GB of RAM** — a gaming laptop, not a server.

It works because of how the model is shaped, not because of a trick:

- **51 B of its parameters are an n-gram lookup table.** The model hashes the last three tokens and
  reads about **2.7 KB per token** from it, at a deterministic address. That is a rounding error
  against a 30 ms token budget, so the table can stay on disk. It never needs to be in RAM.
- **The routed experts (~6 B active per token) go in system RAM**, where ExLlamaV3 will run them on
  the CPU for you (`--moe_cpu_offload`).
- **Only the dense path lives in VRAM** — attention, norms, the output head. That is the 6.6 GB.

## Measured

On our bench (RTX Pro 2000 Blackwell 16 GB, one card, experts on CPU, `-cs 4096`), with
turboderp's `3.05bpw` quant:

| | |
|---|---:|
| **VRAM, peak** | **6,779 MiB (6.62 GiB)** |
| **RAM, peak (cgroup)** | **47.8 GiB** |
| decode, code | 34.1 tok/s |
| decode, English | 35.1 tok/s |
| decode, Japanese | 35.5 tok/s |
| perplexity (wikitext-2, 2048 × 20) | 3.6209 |

An 8 GB card has ~7.7 GB usable headless, so 6.62 GiB fits with room. Attach a monitor and you lose
0.3–0.8 GB of that — still fine, but that is where the margin goes.

**What we have not measured, and you should:** our CPU is a 64-core Threadripper. Yours is not. The
experts run on the CPU, so your decode speed is set by your **memory bandwidth**, not your GPU. The
arithmetic: ~6 B active parameters at 3.05 bpw ≈ **2.3 GB read per token**. Dual-channel DDR5-5600
gives ~89 GB/s, which puts the ceiling near **38 tok/s** — close to what we measured, so a laptop
may land in the same range. That is a prediction, not a result. Run `bench.py` and tell us.

## Requirements

- NVIDIA GPU, 8 GB or more, recent driver
- 64 GB RAM (48 GB is the working set; leave the OS some room)
- **~85 GB of free NVMe.** The table is read from disk on every token — a SATA SSD will hurt and a
  network share (SMB/NFS) cannot be memory-mapped at all
- Python 3.10+, PyTorch with CUDA

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install exllamav3==1.4.5
```

## Get the model

turboderp's quants of the official checkpoint. **3.05bpw is the one for an 8 GB card:**

```bash
pip install huggingface_hub
HF_HUB_DISABLE_XET=1 hf download turboderp/Qwen3.8-Flash-Next-exl3 \
  --revision 3.05bpw_h5_ng5 --local-dir ./Qwen3.8-Flash-Next-exl3-3.05bpw
```

| quant | size | fits |
|---|---:|---|
| [`2.05bpw`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/2.05bpw_h4_ng4) | ~60 GB | smaller RAM, lower quality |
| **[`3.05bpw`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/3.05bpw_h5_ng5)** | **85 GB** | **8 GB VRAM + 64 GB RAM — start here** |
| [`4.05bpw`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/4.05bpw_h6_ng6) | ~110 GB | better quality, needs ~64 GB RAM more |

## Run

Interactive chat:

```bash
python chat.py -m ./Qwen3.8-Flash-Next-exl3-3.05bpw
```

An OpenAI-compatible server, so anything that speaks that API can use it:

```bash
python serve.py -m ./Qwen3.8-Flash-Next-exl3-3.05bpw --port 8080
# curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
#   -d '{"model":"flash-next","messages":[{"role":"user","content":"hello"}]}'
```

The model thinks before it answers. `serve.py` separates that out: the thinking arrives as
`reasoning_content` (streamed as `reasoning_content` deltas) and the answer as `content`, so a
client that does not know about reasoning still shows a clean reply.

Reproduce our numbers:

```bash
python bench.py -m ./Qwen3.8-Flash-Next-exl3-3.05bpw
```

All three take the same flags. The ones that matter:

| flag | why |
|---|---|
| `-mcl 64` | routed experts on the CPU. **This is the flag that makes it fit.** Without it ExLlamaV3 wants ~52 GB of VRAM |
| `-cs 4096` | KV cache size in tokens. Raise it for longer context, at the cost of VRAM |
| *(no `-ngr`)* | leaving `--ngram_ram` **off** is what keeps the table on disk. Turn it on only if you have ~35 GB of RAM to spare and want to skip the reads |

## Notes from building this

- **The n-gram table is precision-sensitive.** In a GGUF build of the same model, dropping the table
  from `q8_0` to `q5_0` moved perplexity from 3.53 to 5.20 — 19 GB saved, 47 % of the quality gone.
  turboderp's EXL3 quants keep the table at a higher bitrate than the body for the same reason; if
  you quantize your own, do not economise there.
- **ExLlamaV3's CPU offload spawns worker processes.** Any script you write around it needs an
  `if __name__ == "__main__":` guard, or you get a confusing `ConnectionResetError` at load.
- **Measure RAM with a cgroup, not `ru_maxrss`.** The offload workers are separate processes and the
  parent's RSS does not see them. We got this wrong once.
- **Check your CPU's instruction set.** ExLlamaV3 picks a kernel at load and says so:
  `CPU MoE worker started: 48 layers, avx512-vbmi, 64 threads`. Ours had AVX-512; many consumer
  laptop CPUs do not (Intel disabled it on 12th gen and later; AMD Zen 4/5 has it). If yours falls
  back to AVX2, expect the experts — and therefore the whole model — to be slower. Read that line.
- **More free RAM makes it faster, not just safer.** Spare memory becomes page cache for the table,
  so the reads stop happening. The trade-off is smooth; there is no cliff.

## License and credit

The code here is MIT. The model is Qwen's, under the
[Qwen Community License](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE); the
quants are [turboderp's](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3); the engine is
[ExLlamaV3](https://github.com/turboderp-org/exllamav3). We wrote the scripts and took the
measurements.

Built at [Lna-Lab](https://github.com/lna-lab). Every number above came off one machine — a second
machine is more informative than a second opinion, so if yours disagrees, open an issue.

---

<details>
<summary>日本語</summary>

**177B のモデルを、VRAM 8GB のノートPCで動かす**ための最短経路です。

仕組みは小細工ではなく、モデルの形そのものです。177B のうち **51B は n-gram の索引表**で、
1 トークンにつき約 2.7 KB しか読まれません。だから**表はディスクに置いたままで構いません**。
**experts は RAM**（ExLlamaV3 が CPU で回してくれます）、**VRAM に載るのは dense 部だけ = 6.6 GB**。

実測（うちの機械、GPU 1 枚、3.05bpw）: **VRAM 6,779 MiB / RAM 47.8 GiB / 34-35 tok/s / ppl 3.6209**。

⚠️ **未検証**: うちの CPU は 64 コアの Threadripper です。experts は CPU で回るので、
**あなたの速度はメモリ帯域で決まります**。1トークンあたり約 2.3 GB 読むので、DDR5-5600 の
デュアルチャネル（約 89 GB/s）なら上限 38 tok/s 付近——うちの数字に近いはずですが、これは予測です。
`bench.py` を回して教えてください。

要る物: NVIDIA GPU 8GB 以上 ／ RAM 64GB ／ **NVMe の空き 85GB**（表を毎トークン読むので
SATA は厳しく、ネットワーク共有は mmap できないので不可）。

肝は **`-mcl 64`**（experts を CPU へ）と **`-ngr` を付けないこと**（表をディスクに残す）。

</details>
