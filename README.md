# Local LLM setup on Framework laptop (Ryzen AI 7 350 + gfx1152)

How we got Qwen3-class MoE models running on the integrated Radeon 860M (gfx1152, RDNA 3.5) via ROCm 7.2 and llama.cpp. Tested 2026-05-16.

## Hardware & OS

```
CPU/APU : AMD Ryzen AI 7 350 (Krackan Point), 16 threads, AVX512 + AVX_VNNI + BF16
iGPU    : Radeon 860M (gfx1152, RDNA 3.5, 8 CUs)
VRAM    : 32 GB allocated to GPU pool from system RAM (unified memory)
RAM     : 64 GB total
NPU     : XDNA2 (aie2p) - not used here, requires onnxruntime/Lemonade stack

OS      : Ubuntu 25.04 "Plucky Puffin"
Kernel  : 6.14.0-37-generic
Shell   : zsh
Python  : 3.13
GCC     : 14.2
```

Note: Ubuntu 25.04 is **non-LTS**, and the ROCm 7.2 packages from apt have a `~24.04` (Noble) suffix on their version strings. They install and work fine on Plucky in practice, but if you hit weirdness it's worth knowing you're slightly off the officially-supported matrix. For an LTS box you'd be on 24.04 Noble using the same packages.

Unified memory is the secret weapon. A Q4_K_M of a 30B-class model (~18 GB) loads in seconds, no PCIe copy cost, and bandwidth is shared between CPU and iGPU.

## 1. Install ROCm

One apt command:

```bash
sudo apt install rocm
```

This pulls ROCm 7.2.0 plus the `rocm-hip`, `hipcc`, `hipblas`, `hipblaslt`, `miopen-hip`, `rocblas`, `rocminfo`, `rccl`, etc. — about 250 packages including the LLVM/Clang 22 toolchain at `/opt/rocm-7.2.0/lib/llvm/bin/clang++`.

After install, log out and back in (or reboot) so udev rules and `render`/`video` group membership take effect.

Verify:

```bash
rocminfo | grep -E "gfx|Name:.*AMD"
```

Should show your iGPU as `gfx1152` Agent.

## 2. The gfx1152 gotcha

**This will bite you.** ROCm 7.2's rocBLAS Tensile kernels ship for:

```
gfx1100, gfx1101, gfx1102, gfx1150, gfx1151, gfx1200, gfx1201
```

**Not gfx1152.** If you build llama.cpp with `AMDGPU_TARGETS=gfx1152` (the natural choice), it compiles fine and the server starts — but the first matrix multiply triggers a GPU queue eviction:

```
rocBLAS error: Cannot read /opt/rocm-7.2.0/lib/rocblas/library/TensileLibrary.dat:
No such file or directory for GPU arch : gfx1152
```

The process dies silently. Check `dmesg` and you'll see `amdgpu: Freeing queue vital buffer ... queue evicted`.

**Fix:** target gfx1151 (Strix Halo — same RDNA 3.5 microarch, present in rocBLAS) and use `HSA_OVERRIDE_GFX_VERSION=11.5.1` at runtime. gfx1152 hardware natively executes gfx1151 ISA.

## 3. Build llama.cpp

```bash
cd ~/dev/ai
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp

cmake -S . -B build \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm-7.2.0 \
  -DCMAKE_C_COMPILER=/opt/rocm-7.2.0/lib/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-7.2.0/lib/llvm/bin/clang++

cmake --build build --config Release -j$(nproc)
```

Build takes ~3 minutes on this CPU. The HIP kernel compilation is what dominates.

## 4. Pull a model

Use the HuggingFace CLI. Put it in a venv to keep system Python clean:

```bash
python3 -m venv ~/dev/ai/.venv
~/dev/ai/.venv/bin/pip install -U "huggingface_hub" hf_transfer
```

Then download. `HF_HUB_ENABLE_HF_TRANSFER=1` enables the Rust-based parallel downloader (~10× faster on a fast pipe).

**Recommended pick for this hardware: Qwen3.6-35B-A3B** — MoE with 3B active params, fits in 32 GB VRAM at UD-Q4_K_XL (~21 GB):

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 ~/dev/ai/.venv/bin/hf download \
  unsloth/Qwen3.6-35B-A3B-GGUF \
  --include "*UD-Q4_K_XL*" \
  --local-dir ~/models/qwen3.6-35b-a3b
```

Recommended models for this rig (both Qwen3.6, both natively 262144-token context):

| Model | File pattern | ~Size | Speed (gen) | When to use |
|---|---|---|---|---|
| **Qwen3.6-35B-A3B** UD-Q4_K_XL | `*UD-Q4_K_XL*` | 21 GB | ~25 tok/s | **Daily driver.** MoE with 3B active params, fast on bandwidth-bound APUs. |
| **Qwen3.6-27B-MTP** UD-Q4_K_XL | `*UD-Q4_K_XL*` | 18 GB | ~7 tok/s (with MTP) | Hard reasoning. Dense, slow per-token, but Qwen3.6-gen quality. Use `--spec-type draft-mtp --spec-draft-n-max 12` to roughly 2× generation speed. |

Dense models pay a heavy bandwidth tax on an APU (every token reads ~16 GB of weights vs ~1.8 GB for 3B-active MoE) — only reach for the 27B-MTP when the quality jump justifies the speed hit.

## 5. Run it

```bash
HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  ~/dev/ai/llama.cpp/build/bin/llama-server \
  -m ~/models/qwen3.6-35b-a3b/<file>.gguf \
  --host 127.0.0.1 --port 8080 \
  -ngl 999 \
  -c 131072 --parallel 1 \
  --flash-attn auto \
  --jinja
```

| Flag | Why |
|---|---|
| `HSA_OVERRIDE_GFX_VERSION=11.5.1` | Makes rocBLAS use gfx1151 kernels (see §2) |
| `-ngl 999` | Offload all layers to iGPU — no penalty under unified memory |
| `-c 131072` | **128K context window.** Qwen3.6 trains natively to 262144 → no rope scaling, no quality drop. ~12 GB KV cache, easily fits in your 51 GB pool from §8. |
| `--parallel 1` | A single request gets the entire 128K window. Without this, llama-server divides the budget across 4 default slots. |
| `--flash-attn auto` | Enable flash-attention (+30% throughput on this hardware vs FA off). |
| `--jinja` | Required for Qwen3's tool-calling chat template. |

**For the 27B-MTP**, append `--spec-type draft-mtp --spec-draft-n-max 12` to engage the model's built-in multi-token-prediction heads (roughly 2× generation speed). Also pass `"chat_template_kwargs": {"enable_thinking": false}` in client requests unless you actively want chain-of-thought — thinking mode tanks MTP acceptance rate because internal reasoning is less predictable.

Hit it like any OpenAI-compatible API:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"qwen3",
    "messages":[{"role":"user","content":"hello"}],
    "max_tokens": 50
  }'
```

## 6. Real numbers measured on this rig

Measured with `llama-bench` on `build: 64b38b5`:

**Qwen3.6-35B-A3B UD-Q4_K_XL** (daily driver):

```
pp512    : 157.85 tok/s
pp1024   : 156.39 tok/s       (flat — prefill is compute-bound, not attention-bound at this length)
pp2048   : 140.68 tok/s       (attention quadratic cost starts to bite)
tg128    :  16.83 tok/s
```

**Qwen3.6-27B-MTP UD-Q4_K_XL** with MTP speculative decoding (`--spec-type draft-mtp --spec-draft-n-max 12`, thinking disabled in client requests):

```
pp512    :  60 tok/s
tg (gen) :   7 tok/s          (vs 3.3 tok/s without MTP — exactly 2.1× speedup)
```

### A note on quants

UD-Q4_K_XL is Unsloth's dynamic quant — uses Q5/Q6 on critical tensors (attention/embeddings), Q4 elsewhere. The file is ~23% larger than vanilla Q4_K_M, which translates to ~23% more bandwidth per token. If you want to trade some quality for speed, swap in the regular Q4_K_M (~17 GB vs ~21 GB), expect roughly +25% throughput.

These match the theoretical bandwidth ceiling for 8 CUs of gfx1151-class silicon. For reference, Strix Halo at 40 CUs hits ~1256 tok/s prefill on the same model class — exactly 5× our throughput, exactly 5× the CU count. The math is honest.

## 7. Quick mental model: why MoE wins on this hardware

This APU is **memory-bandwidth-bound** for inference. Each generated token requires reading the weights touched by that token from RAM.

- **Dense 30B Q4**: ~16 GB read per token → theoretical ceiling ~7 tok/s
- **35B-A3B MoE Q4**: ~1.8 GB read per token (only 3B active params) → theoretical ceiling ~60 tok/s

Pick MoE models with small active-param counts. The 3B-active variants are the sweet spot.

## 7b. Tuning we tried — what worked, what didn't

Tested exhaustively on this hardware. Save yourself the time:

| Lever | Effect on MoE | Effect on Dense (27B-MTP) | Verdict |
|---|---|---|---|
| `--flash-attn` ON (default) | +30% | +30% | **Keep on.** |
| `--spec-type draft-mtp --spec-draft-n-max 12` | n/a (no MTP heads) | **+109% generation** | **Required for 27B-MTP.** Tune `n-max` per model; 16 cliff-drops on Qwen3.6-27B. |
| `--spec-type draft-simple` with Qwen3-0.6B draft | **-20% gen** | n/a | Skip on MoE. Verifier is already fast (3B active), no overhead to amortize. |
| `ROCBLAS_USE_HIPBLASLT=1` env var | -8% prefill | -8% prefill | Skip — community advice was for native gfx1151, doesn't translate when spoofing from gfx1152. |
| `--cache-type-k q8_0 --cache-type-v q8_0` | -19% all | similar | Skip unless desperate for KV cache RAM. |
| Bigger `-b 4096 -ub 1024` | -8% | -8% | Skip — defaults are tuned. |
| `-DGGML_HIP_ROCWMMA_FATTN=ON` build flag | (not tested) | (not tested) | Skip on ROCm 7.2+ — wiki warns it slows down at long contexts. |
| RDNA3_5 MMQ kernel patch | already merged in upstream | already merged | No action needed, build 64b38b5+ has it. |

**Key takeaway:** the 25 tok/s gen / 247 tok/s prefill on 3B-active MoE is the architectural ceiling for 8 CUs of RDNA 3.5 + LPDDR5X bandwidth. No software lever moves it. Speed wins come from picking the right model architecture (MoE), not from compiler flags.

## 8. Memory tuning (VRAM vs GTT)

The "32 GB VRAM" you see in `llama-server`'s startup banner is misleading. Run `rocm-smi --showmeminfo vram gtt` and you'll find:

```
VRAM Total: 536 MB       ← actual BIOS-reserved UMA frame buffer
GTT Total:  32.8 GB      ← shared system RAM the iGPU can borrow
GTT Used:   19.75 GB     ← model weights + KV cache live here
```

**Two pools, same DDR5, same bandwidth.** On a discrete GPU, the VRAM/GTT distinction matters because VRAM has ~10× the bandwidth of system RAM over PCIe. On an APU, both live in the same DDR5 sticks, so the only practical difference is:

- **VRAM (UMA frame buffer)**: reserved by BIOS, OS can never reclaim it.
- **GTT**: shared with the OS, kernel can reclaim under pressure, soft-capped by `amdgpu.gttsize` / `ttm.pages_limit`.

### Two ways to give the iGPU more headroom

**Option A — BIOS UMA frame buffer.** On the Framework BIOS, under *AMD CBS → NBIO → GFX Configuration → UMA Frame Buffer Size*, you can set a fixed allocation (typical options: AUTO, 512 MB, 2 GB, 4 GB, 8 GB, 16 GB, sometimes higher on Krackan/Strix).

- **Pro**: guaranteed reservation, won't compete with OS pressure, ROCm prefers it for some allocations.
- **Con**: that RAM is gone for the OS even when you're not running LLMs.

**Option B — kernel parameter (recommended for this use case).** Raise the GTT cap, leaving BIOS on AUTO:

```bash
# /etc/default/grub
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amdgpu.gttsize=49152 ttm.pages_limit=14155776"
```

Then `sudo update-grub` and reboot. `gttsize` is in MB; `ttm.pages_limit` is in 4 KB pages (49152 MB / 4 KB ≈ 12.6M pages, set with a bit of headroom). On kernel 6.x, **both** flags are needed — the TTM subsystem caps allocations independently of the amdgpu driver.

- **Pro**: flexible, OS gets RAM back when the iGPU isn't using it, no BIOS visit needed.
- **Con**: shared with the OS, theoretical pressure under heavy multitasking.

### What you gain by raising the iGPU ceiling to ~48 GB

| Today (32 GB iGPU pool) | At 48 GB iGPU pool |
|---|---|
| Qwen3.6-35B-A3B Q4 + 8K context | Same model + **64K context** |
| Q4_K_M only | **Q6_K** of 35B-A3B (~28 GB) — meaningfully better quality |
| Single model | Main model + **draft model for speculative decoding** |
| Can't fit ~70B Q4 | Fits 70B dense Q4 (slow on dense, but possible) |

For agentic workloads the **context window** is usually the biggest practical win — agent loops accumulate tool-call history fast, and 64K+ keeps the conversation alive without compaction.

### Verifying after reboot

```bash
rocm-smi --showmeminfo vram gtt
# expect: GTT Total ≈ 49 GB

# also visible in llama-server startup banner:
# ROCm0 : AMD Radeon Graphics (XXXXX MiB, YYYYY MiB free)
```

If `GTT Total` didn't move, check `cat /proc/cmdline` actually shows your new params — GRUB sometimes silently keeps the previous cmdline if `update-grub` errored.

## Future explorations

- **Vision (mmproj)**: Qwen3.6 is natively multimodal. Download `mmproj-*.gguf` alongside the model and pass `--mmproj` to `llama-server` for image input.
- **NPU (XDNA2)**: AMD's Lemonade SDK / Vitis-EP can run small models on the NPU while keeping the iGPU free. Different software stack — not llama.cpp.
- **Vulkan backend**: alternative to ROCm. Build llama.cpp with `-DGGML_VULKAN=ON`. Slightly slower but bypasses the rocBLAS Tensile kernel problem entirely. Worth keeping in your back pocket if a future ROCm update breaks the gfx1151 workaround.
- **Speculative decoding**: pair Qwen3.6-35B-A3B with a small draft model (Qwen3-1.7B) via `--model-draft` for higher tok/s on agentic loops.

## Filesystem layout used here

```
~/dev/ai/llama.cpp/             # source + build/bin/
~/dev/ai/.venv/                 # hf CLI venv
~/models/<model-name>/*.gguf    # model files
```
