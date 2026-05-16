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

Other models on this rig:

| Model | File pattern | ~Size | Speed (gen) | When to use |
|---|---|---|---|---|
| Qwen3-30B-A3B Q4_K_M | `Qwen3-30B-A3B-Q4_K_M.gguf` | 18 GB | ~25 tok/s | Baseline MoE, fast |
| Qwen3.6-35B-A3B UD-Q4_K_XL | `*UD-Q4_K_XL*` | 21 GB | ~15-25 tok/s | Daily driver |
| Qwen3.6-27B-MTP UD-Q4_K_XL | `*UD-Q4_K_XL*` | 18 GB | ~3-6 tok/s | Hard reasoning, dense + speculative decode |

Dense models pay a heavy bandwidth tax on an APU — only reach for them when the quality jump justifies a ~5× speed hit.

## 5. Run it

```bash
HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  ~/dev/ai/llama.cpp/build/bin/llama-server \
  -m ~/models/qwen3.6-35b-a3b/<file>.gguf \
  --host 127.0.0.1 --port 8080 \
  -ngl 999 \
  -c 8192 \
  --flash-attn auto \
  --jinja
```

| Flag | Why |
|---|---|
| `HSA_OVERRIDE_GFX_VERSION=11.5.1` | Makes rocBLAS use gfx1151 kernels (see §2) |
| `-ngl 999` | Offload all layers to iGPU — no penalty under unified memory |
| `-c 8192` | Context window; Qwen3 trained to 40960, bump if RAM allows |
| `--flash-attn auto` | Enable flash-attention where supported |
| `--jinja` | Required for Qwen3's tool-calling chat template |

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

## 6. Real numbers (Qwen3-30B-A3B Q4_K_M, this rig)

```
Prefill   : 64.7 tok/s
Generate  : 24.7 tok/s
Load time : ~4 sec
```

These match the theoretical ceiling for a 3B-active MoE on LPDDR5X bandwidth pretty closely. Dense ~30B at Q4 will be roughly 3-6 tok/s for the same reason.

## 7. Quick mental model: why MoE wins on this hardware

This APU is **memory-bandwidth-bound** for inference. Each generated token requires reading the weights touched by that token from RAM.

- **Dense 30B Q4**: ~16 GB read per token → theoretical ceiling ~7 tok/s
- **30B-A3B MoE Q4**: ~1.8 GB read per token (only 3B active params) → theoretical ceiling ~60 tok/s

Pick MoE models with small active-param counts. The 3B-active variants are the sweet spot.

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
