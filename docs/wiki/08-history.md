# 08 — history (chronological discoveries + dead ends)

What's been tried on this box, what worked, what didn't, in roughly the order
it happened. Useful for "have we tried X before?" lookups.

## Phase 1 — bring-up (multiple weeks)

**Goal:** Get a single vLLM endpoint running on one Gaudi 3.

**Hardware/OS layer pain:**
- Tried kernel 6.11 (newer Ubuntu default) — Habana driver build fails. **Pinned to 6.8.0-110-generic.**
- Forgot `iommu=pt intel_iommu=on` — got "Bad pagetable" kernel oopses. **Now in GRUB persistently.**
- Missing `ib_uverbs` autoload → 24 internal NICs didn't come up at boot. **Added to `/etc/modules-load.d/`.**
- Hugepages not set → allocation failed under load. **24640 hugepages now in sysctl.**
- Sapphire Rapids power-save defaults limited throughput. **MSRs pinned via `gaudi-tune.service`.**
- Docker daemon couldn't reach `vault.habana.ai` through corp proxy → couldn't pull images. **Proxy systemd drop-in fixed.**

**vLLM image selection:**
- Started with `vllm-0.17.1-ptupstream` — graph capture bug, needed `--enforce-eager` (slow). 
- Moved to `vllm-0.17.1-ptfork` — graph capture works, `--enforce-eager` removed. **49 tok/s on Qwen 32B-Thinking, single user. Pinned.**

**First env-var fight:**
- `ValidateSyncInputTensors` crash on Qwen graph capture. Tried many env permutations.
- Root cause: recipe cache poisoned. **Fix: `PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false`.** Now baked in.

**First endpoint live:** Qwen3-VL-32B-Thinking-FP8 on Gaudi 0, port 8000.

## Phase 2 — multi-model + perf (1-2 weeks)

- Added 32B-Instruct on Gaudi 1, port 8001. Same image, same flags.
- Added 30B-A3B and 8B-Thinking presets (Gaudi 0, less commonly used).
- Tried `--block-size 128` (Habana docs recommend) and `--kv-cache-dtype fp8_inc` — pending, **not yet in main presets** (see GAP_ANALYSIS.md).
- Added `--reasoning-parser qwen3` for *-thinking* variants to extract `<think>` blocks into `reasoning_content`.

**Dead ends explored:**
- `--enforce-eager` ladder: confirmed 0.17.1-ptfork doesn't need it.
- Tried Open WebUI on port 3000 — works, but needs SQLite patch to add new endpoints (documented in PLAYBOOK.md).

## Phase 3 — Qwen3-VL-235B-A22B (TP=4 + TP=8) (~1 week)

- Downloaded the 235B-A22B-Thinking-FP8 weights (~250 GB).
- Launched TP=4 on Gaudis 4-7. **Worked.**
- Defrag-OOM issues under sustained inference. **Fixes added:**
  - `--gpu-memory-utilization 0.80` (vs 0.9 default)
  - `HABANA_PGM_LRU_MAX=60000`
  - `--restart unless-stopped` for auto-recovery
- TP=8 across all 8 cards also works (port 8006, uses whole box).

## Phase 4 — Gemma 4 31B + Anthropic API (~2 weeks)

**Goal:** Anthropic `/v1/messages` shape on Gaudi 3.

- `RedHatAI/gemma-4-31b-it-FP8-Dynamic` chosen as the FP8 variant.
- Required transformers ≥5.7 (base image had 4.55) → **Patch ①**.
- Crash: `head_size 512 not supported` (Gemma 4's global attention) → **Patch ②**: allow-list 512.
- Crash: `UniformTypeKVCacheSpecs` wrapper not unwrapped → **Patch ③**: detect and unwrap in KV allocator.
- Crash: `'float' object cannot be interpreted as integer` → **Patch ④**: int() coercion.
- Reasoning parser sees empty input → **Patch ⑤**: flip `skip_special_tokens` default.

**Result:** `gaudi-vllm-gemma4:0.19.0` derived image. Both `/v1/chat/completions` and `/v1/messages` work. Same image then powers MiniMax later.

Custom Gemma 4 tool-call parser was started but never finished — **task #49 in the task list still pending**.

## Phase 5 — gpt-oss attempts (failed, ~1 week)

User asked: "run gpt-oss-120b" → "I want full dense BF16" → "try 20b" → "try MXFP4 variant".

**All three tried; all three loaded; all three produce incoherent output.**

Investigated:
- Confirmed `HpuUnquantizedFusedMoEMethod.forward_oot` takes the gpt-oss branch correctly.
- `vllm_gaudi/models/gptoss_mxfp4.py` patches install at import time (verified).
- HBM footprint matches dequantized BF16 (~220 GB on TP=4 for 120b) → dequant ran.
- HPU op `bias_fused_weights` has correct swigluoai defaults (alpha=1.702, limit=7.0).
- Bias plumbed through `VllmMixtureOfExpertsOp._cached_w*_bias_views`.

Toggled, no effect:
- `--enforce-eager`
- `HPU_FUSED_MOE=0`
- `--quantization mxfp4` (errors: "mxfp4 not supported in hpu")
- `--reasoning-parser openai_gptoss` on/off
- TP=4 vs TP=8

**Conclusion:** bug is in Habana's binary HPU MoE kernel or MXFP4 dequant. Not fixable from user-space. Full analysis + upstream-report draft in [docs/GPT-OSS.md](../GPT-OSS.md).

User accepted: "documented, move on."

## Phase 6 — second Gemma 4 instance (Gaudi 3) (~hours)

User wanted Gemma 4 on multiple cards. Added `vllm-gemma4-31b-b` on Gaudi 3, port 8005. Same image, same patches, just different `HABANA_VISIBLE_DEVICES` and `--port`. HF cache shared. **Worked first try.**

## Phase 7 — MiniMax M2 (~hours)

Web-scraped MiniMax model list. `MiniMaxAI/MiniMax-M2` (230B MoE, 128 experts, FP8) chosen.

Discovered vllm-gaudi ships `HpuMiniMaxM2ForCausalLM` (handcrafted, 504 lines) that overrides upstream at import time. Dedicated FP8 GEMM kernel. Uniform attention (no hetero-head-dim drama).

Launched TP=4 on Gaudis 4-7, port 8006. **Worked first try with the Gemma 4 patched image.** No new patches needed.

Verified:
- /v1/chat/completions reasoning + content
- /v1/messages Anthropic shape (thinking + text blocks)
- Parallel tools × 4 different (weather, flights, exchange, hotel — all 4 extracted in one turn)

Throughput: comparable to Qwen 235B-A22B TP=4. ~107 GB/card HBM steady.

## Phase 8 — MiniMax M2.7 (~1 hour cold-load + verification)

User asked: "go M2.7 once M2 works."

M2.7 details (from config.json):
- Same `MiniMaxM2ForCausalLM` arch class → HPU override applies
- 229B total / 10B active
- **256 experts (vs M2's 128), top-8 routing (vs top-4)**
- **3 MTP modules** loaded as layers 62/63/64
- ~230 GB on disk (115 shards in reality; "of-00130" suffix lies)

Adjustments:
- Preset auto-applies `--gpu-memory-utilization 0.85`, `--max-num-seqs 16`, `HABANA_PGM_LRU_MAX=60000` (tighter than M2 because of larger routing state).

**Cold load was 30-40 min** — 256-expert weight processing is CPU-bound and silent. Almost killed the container thinking it was stuck. **Lesson learned: check `docker stats` BLOCK I/O before pulling the plug.**

User then tested quality with L1-Dep prompt → judged not worth the slower cold-load + tighter HBM. Rolled back to M2. M2.7 preset retained for future use.

Saved: `feedback_minimax_m2_preferred.md` memory.

## Phase 9 — operational tooling (this session)

- `bin/launchpad` — interactive picker (replaces "memorize preset names" workflow)
- `scripts/quality-bench.sh` — cross-endpoint quality bench (the L1-Dep test that surfaced M2.7 weakness)
- `docs/index.html` — self-contained dark dashboard for sharing
- `LAUNCH_DEVICES` / `LAUNCH_PORT` env overrides in `vllm-launch`
- TP scaling math section added to `docs/index.html` (why MiniMax can't run TP=6)
- This wiki (you're reading it).

## Active state (snapshot 2026-05-13)

```
ACTIVE
  vllm-gemma4-31b      Gaudi 2  port 8004   ~95 GB    just-restarted, fresh
  vllm-gemma4-31b-b    Gaudi 3  port 8005   ~102 GB   second Gemma instance
  vllm-minimax-m2      Gaudis 4-7  port 8006  ~107 GB each   running

IDLE (cards available for new presets)
  Gaudi 0, Gaudi 1   (Qwen 32B-Thinking + Instruct were stopped)

NOT RUNNING (presets exist but not deployed)
  30b-a3b, 8b-thinking, 235b-tp4, 235b-tp8, minimax-m2.7, gpt-oss-120b
```

## Outstanding tasks (carry-overs)

- **Task #49:** Write a custom Gemma 4 tool-call parser for vLLM. Started long ago, never completed. The current `gemma4` tool parser works for the basic case; custom parser would handle some edge cases more cleanly.
- Performance roadmap (Phases 1-6 in `docs/GAP_ANALYSIS.md`) — pushing toward "90% of NVIDIA" tok/s. Several knobs still untried: `--block-size 128`, `--kv-cache-dtype fp8_inc`, `PT_HPU_RECIPE_CACHE_CONFIG`, `--enable-prefix-caching`, speculative decoding, real-bucket pre-warm.

## What we've definitively decided AGAINST

- **gpt-oss family** on this stack — broken below user-space, documented, moved on.
- **MiniMax M2.7 as default** — user tested, prefers M2.
- **Newer kernels than 6.8** — Habana driver build fails.
- **Old `vllm-0.17.1-ptupstream`** — replaced by ptfork due to graph capture bug.

## Things to revisit when conditions change

- M2.7 sampling: user used vLLM defaults (temp 0.7). MiniMax card recommends temp=1.0, top_p=0.95, top_k=40. Worth retesting M2.7 with those before final verdict.
- TP=6 on MiniMax: requires a future MiniMax variant with 16 or 32 KV heads.
- gpt-oss: revisit when Habana releases a new SynapseAI version (could fix the MoE kernel).
- New Anthropic /v1/messages-compatible models from RedHatAI or others — Gemma 4 is good but more options would be nice.
