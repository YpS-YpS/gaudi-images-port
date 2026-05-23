# 03 — patches and overrides

The load-bearing modifications that make non-Habana-validated models work on
this stack. If you're patching a new model, read this first to understand
what the existing patches do and how they're applied — then mirror the
pattern.

## Layered structure

```
┌────────────────────────────────────────────────────────────┐
│  Habana base image                                         │
│  vllm-0.19.0-ptfork-2.10.0:1.24.0-1007                      │  ← we don't touch this
├────────────────────────────────────────────────────────────┤
│  Dockerfile.gemma4 layers on top:                          │
│  ① pip install transformers==5.7.0.dev (bumps from 4.55)    │  ← we add this
│  ② patch_kv_divisibility.py runs at build (UniformType-)    │  ← we add this
│  ③ HEAD_SIZES allow-list patched to include 512             │  ← we add this
│  ④ int() coercion in upstream float→int bug                 │  ← we add this
│  ⑤ skip_special_tokens default flipped to False             │  ← we add this
├────────────────────────────────────────────────────────────┤
│  Result: gaudi-vllm-gemma4:0.19.0                          │
│  (used by Gemma 4 + MiniMax M2/M2.7 + gpt-oss attempts)    │
└────────────────────────────────────────────────────────────┘
```

The 5 patches together = roughly 50 lines of changes (a few sed-edits, one
small Python monkey-patch, one pip install). The source of truth is
[`dockerfiles/Dockerfile.gemma4`](../../dockerfiles/Dockerfile.gemma4) + the
patch script [`dockerfiles/patch_kv_divisibility.py`](../../dockerfiles/patch_kv_divisibility.py).

## The 5 Gemma 4 patches — symptom / fix

For full per-patch reasoning see [docs/GEMMA4.md](../GEMMA4.md). Summary:

| # | Patch | Symptom without | Fix |
|---|---|---|---|
| ① | Bump transformers 4.55 → 5.7 | `KeyError: Gemma4ForCausalLM` at config load | `pip install transformers==5.7.0.dev` in Dockerfile |
| ② | Allow-list head_size=512 | `ValueError: head_size 512 not supported` | Add 512 to the HEAD_SIZES tuple in HPU attention backend |
| ③ | UniformTypeKVCacheSpecs unwrap | Crash in KV cache allocator (wrong type) | Detect wrapper in allocator, unwrap to `.kv_cache_spec` |
| ④ | int() coercion on tensor.size | `TypeError: 'float' object cannot be interpreted as integer` | Wrap in `int(...)` at the offending site |
| ⑤ | skip_special_tokens default → False | Gemma's `<|channel|>` markers stripped, reasoning parser sees nothing | Flip the default for this model family |

Plus one launch-flag (not a Dockerfile patch):
- `--default-chat-template-kwargs '{"enable_thinking":true}'` — applied by `bin/vllm-launch` for the `gemma4-31b` preset.

## MiniMax M2 / M2.7 — no patches, but a handcrafted HPU class

`vllm-gaudi` 0.19 ships a complete `HpuMiniMaxM2ForCausalLM` class at:

```
/usr/local/lib/python3.12/dist-packages/vllm_gaudi/models/minimax_m2.py  (504 lines)
```

At import time it overrides upstream `MiniMaxM2ForCausalLM`:

```
WARNING [registry.py:915] Model architecture MiniMaxM2ForCausalLM is already
registered, and will be overwritten by the new model class
vllm_gaudi.models.minimax_m2:HpuMiniMaxM2ForCausalLM.
```

**This warning is expected** — it means the HPU path is active. Don't try to
silence it. Both M2 and M2.7 share this class (same `architectures` field,
same `model_type: minimax_m2`).

MTP support: lines 498-504 of `minimax_m2.py`, `get_spec_layer_idx_from_weight_name()`
maps M2.7's 3 MTP weight tensors to layer indices `num_hidden_layers + i`
(62, 63, 64), distinct from the main transformer stack.

## gpt-oss — the un-patchable bug

vllm-gaudi has dedicated machinery for gpt-oss:
- `/usr/local/lib/python3.12/dist-packages/vllm_gaudi/models/gptoss_mxfp4.py` —
  monkey-patches `GptOssModel.load_weights` + dequantizes MXFP4 → BF16 at load time
- `HPUUnquantizedFusedMoEMethod.forward_oot` has an `if self.model_type in ["gpt_oss"]:` branch using top-k → softmax routing
- The HPU op `mixture_of_experts.bias_fused_weights` has `alpha=1.702, limit=7.0` defaults — exactly gpt-oss's swigluoai values

Everything **looks** wired correctly, but output is incoherent. The bug is
either in:
- the binary HPU MoE kernel itself (closed-source, not patchable), or
- the MXFP4 dequant in `convert_moe_packed_tensors` (open-source, looks
  correct but might have a subtle numerical issue)

Either way, not fixable from this stack. Documented in [docs/GPT-OSS.md](../GPT-OSS.md).

## Env-var overrides — added to `bin/vllm-launch` this session

```bash
LAUNCH_DEVICES="0,1,2,3"   # override preset's default cards. TP auto-derives from count.
LAUNCH_PORT=8500            # override port (for conflict avoidance)
```

Containers using these overrides get suffixed names like `vllm-minimax-m2-c0_1_2_3`,
so they don't clobber the default-cards instance. Used by `bin/launchpad` internally,
or invoke directly:

```bash
LAUNCH_DEVICES="0,1" vllm-launch 32b-thinking  # would try TP=2; but Qwen 32B FP8 won't fit at TP=2
```

Source: [`bin/vllm-launch`](../../bin/vllm-launch) lines 53-65.

## Per-preset auto-tightened knobs

Some presets get extra flags automatically — these live in `bin/vllm-launch`:

| Preset | Auto-applied | Why |
|---|---|---|
| `gemma4-31b` | `--tool-call-parser gemma4 --reasoning-parser gemma4 --default-chat-template-kwargs '{"enable_thinking":true}'` | Channel markers + enable thinking by default |
| `gpt-oss-120b` | `--reasoning-parser openai_gptoss`, NO `--tool-call-parser` (Harmony handles tools) | Harmony format |
| `minimax-m2*` | `--tool-call-parser minimax_m2 --reasoning-parser minimax_m2` | Both parsers registered as minimax_m2 |
| `minimax-m2.7` | `--gpu-memory-utilization 0.85 --max-num-seqs 16` + `HABANA_PGM_LRU_MAX=60000` | 256 experts + 3 MTP layers push HBM tighter than M2 |
| `235b-tp*` | `--gpu-memory-utilization 0.80` + `HABANA_PGM_LRU_MAX=60000` | Chronic HBM fragmentation under sustained inference |
| `*thinking*` and `235b-tp*` | `--reasoning-parser qwen3` | Parse `<think>` blocks into `reasoning_content` |

If you add a new preset, decide which of these knob-blocks apply.

## Common env vars across all presets

These are baked into every `docker run` in `bin/vllm-launch`:

```
PT_HPU_LAZY_MODE=1                          # required by Habana for vLLM
PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false      # REQUIRED for Qwen — was the ValidateSyncInputTensors fix
PT_HPU_ENABLE_LAZY_COLLECTIVES=true
PT_HPU_WEIGHT_SHARING=0
VLLM_GRAPH_RESERVED_MEM=0.3                 # carves 30% of free-HBM for graph capture buffers
VLLM_SKIP_WARMUP=true                       # saves 5-15 min warmup; first request pays the cost
VLLM_ENGINE_ITERATION_TIMEOUT_S=600
VLLM_RPC_TIMEOUT=600000
```

**Don't** change these without a specific reason — they're the result of
trial-and-error during bring-up.

## Patches that did NOT work / dead ends

- `--enforce-eager` for gpt-oss — toggling didn't change incoherent output
- `--quantization mxfp4` for gpt-oss — vLLM errors "mxfp4 quantization is currently not supported in hpu"
- `HPU_FUSED_MOE=0` — disables the fused path; gpt-oss still incoherent
- TP=8 for gpt-oss — same incoherence
- Trying `unsloth/gpt-oss-120b-BF16` (dense, no MXFP4) — same incoherence; this disproves "MXFP4 dequant is the bug"

These are all documented in [docs/GPT-OSS.md](../GPT-OSS.md) so future sessions don't repeat them.
