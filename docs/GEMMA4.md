# GEMMA 4 31B on Gaudi 3 — the five-patch stack

How to run Google **Gemma 4 31B Instruct (FP8)** on a single Intel Gaudi 3 (HL-325),
with full Anthropic `/v1/messages` shape + reasoning blocks + parallel tool calls.

> **TL;DR:** Use a derived Docker image (`gaudi-vllm-gemma4:0.19.0`) that layers five patches on top of Habana's `vllm-0.19.0-ptfork`. The Gaudi software stack does not yet ship handlers for vLLM 0.19's heterogeneous-head-dim machinery; the patches add them.

```
Base image:    vault.habana.ai/.../vllm-0.19.0-ptfork-2.10.0:1.24.0-1007
Derived image: gaudi-vllm-gemma4:0.19.0              ← what this doc builds
Model:         RedHatAI/gemma-4-31b-it-FP8-Dynamic   (60 layers, hetero head_dim 256/512)
Device:        1× Gaudi 3 (HL-325, 128 GiB HBM) — steady ~95 GiB
Endpoints:     /v1/chat/completions   OpenAI shape, tool_calls + reasoning
               /v1/messages           Anthropic shape, content blocks [thinking, text, tool_use]
First request: ~15-20 s warmup with VLLM_SKIP_WARMUP=true
```

## Build

```bash
cd dockerfiles
docker build \
  --build-arg HTTP_PROXY=$HTTP_PROXY \
  --build-arg HTTPS_PROXY=$HTTPS_PROXY \
  --build-arg NO_PROXY=$NO_PROXY \
  -t gaudi-vllm-gemma4:0.19.0 \
  -f Dockerfile.gemma4 .
```

**Why `--build-arg`:** Intel corp network requires the proxy *inside the build container*, not just the dockerd daemon. Without these args `pip install transformers==5.7.0` hangs at PyPI for forever.

## Launch

```bash
vllm-launch gemma4-31b
```

The preset (defined in `bin/vllm-launch`) sets `HABANA_VISIBLE_DEVICES=2`, port 8004, the derived image, and the flags `--tool-call-parser gemma4 --reasoning-parser gemma4 --default-chat-template-kwargs '{"enable_thinking":true}'`.

## Smoke test

```bash
scripts/smoke.sh http://localhost:8004 gemma4-31b 4000
```

Six checks: `/v1/models` → text → vision → single tool → parallel tools × 4 cities → parallel mixed × 4 different tools.

## The five patches — what and why

Gemma 4 is the **first hetero-head_dim model on vllm-gaudi 0.19**. Local-attention layers use `head_dim=256` with 16 KV heads; global layers use `head_dim=512` with 4 KV heads, alternating in a 5:1 sliding-window-to-global pattern across 60 layers. vLLM 0.19 introduced new machinery to handle this, but vllm-gaudi 0.19 was released without the matching handlers. Each patch closes one gap.

### Patch 1 — `head_size=512` accepted by HPU paged attention

**Symptom**
```
ValueError: Head size 512 is not supported by PagedAttention.
Supported head sizes are: [1..256].
```

**Where**
`/usr/local/lib/python3.12/dist-packages/vllm_gaudi/attention/ops/hpu_paged_attn.py`,
function `HPUPagedAttention.get_supported_head_sizes()`.

**Fix**
```python
return list(range(1, 257)) + [320, 384, 448, 512]
```

**Why**
The HPU paged-attention kernel hard-codes its allowed head sizes. Gemma 4's global layers need 512; 320/384/448 are added speculatively for related Gemma derivatives. Adding values to the allow-list doesn't change the kernel — it just stops vllm-gaudi from refusing the model upfront.

### Patch 2 — `UniformTypeKVCacheSpecs` unwrap + integer coercion

**Symptom (stage 1)**
```
ValueError: Unknown KV cache spec type for layer language_model.model.layers.0.self_attn.attn:
<class 'vllm.v1.kv_cache_interface.UniformTypeKVCacheSpecs'>
```

**Symptom (stage 2, after stage-1 fix)**
```
TypeError: zeros() received an invalid combination of arguments
- got (tuple, device=str, dtype=torch.dtype),
  but expected one of: (tuple of ints size, *, dtype=..., device=..., ...)
```

**Where**
`/usr/local/lib/python3.12/dist-packages/vllm_gaudi/v1/worker/hpu_model_runner.py`,
`initialize_kv_cache()` around line 6012, in the non-hybrid allocation branch.

**Fix**
```python
# Detect the wrapper; pull out the inner per-layer FullAttentionSpec.
_g4_inner_specs = getattr(kv_cache_spec, 'kv_cache_specs', None)
if isinstance(_g4_inner_specs, dict) and layer_name in _g4_inner_specs:
    kv_cache_spec = _g4_inner_specs[layer_name]

num_blocks = kv_cache_tensor.size // kv_cache_spec.page_size_bytes

# Coerce both shape and device — kv_cache_tensor.size is float-valued upstream;
# self.device may be a bare string in some configs.
_g4_shape = tuple(int(x) for x in kv_cache_shape)
_g4_dev = torch.device(self.device) if not isinstance(self.device, torch.device) else self.device
key_cache = torch.zeros(_g4_shape, dtype=dtype, device=_g4_dev)
```

**Why**
vLLM 0.19 introduced `UniformTypeKVCacheSpecs` to represent groups of layers that
share an attention *type* (FullAttention, SlidingWindow, …) but have *different*
shape parameters (head_dim, num_kv_heads). For Gemma 4 that's 60
`FullAttentionSpec` objects keyed by layer name inside one wrapper.

vllm-gaudi's allocator was written before this class existed. It tries to read
`page_size_bytes` from a single spec — for the wrapper that's the **sum** across
all layers, not the per-layer page. The patch swaps `kv_cache_spec` to the inner
per-layer spec so all downstream math uses the right page (≈2 MiB local, ≈1 MiB
global, vs the ≈110 MiB sum).

The `int()` coercion at the `torch.zeros` call is needed because PyTorch's
C++ argument parser rejects a shape tuple containing float values — even ones
like `70784.0` that are mathematically integers. The float originates upstream
in `available_memory // page_size` calculations (see Patch 4).

### Patch 3 — `skip_special_tokens` default flipped to `False`

**Symptom**
```
POST /v1/chat/completions → 200
{"choices":[{"message":{"content":"<think>… answer …",
                        "reasoning_content": null}}]}
              ▲                              ▲
              reasoning leaked into content  reasoning_content empty
```

**Where**
Four files: `vllm/sampling_params.py` + the three OpenAI request protocols
(`chat_completion/protocol.py`, `completion/protocol.py`, `responses/protocol.py`).

**Fix (one sed across all four)**
```bash
sed -i 's|skip_special_tokens: bool = True|skip_special_tokens: bool = False|g' \
       /usr/local/lib/python3.12/dist-packages/vllm/sampling_params.py \
       /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/protocol.py \
       /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/completion/protocol.py \
       /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/responses/protocol.py
```

**Why**
`Gemma4ReasoningParser.extract_reasoning()` looks for the literal token strings
`<|channel>` / `<channel|>` (token IDs 100/101 in Gemma's tokenizer) inside the
decoded text. With the default `skip_special_tokens=True`, those tokens are
stripped before the parser runs:

```python
def extract_reasoning(self, model_output: str, request) -> tuple[str | None, str]:
    if self.start_token not in model_output and self.end_token not in model_output:
        return None, model_output            # ← bails, returns raw content
    ...
```

There's **no server-side flag** to flip `skip_special_tokens` globally. Per-request
overrides work via the request body, but every caller would need to remember to
send it. Anthropic `/v1/messages` doesn't expose the field at all — it inherits
the `SamplingParams` default. Flipping all four defaults at the protocol layer is
the only place that affects every entry point uniformly.

**Side effect** Some special tokens that *should* stay hidden (e.g. `<|endoftext|>`)
now appear in raw responses. The model's chat template handles the EOS tokens
correctly, so this is cosmetic for normal completions and load-bearing for
reasoning extraction.

### Patch 4 — upstream `int(...)` coercion on `num_blocks` / `tensor.size`

**Symptom**
```
AssertionError                       # block_pool.py:156
assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
```

**Where**
`/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py`, three places:

1. Line ~1113 — the special-case branch for `UniformTypeKVCacheSpecs` planning
2. Line ~1607 — `kv_cache_config.num_blocks = min_num_blocks`
3. Line ~1612 — `tensor.size = tensor.size // num_blocks_old * min_num_blocks`

**Fix** Wrap each in `int(...)`. See `dockerfiles/patch_kv_divisibility.py` for the
exact substitutions and idempotency check.

**Why**
`available_memory * gpu_memory_utilization` produces a Python float, which
propagates through `//` and `*` into `KVCacheTensor.size`. Downstream:

- `block_pool.py:156` asserts `isinstance(num_gpu_blocks, int)` → fails.
- `torch.zeros((tensor.size // page,...), ...)` builds a shape tuple with a float
  in position 0 → PyTorch rejects (Patch 2 catches it locally, but the root cause
  lives here).

The Patch 2 coercions catch the float at the leaf where torch.zeros sees it; the
Patch 4 coercions catch it at the source so other readers of the same value
(`block_pool`, the scheduler, the metrics layer) also see an int. Both are needed.

### Patch 5 — chat template flag for thinking mode

This is technically a **launch flag**, not a code patch — listed here for
completeness because it's required for reasoning blocks to fire:

```
--default-chat-template-kwargs '{"enable_thinking":true}'
```

Gemma 4's tokenizer chat template has a conditional `{% if enable_thinking %}`
that, when true, prepends the `<|channel>thinking<channel|>` opener so the model
emits a reasoning channel. The server-default form means every request gets it
without the client needing to send `chat_template_kwargs.enable_thinking`.

## Verification matrix (post-patch-5)

| Endpoint | Plain text | Reasoning | Single tool | Parallel tools | Mixed parallel | Vision |
|---|---|---|---|---|---|---|
| `/v1/chat/completions` | ✓ | ✓ (`reasoning` field) | ✓ (`tool_calls[]`) | ✓ | ✓ | ✓ |
| `/v1/messages` | ✓ | ✓ (`thinking` content block) | ✓ (`tool_use` block) | ✓ | ✓ | ✓ |

Drop-in compatible with any Anthropic-SDK consumer:

```python
import anthropic
client = anthropic.Anthropic(base_url="http://<host>:8004", api_key="dummy")
resp = client.messages.create(model="gemma4-31b", max_tokens=1024,
                              system="You are concise.",
                              messages=[{"role":"user","content":"hi"}])
```

## Reproducing on a fresh box

```bash
# 1. Base bring-up (kernel 6.8, Habana 1.24, hugepages, NICs, docker)
sudo bash install.sh

# 2. Pull the Habana base image
docker pull vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/vllm-0.19.0-ptfork-2.10.0:1.24.0-1007

# 3. Build the patched image (one-time, ~10 min cold, ~30 s warm)
cd dockerfiles
docker build --build-arg HTTP_PROXY=$HTTP_PROXY --build-arg HTTPS_PROXY=$HTTPS_PROXY \
             --build-arg NO_PROXY=$NO_PROXY \
             -t gaudi-vllm-gemma4:0.19.0 -f Dockerfile.gemma4 .

# 4. Pre-download model (~30 GB FP8)
huggingface-cli download RedHatAI/gemma-4-31b-it-FP8-Dynamic --local-dir-use-symlinks=False

# 5. Launch
vllm-launch gemma4-31b

# 6. Smoke
scripts/smoke.sh http://localhost:8004 gemma4-31b 4000
```

## Idempotency markers

The patches are designed to be re-applicable across rebuilds:

- Patch 2 marker: `GEMMA4_TRIM_OK` in `hpu_model_runner.py`.
- Patch 4: looks for `int(  # GEMMA4_INT_COERCE` in `kv_cache_utils.py`.
- Patches 1 and 3: pure `sed -i` — the script checks for the post-state string before re-applying so a second run is a no-op.

If `Dockerfile.gemma4` is rebuilt with `--no-cache`, the patch script handles
the pristine-source case. If a layer is cached but the script changed, bump the
`patch_kv_divisibility.py` content (e.g. a comment edit) to invalidate cache.

## When this doc goes stale

These five patches are written against **vllm-gaudi 0.19.0 / vLLM 0.19.0**.
When Habana ships vllm-gaudi 0.20+ with native `UniformTypeKVCacheSpecs` support,
Patches 2 and 4 are likely to become unnecessary. Patches 1 and 3 may still apply
depending on whether Habana lands the head_size table widening upstream and whether
the upstream `skip_special_tokens` default flips.

Smoke test on a fresh image before tearing patches out. If any of the six
smoke-test rows go red, walk the patch list above to find the regression.
