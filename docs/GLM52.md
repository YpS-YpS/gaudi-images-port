# GLM-5.2-FP8 on Gaudi 3 — the "force-dense MLA" bring-up

How to run **Z.AI GLM-5.2-FP8** (`zai-org/GLM-5.2-FP8`), a **753B MoE / ~40B
active** frontier agentic model with **DeepSeek Sparse Attention (DSA)**, on **all
eight** Intel Gaudi 3 cards (HL-325, TP=8) — a model that officially *cannot* run on
Gaudi because HPU has no sparse-attention-indexer kernel.

> **TL;DR:** GLM-5.2's attention is MLA + a DSA *indexer* that top-k-selects 2048 key
> positions per query. HPU has **no kernel** for that indexer (CUDA/Ascend only →
> `NotImplementedError`), so the model won't run sparse. The insight: **top-2048-of-N
> == all-of-N for N ≤ 2048 tokens**, so at short/mid context the trained sparse
> attention is *identical to plain dense MLA*. We drop the indexer and run dense — one
> patch, 3 hunks, marker `GLM52_FORCE_DENSE`. It booted coherent on the first attempt.

```
Base image:    gaudi-vllm-gemma4:0.19.0            (transformers 5.7.0 → ships GlmMoeDsaConfig)
Derived image: gaudi-vllm-glm52:0.19.0             ← what this doc builds
Model:         zai-org/GLM-5.2-FP8                  (GlmMoeDsaForCausalLM — rides vLLM's deepseek_v2.py)
Size:          753B MoE / ~40B active               (DeepSeek-V3.2-style: MLA + DSA + IndexShare)
Attention:     MLA (dense here) · DSA indexer index_topk=2048 (DROPPED) · designed for 1M ctx
Quant:         FP8 compressed-tensors (F8_E4M3) — loads natively, no INC
Device:        8× Gaudi 3 (cards 0-7, TP=8, --enable-expert-parallel) — ~119 GB/card
Port:          8010  ·  served-model-name: glm-5.2  ·  parsers: glm47 tool / glm45 reasoning
Throughput:    ~20-21 tok/s single-stream decode   ·  KV cache 313,472 tokens
Startup:       ~75 s with weights page-cache-hot + VLLM_SKIP_WARMUP=true
Context:       configured 8192 (force-dense is EXACT ≤ 2048 tokens; empirically coherent to 3.1k+)
```

## What GLM-5.2 is (and why it "can't" run on Gaudi)

GLM-5.2 is Z.AI's frontier agentic flagship — a 753B-parameter Mixture-of-Experts
with ~40B active per token, built for long-horizon autonomous execution and ~1M-token
context. Architecturally it is a **DeepSeek-V3.2-style** model, so vLLM registers
`GlmMoeDsaForCausalLM` onto the existing `deepseek_v2.py` model class. Its attention
has three pieces:

1. **MLA (Multi-head Latent Attention)** — the low-rank KV compression from DeepSeek.
   HPU has a working MLA backend. This part is fine.
2. **DSA (DeepSeek Sparse Attention)** — a lightweight *indexer* network computes a
   score per (query, key) pair and keeps only the **top `index_topk` = 2048** keys for
   the expensive attention. This is what makes 1M context affordable.
3. **IndexShare** — the indexer's projections are shared across layers to cut its cost.

The blocker: **HPU has no kernel and no attention backend for the DSA indexer.**
`SparseAttnIndexer.forward_native` raises `NotImplementedError` on the HPU platform,
and vllm-gaudi 0.19 ships no `DeepseekV32IndexerBackend`. On CUDA and Ascend the
indexer has native kernels; on Gaudi it simply doesn't exist. So the stock model
dies at construction/first-forward. This is the "unsupported" that the vendor matrix
means — *not on the validated list*, not *physically impossible*.

## The force-dense insight

`index_topk = 2048` means: for each query, attend to the **top 2048** key positions.
But if the whole sequence is **≤ 2048 tokens**, "top 2048 of ≤2048" is **every** key —
so the sparse attention and plain dense MLA compute the **exact same thing**. The
indexer is pure overhead in that regime.

So for short/mid context we can **drop the indexer entirely and run dense MLA** on the
HPU MLA backend that already works:

- **Exact** for any sequence ≤ `index_topk` (2048 tokens).
- **An approximation** beyond 2048 (dense attends to *all* keys where trained-sparse
  would have kept only the top 2048 — a superset, so information is not lost, but the
  softmax normalization differs from what the model saw in training). Empirically the
  output stayed coherent and needle-correct at **3.1k+ tokens**, well past the exact
  boundary — but see *Limits* before trusting the configured 8K.

This is the same bet that made DeepSeek-R1 practical on Gaudi: reuse the MLA/FP8
machinery, sidestep the one CUDA-only piece.

## Two offline-found blockers (before a single card was touched)

Both were found by reading the config + safetensors shapes offline, not by crashing:

1. **`qk_rope_head_dim` attribute_map collision → `--hf-overrides`.**
   transformers 5.7.0's `GlmMoeDsaConfig` has `attribute_map = {"head_dim":
   "qk_rope_head_dim"}`. The checkpoint's `head_dim = 192` therefore **clobbers** the
   real `qk_rope_head_dim = 64`, yielding `qk_head_dim = qk_nope_head_dim(128) +
   qk_rope_head_dim(192) = 320`... wrong. That mismatches the real weight shapes
   (`q_b_proj[16384, 2048]`, `kv_a_proj_with_mqa[576, 6144]`, which imply 256/576).
   **Fix:** launch with `--hf-overrides '{"qk_rope_head_dim": 64}'` to restore 64,
   giving the correct 256 (`128+128`) qk_head_dim and 576 kv_lora path.

2. **Indexer weights have no home → loader `KeyError`.**
   The FP8 checkpoint carries per-layer `…​.self_attn.indexer.{wq_b, wk, weights_proj,
   k_norm}.*` tensors (present only on "full" layers). Once force-dense skips building
   the indexer modules, those tensors have **no destination parameter**, so
   `params_dict[name]` raises `KeyError` on the first one during `load_weights`. **Fix:**
   the loader skips any tensor whose name contains `.self_attn.indexer.` when
   force-dense is active.

(MTP — the layer-78 `num_nextn_predict_layers=1` multi-token-prediction module — needs
no special handling: the main-model loader already skips `model.layers.78.*` via
`get_spec_layer_idx_from_weight_name`.)

## The patch — `GLM52_FORCE_DENSE` (3 hunks)

Applied to `vllm/model_executor/models/deepseek_v2.py`. Gated on env
`VLLM_GLM_DSA_FORCE_DENSE=1` **and** the config having `index_topk` (so it's a no-op
for ordinary DeepSeek-V2/V3). Source of truth:
[`dockerfiles/patch_glm52_force_dense.py`](../dockerfiles/patch_glm52_force_dense.py).

**Hunk 1 — `import os`** at module top (the gate reads an env var).

**Hunk 2 — flip `is_v32` before the indexer is built** (`DeepseekV2MLAAttention.__init__`):
```python
self.is_v32 = hasattr(config, "index_topk")

# GLM52_FORCE_DENSE
if self.is_v32 and os.environ.get("VLLM_GLM_DSA_FORCE_DENSE", "0") == "1":
    self.is_v32 = False      # skip Indexer construction; MLAModules(is_sparse=False)

if self.is_v32:              # ← now False, so no Indexer, no V32IndexerBackend, no KV
    ...
```
With `is_v32` False, the `Indexer` is never constructed, no `DeepseekV32Indexer`
attention layer (and its KV cache) is created, and `mla.py` runs plain dense MLA.

**Hunk 3 — skip indexer checkpoint tensors** (`load_weights`):
```python
_glm52_force_dense = os.environ.get("VLLM_GLM_DSA_FORCE_DENSE", "0") == "1"
for name, loaded_weight in weights:
    if "rotary_emb.inv_freq" in name:
        continue
    if _glm52_force_dense and ".self_attn.indexer." in name:
        continue  # GLM52_FORCE_DENSE — no destination module
    ...
```

**Idempotency + self-assertion.** `patch_glm52_force_dense.py` uses pristine upstream
lines as anchors, refuses to run twice, and exits non-zero if any anchor drifted. The
Dockerfile then re-asserts all four sub-markers via `python3 -c`, so **a rebuild fails
loudly** if the upstream `deepseek_v2.py` ever changes shape — nothing silently ships
unpatched.

## Build

```bash
cd /home/satyajit-gaudi/gaudi-setup/dockerfiles
docker build \
  --build-arg HTTP_PROXY=http://proxy-dmz.intel.com:912 \
  --build-arg HTTPS_PROXY=http://proxy-dmz.intel.com:912 \
  -t gaudi-vllm-glm52:0.19.0 \
  -f Dockerfile.glm52 .
```

The Dockerfile is thin: `FROM gaudi-vllm-gemma4:0.19.0`, `COPY` the patch script, one
`RUN` that applies + asserts, and two `ENV`s. **No packages are installed and no
network is needed** — the base already ships `transformers 5.7.0` (which carries
`GlmMoeDsaConfig`/`GlmMoeDsaForCausalLM`), and the patch is a local source edit. The
`--build-arg` proxies are accepted only for parity with the Gemma 4 build.

`ENV VLLM_GLM_DSA_FORCE_DENSE=1` is baked into the image. **This is safe** because the
patch's runtime gate is `hasattr(config, "index_topk") AND env==1`: ordinary
DeepSeek-V2/V3 checkpoints have no `index_topk`, so the flip is a no-op and the loader
skip matches no tensors. The only models the env affects are DSA-sparse ones
(GLM-5.x, DeepSeek-V3.2) — for which force-dense is the *only* way to run on HPU anyway.

## Launch

```bash
glm52-launch          # start on all 8 cards, port 8010
glm52-launch smoke    # models + France + palindrome
glm52-launch logs     # follow
glm52-launch stop     # remove
```

Dedicated launcher ([`bin/glm52-launch`](../bin/glm52-launch)), **not** a
`bin/vllm-launch` preset — the same precedent as `sglang-launch`. GLM-5.2 needs
`--hf-overrides '{"qk_rope_head_dim": 64}'` (JSON), `--enable-expert-parallel`,
`--block-size 128`, and several extra env vars that the vllm-launch preset template
(pipe-delimited rows fed into `-lc "exec vllm serve ..."`) can't express without
fragile nested-quote/JSON escaping.

### The verified command (recorded verbatim)

`--entrypoint vllm` overrides the image's env-driven autocalc entrypoint to reach
`vllm serve` directly. Weights load offline from the HF cache (`HF_HUB_OFFLINE=1`).

```bash
docker run -d --name vllm-glm52 --runtime=habana --net=host --ipc=host --cap-add=sys_nice \
  -e HABANA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e PT_HPU_LAZY_MODE=1 \
  -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
  -e PT_HPU_WEIGHT_SHARING=0 \
  -e VLLM_SKIP_WARMUP=true \
  -e VLLM_HPU_FORCE_CHANNEL_FP8=true \
  -e VLLM_GLM_DSA_FORCE_DENSE=1 \
  -e VLLM_SKIP_TRANSFORMERS_VERSION_CHECK=1 \
  -e HF_HOME=/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -v /home/satyajit-gaudi/hf-cache:/root/.cache/huggingface \
  --entrypoint vllm \
  gaudi-vllm-glm52:0.19.0 \
  serve zai-org/GLM-5.2-FP8 \
    --tensor-parallel-size 8 --enable-expert-parallel --block-size 128 \
    --max-model-len 8192 --max-num-seqs 4 --gpu-memory-utilization 0.85 \
    --trust-remote-code --hf-overrides '{"qk_rope_head_dim": 64}' \
    --served-model-name glm-5.2 --port 8010 \
    --tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice
```

## Results — reproduced from the baked image

The bring-up was proven with a read-only bind-mounted `deepseek_v2.py`; the baked
`gaudi-vllm-glm52:0.19.0` image then reproduced it with **no bind-mount** (only the HF
cache volume):

- **Boot:** ready in ~75 s (page-cache-hot weights), `Application startup complete`,
  `/v1/models` → 200 on the first attempt.
- **KV cache:** `GPU KV cache size: 313,472 tokens` — 38.27× max concurrency at 8192/req.
- **HBM:** ~119 GB/card steady-state at `--gpu-memory-utilization 0.85`.
- **Throughput:** ~20-21 tok/s single-stream decode (observed 19.7-23.0 tok/s).
- **Text** (`max_tokens 512`): *"The capital of France is Paris."* — plus a populated
  `reasoning` channel (glm45 reasoning parser working).
- **Coding** (`max_tokens 1024`): a correct, clean `is_palindrome(s)` that strips
  non-alphanumerics, lowercases, and compares `cleaned == cleaned[::-1]`.
- **Needle recall:** correct at 3.1k tokens (past the 2048 exact boundary).

> **Thinking model — `max_tokens` must be ≥ 512.** GLM-5.2 emits a reasoning channel
> first; a small budget gets consumed by reasoning and returns empty/truncated content.

## Limits & open items

- **8K configured, 1M is not real here.** `--max-model-len 8192` is a deliberate cap.
  True 1M context needs the **real DSA indexer kernels**, which HPU doesn't have —
  force-dense's cost is quadratic and its equivalence is only exact ≤ 2048 tokens.
- **Quality beyond 2048 is empirical.** Coherent at 3.1k in spot checks, but the
  configured 8K is *unvalidated*. **Run a long-context / needle sweep (2k→8k) before
  trusting 8K** for anything load-bearing.
- **MTP speculative decoding is untested.** GLM-5.2 ships a multi-token-prediction head
  (layer 78); it's currently skipped. Enabling MTP speculative decoding is a plausible
  future speedup over the ~20 tok/s single-stream baseline.
- **Real warmup.** Booted with `VLLM_SKIP_WARMUP=true`; a real bucket pre-warm would
  give a production profile and kill first-shape JIT recompiles.
- **Tool-calling** (`glm47` parser + `--enable-auto-tool-choice`) is wired but not yet
  validated end-to-end.

## Credits / lineage

This rides the **DeepSeek-R1-on-Gaudi** precedent: the MLA + FP8 compressed-tensors
machinery that made DeepSeek practical on HPU is exactly what GLM-5.2 reuses via
`deepseek_v2.py`. The only GLM-5.2-specific work was recognizing that the one
CUDA-only piece (the DSA indexer) is *redundant* at short context and could be dropped.
It also builds `FROM gaudi-vllm-gemma4:0.19.0`, inheriting transformers 5.7.0 and the
Gemma 4 patch stack — the same base that carried the [GLM-4.6V bring-up](GLM46V.md).

## When this doc goes stale

The patch is written against **vllm-gaudi 0.19.0 / vLLM 0.19.0 / transformers 5.7.0**.
When Habana ships a native HPU **DSA indexer** kernel + `DeepseekV32IndexerBackend`,
`GLM52_FORCE_DENSE` becomes unnecessary (and long context becomes real) — drop the
patch, drop the `--hf-overrides` if transformers fixes the `attribute_map` collision,
and re-run the smokes. If text + palindrome + a long-context needle still pass, the
force-dense workaround can retire.
