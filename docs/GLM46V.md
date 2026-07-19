# GLM-4.6V-FP8 on Gaudi 3 — the one-patch bring-up

How to run **Z.AI GLM-4.6V-FP8** (`zai-org/GLM-4.6V-FP8`), a ~108B vision MoE, on
two Intel Gaudi 3 cards (HL-325) with OpenAI-compatible chat + vision + the glm45
tool/reasoning parsers.

> **TL;DR:** GLM-4.6V is a *pleasant surprise* after Gemma 4. It reuses the Gemma 4
> patched image but needs **exactly one** patch, not five — a shape fix in the MoE
> block. The FP8 checkpoint, the vision encoder, and the KV-cache machinery all
> worked on the generic upstream path with no HPU-specific work.

```
Base image:    gaudi-vllm-gemma4:0.19.0            (already carries transformers 5.7.0 + Gemma 4 patches)
Derived image: gaudi-vllm-glm46v:0.19.0            ← what this doc builds
Model:         zai-org/GLM-4.6V-FP8                 (Glm4vMoeForConditionalGeneration / glm4v_moe)
Size:          ~108B MoE — 128 routed + 1 shared experts, 8/token, ~12B active
Shape:         46 layers · 96 heads / 8 KV heads · head_dim 128 (UNIFORM) · 128K native ctx
Quant:         FP8 compressed-tensors (F8_E4M3) · 103 GB on disk · MIT license
Device:        2× Gaudi 3 (cards 1,2, TP=2) — steady ~105 GiB/card, 46.6 GiB KV reserved
Port:          8006  ·  served-model-name: glm-4.6v
Throughput:    ~50-52 tok/s single-stream decode
Startup:       ~90 s with VLLM_SKIP_WARMUP=true
```

## Build

```bash
cd dockerfiles
docker build \
  --build-arg HTTP_PROXY=$HTTP_PROXY \
  --build-arg HTTPS_PROXY=$HTTPS_PROXY \
  --build-arg NO_PROXY=$NO_PROXY \
  -t gaudi-vllm-glm46v:0.19.0 \
  -f Dockerfile.glm46v .
```

The Dockerfile is thin: `FROM gaudi-vllm-gemma4:0.19.0`, one `sed` patch, one
verification `python3 -c`. No packages installed (the Gemma 4 base already ships
`transformers 5.7.0`, which carries the `glm4v_moe` arch). The `--build-arg`
proxies are accepted for parity with the Gemma 4 build even though nothing is
downloaded — see [GEMMA4.md](GEMMA4.md#build) for the why.

## Launch

```bash
vllm-launch glm46v
```

The preset (`bin/vllm-launch`) sets `HABANA_VISIBLE_DEVICES=1,2`, TP=2, port 8006,
the derived image, `max-model-len 32768`, `max-num-seqs 8`, and the flags
`--tool-call-parser glm45 --reasoning-parser glm45 --enable-auto-tool-choice
--mm-encoder-tp-mode data`.

### The verified ad-hoc command (recorded verbatim)

The bring-up was proven with this `docker run` before the preset was added. Note
`--entrypoint vllm` — the image's *default* entrypoint is the env-driven autocalc
launcher, so it must be overridden to reach `vllm serve` directly:

```bash
docker run -d --name vllm-glm46v --runtime=habana --net=host --ipc=host --cap-add=sys_nice \
  -e HABANA_VISIBLE_DEVICES=1,2 \
  -e PT_HPU_LAZY_MODE=1 -e PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false \
  -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true -e PT_HPU_WEIGHT_SHARING=0 \
  -e VLLM_GRAPH_RESERVED_MEM=0.3 -e VLLM_SKIP_WARMUP=true \
  -e VLLM_ENGINE_ITERATION_TIMEOUT_S=600 -e VLLM_RPC_TIMEOUT=600000 \
  -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
  -e VLLM_SKIP_TRANSFORMERS_VERSION_CHECK=1 \
  -e HF_HOME=/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -v /home/satyajit-gaudi/hf-cache:/root/.cache/huggingface \
  --entrypoint vllm \
  gaudi-vllm-glm46v:0.19.0 \
  serve zai-org/GLM-4.6V-FP8 --tensor-parallel-size 2 --port 8006 \
  --served-model-name glm-4.6v --max-model-len 32768 --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 --trust-remote-code \
  --tool-call-parser glm45 --reasoning-parser glm45 --enable-auto-tool-choice \
  --mm-encoder-tp-mode data
```

> **Launcher vs ad-hoc — two harmless differences.** `vllm-launch` reaches the same
> `vllm serve` invocation by overriding the entrypoint with
> `/bin/bash -lc "exec vllm serve ..."` (equivalent to `--entrypoint vllm` — both
> bypass the image's autocalc entrypoint), and it fetches weights **online via the
> corp proxy** rather than `HF_HUB_OFFLINE=1`. If the weights are already cached the
> proxy fetch is a no-op. `VLLM_SKIP_TRANSFORMERS_VERSION_CHECK=1` is baked into the
> image (`ENV` in `Dockerfile.glm46v`), so the launcher doesn't re-pass it.

## The one patch — `GLM46V_HPU_3D_OK`

GLM-4.6V routes through the **generic upstream path**
(`vllm/model_executor/models/glm4_1v.py` + `glm4_moe.py`) because vllm-gaudi 0.19
ships **no `glm4v_moe` HPU class**. Almost all of that path is HPU-clean. The one
exception is the MoE block's shape assumption.

**Symptom** (weights loaded fine, server came *up*, crash only on the first forward):
```
ValueError: too many values to unpack (expected 2)
  at vllm/model_executor/models/glm4_moe.py:204
```

**Where**
`/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm4_moe.py`,
`Glm4MoeSparseMoeBlock.forward()`.

**Fix** — make the block shape-agnostic:
```python
# was:
num_tokens, hidden_dim = hidden_states.shape          # assumes 2D [num_tokens, hidden]
...
return final_hidden_states.view(num_tokens, hidden_dim)

# now (GLM46V_HPU_3D_OK):
orig_shape = hidden_states.shape                       # capture full shape
hidden_dim = orig_shape[-1]                            # flatten by last dim
...
return final_hidden_states.view(orig_shape)            # restore original shape
```

**Why**
Upstream wrote `Glm4MoeSparseMoeBlock.forward` for a 2D `[num_tokens, hidden]`
tensor. The **HPU model runner passes 3D `[batch, seq, hidden]`**, so the two-target
unpack `num_tokens, hidden_dim = hidden_states.shape` gets three values and raises.
Capturing `orig_shape`, indexing the last dim for `hidden_dim`, and returning
`.view(orig_shape)` works for both 2D and 3D inputs — no behavior change on GPU.

**Idempotency marker:** `GLM46V_HPU_3D_OK` in `glm4_moe.py`. The Dockerfile's
verification step asserts the marker is present, the return was rewritten, and the
old 2D unpack is gone — a rebuild fails loudly if the upstream source drifted.

### Attempt log
1. **Plain launch** on the Gemma 4 image → server came up, crashed at first inference
   (`too many values to unpack`).
2. **Bind-mounted patch** (`-v` the edited `glm4_moe.py` into the running container) →
   all smoke tests passed. Confirmed the one-line fix is sufficient.
3. **Baked image** (`Dockerfile.glm46v`) → reproduced the passing smokes from a clean build.

## What did NOT need patching (the good news)

These are the load-bearing **negative results** — each is a thing Gemma 4 forced us to
patch that GLM-4.6V got for free:

- **FP8 compressed-tensors checkpoint** loaded **natively** on HPU. No INC calibration,
  no dequant pass, no `--quantization` gymnastics. The `F8_E4M3` weights just loaded.
- **The 24-layer ViT vision encoder** worked through the **generic upstream path**
  (`TORCH_SDPA`) with `--mm-encoder-tp-mode data`. No HPU-specific multimodal-encoder work.
- **No head-size patch** (Gemma 4 Patch 1) and **no KV-divisibility patch**
  (Gemma 4 Patches 2/4) — GLM-4.6V has a **uniform `head_dim` 128**, so the whole
  `UniformTypeKVCacheSpecs` / hetero-head_dim machinery that Gemma 4 tripped over
  never engages.
- **`transformers 5.7.0`** (already in the Gemma 4 base image) already carries the
  `glm4v_moe` architecture — no transformers bump needed.

That's why the derived image is `FROM gaudi-vllm-gemma4:0.19.0`: it inherits
`skip_special_tokens=False`, the widened head-size table, and the KV unwrap for free,
and just layers the single MoE shape fix on top.

## Smoke evidence

- **Text:** "The capital of France is Paris."
- **Multimodal:** the classic boardwalk photo described accurately —
  > "A wooden boardwalk cuts through a lush, green grassy field, leading toward a
  > distant tree line under a bright blue sky dotted with wispy clouds..."

## Known quirks

- **`<|begin_of_box|>...<|end_of_box|>` delimiters.** GLM-4.6V wraps its final answer
  in literal `<|begin_of_box|>answer<|end_of_box|>` markers. **These are TEXT, not
  special tokens** — `skip_special_tokens: true` does *not* remove them, and the vLLM
  0.19 `glm45` reasoning parser doesn't strip them either. Cosmetic. Strip client-side,
  or patch the chat template, if a clean answer string is needed.
- **generation_config overrides sampling defaults.** The model ships
  `temperature 0.8`, `top_k 2`, `top_p 0.6`; these win over vLLM's defaults unless the
  request overrides them.
- **`reasoning_content` stays empty on simple prompts.** The glm45 reasoning parser is
  wired but the model doesn't emit a separate reasoning channel for trivial inputs.
- **Tool-calling is wired but untested.** The `glm45` tool parser +
  `--enable-auto-tool-choice` are set; end-to-end tool-call validation is still open.

## Open items

- [ ] Validate tool-calling end-to-end with the `glm45` parser.
- [ ] Concurrent-load / batched-throughput benchmark (single-stream is ~51 tok/s).
- [ ] Real warmup (drop `VLLM_SKIP_WARMUP=true`) for a production profile.
- [ ] Optional **TP4 on cards 4-7** to serve >32K context (the model is native 128K;
      the preset caps at 32768 to keep the KV budget flat at `max-num-seqs 8`).

## When this doc goes stale

The single patch is written against **vllm-gaudi 0.19.0 / vLLM 0.19.0**. When Habana
ships a native `glm4v_moe` HPU class (or upstream makes `Glm4MoeSparseMoeBlock.forward`
shape-agnostic), `GLM46V_HPU_3D_OK` becomes unnecessary. Rebuild without the patch and
re-run the smokes; if text + the boardwalk vision check still pass, the patch can be
dropped.
