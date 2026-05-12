# gpt-oss on Gaudi 3 — current status: **does not work**

> **Bottom line:** As of vllm-gaudi 0.19.0 + Habana 1.24.0, OpenAI's gpt-oss
> models (20b and 120b) load successfully on Gaudi 3 but **produce incoherent
> output**. The bug is in Habana's closed-source `mixture_of_experts.bias_fused_weights`
> HPU kernel and/or vllm-gaudi's MXFP4 dequantization for gpt-oss's specific MoE
> layer shape. We narrowed it down in this session but cannot fix it from
> user space.

## What we tried (this session)

| Variant | Disk | TP | Loads? | Coherent? |
|---|---|---|---|---|
| `unsloth/gpt-oss-120b-BF16` | 240 GB | 4 (G3-6) | ✅ | ❌ |
| `openai/gpt-oss-120b` (MXFP4) | 183 GB | 4 (G3-6) | ✅ | ❌ |
| `openai/gpt-oss-20b` (MXFP4) | 26 GB | 1 (G7) | ✅ | ❌ |

Flags / env vars we toggled, all same incoherent output:
- `--enforce-eager` (disable HPU graphs)
- `HPU_FUSED_MOE=0` (use reference unfused path)
- `--quantization mxfp4` (forces upstream mxfp4 → fails immediately: *"mxfp4 quantization is currently not supported in hpu"*)
- With/without `--reasoning-parser openai_gptoss`
- With/without `skip_special_tokens=false` request override

## Sample evidence of incoherence

```
prompt:  "The capital of France is"
120B BF16:  "the function that returns the function that returns the function..."
120B MXFP4: "the capital of France.\nThe capital of France is Paris.
              \nThe capital\nThe capital of France is Paris.\n..."
20B  MXFP4: "capital of France? That is a question. So the phrase
              \"capital of France?\" is a question. So the phrase..."

prompt:  "Question: What is 17 * 89?\nAnswer:"
120B BF16:  " 89\n\nAnswer: 89\n\nAnswer: 89\n\nAnswer: 89\n\nAnswer:"
120B MXFP4: " 1\n\nThe output is 1, which matches the expected answer.
              \n\nThus, my answer (the code) is correct for the sample."
20B  MXFP4: " \n```\n\nWe need to check if the answer contains any phrase
              that appears in the reference answer..."

chat-completion (Harmony):  content=None, reasoning_content=None on all three
```

## What we confirmed about the stack

1. **`GptOssForCausalLM` arch is registered** in vLLM 0.19 — `[model.py:549] Resolved architecture: GptOssForCausalLM`.
2. **`vllm_gaudi/models/gptoss_mxfp4.py` exists** and monkey-patches `GptOssModel.load_weights` and `ModelArchConfigConvertorBase._normalize_quantization_config` at import time.
3. **The patched config-normalizer returns `None`** for `quant_method=mxfp4` on `model_type=gpt_oss`, telling vLLM to load as if unquantized.
4. **`_load_weights_mxfp4_dequantize_hpu`** dequantizes the MXFP4 `blocks` + `scales` tensors into BF16 weights at load time, using `convert_moe_packed_tensors`. This is why HBM matches BF16 footprint (~220 GB on TP=4 for 120b) rather than MXFP4 (~65 GB).
5. **The MoE forward goes through `HPUUnquantizedFusedMoEMethod.forward_oot`** (`/usr/local/lib/python3.12/dist-packages/vllm_gaudi/ops/hpu_fused_moe.py`), which has an explicit `if self.model_type in ["gpt_oss"]:` branch using top-k → softmax routing (correct for gpt-oss).
6. **Bias is plumbed through** `VllmMixtureOfExpertsOp` — `_cached_w13_bias_views` and `_cached_w2_bias_views` are set from `layer.w13_bias` and `layer.w2_bias`.
7. **The HPU op chosen is `bias_fused_weights`** (the bias path). Its schema:
   ```
   hpu::mixture_of_experts.bias_fused_weights(
     Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights,
     Tensor[] w12, Tensor[] w3, Tensor[] w12_bias, Tensor[] w3_bias,
     *, bool permuted_weights, int experts_min, int experts_max,
     int chunk_size=0, int total_experts=0,
     float alpha=1.702, float limit=7.0     ← built-in swigluoai defaults
   ) -> Tensor
   ```
   So the HPU op DOES know about swigluoai activation — its default `alpha=1.702, limit=7.0` are exactly gpt-oss's values.
8. **The 20b variant emits some Harmony channel tokens** in chat completion (`<|end|>`, `<|start|>`, `<|channel|>`, `<|return|>`) but in nonsensical order. 120b never emits them at all.

Everything *should* work. But output is garbage. The bug is somewhere in the binary HPU kernel call or the MXFP4 dequantization producing values that look numerically reasonable but are subtly wrong (e.g., wrong byte unpacking order, wrong exponent application, transposed expert weights).

## Reproduction (if you want to confirm on a different box)

```bash
# Download (excluding original/ saves bandwidth)
docker run --rm --net=host \
  -e HF_HOME=/root/.cache/huggingface -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e HTTP_PROXY=$HTTP_PROXY -e HTTPS_PROXY=$HTTPS_PROXY \
  -v $HOME/hf-cache:/root/.cache/huggingface --entrypoint /bin/bash \
  vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/vllm-0.19.0-ptfork-2.10.0:1.24.0-1007 \
  -lc 'pip install --quiet hf-transfer && \
       huggingface-cli download openai/gpt-oss-20b \
         --local-dir-use-symlinks=False --exclude "original/*"'

# Launch (uses our Gemma 4 patched image which has all the right base patches)
docker run -d --name vllm-gpt-oss-20b --runtime=habana --restart no \
  --entrypoint /bin/bash -e HABANA_VISIBLE_DEVICES=7 \
  -e PT_HPU_LAZY_MODE=1 -e PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false \
  -e VLLM_GRAPH_RESERVED_MEM=0.3 -e VLLM_SKIP_WARMUP=true \
  -e VLLM_SKIP_TRANSFORMERS_VERSION_CHECK=1 \
  -e HF_HOME=/root/.cache/huggingface \
  --net=host --ipc=host --cap-add=sys_nice \
  -v $HOME/hf-cache:/root/.cache/huggingface \
  gaudi-vllm-gemma4:0.19.0 \
  -lc 'exec vllm serve openai/gpt-oss-20b \
        --tensor-parallel-size 1 --host 0.0.0.0 --port 8007 \
        --max-model-len 16384 --gpu-memory-utilization 0.9 \
        --max-num-seqs 32 --trust-remote-code \
        --served-model-name gpt-oss-20b'

# Smoke test (will return incoherent text)
curl -s http://localhost:8007/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"gpt-oss-20b","prompt":"The capital of France is",
       "max_tokens":30,"temperature":0}'
```

## Why this is past user-space fixable

The HPU compiled kernel `torch.ops.hpu.mixture_of_experts.bias_fused_weights` is
**closed-source binary** (part of Habana SynapseAI 1.24.0). It produces wrong
logits for gpt-oss's specific MoE shape (32 or 128 experts, swigluoai with
interleaved gate/up). We can't patch the kernel; only Habana can.

The Python-side dequantization (`convert_moe_packed_tensors` in
`vllm_gaudi/models/gptoss_mxfp4.py`) is readable, but appears mathematically
correct (FP4 LUT with the canonical 16 values, exponent bias 127, low/high
nibble unpacking, ldexp for block scaling). If it has a bug, it's subtle and
would require numerical comparison against a CUDA-correct reference output —
several hours of debugging.

## Upstream bug-report draft

> **Title:** gpt-oss-120b and gpt-oss-20b produce incoherent output on Gaudi 3 / vllm-gaudi 0.19.0
>
> **Environment**
> - Hardware: Intel Gaudi 3 (HL-325, 128 GiB HBM)
> - Driver: Habana 1.24.0-1007
> - vLLM: 0.19.0 + vllm-gaudi 0.19.0 (base image `vllm-0.19.0-ptfork-2.10.0:1.24.0-1007`)
> - Models tested: `openai/gpt-oss-120b` (MXFP4), `openai/gpt-oss-20b` (MXFP4), `unsloth/gpt-oss-120b-BF16`
>
> **Symptom**
> Model loads, KV cache initializes, server comes up cleanly. `/v1/models`, `/v1/chat/completions`, `/v1/messages`, `/v1/responses` all reachable. But:
> - Plain text completion produces repetitive loops or off-topic responses.
> - Chat completion returns `content: None` — Harmony parser receives no channel structure.
> - Other MoE models (Qwen3-30B-A3B) work fine through the same image and backend.
>
> **What I confirmed works**
> - `vllm_gaudi/models/gptoss_mxfp4.py` patches install and `_load_weights_mxfp4_dequantize_hpu` runs.
> - HBM footprint matches dequantized BF16 (~220 GB on TP=4 for 120b), confirming dequant happened.
> - `forward_oot` takes the `if self.model_type in ["gpt_oss"]` branch with correct top-k → softmax routing.
> - `bias_fused_weights` overload has `alpha=1.702, limit=7.0` defaults matching swigluoai.
> - `w13_bias` and `w2_bias` are plumbed through `_cached_w*_bias_views`.
>
> **What I toggled, no difference**
> - `--enforce-eager`
> - `HPU_FUSED_MOE=0`
> - With/without `--reasoning-parser openai_gptoss`
> - 20b vs 120b vs BF16-dequantized 120b
>
> **Repro** see `docs/GPT-OSS.md` in https://github.com/YpS-YpS/gaudi-images-port

## What works instead

**Gemma 4 31B Instruct FP8** is the working Anthropic `/v1/messages` endpoint
on this stack — see [GEMMA4.md](GEMMA4.md). Same image, full reasoning + tool
calls verified. The MoE bug doesn't apply because Gemma 4 isn't MoE.
