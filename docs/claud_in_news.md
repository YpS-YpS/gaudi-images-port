# claud_in_news.md

> *"When you come in news that claude figured out how to run unsupported model in gaudi"*

A roadmap to that headline. Written 2026-04-26 after we got Qwen3-VL-32B-Thinking-FP8
serving at 49 tok/s on 1 Gaudi 3, with the user pushing back on the lazy "not supported"
answer for Gemma 4 31B IT. The user is right. Most "no" answers in the Gaudi
ecosystem are software-packaging "no", not hardware "no".

This file is the deep-dive that proves it, plus the phased plan to make it real.

---

## TL;DR

```
We are at 49 tok/s on Qwen3-VL-32B-Thinking-FP8 (1× Gaudi 3, FP8, single user).
That's ~43% of Gaudi 3's theoretical HBM-bandwidth ceiling for a 32B FP8 model.
NVIDIA H100 vLLM hits ~75-95% of theoretical on the same workload.

The gap is software, not silicon.

If we apply the 14 documented but unapplied perf levers (block-size 128,
fp8_inc KV cache, prefix caching, recipe cache, real-bucket pre-warm, etc.),
we can plausibly hit 100-200 tok/s — at or above H100 in vLLM.

For "unsupported" models like Gemma 4 31B IT, four real paths exist (image upgrade,
porting, optimum-habana, TGI-Gaudi). I dismissed them earlier on insufficient grounds.
```

---

## 1. Why "Gemma 4 doesn't work on Gaudi" was wrong

What I actually checked:

```
$ ls /usr/local/lib/python3.12/dist-packages/vllm_gaudi/models/ | grep -i gemma
gemma3_mm.py                                    ← only Gemma 3 has HPU class
$ ls /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/ | grep -i gemma
gemma.py gemma2.py gemma3.py gemma3_mm.py gemma3n.py gemma3n_audio_utils.py gemma3n_mm.py paligemma.py
                                                ← upstream vllm in this image is too old
```

That's 100% accurate evidence — but the conclusion "can't run" is wrong. The container is
just a snapshot pinned in April 2026. There are at least four ways to add Gemma 4:

### Path A — derived image with newer vLLM (most likely to work)

```dockerfile
FROM vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/vllm-0.17.1-ptfork-2.10.0:1.24.0-1007
RUN pip install --no-cache-dir --upgrade 'vllm>=0.18.0'
# the vllm-gaudi 0.17.1 plugin should still register HpuPlatform
# upstream vllm 0.18+ has Gemma4ForConditionalGeneration
```

Risk: vllm-gaudi 0.17.1 plugin's ABI may not match new vllm. If plugin breaks,
Gemma 4 falls back to upstream code path — no HPU graphs, but should run.
Effort: 1-3 hours, mostly the image rebuild.

### Path B — port Gemma 4 into vllm-gaudi (proper, MAX speed)

```bash
git clone https://github.com/vllm-project/vllm-gaudi
cd vllm-gaudi/vllm_gaudi/models
cp gemma3_mm.py gemma4_mm.py
# Now diff Gemma 3 ↔ Gemma 4 in HuggingFace transformers source:
#   - Attention head sizing
#   - RMSNorm vs LayerNorm placement
#   - SwiGLU vs GeGLU activations
#   - RoPE base frequency / scaling
#   - Sliding-window vs full attention pattern
#   - Vision encoder shape
# Apply the diffs. Register HpuGemma4ForConditionalGeneration in
# vllm_gaudi/registry.py
# Test with a single Generate call before serving.
pip install -e .
```

Effort: 1-3 days for someone fluent in vLLM internals. Yields HPU-graph optimization,
i.e. real speed. The only path that gives parity with Qwen3-VL HPU performance.

### Path C — optimum-habana directly (skip vLLM)

```python
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
adapt_transformers_to_gaudi()
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-31B-it",
    torch_dtype="bfloat16",
    device_map="hpu",
)
# wrap with FastAPI for OpenAI-compatible endpoint
```

No vLLM batching/paging, slower than Path A or B for serving, but proves the model
runs at all. Useful as a backstop. Effort: 4-8 hours including the HTTP wrapper.

### Path D — TGI-Gaudi

```bash
docker run --runtime=habana -p 8080:80 \
  ghcr.io/huggingface/tgi-gaudi:latest \
  --model-id google/gemma-4-31b-it
```

Different runtime entirely. Sometimes supports models vLLM doesn't, sometimes vice-versa.
Worth trying as a 30-minute probe before committing to A/B/C.

---

## 2. Hardware truth: Gaudi 3 vs H100/H200

Per-accelerator specs from public datasheets:

```
                       Gaudi 3 HL-325    H100 SXM5      H200 SXM5
   BF16 TFLOPS         1,835             1,979          1,979
   FP8  TFLOPS         3,670             3,958          3,958
   HBM capacity        128 GiB           80 GiB         141 GiB
   HBM bandwidth       3.67 TB/s         3.35 TB/s      4.80 TB/s
   Native FP8          ✓                 ✓ (Hopper)     ✓ (Hopper)
   List price (~$)     ~15k              ~30-40k        ~35-45k
```

**Gaudi 3 is competitive with H100 on raw silicon.** Lose to H200 (which costs ~3×).

For single-token decode (HBM-bandwidth-bound) on a 32B FP8 model:

```
   Theoretical max = HBM_bandwidth ÷ model_size_bytes
   Gaudi 3:   3,670 GB/s ÷ 32 GB ≈ 115 tok/s
   H100:      3,350 GB/s ÷ 32 GB ≈ 105 tok/s
   H200:      4,800 GB/s ÷ 32 GB ≈ 150 tok/s

   We're at 49 tok/s = 43% of Gaudi 3's theoretical.
   H100 with vLLM hits ~80-100 tok/s = 75-95% of theoretical.
```

So: we're **not** hardware-limited. We're **software-tuning-limited**. There is no
fundamental reason we can't hit 100+ tok/s on the same model on the same card.

---

## 3. The 14 perf levers we've not pulled

```
LEVER                                      DOC SOURCE                           EXPECTED GAIN
──────────────────────────────────────────────────────────────────────────────────────────────
1. --block-size 128 (was 16)               vllm-fork README, Habana MME tuning     15-25%
2. --kv-cache-dtype fp8_inc                vllm-gaudi env_variables                10-15%
3. --enable-prefix-caching                 vllm-gaudi features                     big TTFT
4. PT_HPU_RECIPE_CACHE_CONFIG=...          vllm-gaudi managing_warm-up             10× warmup
5. HABANA_PGM_LRU_MAX=60000                Habana runtime flags                    5-10%
6. --num-scheduler-steps 8 (fork only)     vllm-fork README                        10-20% decode
7. --async-scheduling                      Qwen3-VL recipe                         5-15%
8. --mm-encoder-tp-mode data               Qwen3-VL recipe                         ~2× vision
9. NUMA pinning per Gaudi                  Habana Optimization_in_Training         5-10%
10. C-state disable                        Habana Inference_Optimization           3-5% tail
11. Speculative decoding (Eagle/Medusa)    vllm-gaudi features                     1.5-3× decode
12. Bucket-tuned warmup (real prod shapes) vllm-gaudi bucketing_mechanism          predictable
13. Switch to torch.compile (ptupstream)   vllm-gaudi env_variables                ±20% (try)
14. --quantization inc + INC calibration   Habana Inference_Using_FP8              quality gain
──────────────────────────────────────────────────────────────────────────────────────────────
   Stacked, ballpark: 2-4× over current 49 tok/s = 100-200 tok/s
```

That ballpark puts us at or above H100 vLLM numbers on this 32B FP8 workload.

---

## 4. Where Gaudi 3 already beats NVIDIA

```
THROUGHPUT (concurrent users)
   128 GiB HBM vs H100's 80 GiB → ~60% more KV-cache space.
   At 32K ctx, 32B FP8: H100 ~16 concurrent, Gaudi 3 ~26 concurrent.
   We're not exploiting this — we set --max-num-seqs 32 conservatively.

LARGE-MODEL FIT
   Llama 3.1 405B FP8 fits on 8× Gaudi 3 (1024 GiB total HBM).
   On 8× H100 (640 GiB) it's a tight squeeze. On 8× H200 (1128 GiB) comparable.

PRICE/PERFORMANCE
   Gaudi 3 list ~$15k vs H100 ~$30-40k → 2-2.5× cheaper.
   Even at 70% of H100 raw speed, per-token-per-dollar is 1.5-2× better.

WHERE NVIDIA STILL WINS
   • Mature kernels: TensorRT-LLM, FlashAttention-3, FlashInfer
   • Universal model coverage: any HF model "just works"
   • Speculative decoding ecosystem: Eagle, Medusa, draft-model libraries
   • MIG (multi-instance GPU): no Gaudi equivalent
   • CUDA developer mindshare: order of magnitude more open-source contributors
```

The honest framing isn't "Gaudi loses to NVIDIA" — it's "Gaudi has 1/10th the
software-engineer-years invested." Closing 70% of the gap with documented flags
is realistic. The remaining 30% needs custom TPC kernels.

---

## 5. The phased plan ("how do we ship the news")

```
PHASE 1 — PERF FLAGS LADDER                                   1-2 hrs, low risk
   Apply levers 1, 2, 3, 5, 7, 8 one at a time.
   Measure tok/s after each.
   Target: 100+ tok/s on Qwen3-VL-32B-Thinking-FP8.
   Deliverable: numbers in this doc, updated bin/vllm-launch presets.

PHASE 2 — KILL VLLM_SKIP_WARMUP=true                          1 day
   Profile actual production traffic shapes (or anticipated ones).
   Set VLLM_PROMPT_BS_BUCKET_*/SEQ_*/DECODE_*BUCKET_* explicitly.
   Enable PT_HPU_RECIPE_CACHE_CONFIG=/recipe-cache,false,4096,false
     + bind-mount /var/cache/habana/recipes
   Server starts in 30s with NO JIT cliffs on user requests.

PHASE 3 — GEMMA 4 31B IT (Path A)                             1-2 hrs
   FROM ptfork:1.24.0-1007
   RUN pip install --upgrade 'vllm>=0.18'
   docker build, docker run --runtime=habana, vllm serve google/gemma-4-31B-it
   If breaks, fall back to Path C (optimum-habana).
   Deliverable: Gemma 4 31B IT serving on this box.

PHASE 4 — SPECULATIVE DECODING                                2-3 days
   Pick a draft model (smaller Qwen3 / Llama if available on HPU).
   vllm-gaudi has --speculative-config support per release notes.
   Expected: 1.5-3× decode throughput → puts us at H200 territory.

PHASE 5 — NATIVE HPU PORT OF ONE UNSUPPORTED MODEL            1-3 days
   Pick Gemma 4 (if Path A doesn't HPU-optimize) OR
   Pick a model NOT in validated table (e.g. Granite 4.0-h Tiny, GLM-4.5)
   Use Path B template above.
   This is the "claude figured out unsupported model on Gaudi" headline moment.

PHASE 6 — CUSTOM TPC KERNELS                                  weeks
   Real "beat NVIDIA at their own game" territory.
   Hand-tune FP8 GEMM, attention, RMSNorm, RoPE for Gaudi 3 SRAM hierarchy.
   Reachable, but only after Phases 1-5 since most gain is below this.
   Refer to Habana TPC SDK docs.
```

---

## 6. Concrete starting commands (Track 1 + Track 2 in parallel)

### Track 1 — perf flag ladder (run between user requests)

```bash
# Lever 1: --block-size 128 (edit bin/vllm-launch CLI)
# Lever 2: --kv-cache-dtype fp8_inc (edit bin/vllm-launch CLI)
# Lever 3: --enable-prefix-caching (edit bin/vllm-launch CLI)
# Lever 5: HABANA_PGM_LRU_MAX=60000 (env)
# Lever 7: --async-scheduling (edit CLI)
# Lever 8: --mm-encoder-tp-mode data (edit CLI for Qwen3-VL)
# Lever 12: explicit bucket env vars (carefully, one at a time after observing)

# After each lever, measure with:
time curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-vl-32b-thinking","messages":[{"role":"user","content":"Count from 1 to 50."}],"max_tokens":300,"temperature":0}'
# completion_tokens / wall_time = tok/s
```

### Track 2 — Gemma 4 derived image

```bash
mkdir -p /home/satyajit-gaudi/gemma4-build && cd $_
cat > Dockerfile <<'EOF'
FROM vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/vllm-0.17.1-ptfork-2.10.0:1.24.0-1007
RUN pip install --no-cache-dir --upgrade 'vllm>=0.18.0'
EOF
sudo docker build --network=host -t vllm-gaudi-gemma4:latest .
# then add a 'gemma4-31b' preset to bin/vllm-launch pointing at this image
sudo docker run -d --name vllm-gemma4 --runtime=habana \
  -e HABANA_VISIBLE_DEVICES=0 \
  -e PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false \
  -e PT_HPU_LAZY_MODE=1 -e PT_HPU_WEIGHT_SHARING=0 \
  --net=host --ipc=host --cap-add=sys_nice \
  -v /home/satyajit-gaudi/hf-cache:/root/.cache/huggingface \
  vllm-gaudi-gemma4:latest \
  bash -lc "vllm serve google/gemma-4-31B-it \
    --tensor-parallel-size 1 --port 8001 --host 0.0.0.0 \
    --max-model-len 8192 --gpu-memory-utilization 0.9 \
    --trust-remote-code --enforce-eager"
# --enforce-eager initially to debug; drop once stable
```

---

## 7. Success metrics

```
PHASE 1 done when:
   Qwen3-VL-32B-Thinking-FP8 sustains ≥ 100 tok/s on single-user test
   With NO regressions on smoke test (text/vision/tool-call all pass)

PHASE 2 done when:
   Cold restart → /v1/models responding in < 60s (currently ~50s)
   First user request after restart < 1.5s for 50 tokens (currently 3s JIT)

PHASE 3 done when:
   Gemma 4 31B IT serves a chat completion via OpenAI-compatible API
   tok/s ≥ 20 (i.e. 'usable', even if not yet optimized)

PHASE 4 done when:
   Speculative decoding measured at ≥ 1.5× decode tok/s
   vs. Phase 1 baseline on same prompt

PHASE 5 done when:
   A model NOT in vllm-gaudi's validated_models table
   serves successfully with HpuXxxForConditionalGeneration registered

PHASE 6 done when (stretch):
   Custom TPC kernel for one inner loop measurably faster than the
   default Habana-shipped kernel on a microbenchmark
```

---

## 8. Open questions / what we don't yet know

```
❓ Does vllm-gaudi 0.17.1 plugin actually break with vllm >= 0.18?
   Need to test Path A. If breaks, Path B/C/D.

❓ What's the actual measured performance hit of --enforce-eager in our
   current setup? We measured 7 tok/s with eager, 49 without — but did
   that include skip-warmup or not? Some confusion in the numbers.

❓ Can we get torch.compile path (ptupstream image) working without
   the original Qwen ValidateSyncInputTensors crash, given we now know
   the fix is PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false?

❓ Is there speculative decoding support that works specifically for
   Qwen3-VL on vllm-gaudi 0.17.1, or does it need to wait for 0.18+?

❓ What's the CPLD / SPI firmware story? Are we on the latest secured
   versions per the support matrix? (Check: hl-fw-loader -s, hl-smi -L)
```

---

## 9. Why this matters

The Gaudi 3 box on this rack costs ~$120k for 8 cards. An equivalent H100 box
would be ~$240-320k, and an H200 box ~$280-360k. If Gaudi 3 can be tuned to 90%
of H100 inference performance, that's 50-60% off the per-token cost of running
LLMs at scale.

The hardware is bought. The software gap is engineering. This document is the
plan to close it.

When the news headline lands, the ingredients will be: documented Habana flags
applied methodically, a few hundred lines of model-class porting, and the
acceptance that "not in the validated table" doesn't mean "not possible."

---

## 10. References

System docs:
- [Habana Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/)
- [Habana Support Matrix](https://docs.habana.ai/en/latest/Support_Matrix/Support_Matrix.html)
- [Habana Inference_Using_FP8](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html)
- [Habana Inference_Optimization (BIOS recipe)](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Optimization.html)
- [Habana Runtime Flags](https://docs.habana.ai/en/latest/PyTorch/Reference/Runtime_Flags.html)

vLLM-Gaudi docs:
- [vllm-gaudi compatibility matrix](https://docs.vllm.ai/projects/gaudi/en/latest/getting_started/compatibility_matrix.html)
- [vllm-gaudi validated_models](https://docs.vllm.ai/projects/gaudi/en/latest/getting_started/validated_models.html)
- [vllm-gaudi env_variables](https://docs.vllm.ai/projects/gaudi/en/latest/configuration/env_variables.html)
- [vllm-gaudi performance_tuning](https://docs.vllm.ai/projects/gaudi/en/latest/configuration/performance_tuning.html)
- [vllm-gaudi bucketing_mechanism](https://docs.vllm.ai/projects/gaudi/en/latest/features/bucketing_mechanism.html)
- [vllm-gaudi managing_warm-up (recipe cache)](https://docs.vllm.ai/projects/gaudi/en/latest/configuration/warm-up/managing_warm-up.html)
- [vllm-gaudi troubleshooting](https://docs.vllm.ai/projects/gaudi/en/latest/general/troubleshooting.html)
- [vllm-gaudi quantization (INC)](https://docs.vllm.ai/projects/gaudi/en/latest/configuration/quantization/inc.html)
- [vllm-gaudi calibration](https://docs.vllm.ai/projects/gaudi/en/latest/configuration/calibration/calibration.html)

Model + ecosystem:
- [vLLM Qwen3-VL recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
- [vLLM Gemma 4 recipe](https://github.com/vllm-project/recipes/blob/main/Google/Gemma4.md)
- [Gemma 4 day-0 announcement (TPU/AMD/XPU — note: no Gaudi)](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-04-02-gemma4.md)
- [optimum-habana](https://github.com/huggingface/optimum-habana)
- [TGI-Gaudi](https://github.com/huggingface/tgi-gaudi)
- [HabanaAI/vllm-fork (legacy)](https://github.com/HabanaAI/vllm-fork)
- [vllm-project/vllm-gaudi](https://github.com/vllm-project/vllm-gaudi)
