# GLM-5.2 Phase 2 Plan — MTP Speculative Decoding + Long Context

**Status:** IN PROGRESS (2026-07-17). Baseline: `vllm-glm52` from baked image
`gaudi-vllm-glm52:0.19.0`, TP8, port 8010, `--max-model-len 8192`, ~20–23 tok/s
single-stream. See [GLM52.md](GLM52.md) for the force-dense bring-up story.

## Goals

1. **Speed — MTP speculative decoding.** GLM-5.2 ships a multi-token-prediction
   layer (checkpoint layer 78, `num_nextn_predict_layers: 1`). The official vLLM
   recipe runs `{"method": "mtp", "num_speculative_tokens": 5}` — on CUDA this
   is a 2–3× decode lever. Target: measure honestly on HPU at N=3 and N=5.
2. **Fairness — context length.** 8K was a conservative first-boot cap, not the
   model's limit (native design: 1M with real DSA kernels). MLA KV is cheap
   (~313K-token pool already allocated), so raise `--max-model-len` to 65536
   (fallback 32768) at no weight-memory cost.

## Known risks (identified up front)

| Risk | Mitigation |
|---|---|
| Spec decode may be unimplemented in vllm-gaudi 0.19's HPU model runner | Gate check first, offline, before any relaunch. If absent: stop, document, revisit on plugin 0.21/0.24. |
| MTP layer 78 carries its own `self_attn.indexer.*` weights + DSA gating; the force-dense patch covers `deepseek_v2.py` only | Read the MTP model file in the image; prepare `GLM52_FORCE_DENSE_MTP` patch before enabling. |
| Dense attention is O(n²) — long-prompt prefill cost | Measure prefill wall-time at ~30K tokens; publish the number rather than hide it. |
| Force-dense quality is exact only ≤2048 tokens (`index_topk`); 3.1K verified; 32–64K unproven | Needle-in-haystack sweep at 4K / 8K / 16K / 30K with distractors before trusting the raised cap. |

## Protocol (one variable at a time)

1. **Phase 1 @ 8K:** MTP on, everything else identical. Coherence smoke → 3-run
   tok/s at N=3 and N=5 → acceptance rate from vLLM metrics. Compare vs 20–23 baseline.
2. **Phase 2:** raise `--max-model-len` to 65536 (`--max-num-seqs 8`), keep MTP
   only if Phase 1 proved it. Prefill timing + needle sweep decide the final cap.
3. **Ship:** best validated config baked into `gaudi-vllm-glm52:0.19.0` (no
   bind-mounts), `bin/glm52-launch` updated, GLM52.md updated with results,
   this plan file updated with outcomes.

## Outcome

_To be filled when the runs complete: MTP verdict, tok/s before/after,
acceptance rate, prefill timing, needle accuracy table, final shipped config._
