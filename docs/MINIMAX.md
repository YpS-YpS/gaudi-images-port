# MiniMax M2 and M2.7 on Gaudi 3

How to run **MiniMaxAI/MiniMax-M2** (230B MoE, ~10B active, FP8) and the newer
**MiniMaxAI/MiniMax-M2.7** (229B MoE, 256 experts, FP8) on 4× Intel Gaudi 3 (HL-325).

> **Bottom line:** Works on the **first try** with the Gemma 4 patched image — no
> new patches required. `vllm-gaudi` ships a handcrafted
> `HpuMiniMaxM2ForCausalLM` class plus a dedicated FP8 GEMM kernel
> (`HPUChannelWiseTorchFP8ScaledMMLinearKernel`) for this model. Full reasoning,
> Anthropic `/v1/messages` shape, and 4-tool parallel tool calls all verified.

```
Variant:       MiniMaxAI/MiniMax-M2           (FP8, official MiniMaxAI release)
Architecture:  MiniMaxM2ForCausalLM            → overridden by vllm-gaudi HpuMiniMaxM2ForCausalLM
Params:        230B total, ~10B active (MoE)
Disk:          ~215 GB on disk (130 safetensors shards, FP8)
HBM (TP=4):    ~107 GB/card on Gaudis 4-7   (weights + KV cache pool)
Devices:       4 (Gaudis 4, 5, 6, 7)
Port:          8006
Image:         gaudi-vllm-gemma4:0.19.0       (reused — no patches needed for MiniMax)
Parsers:       --reasoning-parser minimax_m2
               --tool-call-parser minimax_m2
```

## Launch

```bash
vllm-launch minimax-m2
```

Preset is defined in [`bin/vllm-launch`](../bin/vllm-launch):

```bash
[minimax-m2]="MiniMaxAI/MiniMax-M2|4|4,5,6,7|8006|16384|gaudi-vllm-gemma4:0.19.0"
```

The launcher auto-selects `minimax_m2` for both tool and reasoning parsers when
the preset matches `minimax-m2*`.

## API

### Chat completion (OpenAI shape)

```bash
curl -sX POST http://localhost:8006/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"minimax-m2",
    "max_tokens":2000,
    "messages":[{"role":"user","content":"Plan a Paris trip: weather, flight, exchange rate, hotel."}],
    "tools":[
      {"type":"function","function":{"name":"get_weather", ...}},
      {"type":"function","function":{"name":"search_flights", ...}},
      {"type":"function","function":{"name":"get_exchange_rate", ...}},
      {"type":"function","function":{"name":"book_hotel", ...}}
    ]
  }'
```

### Anthropic `/v1/messages` shape

```bash
curl -sX POST http://localhost:8006/v1/messages \
  -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model":"minimax-m2",
    "max_tokens":1000,
    "system":"You are concise.",
    "messages":[{"role":"user","content":"Capital of France?"}]
  }'
```

Anthropic SDK drop-in:

```python
import anthropic
c = anthropic.Anthropic(base_url="http://<box>:8006", api_key="dummy")
r = c.messages.create(model="minimax-m2", max_tokens=1024,
                      messages=[{"role":"user","content":"..."}])
# r.content is a list of [{type:"thinking", thinking:"..."}, {type:"text", text:"..."}]
```

## Verified behaviour (May 2026)

```
✅ /v1/models                       served: minimax-m2  max_len: 16384
✅ /v1/messages reasoning           thinking + text blocks, stop_reason=end_turn
✅ Parallel tools × 2 (cities)      both get_weather calls, finish=tool_calls
✅ Parallel mixed × 4 different     weather + flights + exchange + hotel
                                    all four extracted into tool_calls[]
```

Sample 4-tool output:

```json
"tool_calls": [
  {"function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}},
  {"function":{"name":"search_flights","arguments":"{\"origin\":\"JFK\",\"destination\":\"CDG\",\"date\":\"2023-06-23\"}"}},
  {"function":{"name":"get_exchange_rate","arguments":"{\"from_currency\":\"EUR\",\"to_currency\":\"USD\"}"}},
  {"function":{"name":"book_hotel","arguments":"{\"city\":\"Paris\",\"area\":\"Eiffel Tower\",\"nights\":2}"}}
]
```

## Why it Just Worked

Three pieces of pre-existing Habana engineering align here:

1. **`vllm_gaudi/models/minimax_m2.py`** — handcrafted HPU class registered as `HpuMiniMaxM2ForCausalLM`. At import time, vllm-gaudi overrides the upstream `MiniMaxM2ForCausalLM` registration:
   ```
   WARNING [registry.py:915] Model architecture MiniMaxM2ForCausalLM is already 
   registered, and will be overwritten by the new model class 
   vllm_gaudi.models.minimax_m2:HpuMiniMaxM2ForCausalLM.
   ```
2. **Dedicated FP8 kernel** — `HPUChannelWiseTorchFP8ScaledMMLinearKernel` selected automatically for the FP8 weights.
3. **No `UniformTypeKVCacheSpecs` drama** — MiniMax M2's `attn_type_list` is uniform (all full attention), so no hetero-head-dim machinery kicks in. Our Gemma 4 patches sit idle.

## Per-card HBM breakdown (TP=4)

```
Per Gaudi 3 (128 GiB HBM):
  Weights (FP8, 215 GiB / 4)        ~ 54 GiB
  KV cache pool                     ~ 50 GiB
  HPU graph reserve (0.3 of free)   ~ 23 GiB
  ───────────────────────────────────────────
  Total                              ~107 GiB  (comfortable margin)
```

## Download

The official `MiniMaxAI/MiniMax-M2` repo ships with FP8 weights directly — no
need for a third-party quantizer. 130 shards averaging ~1.7 GB each.

```bash
docker run --rm --net=host \
  -e HF_HOME=/root/.cache/huggingface -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e HTTP_PROXY=$HTTP_PROXY -e HTTPS_PROXY=$HTTPS_PROXY \
  -v $HOME/hf-cache:/root/.cache/huggingface --entrypoint /bin/bash \
  vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/vllm-0.19.0-ptfork-2.10.0:1.24.0-1007 \
  -lc 'pip install --quiet hf-transfer && \
       huggingface-cli download MiniMaxAI/MiniMax-M2 --local-dir-use-symlinks=False'
```

Typical download time on Intel network: ~30 min at 110-180 MiB/s.

## Smoke test

```bash
scripts/smoke.sh http://localhost:8006 minimax-m2 4000
```

Or one-liner:

```bash
curl -s http://localhost:8006/v1/messages \
  -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"minimax-m2","max_tokens":400,
       "messages":[{"role":"user","content":"What is 17 * 89?"}]}' \
  | python3 -m json.tool
```

Expected: a `content` array with a `thinking` block (the model's CoT) followed
by a `text` block ("1,513" or similar).

## Caveats

- **First-request warmup** — `VLLM_SKIP_WARMUP=true` is set, so the first request
  pays the graph-compile cost (~30-60 s). After that, subsequent requests are
  fast. If you need consistent first-token latency, remove `VLLM_SKIP_WARMUP=true`
  and pay the 10-15 min eager warmup at launch instead.
- **Long-reasoning chat completion `finish=length`** — when a chat-completion
  hits `max_tokens` mid-reasoning, the parser can't extract because the closing
  `<|end|>` marker never emits. Either bump `max_tokens` or use `/v1/messages`
  (Anthropic shape) which streams typed content blocks cleanly.
- **No reasoning_content seeded in incomplete responses** — same root cause as
  above; both `reasoning_content` and `content` come back as empty/None if the
  full Harmony structure didn't emit.

## Comparison with other endpoints on this box

| Endpoint | Port | Reasoning | Tools | Notes |
|---|---|---|---|---|
| Qwen3-VL-32B-Thinking | 8000 | ✅ `<think>` | hermes | OpenAI shape only |
| Qwen3-VL-32B-Instruct | 8001 | — | hermes | Fast direct answers |
| Gemma 4 31B (#1) | 8004 | ✅ gemma4 channels | gemma4 | Anthropic `/v1/messages` shape |
| Gemma 4 31B (#2) | 8005 | ✅ gemma4 channels | gemma4 | Second instance (Gaudi 3) |
| **MiniMax M2** | 8006 | ✅ minimax_m2 | minimax_m2 | 230B MoE, 128 experts |
| **MiniMax M2.7** | **8006** | ✅ minimax_m2 | **minimax_m2** | **229B MoE, 256 experts, 3 MTP modules** (swap with M2) |

# Part 2 — MiniMax M2.7 (Apr 2026)

**M2.7 is M2's successor.** Same architecture class (`MiniMaxM2ForCausalLM`), so
the existing `HpuMiniMaxM2ForCausalLM` HPU override applies and no new patches
are needed. But the model has been re-shaped to scale routing:

| Spec | M2 | **M2.7** |
|---|---|---|
| Total params | 230B | 229B |
| Active params | ~10B | ~10B |
| Experts (`num_local_experts`) | 128 | **256** |
| Top-k routing | 4 | **8** |
| Hidden layers | 62 | 62 |
| MTP modules (`num_mtp_modules`) | 0 | **3** |
| Disk size | ~215 GB | ~230 GB |
| Shards in index | 125 | 125 |
| Steady-state HBM/card (TP=4) | ~107 GB | **~101 GB** (with tighter knobs below) |

```
Variant:       MiniMaxAI/MiniMax-M2.7
Architecture:  MiniMaxM2ForCausalLM            → HpuMiniMaxM2ForCausalLM (handcrafted)
Devices:       4 (Gaudis 4, 5, 6, 7)            ← SAME as M2 (only one of them runs at a time)
Port:          8006                              ← SAME as M2 (can't co-run with M2)
Image:         gaudi-vllm-gemma4:0.19.0          (reused — no patches needed)
Parsers:       --reasoning-parser minimax_m2     (same parser names — both registered as minimax_m2)
               --tool-call-parser minimax_m2
```

## Launch

```bash
vllm-launch stop minimax-m2          # if M2 was running
vllm-launch minimax-m2.7              # tighter knobs auto-applied (see below)
```

Preset:

```bash
[minimax-m2.7]="MiniMaxAI/MiniMax-M2.7|4|4,5,6,7|8006|16384|gaudi-vllm-gemma4:0.19.0"
```

The launcher auto-applies tighter HBM knobs for `minimax-m2.7` (vs M2's defaults):

| Flag | M2 default | M2.7 (auto) |
|---|---|---|
| `--gpu-memory-utilization` | 0.9 | **0.85** |
| `--max-num-seqs` | 32 | **16** |
| `HABANA_PGM_LRU_MAX` env | — | **60000** (recipe-cache, blunts defrag-OOM) |

Why: M2.7 has ~3.5 GB/card more weights AND larger routing state (256 vs 128
experts). Without the tighter knobs, steady-state runs ~115 GB/card which
leaves only ~13 GB defrag headroom. With the tighter knobs it lands at ~101
GB/card — same as M2 in practice.

## Why It Worked the Same Way as M2

1. **Architecture class is identical** — `config.architectures: ["MiniMaxM2ForCausalLM"]`
   → the same `HpuMiniMaxM2ForCausalLM` override fires at import time.
2. **MTP layers handled by `get_spec_layer_idx_from_weight_name`** in
   `vllm_gaudi/models/minimax_m2.py:498-504` — MTP weight tensors are mapped to
   layer indices `num_hidden_layers + i` (62, 63, 64), distinct from the main
   transformer stack.
3. **FP8 kernel selection identical** — `HPUChannelWiseTorchFP8ScaledMMLinearKernel`
   logged the same way as M2: `[__init__.py:261] Selected
   HPUChannelWiseTorchFP8ScaledMMLinearKernel for Fp8LinearMethod`.
4. **256 experts works** because the HPU MoE op (`bias_fused_weights`) takes
   `total_experts` as a runtime tensor-list length, not a compiled-in constant.
5. **Uniform attention (`attn_type_list` all-1)** — no `UniformTypeKVCacheSpecs`
   drama, our Gemma 4 patches sit idle here too.

## Verified behaviour (May 2026)

```
✅ /v1/models                       served: minimax-m2.7  max_len: 16384
✅ /v1/chat/completions  17*89      content: "1513"  finish: stop
✅ /v1/messages reasoning           thinking + text blocks  stop_reason: end_turn
                                    (snail problem CoT: "net gain per full
                                    day-night cycle = 3 - 2 = 1m...")
✅ Parallel tools × 4 different     weather + flights + exchange + hotel
                                    all four called in parallel, finish=tool_calls
```

Sample parallel-tool output (4 distinct functions in one user turn):

```json
"tool_calls": [
  {"function":{"name":"get_weather","arguments":"{\"city\": \"Paris\"}"}},
  {"function":{"name":"search_flights","arguments":"{\"origin\": \"JFK\", \"destination\": \"CDG\", \"date\": \"2024-06-23\"}"}},
  {"function":{"name":"get_exchange_rate","arguments":"{\"from_currency\": \"EUR\", \"to_currency\": \"USD\"}"}},
  {"function":{"name":"book_hotel","arguments":"{\"city\": \"Paris\", \"area\": \"Eiffel Tower\", \"nights\": 2}"}}
]
```

## What's different about the load timeline

**M2.7 loads slower than M2.** Plan for ~30-40 min from `vllm-launch` to
"Application startup complete":

```
0:00     container start, plugin discovery (~1 min)
0:01     Workers spawn, model registry overwrite logged
0:01-0:35  Workers PIN at 99% CPU each, iterating safetensors
         shards from disk into RAM and through dequant. HBM is 0
         throughout this phase. Block I/O climbs to ~215 GB read.
         No useful log lines emit during this phase — silent grind.
0:35     HBM takeoff: ~50 GB across 4 cards
0:36     HBM bulk-pushed: ~195 GB across 4 cards (~50 GB/card)
0:37     KV cache allocation, server up.
```

The 30 min CPU-bound silent phase is caused by the 256-expert × 62-layer × TP=4
weight processing loop. M2 had 128 experts so its loop was ~half as long
(~15 min). Don't kill the container during the silent phase — it IS progressing
(check `docker stats <c>` Block IO column).

## Disk layout note (130-shard naming, 125-shard index)

The repo's safetensors files use the filename pattern `model-NNNNN-of-00130.safetensors`,
but `model.safetensors.index.json` only references shards 00000-00124 (125 total).
This is a naming-pattern holdover from training/planning — the "-of-00130" is
*not* the true count. Shards 00125-00129 don't exist. If you see
`huggingface-cli download` exit with 125 files instead of 130, that's the
correct state.

## Caveats specific to M2.7

- **First launch is slow.** ~30 min CPU-bound phase before any HBM activity.
  Subsequent launches reuse the on-disk cache so the disk I/O is faster but the
  CPU-side processing remains the bottleneck.
- **MTP layers load but are not used for speculative decoding** by default.
  vLLM's speculative-decode path needs explicit `--speculative-config` to
  activate. Without it, the 3 MTP transformer blocks are loaded into HBM as
  layers 62/63/64 but inference still runs greedily through layers 0-61. (~3
  GB/card of HBM is "spent" on these unused-by-default MTP weights.)
- **Cannot co-run with M2.** Same Gaudis 4-7 and port 8006. Stop one before
  launching the other.
- **First-request warmup** — same as M2 (~30-60 s for graph compile due to
  `VLLM_SKIP_WARMUP=true`).
