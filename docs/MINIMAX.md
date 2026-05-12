# MiniMax M2 on Gaudi 3

How to run **MiniMaxAI/MiniMax-M2** (230B MoE, ~10B active, FP8) on 4× Intel Gaudi 3 (HL-325).

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
| **MiniMax M2** | **8006** | ✅ minimax_m2 | **minimax_m2** | **Anthropic shape**, **230B MoE** |

## Next: MiniMax M2.7

Same `MiniMaxM2ForCausalLM` architecture, FP8, similar size (~230 GB). Should
drop in by swapping the model name. Preset stub for when ready:

```bash
[minimax-m2.7]="MiniMaxAI/MiniMax-M2.7|4|4,5,6,7|8007|16384|gaudi-vllm-gemma4:0.19.0"
```
