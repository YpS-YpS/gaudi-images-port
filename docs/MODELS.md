# MODELS — what runs, where, with what

All presets are defined in [`bin/vllm-launch`](../bin/vllm-launch). Launch any of
them with `vllm-launch <preset>` after running `install.sh`. The vLLM container
is created fresh per preset; restart policy `unless-stopped` recovers from
defrag-OOM and similar transient failures.

## Active presets

| Preset | Model | TP | Device | Port | Max-len | Image | Notes |
|---|---|---|---|---|---|---|---|
| `32b-thinking` | `Qwen/Qwen3-VL-32B-Thinking-FP8` | 1 | 0 | 8000 | 16384 | vllm-0.17.1 ptfork | `<think>` blocks via `--reasoning-parser qwen3` |
| `32b-instruct` | `Qwen/Qwen3-VL-32B-Instruct-FP8` | 1 | 1 | 8001 | 16384 | vllm-0.17.1 ptfork | direct answers, hermes tool parser |
| `30b-a3b` | `Qwen/Qwen3-VL-30B-A3B-Thinking-FP8` | 1 | 0 | 8002 | 32768 | vllm-0.17.1 ptfork | MoE, 3B active params |
| `8b-thinking` | `Qwen/Qwen3-VL-8B-Thinking-FP8` | 1 | 0 | 8003 | 32768 | vllm-0.17.1 ptfork | small / fast iteration |
| `gemma4-31b` | `RedHatAI/gemma-4-31b-it-FP8-Dynamic` | 1 | 2 | 8004 | 16384 | **gaudi-vllm-gemma4:0.19.0** (derived) | Anthropic /v1/messages, see [GEMMA4.md](GEMMA4.md) |
| `gpt-oss-120b` | `unsloth/gpt-oss-120b-BF16` | 4 | 3,4,5,6 | 8005 | 16384 | **gaudi-vllm-gemma4:0.19.0** (derived) | ⚠️ Loads but incoherent output — see [GPT-OSS.md](GPT-OSS.md) |
| `minimax-m2` | `MiniMaxAI/MiniMax-M2` | 4 | 4,5,6,7 | 8006 | 16384 | **gaudi-vllm-gemma4:0.19.0** (derived) | 230B MoE FP8, 128 experts, Anthropic /v1/messages + 4-tool parallel verified, see [MINIMAX.md](MINIMAX.md) |
| `minimax-m2.7` | `MiniMaxAI/MiniMax-M2.7` | 4 | 4,5,6,7 | 8006 | 16384 | **gaudi-vllm-gemma4:0.19.0** (derived) | 229B MoE FP8, **256 experts + 3 MTP modules**, port conflicts with `minimax-m2`, see [MINIMAX.md](MINIMAX.md#part-2--minimax-m27-apr-2026) |
| `235b-tp4` | `Qwen/Qwen3-VL-235B-A22B-Thinking-FP8` | 4 | 4,5,6,7 | 8004 | 8192 | vllm-0.17.1 ptfork | 235B MoE on 4 cards. **Port conflicts with `gemma4-31b`** — only one at a time. |
| `235b-tp8` | `Qwen/Qwen3-VL-235B-A22B-Thinking-FP8` | 8 | 0..7 | 8006 | 32768 | vllm-0.17.1 ptfork | maximum context on 8 cards |

## Per-model patch & flag summary

### Qwen3-VL family (32B-Thinking / 32B-Instruct / 30B-A3B / 8B / 235B)

- **Image:** `vault.habana.ai/.../vllm-0.17.1-ptfork-2.10.0:1.24.0-1007` (stock Habana — no patches)
- **Env:** `PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false` (required — without it, recipe cache gets poisoned on Qwen graph capture)
- **Env:** `VLLM_SKIP_WARMUP=true` (saves ~5 min warmup; first request takes the hit instead)
- **Flag:** `--enable-auto-tool-choice --tool-call-parser hermes` (Qwen tools are hermes-format)
- **Flag (thinking variants only):** `--reasoning-parser qwen3` (extracts `<think>…</think>` into the `reasoning_content` field)
- **Throughput:** ~49 tok/s on 32B-Thinking, FP8, single Gaudi 3, 1 user
- **Caveat:** `--enforce-eager` is NOT needed on 0.17.1 ptfork (the older 0.17.1-ptupstream image had a graph capture bug; ptfork fixes it)

### MiniMax M2 FP8 (`MiniMaxAI/MiniMax-M2`)

- **Image:** `gaudi-vllm-gemma4:0.19.0` — reuses Gemma 4 patched image, no MiniMax-specific patches needed
- See [MINIMAX.md](MINIMAX.md) for full launch + smoke-test walkthrough
- **Architecture:** `MiniMaxM2ForCausalLM` — vllm-gaudi ships handcrafted `HpuMiniMaxM2ForCausalLM` (overrides upstream registration at import time)
- **FP8 kernel:** `HPUChannelWiseTorchFP8ScaledMMLinearKernel` selected automatically
- **MoE shape:** 128 experts, top-4
- **TP=4** on Gaudis 4-7; ~107 GB/card steady-state
- **Flags:** `--tool-call-parser minimax_m2 --reasoning-parser minimax_m2`
- **Verified:** `/v1/messages` reasoning + text blocks, parallel tool calls × 4 different tools (weather/flights/exchange/hotel)
- **Disk:** ~215 GB (125 FP8 safetensors shards from MiniMaxAI directly)

### MiniMax M2.7 FP8 (`MiniMaxAI/MiniMax-M2.7`)

- **Image:** `gaudi-vllm-gemma4:0.19.0` — same image as M2; no new patches
- See [MINIMAX.md § Part 2](MINIMAX.md#part-2--minimax-m27-apr-2026) for the deep-dive on the 256-expert/MTP shape
- **Architecture:** `MiniMaxM2ForCausalLM` (same class → same HPU override fires)
- **MoE shape:** **256 experts, top-8** (vs M2's 128 / top-4) + **3 MTP modules** loaded as layers 62/63/64
- **TP=4** on Gaudis 4-7; ~101 GB/card steady-state with tightened knobs
- **Auto-applied knobs** (different from M2 because per-card weights + routing state are bigger):
  - `--gpu-memory-utilization 0.85`
  - `--max-num-seqs 16`
  - `HABANA_PGM_LRU_MAX=60000`
- **Same parsers as M2:** `--tool-call-parser minimax_m2 --reasoning-parser minimax_m2` (parser registry uses the model_type `minimax_m2`, which both versions share)
- **Verified:** 17*89, snail-problem reasoning, parallel tools × 4 different tools (May 2026)
- **Disk:** ~215 GiB on disk / ~230 GB nominal (125 FP8 safetensors shards — note the `of-00130` filename pattern is misleading; only 125 shards actually exist per the index)
- **Cold load time:** ~30-40 min (vs M2's ~15 min) — 256-expert weight processing is CPU-bound and silent for ~30 min before HBM takeoff
- **Port-conflicts with `minimax-m2`** (both use 8006 + Gaudis 4-7) — one at a time

### gpt-oss-120b BF16 (`unsloth/gpt-oss-120b-BF16`)

- **Image:** `gaudi-vllm-gemma4:0.19.0` — reuses the same 5-patch image as Gemma 4 (the wrapper-unwrap and skip_special_tokens patches cover gpt-oss too)
- See [GPT-OSS.md](GPT-OSS.md) for the full deep-dive
- **Architecture:** GptOssForCausalLM, native vLLM 0.19 — MoE with 128 experts, top-4 per token, 36 alternating sliding(128)/full attention layers
- **TP=4** on Gaudis 3-6; ~60 GB weights/card + ~48 GB KV cache budget
- **Flag:** `--reasoning-parser gptoss` (no tool-call-parser; Harmony handles tools)
- **Endpoints:** `/v1/chat/completions` (Harmony output → `reasoning_content` + `content`) AND `/v1/responses` (canonical for tools)
- **HBM:** ~115 GB / 128 GB per card — tight; lower `--gpu-memory-utilization` or fall back to TP=8 if defrag-OOM appears
- **Disk:** ~240 GB (73 safetensors shards, unquantized BF16)

### Gemma 4 31B Instruct FP8 (`RedHatAI/gemma-4-31b-it-FP8-Dynamic`)

- **Image:** `gaudi-vllm-gemma4:0.19.0` — derived image with **five patches** on top of Habana's vllm-0.19.0-ptfork
- See [GEMMA4.md](GEMMA4.md) for the full deep-dive (what each patch does and why)
- **Env:** same as Qwen + `VLLM_SKIP_TRANSFORMERS_VERSION_CHECK=1`
- **Flag:** `--tool-call-parser gemma4 --reasoning-parser gemma4 --default-chat-template-kwargs '{"enable_thinking":true}'`
- **Endpoints:** both `/v1/chat/completions` (OpenAI shape) AND `/v1/messages` (Anthropic shape) work
- **HBM:** ~95 GiB steady on a single Gaudi 3
- **Special:** 60 layers with heterogeneous attention (local head_dim=256 / global head_dim=512, alternating 5:1)

## How to add a new model preset

1. **Verify it runs** in a one-off container first:
   ```bash
   docker run --rm -it --runtime=habana -e HABANA_VISIBLE_DEVICES=0 \
     -v ~/hf-cache:/root/.cache/huggingface \
     <base-image> bash -lc "vllm serve <model> --tensor-parallel-size 1 --port 8000"
   ```
2. **If it needs patches**, write a `Dockerfile.<modelname>` in `dockerfiles/` that layers on top of the matching `vllm-X.Y.Z-ptfork` base.
3. **Add the preset** to the `PRESETS` associative array in `bin/vllm-launch`:
   ```bash
   [my-preset]="<model-id>|<tp>|<devices>|<port>|<maxlen>|<optional-image>"
   ```
   The 6th `image` field is optional — when omitted, the default `vllm-0.17.1-ptfork` is used.
4. **Document it** in this file (this table) + write a `docs/MYMODEL.md` if patches are involved.
5. **Smoke test:**
   ```bash
   scripts/smoke.sh http://localhost:<port> <served-name> 4000
   ```

## Multi-model concurrency

You can run multiple presets in parallel, each binding to a different Gaudi card.
Devices currently in use by active presets:

```
Gaudi 0    32b-thinking      (port 8000)
Gaudi 1    32b-instruct      (port 8001)
Gaudi 2    gemma4-31b        (port 8004)   ← Anthropic /v1/messages compatible
Gaudi 3-7  free              (can host 30b-a3b, 8b-thinking, 235b-tp4, or your own preset)
```

Open WebUI on port 3000 lists all running endpoints; add new ones via the SQLite
config patch in [docs/PLAYBOOK.md](PLAYBOOK.md#open-webui).

## Performance notes

- **Qwen 32B-Thinking on 1× Gaudi 3:** ~49 tok/s steady, 16k context, hermes tools, FP8.
- **Gemma 4 31B on 1× Gaudi 3:** first request ~15-20 s warmup (`VLLM_SKIP_WARMUP=true`); steady-state similar to Qwen 32B once HPU graphs are cached.
- **235B-A22B TP=4:** chronic HBM fragmentation under sustained load. Mitigations in preset: `--gpu-memory-utilization 0.80` + `HABANA_PGM_LRU_MAX=60000` + `--restart unless-stopped`. Don't run alongside Gemma 4 if you need port 8004 free.
