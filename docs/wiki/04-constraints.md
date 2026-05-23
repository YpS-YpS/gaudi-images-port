# 04 — constraints

The math behind what's possible / not on this box. If you're sizing a new
model or planning a TP layout, start here.

## TP divisibility (the most common gotcha)

For tensor parallelism to work, **TP must evenly divide every parallelizable
dimension**:

1. `num_attention_heads mod TP == 0`
2. `num_key_value_heads mod TP == 0`  ← usually the binding one
3. `num_local_experts mod TP == 0` (MoE only)

If any one fails, vLLM refuses to load.

### MiniMax M2 / M2.7 worked example

| Field | M2 | M2.7 | Divisors ≤ 8 |
|---|---|---|---|
| `num_attention_heads` | 48 | 48 | {1, 2, 3, 4, 6, 8} |
| `num_key_value_heads` | **8** | **8** | **{1, 2, 4, 8} ← binding** |
| `num_local_experts` | 128 | 256 | {1, 2, 4, 8, …} |

**Intersection = {1, 2, 4, 8}**. So **TP=6 fails** because 8 KV heads ÷ 6 = 1.33.

### Quick check for any HF model

```bash
python3 -c '
import json, sys
c = json.load(open(sys.argv[1]))
kv  = c.get("num_key_value_heads")
ah  = c.get("num_attention_heads")
exp = c.get("num_local_experts", float("inf"))
valid = [tp for tp in (1,2,3,4,5,6,7,8) if (kv%tp==0 and ah%tp==0 and (exp==float("inf") or exp%tp==0))]
print(f"kv_heads={kv}, attn_heads={ah}, experts={exp}")
print(f"valid TP ≤ 8: {valid}")
' ~/hf-cache/hub/models--<ORG>--<MODEL>/snapshots/*/config.json
```

### Per-card-cost table for MiniMax M2.7 (230 GB on disk)

| TP | Cards | Weights/card | Verdict |
|---|---|---|---|
| 1 | 1 | 230 GB | doesn't fit (128 GiB HBM per card) |
| 2 | 2 | 115 GB | fits weights but no KV room (need ~75 GB) |
| 4 | 4 | 58 GB | ✓ default, ~101 GB/card steady-state |
| 6 | 6 | — | invalid (8 KV heads ÷ 6 ≠ int) |
| 8 | 8 (full box) | 29 GB | ✓ maximum throughput, takes whole box |

The full graphical version of this section lives in [docs/index.html § TP math](../index.html#tp-math).

## HBM per-card budget — how to estimate

```
per-card HBM = weights + KV cache pool + HPU graph reserve

weights         ≈ (model_disk_size_GB / TP)
KV cache pool   ≈ (gpu-memory-utilization × 128 GiB) − weights
graph reserve   ≈ VLLM_GRAPH_RESERVED_MEM × (128 GiB − weights)
                   (default 0.3 × free)
```

Typical defaults: `gpu-memory-utilization=0.9`, `VLLM_GRAPH_RESERVED_MEM=0.3`.
That gets us:

- weights ≈ disk/TP
- KV pool ≈ ~0.9 × 128 − weights ≈ 115 − weights GB
- graph reserve ≈ 0.3 × (128 − weights) GB

Pool reservation is what `hl-smi` reports — it doesn't grow at runtime.

### Active-cards reference (May 2026)

| Card | Model | Used / Free |
|---|---|---|
| 0 | (idle, Qwen removed) | 670 MiB / 127 GiB |
| 1 | (idle) | 670 MiB / 127 GiB |
| 2 | Gemma 4 31B (#1) | ~95 GiB / ~33 GiB (after restart) |
| 3 | Gemma 4 31B (#2) | ~102 GiB / ~25 GiB |
| 4-7 | MiniMax M2 TP=4 | ~107 GiB / ~20 GiB each |

## Defrag-OOM — the runtime failure mode

Even though vLLM pre-allocates the HBM pool, **internal fragmentation
accumulates** over days of inference with varied sequence shapes. Symptom:
`hl-smi` shows the same pool reservation but new graph captures or KV
allocations fail with "out of memory" — even though there's nominally free space.

Mitigations:
1. `--restart unless-stopped` — container auto-recovers (set on all FP8 MoE presets)
2. `--gpu-memory-utilization 0.85` (vs 0.9) — gives the allocator more slack
3. `HABANA_PGM_LRU_MAX=60000` — larger recipe-cache reduces eviction churn
4. **Weekly container restart** — reclaims fragmented allocations. We saw 35 GiB reclaimed on a 5-day-old Gemma 4 container.

## Conflict pairs

These pairs cannot run simultaneously:

| Pair | Conflict |
|---|---|
| `gemma4-31b` ↔ `235b-tp4` | both want port 8004 |
| `minimax-m2` ↔ `minimax-m2.7` | same port 8006 + same Gaudis 4-7 |
| `235b-tp8` ↔ everything else | uses all 8 cards |
| any two presets on the same Gaudi | hl-smi will show the second's allocation fail |

## Cold-load timing reference

| Preset | Disk size | Cold load (launch → "Application startup complete") |
|---|---|---|
| `32b-thinking` / instruct | ~70 GB | ~5-10 min |
| `8b-thinking` | ~16 GB | ~3-5 min |
| `gemma4-31b` | ~32 GB | ~10-15 min (5-patch image is heavier) |
| `minimax-m2` | ~215 GB | ~15 min |
| `minimax-m2.7` | ~230 GB | **~30-40 min** (256-expert weight loop is CPU-bound and silent for ~30 min before HBM takeoff) |
| `235b-tp4` | ~250 GB | ~20-25 min |
| `235b-tp8` | ~250 GB | ~15-20 min (TP=8 splits load work across more workers) |

**Important:** `VLLM_SKIP_WARMUP=true` is set on all presets, so the FIRST
inference request after "Application startup complete" pays an additional
30-60 s graph-capture cost. Subsequent requests are fast.

## Network constraints

- **Internal scale-out NICs (24× per box)**: Habana-proprietary mesh, used for TP communication. Not Ethernet. Must be up before any TP>1 run — `gaudi-tune.service` brings them up at boot.
- **External NIC**: standard. Used for HF downloads, OpenAI-compat API serving, Open WebUI.
- **Intel corp proxy `http://proxy-dmz.intel.com:912`** — required for HF egress. Set on dockerd + every launch container. `NO_PROXY` must include `localhost,127.0.0.1,.intel.com` or local API calls go through proxy and time out.
