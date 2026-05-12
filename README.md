# gaudi-images-port

One-click bring-up + ops kit for an **Ubuntu 24.04** Intel **Gaudi 3** (HL-325) box
running **vLLM** to serve Qwen3-VL and **Gemma 4** with vision, tool-calling,
reasoning, and Anthropic `/v1/messages` compatibility.

Battle-tested on a Dell PowerEdge XE9680 (8× Gaudi 3, 2× Xeon 8480+, 2 TB RAM).

```
fresh Ubuntu 24.04 box
        │
        │   sudo ./bootstrap.sh --with-gemma4
        │
        ▼
  ┌────────────────────────────────────────────────────────────┐
  │  installs kernel 6.8, Habana driver, hugepages, NICs,      │
  │  Docker; pulls vllm-0.17.1 + vllm-0.19.0 base images;       │
  │  builds the patched Gemma 4 image; downloads both models;  │
  │  launches Qwen 32B-Thinking on :8000 + Gemma 4 on :8004    │
  └────────────────────────────────────────────────────────────┘
```

## tl;dr — one-click on a fresh box

```bash
git clone https://github.com/YpS-YpS/gaudi-images-port.git
cd gaudi-images-port
sudo ./bootstrap.sh --with-gemma4          # one command; ~30-60 min depending on bandwidth
                                            # reboots once if the kernel needs swapping
```

After bootstrap exits green:

```bash
# Endpoints
http://<box>:8000/v1/chat/completions        # Qwen 32B-Thinking (OpenAI shape)
http://<box>:8004/v1/chat/completions        # Gemma 4 31B (OpenAI shape, tool_calls + reasoning)
http://<box>:8004/v1/messages                # Gemma 4 31B (Anthropic shape, content blocks)

# Smoke
scripts/smoke.sh http://localhost:8000 qwen3-vl-32b-thinking
scripts/smoke.sh http://localhost:8004 gemma4-31b
```

## What you get

| Layer | Setup |
|---|---|
| Kernel | `6.8.0-110-generic` installed and pinned; old kernels remain in GRUB fallback |
| GRUB | `iommu=pt intel_iommu=on` baked in — required by Habana on kernel 6.8 |
| Habana driver | `1.24.0` with all 7 modules loaded (incl. `habanalabs_ib` + `ib_uverbs`) |
| Hugepages | `vm.nr_hugepages=24640` (~48 GB) per Habana's installer formula |
| MSRs (Sapphire Rapids) | `IA32_ENERGY_PERF_BIAS=0`, `HWP_REQUEST=max` |
| Networking | All 24 internal Gaudi scale-out NICs brought up |
| Docker | Docker engine + Habana container runtime + corp proxy drop-in (Intel) |
| vLLM image (Qwen) | Habana's `vllm-0.17.1-ptfork` (stock, no patches) |
| vLLM image (Gemma) | `gaudi-vllm-gemma4:0.19.0` — derived image with **5 patches** (see [docs/GEMMA4.md](docs/GEMMA4.md)) |
| Boot persistence | `gaudi-tune.service` re-applies MSRs + NIC bring-up every boot |
| Tools | `gaudi-dash` tmux dashboard (btop + htop + hl-smi), `vllm-launch` multi-preset launcher |

## Documentation map

| Doc | When to read it |
|---|---|
| **`docs/MODELS.md`** | List of all presets, ports, devices, flags, and per-model patches |
| **`docs/GEMMA4.md`** | Deep-dive on the five patches required for Gemma 4 on vllm-gaudi 0.19 — what each does and why |
| `docs/REQUIREMENTS.md` | Hardware / OS / BIOS / network checklist before running `bootstrap.sh` |
| `docs/PLAYBOOK.md` | Day-1 walkthrough, day-2 ops (start/stop/swap models, multi-model parallel) |
| `docs/CHECKPOINT.md` | Snapshot of a known-good system (exact versions, paths, configs) |
| `docs/TROUBLESHOOTING.md` | Every problem hit during bring-up + the actual fix (Bad pagetable, ibv init failure, FP8 GEMM crash, hf-xet re-download, defrag-OOM, …) |
| `docs/API_ACCESS.md` | Pointing IDEs / Open WebUI / Anthropic-SDK code at the endpoints |
| `docs/GAP_ANALYSIS.md` | What's still missing vs the NVIDIA stack (perf, model coverage, ecosystem) |

## Layout

```
gaudi-images-port/
├── bootstrap.sh                # ⬅ one-click entrypoint (this doc's tl;dr)
├── install.sh                  # idempotent base bring-up (kernel, driver, docker)
├── docs/                       # see the map above
├── scripts/
│   ├── check.sh                # readiness — green/red checklist
│   ├── fix.sh                  # re-apply runtime tunings (no reboot)
│   ├── verify.sh               # readiness + hl_qual + API smoke
│   └── smoke.sh                # API-only smoke (text + vision + tools)
├── systemd/                    # configs that install.sh drops into /etc/
│   ├── gaudi-tune.service      # MSRs + NIC bring-up on every boot
│   ├── docker-http-proxy.conf  # corp proxy for dockerd
│   ├── habana-modules.conf     # ib_uverbs + habanalabs_ib autoload
│   └── habana-hugepages.conf   # vm.nr_hugepages persistence
├── dockerfiles/                # derived images that patch the Habana base
│   ├── Dockerfile.gemma4       # 5 patches → gaudi-vllm-gemma4:0.19.0
│   └── patch_kv_divisibility.py
└── bin/
    ├── gaudi-dash              # tmux dashboard
    └── vllm-launch             # multi-preset vLLM serving launcher
```

## Quick API examples

After `vllm-launch 32b-thinking` and `vllm-launch gemma4-31b`:

```bash
# 1. Qwen — plain text on :8000
curl -sX POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-vl-32b-thinking",
       "messages":[{"role":"user","content":"What is 17 * 89?"}],
       "max_tokens":200}'

# 2. Gemma 4 — Anthropic /v1/messages on :8004
curl -sX POST http://localhost:8004/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"gemma4-31b","max_tokens":500,
       "system":"You are concise.",
       "messages":[{"role":"user","content":"Solve: snail climbs 3m/day, slides 2m/night, 10m wall — how many days?"}]}'

# 3. Anthropic SDK (drop-in — zero code changes)
python3 - <<'EOF'
import anthropic
c = anthropic.Anthropic(base_url="http://localhost:8004", api_key="dummy")
r = c.messages.create(model="gemma4-31b", max_tokens=1024,
                     messages=[{"role":"user","content":"hello"}])
print(r.content[0].text)
EOF
```

## Per-model details

See [`docs/MODELS.md`](docs/MODELS.md) for the full table. Short version:

| Preset | Model | Port | Device | Special |
|---|---|---|---|---|
| `32b-thinking` | Qwen3-VL-32B-Thinking-FP8 | 8000 | Gaudi 0 | `<think>` reasoning |
| `32b-instruct` | Qwen3-VL-32B-Instruct-FP8 | 8001 | Gaudi 1 | direct answers |
| `8b-thinking` | Qwen3-VL-8B-Thinking-FP8 | 8003 | Gaudi 0 | small / fast |
| `30b-a3b` | Qwen3-VL-30B-A3B-Thinking-FP8 | 8002 | Gaudi 0 | MoE, 3B active |
| **`gemma4-31b`** | gemma-4-31b-it-FP8-Dynamic | 8004 | Gaudi 2 | Anthropic `/v1/messages`, 5 patches ([GEMMA4.md](docs/GEMMA4.md)) |
| **`gpt-oss-120b`** | unsloth/gpt-oss-120b-BF16 | 8005 | Gaudi 3-6 (TP=4) | OpenAI 120B MoE, BF16 unquantized, Harmony ([GPT-OSS.md](docs/GPT-OSS.md)) |
| `235b-tp4` | Qwen3-VL-235B-A22B-Thinking-FP8 | 8004 | Gaudi 4-7 | 235B MoE on 4 cards (port-conflicts with `gemma4-31b`) |
| `235b-tp8` | Qwen3-VL-235B-A22B-Thinking-FP8 | 8006 | Gaudi 0-7 | full TP=8 |

## Why the patches?

vLLM 0.19 introduced **`UniformTypeKVCacheSpecs`** to support models with
heterogeneous attention head dimensions per layer (Gemma 4 has local head_dim=256
and global head_dim=512 alternating). Habana's `vllm-gaudi` 0.19 plugin shipped
without handlers for this new wrapper class, so out-of-the-box Gemma 4 fails
with one of five distinct errors depending on which subsystem hits the wrapper
first. Each patch closes one gap:

1. `head_size=512` not in the allow-list → add it
2. `UniformTypeKVCacheSpecs` not recognized by the KV cache allocator → unwrap to inner spec
3. `skip_special_tokens=True` strips Gemma's `<|channel>` markers → flip default
4. `tensor.size` is a Python float upstream → coerce to int
5. (launch flag) `chat_template_kwargs.enable_thinking=true` for default reasoning mode

Full reasoning and per-patch reproduction steps in [docs/GEMMA4.md](docs/GEMMA4.md).

## Verification

After bootstrap completes:

```bash
sudo bash scripts/check.sh                              # system green
sudo bash scripts/verify.sh                              # + hl_qual + API smoke
scripts/smoke.sh http://localhost:8000 qwen3-vl-32b-thinking
scripts/smoke.sh http://localhost:8004 gemma4-31b
```

Each smoke test exercises: `/v1/models` → plain text → vision (inline PNG) →
single tool call → parallel tools × 4 cities → parallel mixed × 4 different tools.

## Caveats

- `install.sh` modifies `/etc/`, GRUB, kernel modules, MSRs, hugepages. Read it before running on a box you care about.
- Gemma 4 base image is gated behind Habana's `vault.habana.ai` registry — `docker login vault.habana.ai` may be required (see `docs/REQUIREMENTS.md`).
- 235B-A22B has chronic HBM fragmentation under sustained inference. Preset uses `--gpu-memory-utilization 0.80` + `--restart unless-stopped` to recover.

## License

MIT — see [LICENSE](LICENSE) if present, otherwise this notice. Use at your own
risk. Not affiliated with Intel/Habana/Google/RedHat/Qwen/Anthropic.
