# gaudi-images-port

> **One-click bring-up + multi-model serving kit for an 8× Intel Gaudi 3 box.**
> Run Qwen3-VL, Gemma 4 (Anthropic API), MiniMax M2 / M2.7 — each with one
> command after bootstrap. Battle-tested on a Dell PowerEdge XE9680.

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

---

## tl;dr — one-click on a fresh box

```bash
git clone https://github.com/YpS-YpS/gaudi-images-port.git
cd gaudi-images-port
sudo ./bootstrap.sh --with-gemma4          # ~30-60 min; reboots once if kernel swap is needed
```

After bootstrap exits green, jump to **[Launch any model in one line](#-launch-any-model-in-one-line)** below.

---

## 🚀 Launch any model in one line

After `install.sh` has run (or after bootstrap), every model launches with a
single command. The `vllm-launch` script lives in `bin/` — add it to your PATH
or call it via its full path.

```bash
# add this once
export PATH="$PWD/bin:$PATH"
```

### Cheat sheet

| # | Model | One-line launch | Port | Cards | HBM/card |
|---|---|---|---|---|---|
| 1 | **Qwen3-VL-32B-Thinking** (FP8, `<think>`) | `vllm-launch 32b-thinking` | 8000 | Gaudi 0 | ~76 GB |
| 2 | **Qwen3-VL-32B-Instruct** (FP8, fast) | `vllm-launch 32b-instruct` | 8001 | Gaudi 1 | ~76 GB |
| 3 | **Qwen3-VL-30B-A3B-Thinking** (MoE) | `vllm-launch 30b-a3b` | 8002 | Gaudi 0 | ~70 GB |
| 4 | **Qwen3-VL-8B-Thinking** (small/fast) | `vllm-launch 8b-thinking` | 8003 | Gaudi 0 | ~22 GB |
| 5 | **Gemma 4 31B** (Anthropic `/v1/messages`) | `vllm-launch gemma4-31b` | 8004 | Gaudi 2 | ~95 GB |
| 6 | **MiniMax M2** (230B MoE, 128 experts) | `vllm-launch minimax-m2` | 8006 | Gaudi 4-7 | ~107 GB |
| 7 | **MiniMax M2.7** (229B MoE, 256 experts + MTP) | `vllm-launch minimax-m2.7` | 8006 | Gaudi 4-7 | ~101 GB |
| 8 | Qwen3-VL-235B-A22B-Thinking TP=4 | `vllm-launch 235b-tp4` | 8004 | Gaudi 4-7 | ~110 GB |
| 9 | Qwen3-VL-235B-A22B-Thinking TP=8 | `vllm-launch 235b-tp8` | 8006 | Gaudi 0-7 | ~60 GB |
| ⚠️ | **gpt-oss-120b** (broken — see [GPT-OSS.md](docs/GPT-OSS.md)) | `vllm-launch gpt-oss-120b` | 8005 | Gaudi 3-6 | loads but incoherent |

**Note on conflicts** — these endpoints share resources, only one can run at a time:
- `gemma4-31b` (port 8004) ↔ `235b-tp4` (port 8004)
- `minimax-m2` ↔ `minimax-m2.7` (same port 8006 + same Gaudis 4-7)
- `235b-tp8` ↔ everything else (uses ALL 8 cards)

### Stop, list, follow logs

```bash
vllm-launch list                # show every preset and its config
vllm-launch logs 32b-thinking   # tail logs of one
vllm-launch stop gemma4-31b     # stop one
vllm-launch stop-all            # stop everything starting with vllm-*
```

### Test any endpoint (works for all 9 models)

```bash
# minimal smoke
curl -s http://localhost:<PORT>/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<served-name>","max_tokens":200,
       "messages":[{"role":"user","content":"What is 17 * 89?"}]}' \
  | python3 -m json.tool

# full smoke (text + vision + tools + parallel tools)
scripts/smoke.sh http://localhost:<PORT> <served-name>

# Anthropic-shape smoke (Gemma 4 and MiniMax only)
curl -s http://localhost:<PORT>/v1/messages \
  -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"<served-name>","max_tokens":500,
       "messages":[{"role":"user","content":"hi"}]}'
```

The `<served-name>` for each preset:

| Preset | served name |
|---|---|
| `32b-thinking` | `qwen3-vl-32b-thinking` |
| `32b-instruct` | `qwen3-vl-32b-instruct` |
| `30b-a3b` | `qwen3-vl-30b-a3b` |
| `8b-thinking` | `qwen3-vl-8b-thinking` |
| `gemma4-31b` | `gemma4-31b` |
| `minimax-m2` | `minimax-m2` |
| `minimax-m2.7` | `minimax-m2.7` |
| `235b-tp4` / `235b-tp8` | `qwen3-vl-235b-tp4` / `qwen3-vl-235b-tp8` |

---

## 📚 Per-model details

Each row below shows: **what's special**, **launch**, **smoke**, **deep-dive doc**.

### 1. Qwen3-VL-32B-Thinking (Gaudi 0, port 8000)

Vision + reasoning + tools. Returns `<think>...</think>` parsed into a separate
`reasoning_content` field. Hermes tool format. Steady ~49 tok/s single user.

```bash
vllm-launch 32b-thinking
scripts/smoke.sh http://localhost:8000 qwen3-vl-32b-thinking
```

### 2. Qwen3-VL-32B-Instruct (Gaudi 1, port 8001)

Same model family, no reasoning preamble — faster direct answers. Vision + Hermes tools.

```bash
vllm-launch 32b-instruct
scripts/smoke.sh http://localhost:8001 qwen3-vl-32b-instruct
```

### 3. Qwen3-VL-30B-A3B-Thinking (Gaudi 0, port 8002)

MoE — 30B total, ~3B active. Same Hermes tools + Qwen3 reasoning parser.

```bash
vllm-launch 30b-a3b
```

### 4. Qwen3-VL-8B-Thinking (Gaudi 0, port 8003)

Small, fast iteration model. Same Hermes tools + reasoning parser.

```bash
vllm-launch 8b-thinking
```

### 5. Gemma 4 31B Instruct FP8 — Anthropic-shape (Gaudi 2, port 8004)

**The Anthropic-API-compatible endpoint.** Implements both `/v1/chat/completions`
*and* `/v1/messages` (typed content blocks). Runs via a custom-patched image —
five patches on top of Habana's `vllm-0.19.0-ptfork`.

```bash
vllm-launch gemma4-31b

# Anthropic SDK (drop-in — zero code changes)
python3 - <<'EOF'
import anthropic
c = anthropic.Anthropic(base_url="http://localhost:8004", api_key="dummy")
r = c.messages.create(model="gemma4-31b", max_tokens=512,
                     messages=[{"role":"user","content":"hi"}])
print(r.content[0].text)
EOF
```

Patches → see **[docs/GEMMA4.md](docs/GEMMA4.md)** for the deep-dive.

### 6. MiniMax M2 (Gaudis 4-7 TP=4, port 8006)

230B MoE, **128 experts top-4**, FP8. Anthropic shape + parallel tools verified.
No special patches — vllm-gaudi ships a handcrafted `HpuMiniMaxM2ForCausalLM`.

```bash
vllm-launch minimax-m2
scripts/smoke.sh http://localhost:8006 minimax-m2
```

### 7. MiniMax M2.7 (Gaudis 4-7 TP=4, port 8006) — NEW

229B MoE, **256 experts top-8**, FP8, with 3 MTP modules. Same architecture
class as M2; preset auto-applies tightened HBM knobs (`--gpu-memory-utilization
0.85`, `--max-num-seqs 16`).

```bash
# stop M2 first if it's running (same port, same Gaudis):
vllm-launch stop minimax-m2

vllm-launch minimax-m2.7
scripts/smoke.sh http://localhost:8006 minimax-m2.7
```

**Cold-load warning** — ~30-40 min from launch to ready (256-expert × 62-layer
× TP=4 weight processing is CPU-bound and silent for the first ~30 min — don't
kill it, check `docker stats` for climbing Block I/O).

Quirks → see **[docs/MINIMAX.md](docs/MINIMAX.md)**.

### 8/9. Qwen3-VL-235B-A22B-Thinking (TP=4 or TP=8)

The flagship Qwen MoE. TP=4 puts it on Gaudis 4-7 (port 8004 — conflicts with
Gemma 4); TP=8 spreads across all 8 cards (port 8006 — uses the whole box).

```bash
vllm-launch 235b-tp4         # 4 cards, 8192 ctx
vllm-launch 235b-tp8         # 8 cards, 32768 ctx
```

**HBM caveat:** 235B-A22B has chronic HBM fragmentation under sustained
inference. The preset already lowers `--gpu-memory-utilization` to 0.80,
sets `HABANA_PGM_LRU_MAX=60000`, and uses `--restart unless-stopped` for
auto-recovery from defrag-OOM.

### ⚠️ gpt-oss-120b (broken on this stack — for reference only)

```bash
vllm-launch gpt-oss-120b    # loads but returns incoherent text
```

Full failure analysis + upstream bug report draft in **[docs/GPT-OSS.md](docs/GPT-OSS.md)**.

---

## 🔧 Run any model fully manually (no `vllm-launch` script)

If you want the raw `docker run` (e.g. inside another orchestrator), the
exact env vars + flags for each preset live in **[`bin/vllm-launch`](bin/vllm-launch)**
(lines 20-39 for the `PRESETS` table; lines 60-95 for the per-preset parser
selection; lines 114-152 for the actual `docker run` call). Copy that block
and substitute the preset values.

Quick template for **any FP8 Qwen** model:

```bash
docker run -d --runtime=habana --restart unless-stopped \
  --name vllm-myrun \
  -e HABANA_VISIBLE_DEVICES=0 \
  -e PT_HPU_LAZY_MODE=1 \
  -e PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false \
  -e VLLM_GRAPH_RESERVED_MEM=0.3 \
  -e VLLM_SKIP_WARMUP=true \
  -e HF_HOME=/root/.cache/huggingface \
  --net=host --ipc=host --cap-add=sys_nice \
  -v $HOME/hf-cache:/root/.cache/huggingface \
  --entrypoint /bin/bash \
  vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/vllm-0.17.1-ptfork-2.10.0:1.24.0-1007 \
  -lc 'exec vllm serve Qwen/Qwen3-VL-32B-Thinking-FP8 \
       --tensor-parallel-size 1 --host 0.0.0.0 --port 8000 \
       --enable-auto-tool-choice --tool-call-parser hermes \
       --reasoning-parser qwen3 \
       --max-model-len 16384 --gpu-memory-utilization 0.9 \
       --max-num-seqs 32 --trust-remote-code \
       --served-model-name qwen3-vl-32b-thinking'
```

For **Gemma 4 / MiniMax**, swap the image to `gaudi-vllm-gemma4:0.19.0` (built
via `cd dockerfiles && docker build -t gaudi-vllm-gemma4:0.19.0 -f Dockerfile.gemma4 .`)
and use the corresponding parsers (`--tool-call-parser gemma4 --reasoning-parser
gemma4` for Gemma; `--tool-call-parser minimax_m2 --reasoning-parser minimax_m2`
for MiniMax).

---

## 🗺️ Documentation map

| Doc | When to read it |
|---|---|
| **[docs/wiki/](docs/wiki/)** | **Onboarding wiki** — start here if you're new to this box. Tree of 10 files covering everything: hardware, models, patches, constraints, tools, debugging playbook, gotchas, history. |
| **[docs/MODELS.md](docs/MODELS.md)** | Per-model patches, flags, performance notes, how to add a new preset |
| **[docs/GEMMA4.md](docs/GEMMA4.md)** | Deep-dive on the 5 patches required for Gemma 4 on vllm-gaudi 0.19 |
| **[docs/MINIMAX.md](docs/MINIMAX.md)** | MiniMax M2 + M2.7 — handcrafted HPU class, 256-expert/MTP quirks, Anthropic + parallel tools |
| **[docs/GPT-OSS.md](docs/GPT-OSS.md)** | gpt-oss 20b/120b failure analysis (MoE backend bug, upstream-report draft) |
| [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) | Hardware / OS / BIOS / network checklist before running `bootstrap.sh` |
| [docs/PLAYBOOK.md](docs/PLAYBOOK.md) | Day-1 walkthrough, day-2 ops (start/stop/swap models, multi-model parallel) |
| [docs/CHECKPOINT.md](docs/CHECKPOINT.md) | Snapshot of a known-good system (exact versions, paths, configs) |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Every problem hit during bring-up + the actual fix |
| [docs/API_ACCESS.md](docs/API_ACCESS.md) | Pointing IDEs / Open WebUI / Anthropic-SDK code at the endpoints |
| [docs/GAP_ANALYSIS.md](docs/GAP_ANALYSIS.md) | What's still missing vs the NVIDIA stack (perf, model coverage, ecosystem) |

---

## ⚙️ What `install.sh` sets up

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
| vLLM image (Gemma/MiniMax) | `gaudi-vllm-gemma4:0.19.0` — derived image with **5 patches** (see [docs/GEMMA4.md](docs/GEMMA4.md)) |
| Boot persistence | `gaudi-tune.service` re-applies MSRs + NIC bring-up every boot |
| Tools | `gaudi-dash` tmux dashboard (btop + htop + hl-smi), `vllm-launch` multi-preset launcher |

---

## 📂 Layout

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

---

## 🩺 Verification

After bootstrap completes:

```bash
sudo bash scripts/check.sh                              # system green
sudo bash scripts/verify.sh                              # + hl_qual + API smoke
scripts/smoke.sh http://localhost:8000 qwen3-vl-32b-thinking
scripts/smoke.sh http://localhost:8004 gemma4-31b
```

Each smoke test exercises: `/v1/models` → plain text → vision (inline PNG) →
single tool call → parallel tools × 4 cities → parallel mixed × 4 different tools.

---

## ⚠️ Caveats

- `install.sh` modifies `/etc/`, GRUB, kernel modules, MSRs, hugepages. Read it before running on a box you care about.
- Gemma 4 base image is gated behind Habana's `vault.habana.ai` registry — `docker login vault.habana.ai` may be required (see [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md)).
- gpt-oss family (`gpt-oss-120b`, `gpt-oss-20b`) **loads but produces incoherent output** on this stack — see [docs/GPT-OSS.md](docs/GPT-OSS.md).
- MiniMax M2 and M2.7 share port 8006 + Gaudis 4-7 — only one at a time.
- MiniMax M2.7 cold load is ~30-40 min (256-expert weight loop is CPU-bound and silent for the first ~30 min — don't kill it).

---

## License

MIT — see [LICENSE](LICENSE) if present, otherwise this notice. Use at your own
risk. Not affiliated with Intel/Habana/Google/RedHat/Qwen/MiniMax/OpenAI/Anthropic.
