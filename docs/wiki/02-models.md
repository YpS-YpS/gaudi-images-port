# 02 — models

Every model that's been tried on this box, what works, what doesn't, and why.

## TL;DR table

| Preset | Model | Port | Cards | TP | Image | Status |
|---|---|---|---|---|---|---|
| `32b-thinking` | `Qwen/Qwen3-VL-32B-Thinking-FP8` | 8000 | 0 | 1 | 0.17.1-ptfork | ✓ working |
| `32b-instruct` | `Qwen/Qwen3-VL-32B-Instruct-FP8` | 8001 | 1 | 1 | 0.17.1-ptfork | ✓ working |
| `30b-a3b` | `Qwen/Qwen3-VL-30B-A3B-Thinking-FP8` | 8002 | 0 | 1 | 0.17.1-ptfork | ✓ working |
| `8b-thinking` | `Qwen/Qwen3-VL-8B-Thinking-FP8` | 8003 | 0 | 1 | 0.17.1-ptfork | ✓ working |
| `gemma4-31b` | `RedHatAI/gemma-4-31b-it-FP8-Dynamic` | 8004 | 2 | 1 | **gaudi-vllm-gemma4:0.19.0** | ✓ working, 5 patches |
| [`glm46v`](../GLM46V.md) | `zai-org/GLM-4.6V-FP8` | 8006 | 1,2 | 2 | **gaudi-vllm-glm46v:0.19.0** | ✓ working, 1 patch, ~51 tok/s |
| [`glm52`](../GLM52.md) | `zai-org/GLM-5.2-FP8` | 8010 | 0-7 | 8 | **gaudi-vllm-glm52:0.19.0** | ✓ working*, ~20 tok/s — experimental force-dense MLA (`glm52-launch`, not a preset) |
| `minimax-m2` | `MiniMaxAI/MiniMax-M2` | 8006 | 4-7 | 4 | gaudi-vllm-gemma4:0.19.0 | ✓ preferred |
| `minimax-m2.7` | `MiniMaxAI/MiniMax-M2.7` | 8006 | 4-7 | 4 | gaudi-vllm-gemma4:0.19.0 | ✓ but slower cold-load, user rejected quality |
| `235b-tp4` | `Qwen/Qwen3-VL-235B-A22B-Thinking-FP8` | 8004 | 4-7 | 4 | 0.17.1-ptfork | ✓ defrag-OOM tendency |
| `235b-tp8` | same as above | 8006 | 0-7 | 8 | 0.17.1-ptfork | ✓ uses whole box |
| `gpt-oss-120b` | `unsloth/gpt-oss-120b-BF16` etc | 8005 | 3-6 | 4 | gaudi-vllm-gemma4:0.19.0 | **✗ INCOHERENT** |

Source: [`bin/vllm-launch`](../../bin/vllm-launch) PRESETS array, line 20-39.
`glm52` is **not** in that array — it has a dedicated launcher [`bin/glm52-launch`](../../bin/glm52-launch) (like `sglang-launch`), because its `--hf-overrides` JSON + extra flags don't fit the preset template.

\* **`glm52` = working, with an asterisk.** The DeepSeek Sparse Attention (DSA) indexer has no HPU kernel, so it runs **force-dense MLA** (`GLM52_FORCE_DENSE` patch): *exact* for sequences ≤ 2048 tokens, empirically coherent to ~3.1k+, and configured at 8K (unvalidated past 2048 — needs a long-context sweep). Real 1M context needs native DSA kernels HPU doesn't have. Full story: [docs/GLM52.md](../GLM52.md).

## Why two different vLLM images

| Image | Used for | Why |
|---|---|---|
| `vault.habana.ai/.../vllm-0.17.1-ptfork-2.10.0:1.24.0-1007` | All Qwen models | Stock Habana, no patches needed. Stable. 49 tok/s on 32B-Thinking. |
| `gaudi-vllm-gemma4:0.19.0` (locally built) | Gemma 4, MiniMax M2/M2.7, gpt-oss attempts | Derived from `vllm-0.19.0-ptfork`. Adds 5 patches needed for Gemma 4's heterogeneous attention. MiniMax happens to also need 0.19+ (handcrafted HpuMiniMaxM2ForCausalLM class lives there). |
| `gaudi-vllm-glm46v:0.19.0` (locally built) | GLM-4.6V-FP8 | Derived `FROM gaudi-vllm-gemma4:0.19.0`. Adds 1 patch (`GLM46V_HPU_3D_OK`) — a 3D-shape fix in the glm4v_moe MoE block. Inherits the Gemma 4 patches (unused here — GLM has uniform head_dim). |
| `gaudi-vllm-glm52:0.19.0` (locally built) | GLM-5.2-FP8 | Derived `FROM gaudi-vllm-gemma4:0.19.0`. Adds 1 patch (`GLM52_FORCE_DENSE`, 3 hunks in `deepseek_v2.py`) that drops the DSA sparse-attention indexer (no HPU kernel) and runs dense MLA. Bakes `ENV VLLM_GLM_DSA_FORCE_DENSE=1`. |

Building the derived image:

```bash
cd /home/satyajit-gaudi/gaudi-setup/dockerfiles
docker build -t gaudi-vllm-gemma4:0.19.0 -f Dockerfile.gemma4 .
```

## Per-model deep links

- **Qwen3-VL family** — no special docs, works stock. Throughput notes in [docs/MODELS.md](../MODELS.md).
- **Gemma 4 31B** — the 5 patches are documented in [docs/GEMMA4.md](../GEMMA4.md). Anthropic `/v1/messages` shape works out of the box once patched.
- **GLM-4.6V-FP8** — ~108B vision MoE, brought up with a single patch on top of the Gemma 4 image. Full story in [docs/GLM46V.md](../GLM46V.md). FP8 checkpoint + ViT vision encoder worked on the generic upstream path with no HPU-specific work.
- **GLM-5.2-FP8** — 753B MoE / ~40B active DSA agentic flagship, running on all 8 cards via the **force-dense MLA** patch (`GLM52_FORCE_DENSE`) because HPU has no DSA indexer kernel. Full story in [docs/GLM52.md](../GLM52.md). Exact ≤ 2048-token context, ~20 tok/s, 313k-token KV. Dedicated launcher `bin/glm52-launch`.
- **MiniMax M2** — works stock with the patched image. Full walkthrough in [docs/MINIMAX.md](../MINIMAX.md).
- **MiniMax M2.7** — same architecture as M2 but 256 experts / top-8 / 3 MTP modules. Cold-load is ~30-40 min (256-expert weight processing is CPU-bound and silent). User tested and rejected on quality grounds. Preset stays for future. [Part 2 of MINIMAX.md](../MINIMAX.md#part-2--minimax-m27-apr-2026).
- **gpt-oss** — does not work on this stack. The HPU MoE kernel produces incoherent logits. Full analysis + upstream-report draft in [docs/GPT-OSS.md](../GPT-OSS.md).

## Conflict pairs (mutually exclusive)

These pairs cannot run simultaneously:

| Pair | Why |
|---|---|
| `gemma4-31b` ↔ `235b-tp4` | both want port 8004 |
| `minimax-m2` ↔ `minimax-m2.7` | same port 8006 + same Gaudis 4-7 |
| `glm46v` ↔ `minimax-m2` / `minimax-m2.7` / `235b-tp8` | all want port 8006 |
| `glm46v` ↔ `gemma4-31b` | both use Gaudi 2 |
| `235b-tp8` ↔ everything else | uses all 8 cards |
| `glm52` ↔ everything else | uses all 8 cards (TP=8, cards 0-7) |

## Served-name reference

For curl / SDK calls (the `model` field), use these strings:

| Preset | Served-as |
|---|---|
| `32b-thinking` | `qwen3-vl-32b-thinking` |
| `32b-instruct` | `qwen3-vl-32b-instruct` |
| `30b-a3b` | `qwen3-vl-30b-a3b` |
| `8b-thinking` | `qwen3-vl-8b-thinking` |
| `gemma4-31b` | `gemma4-31b` |
| `glm46v` | `glm-4.6v` |
| `glm52` (via `glm52-launch`) | `glm-5.2` |
| `minimax-m2` | `minimax-m2` |
| `minimax-m2.7` | `minimax-m2.7` |
| `235b-tp4` / `235b-tp8` | `qwen3-vl-235b-tp4` / `qwen3-vl-235b-tp8` |

Or query `/v1/models` on any port to see what that endpoint reports.

## Models tested but NOT in presets (one-offs)

- **`openai/gpt-oss-120b` (MXFP4)** — same incoherence as the BF16 variant.
- **`openai/gpt-oss-20b`** — same.
- **`Qwen3-VL-30B-A3B-Instruct-FP8`** — never tried; would just add a preset row mirroring the Thinking variant.
- **Qwen3-VL-2B / 4B / 14B** — not pursued (too small to be interesting on this box).

## How to add a new model

Process documented at the bottom of [docs/MODELS.md](../MODELS.md). Short version:

1. Run a one-off `docker run` test to confirm the model loads.
2. If it needs patches: write a `Dockerfile.<modelname>` layering on top of the right base.
3. Add a preset row to `bin/vllm-launch` (`PRESETS` array).
4. If the model uses a non-standard tool/reasoning parser, add a per-preset branch in the parser-selection block (line 60-95 of `vllm-launch`).
5. Document in `docs/MODELS.md` + per-model `docs/<MODEL>.md` if patches are involved.
6. Smoke test: `scripts/smoke.sh http://localhost:<port> <served-name>`.
7. Add a row to this wiki's table above.
