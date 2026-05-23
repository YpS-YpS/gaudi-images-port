# 09 — future roadmap: open-weight agentic-reasoning models + 5-box fleet plan

**Snapshot date:** 2026-05-13. Open-weight agentic-AI is moving fast — re-validate before acting if you're reading this >1 month after that date.

**Context:** Satyajit is getting **4 more Gaudi 3 boxes** (XE9680 = 8 cards
each), bringing the fleet to **5 boxes / 40 cards / 5 TB HBM total**. This
file is the deployment plan for that scale-up plus the open-weight model
landscape that informs it.

## The open-weight agentic-reasoning leaderboard (May 2026)

Ranked by **SWE-Bench Pro** (the leak-resistant version) and agentic-task
scores. The top three are the open-weight answers to GPT-5.5 and Claude Opus
4.7 on agent workloads.

| # | Model | License | Params | Key scores | Notable |
|---|---|---|---|---|---|
| 🥇 | **GLM-5.1** (Z.AI) | MIT | **744B MoE / 40B active** | SWE-Pro **58.4 (SOTA over GPT-5.4, Opus 4.6, Gemini 3.1 Pro)** · τ³-Bench 70.6 · BrowseComp 68 · MCP-Atlas 71.8 | **8-hour autonomous execution**, 200K ctx, agent-first design |
| 🥈 | **DeepSeek V4-Pro** | MIT | ~671B MoE / 37B active | SWE-Verified 80.6 | **1M context**, strongest general open reasoning |
| 🥉 | **Kimi K2.6** (Moonshot) | open | **1T MoE** | highest open SWE-Pro to date | **Native 300-sub-agent swarms** — agent-first arch |
| 4 | **Qwen3-Coder-Next** | Apache 2.0 | size varies | SWE-Verified 70.6 | 256K ctx, dedicated agent-coding tune |
| 5 | **Llama 4 Maverick** | Llama-4 license | 400B / 17B active | competitive w/ GPT-5.4 | **1M context**, 128 experts |
| 6 | **Llama 4 Scout** | Llama-4 license | 109B / 17B active | strong reasoning | **10M context** (largest open) |
| 7 | **Qwen3-VL-235B-A22B** *(we have)* | Apache 2.0 | 235B / 22B | thinking variant good | already in our preset table |
| 8 | **MiniMax M2** *(we have)* | open | 230B MoE | parallel-tool verified | already running on Box 1 |
| 9 | Qwen3.6-35B-A3B | Apache 2.0 | 35B MoE / 3B active | SWE-Verified 73.4 | **fits 1 card** — best throughput option |

**Note on closed/API-only:** Qwen3.7-Max (announced May 2026) is the strongest
proprietary Chinese model — 35-hour autonomous, 1M context, ranked 13th on
Artificial Analysis Intelligence Index — but **NO open weights, API-only**
($2.50 in / $7.50 out per 1M tokens via OpenRouter). Not deployable on this
fleet. GLM-5.1 is the closest open analogue.

## Hardware fit on the future 5-box fleet

40 Gaudi 3 cards × 128 GiB HBM = **5,120 GiB total HBM**. FP8 weights ≈ 1
byte/param, so the largest open MoEs fit comfortably with our defaults
(`--gpu-memory-utilization 0.85-0.9`, `VLLM_GRAPH_RESERVED_MEM=0.3`).

| Model | FP8 weights | Min TP | Single-box TP=8 HBM/card | Multi-box needed? |
|---|---|---|---|---|
| GLM-5.1 (744B) | ~744 GB | TP=8 | ~93 GB/card · **tight but works** | optional TP=16 across 2 boxes for headroom |
| DeepSeek V4-Pro (671B) | ~671 GB | TP=8 | ~84 GB/card · comfortable | not required |
| Kimi K2.6 (1T) | ~1000 GB | **TP=16** | doesn't fit one box | **yes — 2 boxes Ray cluster** |
| Llama 4 Maverick (400B) | ~400 GB | TP=4 or 8 | TP=8: ~50 GB/card · plenty of room | not required |
| Llama 4 Scout (109B) | ~109 GB | TP=2 | TP=2: ~55 GB/card | not required |
| Qwen3-Coder-Next | ~50-100 GB est. | TP=1-2 | trivial | no |
| Qwen3.6-35B-A3B | ~35 GB | TP=1 | trivial — **run 40 parallel instances if you want** | no |
| Qwen3-VL-235B (existing) | ~235 GB | TP=4 | already running TP=4 | no |
| MiniMax M2 (existing) | ~215 GB | TP=4 | already running TP=4 | no |

## Recommended fleet layout

### Box 1 (current production — keep as-is)
```
Gaudi 0,1       idle (slots for new model trials)
Gaudi 2-3       Gemma 4 31B × 2          (Anthropic /v1/messages — ports 8004, 8005)
Gaudi 4-7       MiniMax M2 TP=4          (port 8006)
```

### Box 2 — Agent flagship: GLM-5.1
- All 8 cards · TP=8 · single endpoint
- **The crown jewel.** Open-weight answer to Qwen3.7-Max
- Point Claude Code / IDEs at this for long-horizon autonomous work
- Expect 1-week patch investigation (Z.AI ChatGLM family, vllm-gaudi may need Gemma-4-style patches)

### Box 3 — Reasoning flagship: DeepSeek V4-Pro
- All 8 cards · TP=8 · ~84 GB/card (comfortable)
- 1M context, strongest general open reasoning
- Use for hard math, research, deep analysis
- **Lowest patch risk** — DeepSeek V3 already has good HPU support, V4 likely a small delta

### Box 4 — Long context + scale: Llama 4 Maverick
- All 8 cards · TP=8 · ~50 GB/card (TONS of room)
- 1M context, 400B/17B active
- Use for retrieval-heavy / document-heavy workloads
- Many concurrent users supported

### Box 5 — Throughput specialist (many parallel small endpoints)
```
Gaudi 0-3       4× Qwen3.6-35B-A3B       (4 independent endpoints, ~35 GB each)
Gaudi 4-7       2× Llama 4 Scout TP=2    (10M context, 2 endpoints)
```
- Designed for: parallel tool calls from many agents, high-throughput API serving
- Each endpoint can handle different model families (parallel diversity)

### Cross-box experiment (later, when basics are stable)
- **Kimi K2.6 TP=16** across Boxes 2+3 with vLLM Ray cluster
- 1T MoE, native sub-agent swarms
- Needs Ray-cluster setup, multi-node TP testing — 1-week project
- Validate the path with DeepSeek V4-Pro TP=16 first (lower-risk model)

## Per-model risk assessment (patches/HPU support)

| Model | Architecture family | HPU support risk | Patch expectation |
|---|---|---|---|
| **GLM-5.1** | Z.AI ChatGLM | Medium-High | vllm-gaudi may need Gemma-4-style patches for new attention shapes. **Budget ~1 week.** |
| **DeepSeek V4-Pro** | DeepSeek MoE (extends V3) | Low | V3 already supported. V4 likely minor delta. **Few days.** |
| **Kimi K2.6** | Moonshot custom 1T MoE | High | Newest, no known HPU port. Multi-node TP=16 adds complexity. **Plan as a research project.** |
| **Llama 4 Maverick/Scout** | Llama 4 MoE | **Low** | Meta + Habana have training history; vLLM has the arch class; HPU port likely lands first. **Hours to days.** |
| **Qwen3-Coder-Next** | Qwen3 | **Trivial** | Same arch family as our running Qwen3-VL-32B. Drop-in. |
| **Qwen3.6-35B-A3B** | Qwen3 MoE | **Trivial** | Same arch as `30b-a3b` preset already in vllm-launch. |

## Action plan (in order of risk-adjusted value)

### Phase 1 — Easy wins (do these the day each new box arrives)
1. `bootstrap.sh --with-gemma4` on each new box. Each comes up with the Qwen + Gemma 4 baseline.
2. **Add `qwen3-coder-next` preset** to `bin/vllm-launch` — same parsers as Qwen3-VL. Test on a single card. (~30 min)
3. **Add `qwen3.6-35b-a3b` preset** — single card, near-instant deploy once weights download. (~30 min)
4. **Launch Llama 4 Maverick on Box 4** — likely works first try with the vllm-0.19 stack. Allocate 1 day for download + verify.

### Phase 2 — High-value medium-risk (1-2 days each)
5. **DeepSeek V4-Pro on Box 3**. Lower risk than GLM-5.1. Validates the "big MoE on a single box" path.
6. **GLM-5.1 on Box 2** — the crown jewel. Allocate 1 week. Patches may be needed. Document in a new `docs/GLM5.md` mirroring the GEMMA4.md format.

### Phase 3 — Frontier experiments (1+ week each)
7. **Multi-node vLLM Ray cluster** between Boxes 2+3 (test with DeepSeek V4-Pro at TP=16). Networking/setup is the hard part — model load is easier.
8. **Kimi K2.6 TP=16** once Ray cluster proven. This unlocks the trillion-param tier.

### Phase 4 — Optimization (anytime)
9. Push toward "90% of NVIDIA" tok/s with the knobs in `docs/GAP_ANALYSIS.md`:
   - `--block-size 128`
   - `--kv-cache-dtype fp8_inc`
   - `PT_HPU_RECIPE_CACHE_CONFIG` (save 5-15 min warmup per restart)
   - `--enable-prefix-caching`
   - Speculative decoding (Eagle/Medusa)
10. Real-bucket pre-warm instead of `VLLM_SKIP_WARMUP=true` (eliminates JIT recompile on every new shape).

## Decision matrix — "I want a model for X"

| Need | Best on this fleet | Box / Preset |
|---|---|---|
| Long-horizon autonomous agent (Claude Code-style) | **GLM-5.1** | Box 2, TP=8 |
| 1M context document reasoning | **DeepSeek V4-Pro** or **Llama 4 Maverick** | Box 3 or 4 |
| 10M context (longest open) | **Llama 4 Scout** | Box 5, TP=2 |
| SWE-Bench coding | **GLM-5.1** | Box 2 |
| Anthropic `/v1/messages` API | Gemma 4 31B *(already have)* | Box 1, port 8004/8005 |
| Parallel tools, fast turnaround | MiniMax M2 *(already have)* | Box 1, port 8006 |
| Vision + reasoning | Qwen3-VL-32B-Thinking *(already have)* | Box 1, port 8000 |
| High throughput, many parallel users | Qwen3.6-35B-A3B × N instances | Box 5 |
| Coding-specialist | **Qwen3-Coder-Next** | Box 5 single card |

## Open questions to research before Phase 2

- [ ] Does `vllm-gaudi 0.19` have an HPU class for GLM (ChatGLM5)? Check `/usr/local/lib/python3.12/dist-packages/vllm_gaudi/models/` for `glm*.py` or `chatglm*.py`.
- [ ] Same for Llama 4 (`llama4*.py`).
- [ ] DeepSeek V4 architecture — is `deepseek_v3.py` (HPU) compatible, or are there V4-specific tensor shapes?
- [ ] Kimi K2's architecture — is it upstream in vLLM yet? Check `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/` for `kimi*.py`.
- [ ] Multi-box networking — does Gaudi 3's scale-out NIC fabric support cross-box TP, or do we need separate TCP/RDMA over InfiniBand? Habana docs cover multi-node *training* extensively; need to confirm multi-node *inference* path.

## Snapshot quotes worth keeping

> "GLM-5.1 from Z.ai is the strongest all-around open-source coding model in 2026 for long-horizon agentic engineering."
> — *MarkTechPost*

> "Three open-weight models reach genuine frontier-class capability in May 2026: DeepSeek V4, Kimi K2.6, and GLM-5."
> — *Local AI Master*

> "By locking Qwen3.7-Max behind an API, Alibaba is pivoting to the standard commercial playbook utilized by OpenAI and Anthropic."
> — *Startup Fortune*

## Sources

- [MarkTechPost: GLM-5.1 — SOTA SWE-Bench Pro, 8-hour autonomous](https://www.marktechpost.com/2026/04/08/z-ai-introduces-glm-5-1-an-open-weight-754b-agentic-model-that-achieves-sota-on-swe-bench-pro-and-sustains-8-hour-autonomous-execution/)
- [Codersera: DeepSeek V4-Pro review and benchmarks](https://codersera.com/blog/deepseek-v4-pro-review-benchmarks-pricing-2026/)
- [MindStudio: best open-source LLMs for agentic coding 2026](https://www.mindstudio.ai/blog/best-open-source-llms-agentic-coding-2026)
- [BenchLM: SWE-bench Verified scores for 47 LLMs](https://benchlm.ai/benchmarks/sweVerified)
- [Latent Space: Z.AI GLM-5 new SOTA open weights LLM](https://www.latent.space/p/ainews-zai-glm-5-new-sota-open-weights)
- [Codersera: Best Open-Source LLM May 2026](https://codersera.com/blog/best-open-source-llm-2026-llama-4-qwen-3-5-deepseek-v4-gemma-4-mistral/)
- [VentureBeat: Qwen3.7-Max 35-hour autonomous agent](https://venturebeat.com/technology/alibabas-proprietary-qwen3-7-max-can-run-for-35-hours-autonomously-and-supports-external-harnesses-like-anthropics-claude-code)
- [BenchmarkingAgents: GAIA / AgentBench / WebArena](https://benchmarkingagents.com/agent-benchmarks/)

## Re-validation

When you re-read this file >1 month after the snapshot date above:
1. Re-search for "best open-weight agentic models <current month>" — landscape moves fast
2. Verify each model still tops the leaderboards (newer releases may have shifted rankings)
3. Check vllm-gaudi commit history for new HPU classes — landed support changes priorities
4. **The fleet allocation is the durable part of this doc; the specific model picks are not.**
