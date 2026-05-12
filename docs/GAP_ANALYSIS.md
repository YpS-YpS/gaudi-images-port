# GAP_ANALYSIS — current setup vs Habana official docs

Synthesis of two independent doc-tree crawls of `docs.habana.ai/en/latest/` and
`docs.vllm.ai/projects/gaudi/en/latest/` against our `gaudi-setup` repo.
Each item below cites the doc page that prescribes the value or behavior.

## 0. The headline finding

**Qwen3-VL-32B-Thinking-FP8 at TP=1 on Gaudi 3 IS officially validated** —
listed verbatim in `vllm-gaudi/.../validated_models.html`. So our model+hardware
combo is supported. Our `--enforce-eager` workaround is a config issue, not a
hardware/software-stack issue.

**Most likely root cause:** `PT_HPUGRAPH_DISABLE_TENSOR_CACHE` is not explicitly
set, defaulting to `true` on the ptfork image. Habana docs:

> `PT_HPUGRAPH_DISABLE_TENSOR_CACHE` (default: `false`) —
> **"Must be set to false for LLaVA, Qwen, and RoBERTa models"**
> — [vllm-gaudi env_variables.md](https://docs.vllm.ai/projects/gaudi/en/latest/configuration/env_variables.html)

Our crash signature (`ValidateSyncInputTensors tensor_data is empty` in
`down_proj/hpu__fp8_gemm_v2`) matches this exact bug class:
a tensor handle cached inside a captured HPU graph gets freed before replay.

## 1. The vllm-gaudi compatibility matrix

For SynapseAI `1.24.0` (what we're running):

| vllm-gaudi plugin | 1.22.2 | 1.23.0 | 1.24.0 |
|---|---|---|---|
| 0.11.2 | ✓ | ✓ | ✗ |
| 0.13.0 / 0.14.1 / 0.15.1 / 0.16.0 | ✗ | ✓ | ✗ |
| **0.17.1** | ✗ | ✓ | **✓** |

We're already on 0.17.1 — the only validated version. Don't downgrade.
([compatibility_matrix](https://docs.vllm.ai/projects/gaudi/en/latest/getting_started/compatibility_matrix.html))

## 2. The image variants on `vault.habana.ai`

| Tag | Stack inside | Lazy mode? | torch.compile? |
|---|---|---|---|
| `vllm-0.17.1-ptfork-2.10.0:1.24.0-1007` | Habana PT fork + vllm-gaudi | ✓ | partial |
| `vllm-0.17.1-ptupstream-2.10.0:1.24.0-1007` | upstream PT + vllm-gaudi | ✗ | ✓ |
| `pytorch-installer-2.10.0:1.24.0-1007` | PT only (no vllm) | depends | depends |

We're on **ptfork**. Per Habana, `PT_HPU_LAZY_MODE=1` is supported only on
ptfork; ptupstream forces `PT_HPU_LAZY_MODE=0` (torch.compile).

## 3. Validated model + TP layout (Gaudi 3, vllm-gaudi 0.17.1)

| Model | TP | Datatype |
|---|---|---|
| Qwen/Qwen3-VL-32B-Instruct | 1 | BF16, FP8 |
| Qwen/Qwen3-VL-32B-Thinking | 1 | BF16, FP8 |
| Qwen/Qwen3-VL-235B-A22B-Instruct | 8 | BF16 |
| Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 | **4** | FP8 |
| Qwen/Qwen3-VL-235B-A22B-Thinking | 8 | BF16 |
| Qwen/Qwen3-VL-235B-A22B-Thinking-FP8 | **4** | FP8 |

For the 235B-FP8 variants: TP=**4** is the validated layout, NOT TP=8.
Our `bin/vllm-launch` had a TP=8 preset — change to TP=4 only or document
that TP=8 is unvalidated.
([validated_models](https://docs.vllm.ai/projects/gaudi/en/latest/getting_started/validated_models.html))

## 4. Critical env-var gaps (in order of likely impact)

| Env var | Default (ptfork) | Recommended | Why |
|---|---|---|---|
| `PT_HPUGRAPH_DISABLE_TENSOR_CACHE` | `true` | **`false`** | Required for Qwen/LLaVA/RoBERTa — likely fixes our crash |
| `PT_HPU_ENABLE_LAZY_COLLECTIVES` | `false` | **`true`** | Required for TP>1 + HPU graphs (235B path) |
| `PT_HPU_WEIGHT_SHARING` | unset | **`0`** | Required for any FP8-quantized run ([Inference_Using_FP8](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html)) |
| `RUNTIME_SCALE_PATCHING` | true (compile) / false (lazy) | **`0`** if FP8 + torch.compile crashes with scale-method assert | Per [troubleshooting](https://docs.vllm.ai/projects/gaudi/en/latest/general/troubleshooting.html) |
| `PT_HPU_RECIPE_CACHE_CONFIG` | unset | **`/recipe-cache,false,4096,false`** | Llama-8B FP8 startup 504s → 34s |
| `HABANA_PGM_LRU_MAX` | 30000 | **60000** | Bigger graph cache, fewer evictions on long-context VL |
| `VLLM_GRAPH_RESERVED_MEM` | 0.1 | **0.2** for FP8 32B | More HBM for graph capture |
| `VLLM_ENGINE_ITERATION_TIMEOUT_S` | (default) | **600** | FP8 compile is slow, default times out |
| `VLLM_RPC_TIMEOUT` | (default) | **600000** | Same |
| `OMP_NUM_THREADS` | (=ncores=224) | **224 / TP** | Avoid 224-thread thrash |
| `VLLM_EXPONENTIAL_BUCKETING` | true | leave true | Default since vllm-gaudi 1.21.0-post1 |
| `VLLM_HPU_LOG_STEP_GRAPH_COMPILATION` | false | **true** during diagnosis | Per-step compile log |

## 5. vLLM CLI flag gaps

| Flag | Default | Recommended | Why |
|---|---|---|---|
| `--block-size` | 16 | **128** | Gaudi3 MME utilization (vllm-fork README quote) |
| `--max-num-seqs` | 256 | **32** for long ctx | Default forces preempt/recompile on Gaudi |
| `--kv-cache-dtype` | bf16 | **fp8_inc** for FP8 models | Match weight precision, double KV capacity |
| `--quantization` | (auto) | **inc** for INC-calibrated FP8 | Per fork README FP8 example |
| `--weights-load-device` | hpu | **cpu** | OOM mitigation per plugin v0.16.0 release notes |
| `--enable-prefix-caching` | off | **on** | Big TTFT win for chat / shared system prompt |
| `--enforce-eager` | off | **off** | Drop it once env vars above are set |
| `--num-scheduler-steps` | 1 | n/a | Deprecated in plugin (replaced by async-scheduler) |
| `--mm-encoder-tp-mode` | (default) | `data` for VL TP>1 | Per Qwen3-VL recipe |
| `--enable-expert-parallel` | off | **on** for MoE | Per Qwen3-VL recipe |
| `--limit-mm-per-prompt.video` | 1 | `0` if image-only | Smaller graph, faster compile |

## 6. OS / system tunings

| Item | Now | Recommended | Doc |
|---|---|---|---|
| `iommu=pt intel_iommu=on` | ✓ set | keep | [Driver_Installation](https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html) |
| Kernel | 6.8.0-110 | keep | Support matrix |
| MSR `0x1b0=0` | ✓ via gaudi-tune.service | keep | [Bare_Metal_Fresh_OS](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html) |
| MSR `0x774=0x2708` | ✓ | keep | same |
| CPU governor | sysfs missing on this kernel | re-test: `echo performance > .../scaling_governor` after install of `linux-modules-extra` (HWP path) | same |
| C-state disable | not set | optional for Sapphire Rapids, **required for Granite Rapids**: `echo 1 > /sys/devices/system/cpu/cpu*/cpuidle/state2/disable` | [Inference_Optimization](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Optimization.html) |
| `vm.nr_hugepages=24640` | ✓ | keep — installer formula | Habana installer source |
| ulimits memlock/nofile | not set | `* soft memlock unlimited` + `* hard nofile 1048576` | not in Habana docs verbatim, perennial best practice |
| Recipe cache dir | not present | `mkdir -p /var/cache/habana/recipes` + bind-mount | vllm-gaudi env_variables |

## 7. NUMA / pinning

XE9680 has 2 NUMA nodes; 4 Gaudis attached to each. Per [Optimization_in_Training_Platform](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html):
NUMA affinity is a documented perf lever. For TP=1 on Gaudi 0, pin to NUMA 0:

```
docker run --cpuset-cpus=0-55,112-167 --cpuset-mems=0 ...
```

We currently don't pin. Add per-preset NUMA pinning to `bin/vllm-launch`.

## 8. Verification gaps

| Check | Now | Recommended |
|---|---|---|
| hl_qual functional | `-l simple -t 30` | `-l extreme -t 240` (per [System_Verification](https://docs.habana.ai/en/latest/Installation_Guide/System_Verification_and_Final_Tests.html)) |
| hl_qual NIC | not run | `-i 100 -nic_base -test_type pairs` ([Connectivity_Serdes](https://docs.habana.ai/en/latest/Management_and_Monitoring/Qualification_Library/Connectivity_Serdes_Tests_Plugin.html)) |
| hccl_demo | not run | `--nranks 8 --test all_reduce --size 1G --loop 1000` |
| firmware/CPLD assertion | not in check.sh | parse `hl-smi -L`, expect SPI `1.24.0-fw-62.6.1`, CPLD `03 04` |
| dmesg habana faults | only `Bad pagetable` grep | broaden to `habana.*(err\|fail\|fault)` |
| temperature ≠ 0°C | not checked | `hl-smi --query-aip=temperature.aip` (≠0 means driver is alive) |
| inside-container `pip list \| grep habana` | not run | `verify.sh inside-container` mode |

## 9. SynapseAI 1.24.0 known issues (verbatim)

From the v1.24.0 release-notes index:
- *"Sporadic numerical instability may occur when training with FP8 precision"*
- *"Handling Dynamic shapes can be initiated by setting the `PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES` flag"* — limitation, not feature
- *"In certain models, there is performance degradation when using HPU graphs with Lazy collectives"*
- *"Support for `torch.compile` is in early stage. Models may not work."*
- *"Support for Eager mode is in early stages. Models may not work."*
- *"Enabling IOMMU passthrough is required only for Ubuntu 24.04.2/22.04.5 with Linux kernel 6.8"*

**Notable absence:** Qwen3-VL is NOT listed in 1.24.0 SynapseAI release notes —
its support comes from vllm-gaudi 0.16.0/0.17.1 plugin notes (PRs #994, #1031,
#1060, #1252, #1256). And there is **no public known-issue entry** for
`ValidateSyncInputTensors` — closest reference is
[`huggingface/optimum-habana#1241`](https://github.com/huggingface/optimum-habana/issues/1241).

## 10. PR trail that fixed Qwen3-VL on Gaudi (these are why 0.17.1 exists)

- v0.16.0 PR #1060: *"Change Qwen3-VL to use HPUMMEncoderAttention"*
- v0.16.0 PR #1028: *"Fix qwen3 vl moe execution failure"*
- v0.16.0 PR #1068: *"Enable caching for qwen3 moe op"*
- v0.17.1 PR #1031: re-applied HPUMMEncoderAttention forward-port
- v0.17.1 PR #994: *"Qwen3-VL WarmUp Fix"*
- v0.17.1 PR #1256: *"Fix of Qwen Out of HOST memory (OOM)"*
- v0.17.1 PR #1252: *"Fix OOM crashes during high-concurrency inference"*

Anything older than 0.16.0 will crash on Qwen3-VL warmup.

## 11. Calibration / measurements (only if we want to recalibrate)

- Tool: `calibrate_model.sh -m <model> -d <pickle> -o <out> [-b 32 -l samples -t TP]`
- **Measurements are device-locked** — Gaudi3 measurements not reusable on Gaudi2.
- DeepSeek-R1: ≥512 samples × ≥1024 tokens (NeelNanda/pile-10k) — same expected for any wide-MoE incl. 235B-A22B.
- Per-channel quantization (PCQ) disables compilation optimization — avoid unless needed.
- ([calibration](https://docs.vllm.ai/projects/gaudi/en/latest/configuration/calibration/calibration.html))

## 12. Action plan (prioritized)

```
A. HOTPATH (kill --enforce-eager, ~10 min)
   1. Add env: PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false
   2. Add env: PT_HPU_ENABLE_LAZY_COLLECTIVES=true
   3. Add env: PT_HPU_WEIGHT_SHARING=0
   4. Drop CLI: --enforce-eager
   5. Restart, smoke-test
       ↓
   if A passes  →  3-7× faster than current 7 tok/s

B. PERF FLAGS (after A is green, ~30 min)
   • --block-size 128
   • --max-num-seqs 32
   • --kv-cache-dtype fp8_inc, --quantization inc, --weights-load-device cpu
   • --enable-prefix-caching
   • PT_HPU_RECIPE_CACHE_CONFIG (+ bind-mount /var/cache/habana/recipes)
   • HABANA_PGM_LRU_MAX=60000, VLLM_GRAPH_RESERVED_MEM=0.2
   • OMP_NUM_THREADS=$((224/TP))
   • VLLM_ENGINE_ITERATION_TIMEOUT_S=600, VLLM_RPC_TIMEOUT=600000

C. PLUGIN PATH (if A fails or curiosity, ~30 min)
   • Switch image: ptfork → ptupstream
   • Drop env: PT_HPU_LAZY_MODE=1
   • Add env: RUNTIME_SCALE_PATCHING=0 (FP8 + torch.compile)

D. VERIFICATION HARDENING (any time, ~20 min)
   • hl_qual extreme + NIC pairs + hccl_demo in verify.sh
   • Firmware + CPLD assertion in check.sh
   • Inside-container pip-list mode

E. NUMA + ulimits (any time, low risk)
   • --cpuset-cpus / --cpuset-mems per preset
   • /etc/security/limits.d/99-habana.conf
```

## 13. References (verified live during crawl)

- [Habana docs root](https://docs.habana.ai/en/latest/) — entry point
- [Bare-Metal Fresh OS Install](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html) — BIOS + sysctl recipe
- [Driver Installation](https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html) — `iommu=pt`, package list, postinst flow
- [Platform Readiness](https://docs.habana.ai/en/latest/Installation_Guide/Platform_Readiness.html) — LED color, post-install sanity
- [Firmware Upgrade](https://docs.habana.ai/en/latest/Installation_Guide/Firmware_Upgrade.html) — SPI / CPLD / eROM update path
- [System Verification and Final Tests](https://docs.habana.ai/en/latest/Installation_Guide/System_Verification_and_Final_Tests.html) — `hl_qual -l extreme -t 240` canonical
- [Support Matrix](https://docs.habana.ai/en/latest/Support_Matrix/Support_Matrix.html) — exact firmware/kernel/Python/PT versions
- [Inference Using FP8 (canonical)](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html) — `PT_HPU_WEIGHT_SHARING=0`, INC config
- [Inference Optimization](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Optimization.html) — Granite Rapids BIOS recipe + C-state disable
- [HPU Graphs Inference](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html) — `wrap_in_hpu_graph` API + limitations
- [PyTorch Runtime Flags](https://docs.habana.ai/en/latest/PyTorch/Reference/Runtime_Flags.html) — official PT_HPU_* list (partial)
- [Dynamic Shapes](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Dynamic_Shapes.html) — `PT_HPU_METRICS_FILE`
- [hl_qual Functional Plugin (-f2)](https://docs.habana.ai/en/latest/Management_and_Monitoring/Qualification_Library/Functional_Tests_Plugin.html)
- [hl_qual Connectivity Serdes (-nic_base)](https://docs.habana.ai/en/latest/Management_and_Monitoring/Qualification_Library/Connectivity_Serdes_Tests_Plugin.html)
- [vllm-gaudi compatibility matrix](https://docs.vllm.ai/projects/gaudi/en/latest/getting_started/compatibility_matrix.html) — 0.17.1 ↔ 1.24.0 only
- [vllm-gaudi validated_models](https://docs.vllm.ai/projects/gaudi/en/latest/getting_started/validated_models.html) — Qwen3-VL FP8 row
- [vllm-gaudi env_variables](https://docs.vllm.ai/projects/gaudi/en/latest/configuration/env_variables.html) — every PT_*, VLLM_*, HABANA_*, HCCL_* perf flag
- [vllm-gaudi performance_tuning](https://docs.vllm.ai/projects/gaudi/en/latest/configuration/performance_tuning.html)
- [vllm-gaudi bucketing_mechanism](https://docs.vllm.ai/projects/gaudi/en/latest/features/bucketing_mechanism.html)
- [vllm-gaudi managing_warm-up](https://docs.vllm.ai/projects/gaudi/en/latest/configuration/warm-up/managing_warm-up.html) — recipe cache + exponential bucketing
- [vllm-gaudi multi_node](https://docs.vllm.ai/projects/gaudi/en/latest/configuration/multi_node.html)
- [vllm-gaudi troubleshooting](https://docs.vllm.ai/projects/gaudi/en/latest/general/troubleshooting.html) — `RUNTIME_SCALE_PATCHING=0` quote
- [vllm-gaudi quantization (INC)](https://docs.vllm.ai/projects/gaudi/en/latest/configuration/quantization/inc.html)
- [vllm-gaudi calibration](https://docs.vllm.ai/projects/gaudi/en/latest/configuration/calibration/calibration.html)
- [vllm-gaudi 0.16.0 release notes](https://docs.vllm.ai/projects/gaudi/en/0.16.0/release_notes_v0.16.0.html) — Qwen3-VL HPUMMEncoderAttention PRs
- [vllm-gaudi 0.17.1 quickstart](https://docs.vllm.ai/projects/gaudi/en/0.17.1/getting_started/quickstart/quickstart.html)
- [vllm-fork README_GAUDI (legacy fork — what our image runs)](https://github.com/HabanaAI/vllm-fork/blob/habana_main/README_GAUDI.md) — block_size=128, FP8 timeouts
- [Qwen3-VL recipe (vLLM)](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html) — `--mm-encoder-tp-mode data`, `--enable-expert-parallel`, `--async-scheduling`
- [optimum-habana #1241](https://github.com/huggingface/optimum-habana/issues/1241) — closest public reference for `ValidateSyncInputTensors`
