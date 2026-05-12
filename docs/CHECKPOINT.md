# CHECKPOINT — what works, where, what version

Snapshot of a working bring-up on a Dell PowerEdge XE9680 with 8× Intel Gaudi3 (HL-325).

## Hardware

| Component | Value |
|---|---|
| Chassis | Dell PowerEdge XE9680, BIOS 2.5.4 |
| CPU | 2× Intel Xeon Platinum 8480+ (Sapphire Rapids), 224 logical cores total |
| RAM | 2 TB (1 TB per NUMA node) |
| Accelerators | 8× Intel Gaudi3 (HL-325, PCI 1da3:1060) — 128 GiB HBM each |
| Storage | NVMe, 7 TB / used |
| NIC | 1 Gbps eno8303 (host management); 24× internal Gaudi scale-out per box |

## OS / kernel / driver

| Layer | Version |
|---|---|
| Distro | Ubuntu 24.04.2 LTS (noble) |
| Running kernel | `6.8.0-110-generic` (HWE backport, Habana-supported) |
| Habana driver | `1.24.0-1007` (DKMS — built against running kernel) |
| Habana firmware | mgmt `hl-gaudi3-1.24.0-fw-62.6.1-sec-3` |
| Python | 3.12.3 |
| Docker | 29.1.3 |

Other kernels installed (kept as fallback in GRUB advanced menu): 6.17.0-14, 6.17.0-22.

## Kernel boot params (`/proc/cmdline`)

```
BOOT_IMAGE=/boot/vmlinuz-6.8.0-110-generic
root=UUID=990b7738-e3af-426d-996b-352d042e99f1
ro quiet splash iommu=pt intel_iommu=on
```

`iommu=pt intel_iommu=on` is **required** by Habana on kernel 6.8 — without it, kernel page-table corruption ("Bad pagetable") fires when the driver maps device memory.

## Loaded kernel modules (all 7 must be loaded)

```
habanalabs               # main driver
habanalabs_compat        # backward-compat shim
habanalabs_cn            # compute-network (Gaudi-internal scale-out)
habanalabs_en            # ethernet
habanalabs_ib            # InfiniBand verbs — REQUIRED by HCL collective comm
ib_uverbs                # provides ib_* symbols for habanalabs_ib (ships in linux-modules-extra)
ib_core                  # base RDMA core
```

## Persisted system files (post-bring-up)

| Path | Purpose |
|---|---|
| `/etc/default/grub` (and `.bak.gaudi-setup`) | iommu cmdline + 6.8 pin |
| `/etc/sysctl.d/99-habana-hugepages.conf` | `vm.nr_hugepages=24640` (~48 GB) |
| `/etc/modules-load.d/habana-modules.conf` | autoload `ib_uverbs` + `habanalabs_ib` on boot |
| `/etc/systemd/system/gaudi-tune.service` | re-applies MSRs + Gaudi NICs every boot |
| `/etc/systemd/system/docker.service.d/http-proxy.conf` | corporate proxy for `dockerd` |
| `/etc/docker/daemon.json` | registers `habana` container runtime |
| `/etc/apt/sources.list.d/habanalabs_synapseai.list` | Habana apt repo |

## CPU performance tuning (Sapphire Rapids)

Sysfs `cpufreq/` doesn't exist on this kernel — Intel HWP drives frequency. Tunings via MSR:

```
wrmsr -a 0x1b0 0x0       → IA32_ENERGY_PERF_BIAS = 0 (max performance)
wrmsr -a 0x774 0x2708    → IA32_HWP_REQUEST = HWP max
```

Re-applied on every boot by `gaudi-tune.service`.

## Hugepages

Habana installer formula:

```
HugePages_Total = (110 × 1024 × n_cores × 2) / Hugepagesize_kB + 1
                = (110 × 1024 × 224     × 2) / 2048             + 1
                = 24,640
```

Currently allocated: **24,640 × 2 MiB = 48.1 GB**. Persisted in `/etc/sysctl.d/`.

## Internal Gaudi scale-out NICs

Each Gaudi3 has 24 RDMA ports (21 internal between cards + 3 external). On a standalone box:

* 24 internal ports must be brought up: `manage_network_ifs.sh --up`
* 3 external ports per card stay down (they go to nothing on a single-server box)
* Re-applied on every boot by `gaudi-tune.service`

## Docker

| | |
|---|---|
| Runtimes registered | `runc`, `io.containerd.runc.v2`, **`habana`** |
| Daemon proxy | wired to `http://proxy-dmz.intel.com:912` (Intel-internal — drop the file on non-Intel boxes) |
| User in docker group | `satyajit-gaudi` |

## vLLM container

| | |
|---|---|
| Image | `vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/vllm-0.17.1-ptfork-2.10.0:1.24.0-1007` |
| Image size | 10.4 GB |
| vLLM version | 0.17.1 (with `vllm-gaudi` plugin baked in) |
| PyTorch | 2.10.0 (Habana fork — supports `PT_HPU_LAZY_MODE=1`) |

The **upstream** PyTorch variant (`-ptupstream-`) doesn't support lazy mode. Use the **fork** variant.

## Currently configured presets

```
preset           model                                  TP  devices  port
8b-thinking      Qwen/Qwen3-VL-8B-Thinking-FP8           1  0        8003
30b-a3b          Qwen/Qwen3-VL-30B-A3B-Thinking-FP8      1  0        8002
32b-instruct     Qwen/Qwen3-VL-32B-Instruct-FP8          1  0        8001
32b-thinking     Qwen/Qwen3-VL-32B-Thinking-FP8          1  0        8000  ← validated
235b-tp4         Qwen/Qwen3-VL-235B-A22B-Thinking-FP8    4  0..3     8004
235b-tp8         Qwen/Qwen3-VL-235B-A22B-Thinking-FP8    8  0..7     8005
```

## Models on disk

| Model | Path | Size |
|---|---|---|
| Qwen3-VL-32B-Thinking-FP8 | `~/hf-cache/hub/models--Qwen--Qwen3-VL-32B-Thinking-FP8/` | ~33 GB |
| Qwen3-VL-235B-A22B-Thinking-FP8 | `~/hf-cache/hub/models--Qwen--Qwen3-VL-235B-A22B-Thinking-FP8/` | ~222 GB |

## Required runtime flag

vLLM-Gaudi 0.17.1 has a known crash in HPU-graph capture for Qwen3-VL FP8 GEMM
(`ValidateSyncInputTensors tensor_data is empty`). Workaround applied in
`bin/vllm-launch`:

```
--enforce-eager
```

This disables HPU graphs and runs ops eagerly. Throughput drops vs. graph mode
(roughly **3-7×** slower per token) but the server is stable. Re-evaluate when
vllm-gaudi or the FP8 quant code paths get a fix.

## Validated end-to-end

API: `http://10.234.184.59:8000/v1` (also `localhost:8000` on the host)

| Test | Result |
|---|---|
| `GET /v1/models` | `200 OK` returns `qwen3-vl-32b-thinking` |
| Plain text math (`17 * 89`) | `1513` ✓ + `<think>` reasoning trace |
| Tool calling (`get_weather`) | `tool_calls` with `{"city": "Paris"}` ✓ |
| Vision (base64 PNG) | "red square + blue circle inside" ✓ |
| `Bad pagetable` events post-fix | **0** |

Throughput in eager mode: ~7 tok/s on a single Gaudi3 for 32B. Loading the model takes ~2 min once cached, ~6 min cold (download+load+warmup).

## Repo layout

```
gaudi-setup/
├── README.md
├── LICENSE
├── install.sh                  # one-shot bring-up (idempotent)
├── docs/
│   ├── CHECKPOINT.md           # this file
│   ├── REQUIREMENTS.md         # what to confirm before install.sh
│   ├── PLAYBOOK.md             # day-1 + day-2 ops
│   └── TROUBLESHOOTING.md      # everything we hit and fixed
├── scripts/
│   ├── check.sh                # readiness — green/red checklist
│   ├── fix.sh                  # re-apply runtime tunings (no reboot)
│   ├── verify.sh               # readiness + hl_qual + API smoke
│   └── smoke.sh                # API-only smoke (text/vision/tools)
├── systemd/
│   ├── gaudi-tune.service       # MSRs + NIC bring-up on boot
│   ├── docker-http-proxy.conf   # corp proxy for dockerd
│   ├── habana-modules.conf      # ib_uverbs + habanalabs_ib autoload
│   └── habana-hugepages.conf    # vm.nr_hugepages persistence
└── bin/
    ├── gaudi-dash              # tmux dashboard (btop + htop + hl-smi)
    └── vllm-launch             # multi-preset vLLM serving launcher
```
