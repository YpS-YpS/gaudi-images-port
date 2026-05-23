# 01 — the box itself

Hardware, OS, driver, and persistence layer. Everything in this file is
established and verified; deviating from these versions has caused failures.

## Hardware

| Component | Spec |
|---|---|
| Chassis | Dell PowerEdge XE9680 |
| CPU | 2× Intel Xeon Platinum 8480+ (Sapphire Rapids, 224 cores total) |
| RAM | 2 TB DDR5 |
| Accelerators | 8× Intel Gaudi 3 (HL-325, 128 GiB HBM2e each, 1 TiB total) |
| Storage | 7 TB NVMe (model weights at `~/hf-cache/`) |
| Network | 24× internal Gaudi scale-out NICs (mesh, not Ethernet) + standard NIC for external |

## OS — pinned

- **Ubuntu 24.04.2 LTS**
- **Kernel 6.8.0-110-generic** — pinned in GRUB. **Newer kernels break the
  Habana driver build** (verified). Old kernels remain as GRUB fallback.

## GRUB — required flags

`/etc/default/grub` must include:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash iommu=pt intel_iommu=on"
```

**Why:** without `iommu=pt intel_iommu=on`, kernel page-table corruption fires
when the Habana driver maps device memory. We saw "Bad pagetable" oopses
without these. `install.sh` sets and persists them via `update-grub`.

## Habana driver

- Version: **1.24.0-1007** (SynapseAI 1.24.0)
- Required modules (autoload via `/etc/modules-load.d/habana-modules.conf`):
  - `habanalabs` (main driver)
  - `habanalabs_ib` (InfiniBand for scale-out)
  - `ib_uverbs` (RDMA user-verbs — needed for the Gaudi NICs)
  - 4 more sub-modules loaded transitively
- Verify with `lsmod | grep -E 'habana|ib_uverbs'` — should show 7 lines.

## Hugepages

```
/etc/sysctl.d/99-habana-hugepages.conf:
  vm.nr_hugepages = 24640         # ~48 GB at 2 MiB pages
```

This is Habana's installer formula. Without it, allocations fail under load.

## MSRs (Sapphire Rapids tuning)

Applied at boot via `gaudi-tune.service` (`systemd/gaudi-tune.service`):

```
IA32_ENERGY_PERF_BIAS = 0        # max performance
HWP_REQUEST           = max
```

These are restored on every boot — don't undo them manually.

## NICs

The 24 internal Gaudi scale-out NICs need to be brought up after every
reboot. `gaudi-tune.service` also handles this.

## Docker

- Engine: standard Docker (apt-installed)
- Habana container runtime registered as `--runtime=habana`
- Corp proxy drop-in: `/etc/systemd/system/docker.service.d/http-proxy.conf`
  (Intel proxy is `http://proxy-dmz.intel.com:912`)

## HuggingFace cache

- Location: `~/hf-cache/` (mapped into containers at `/root/.cache/huggingface`)
- Shared by all containers — single download, many readers
- Currently contains: Qwen3-VL family, Gemma 4, MiniMax M2, MiniMax M2.7, gpt-oss
- Watch disk space: `du -sh ~/hf-cache/` — currently sitting at multi-hundred-GB

## Network — Intel corp proxy

All HTTP egress goes through `http://proxy-dmz.intel.com:912`. This is set:
- In each launch container via `HTTP_PROXY` / `HTTPS_PROXY` env vars
- In the Docker daemon via the systemd drop-in (so `docker pull` works)
- **`NO_PROXY` includes** `.intel.com,10.0.0.0/8,192.168.0.0/16,172.16.0.0/12,host.docker.internal,localhost,127.0.0.1`
  — without this, traffic to localhost vLLM ports gets sent through the proxy
  and times out.

## State persistence files in /etc/

| File | What it does |
|---|---|
| `/etc/default/grub` | iommu=pt + kernel pin |
| `/etc/modules-load.d/habana-modules.conf` | Load all 7 Habana kernel modules |
| `/etc/sysctl.d/99-habana-hugepages.conf` | 24640 hugepages |
| `/etc/systemd/system/gaudi-tune.service` | MSRs + NIC bring-up on boot |
| `/etc/systemd/system/docker.service.d/http-proxy.conf` | Proxy for dockerd |

Re-apply all of these at once with `sudo bash scripts/fix.sh` (no reboot needed
for most). To rebuild from scratch: `sudo bash install.sh` (idempotent).

## Verification one-liners

```bash
# Cards visible to driver
hl-smi | head -20

# All 7 modules loaded
lsmod | grep -E 'habana|ib_uverbs' | wc -l   # expect 7

# Hugepages
cat /proc/meminfo | grep HugePages_Total      # expect 24640

# GRUB has iommu=pt
sudo grep iommu /boot/grub/grub.cfg | head -3 # expect iommu=pt intel_iommu=on

# Docker runtime registered
docker info | grep -i habana                  # expect "Runtimes: habana ..."

# Or just run readiness checklist
sudo bash scripts/check.sh                     # one command, green/red
```
