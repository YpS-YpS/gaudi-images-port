# REQUIREMENTS — what you need before `install.sh`

Hard requirements for an Intel Gaudi3 box to run `install.sh` and end up with a working vLLM + 8 accelerators.

## 1. Hardware

| Item | Required | Why |
|---|---|---|
| Gaudi accelerators | **Gaudi 2 or Gaudi 3** (HL-225 or HL-325 SKU). Other vendors (NVIDIA/AMD) won't work. | Driver only supports Habana ASICs. |
| CPU | x86-64. Tested on Intel Sapphire Rapids (8480+); should work on Granite Rapids and AMD Genoa/Bergamo. | MSR tunings in `gaudi-tune.service` assume Intel HWP — adapt for AMD. |
| RAM | ≥ 1 TB recommended for FP8 235B; ≥ 256 GB for 32B and below. Hugepages take ~50 GB. | Model staging buffers + KV cache mirror + hugepage pool. |
| HBM (per Gaudi) | 128 GiB (Gaudi3) or 96 GiB (Gaudi2). | Model + KV cache live here. |
| Disk | ≥ 500 GB free for one frontier model + image. ≥ 50 GB for 32B. | HF cache + Docker layers. |
| Network | ≥ 1 Gbps to HuggingFace + Habana vault. | First-time downloads (image ~10 GB, models 30–235 GB). |

## 2. Operating system

| Item | Required |
|---|---|
| Distro | **Ubuntu 24.04** (noble). 22.04 also supported by Habana but not validated by this repo. |
| Kernel | **6.8.0** series — `linux-image-6.8.0-110-generic` is what `install.sh` pins. Newer kernels (e.g. 6.17 from HWE) **break the driver build** — see TROUBLESHOOTING.md. |
| Root / sudo | Yes (script runs as root). |
| Python | 3.12 (Ubuntu default on 24.04). |

## 3. Kernel boot parameters

Required for Habana on kernel 6.8 (per Habana docs):

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash iommu=pt intel_iommu=on"
```

Without `iommu=pt intel_iommu=on`, you'll see kernel `Bad pagetable` oopses
the moment vLLM (or any HPU workload) maps device memory.

`install.sh` adds these automatically and triggers a reboot if missing.

## 4. BIOS / firmware settings

Confirm in BIOS before bring-up:

| Setting | Value |
|---|---|
| Intel VT-d / IOMMU | **Enabled** |
| SR-IOV | Enabled (often default; required for some virtualized setups) |
| Hyper-threading | Enabled (default) |
| CPU power policy | Performance (or "OS Control" — we override via MSRs at runtime) |
| Above 4G decoding | Enabled |
| Memory mode | Optimizer (NUMA-aware) |
| AC Power Recovery | "Last" or "On" if you want auto-restart after power loss |

Habana firmware versions (mgmt/preboot) should be in the **1.20.1 – 1.24.0**
range. `hl-smi --fw-versions` shows them. Newer is fine — older fails.

## 5. Network

| Need | Detail |
|---|---|
| Outbound to `https://vault.habana.ai` | apt repo + Docker images. Behind a proxy: see "Proxy" below. |
| Outbound to `https://huggingface.co` | model downloads. Speed: ≥ 50 MB/s recommended (235B = 222 GB). |
| Outbound to `https://archive.ubuntu.com` | apt updates + linux-modules-extra. |
| Inbound to your serving port (default `:8000`) | from clients that will hit the API. |

### Proxy

If you're behind a corporate HTTP proxy (e.g. `http://proxy-dmz.intel.com:912`):

* Set `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` in `/etc/environment` and
  `/etc/apt/apt.conf.d/95proxy` (Ubuntu generally picks them up).
* `install.sh` writes a `/etc/systemd/system/docker.service.d/http-proxy.conf`
  drop-in so `dockerd` uses the proxy too — without it, `docker pull` hangs
  on `vault.habana.ai/v2/...`.

## 6. Required packages (covered by `install.sh`)

* `habanalabs-{dkms,firmware,firmware-tools,graph,rdma-core,thunk,tools,container-runtime,qual}` — driver + userspace
* `linux-modules-extra-$(uname -r)` — provides `ib_uverbs` (REQUIRED for `habanalabs_ib`)
* `docker.io` + the Habana `habana-container-runtime` registered in `/etc/docker/daemon.json`
* `msr-tools` for `wrmsr`
* `btop htop glances iotop iftop nethogs ncdu tmux git curl` — operational tools

## 7. Disk layout / mounts

* `/home/<user>/hf-cache` — bind-mounted into vLLM containers as `/root/.cache/huggingface`. Default; change in `bin/vllm-launch` if your storage is elsewhere.
* `/var/lib/docker` — container images. Make sure the partition has the headroom (10–25 GB per image).

## 8. What's *not* required

* No InfiniBand HCA / external RDMA fabric. The 24 internal Gaudi-to-Gaudi NICs
  are scale-out *within* the chassis and use Habana's ICX, not Mellanox/IB.
* No GPU drivers (NVIDIA/AMD).
* No Anaconda / venv on host — vLLM runs only inside the Habana container.
* No HuggingFace token for the Qwen models we use (all ungated).

## 9. Sanity checks before running `install.sh`

```bash
lspci -d 1da3: -nn | wc -l        # → 8
lscpu | grep -E '^(CPU|Model)'    # → 224 cores, Intel(R) Xeon(R) Platinum 8480+
free -h                            # → ≥ 1 TB
df -h /                            # → ≥ 300 GB free
curl -sI https://vault.habana.ai  # → 200/302
curl -sI https://huggingface.co   # → 200/302
```

If any of those fail, fix it before running `install.sh`.
