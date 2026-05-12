# TROUBLESHOOTING — every problem we hit, and how to fix it

Each section: **symptom → root cause → fix**. Keep this updated as new ones come up.

## 0. Diagnose first

```bash
sudo bash scripts/check.sh                         # green/red checklist
sudo dmesg | grep -iE 'habana|hpu|pagetable' | tail
sudo docker logs vllm-32b-thinking 2>&1 | tail -40 # if a container is up
```

---

## 1. `Bad pagetable` kernel oops, vLLM workers become zombies

### Symptom
`dmesg` shows:
```
VLLM::Worker: Corrupted page table at address ...
Bad pagetable: 000f [#1] PREEMPT SMP NOPTI
RIP: at __memset_avx512_unaligned_erms in glibc
```
`ps` shows `EngineCore` and `Worker` processes with state `Z` (zombie). Container is "Up" but no API.

### Root cause
Kernel boot parameters `iommu=pt intel_iommu=on` are missing. Without IOMMU
passthrough, the habana driver's DMA mappings collide with kernel page tables
when the model is loaded.

### Fix
```bash
sudo sed -i 's|^GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"|GRUB_CMDLINE_LINUX_DEFAULT="\1 iommu=pt intel_iommu=on"|' /etc/default/grub
sudo update-grub
sudo reboot
```
Verify: `cat /proc/cmdline | grep -E 'iommu=pt|intel_iommu=on'`.

---

## 2. `habanalabs-dkms` install fails with "DKMS tree already contains"

### Symptom
```
Adding Module to DKMS build system habanalabs 1.24.0-1007
Error! DKMS tree already contains: habanalabs-1.24.0-1007
```
`dpkg -l habanalabs-dkms` shows status `iF` (install-failed).

### Root cause
The previous postinst attempt was killed mid-install (often by a kernel mismatch),
leaving a half-registered dkms entry. The next attempt fails because dkms refuses
to add the same module/version twice.

### Fix
```bash
ver=$(dpkg-query -W -f '${Version}' habanalabs-dkms)
sudo dkms remove "habanalabs/${ver}" --all
sudo dpkg --configure -a
```

---

## 3. `habanalabs_ib` won't load, vLLM dies on `g_ibv.init() == hcclSuccess` failed

### Symptom
```
hcl_device_control_factory.cpp:84(initDevice):
  The condition [ g_ibv.init(deviceConfig) == hcclSuccess ] failed.
  ibv initialization failed
```
`dmesg` also shows:
```
habanalabs_ib: Unknown symbol uverbs_idr_class
habanalabs_ib: Unknown symbol ib_uverbs_get_ucontext_file
```

### Root cause
`habanalabs_ib` depends on `ib_uverbs`, which ships in
`linux-modules-extra-$(uname -r)`. Ubuntu's stock `linux-image` doesn't include
it. The Habana installer (`habanalabs-installer.sh`) handles this; direct apt
install doesn't.

### Fix
```bash
sudo apt install -y "linux-modules-extra-$(uname -r)"
sudo modprobe ib_uverbs
sudo modprobe habanalabs_ib
# Persist:
echo -e "ib_uverbs\nhabanalabs_ib" | sudo tee /etc/modules-load.d/habana-modules.conf
```

---

## 4. vLLM crashes during prompt warmup with `ValidateSyncInputTensors tensor_data is empty`

### Symptom
```
RuntimeError: [Rank:0] FATAL ERROR :: MODULE:PT_LAZY Error,
  ValidateSyncInputTensors tensor_data is empty.
  Failing Graph: ... model/N/mlp/down_proj/hpu__fp8_gemm_v2 ...
```
Container exits with code 1 after sampler warmup completes.

### Root cause
Bug in `vllm-gaudi 0.17.1` HPU-graph capture for Qwen3-VL FP8 GEMMs. A tensor
input gets recorded into the graph IR without its data attached.

### Fix (workaround)
Add `--enforce-eager` to the `vllm serve` command. Disables HPU-graph capture;
ops run eagerly. Throughput drops 3-7× per token but the server is stable.

```
# bin/vllm-launch already has --enforce-eager baked in; no action needed if
# you use the launcher.
```

Re-evaluate when `vllm-gaudi` cuts a fix release.

---

## 5. `docker pull vault.habana.ai/...` times out

### Symptom
```
Error response from daemon: failed to resolve reference "vault.habana.ai/...":
  failed to do request: Head ".../manifests/...": dial tcp X.X.X.X:443: i/o timeout
```
But `curl https://vault.habana.ai/v2/` works.

### Root cause
`dockerd` doesn't read `HTTP_PROXY` env vars from `/etc/environment` —
it needs a systemd drop-in.

### Fix
```bash
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf <<EOF
[Service]
Environment="HTTP_PROXY=$HTTP_PROXY"
Environment="HTTPS_PROXY=$HTTPS_PROXY"
Environment="NO_PROXY=localhost,127.0.0.1,::1,.intel.com,10.0.0.0/8,192.168.0.0/16,172.16.0.0/12"
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
```

⚠️ Don't put `vault.habana.ai` into `NO_PROXY` — it's an external host that
needs to go through the proxy.

---

## 6. `cpufreq` sysfs missing → can't set CPU governor to "performance"

### Symptom
```
$ cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
cat: ... No such file or directory
```

### Root cause
Sapphire / Granite Rapids use Intel HWP. The kernel doesn't expose the
classic cpufreq governor knob — frequency is driven by the CPU itself.

### Fix
Tune via MSR instead. `install.sh` already does this on every boot via
`gaudi-tune.service`:

```bash
sudo modprobe msr
sudo wrmsr -a 0x1b0 0x0      # IA32_ENERGY_PERF_BIAS = max performance
sudo wrmsr -a 0x774 0x2708   # IA32_HWP_REQUEST = max
```

---

## 7. vLLM container hangs at "Starting to load model" for minutes

### Symptom
Last log line:
```
[hpu_model_runner.py:4518] Starting to load model Qwen/...
```
No further output for 2+ minutes. Gaudi memory is still 672 MiB (idle baseline).

### Root cause
The vLLM image uses `hf-xet` (HuggingFace's chunked CDN client) which has a
**different cache layout** than the `hf download` CLI. So even though you
already downloaded the model on the host (mounted at `/root/.cache/huggingface`
in the container), vLLM re-downloads via xet to the same path.

### Fix
Just wait. ~5-10 min for 32B over a 100 MB/s link. Watch:
```bash
sudo docker exec <name> ls /root/.cache/huggingface/hub/models--Qwen--*/blobs/*.incomplete 2>/dev/null
sudo docker exec <name> ss -tnp 2>/dev/null | grep -c ESTAB
```
The `.incomplete` files vanish when each shard finishes; `ESTAB` count drops as the xet pool wraps up.

(The pre-download on host is still useful — xet caches its own chunks under the
same `~/.cache/huggingface/xet/` directory, so subsequent container restarts
don't re-fetch.)

---

## 8. Internal Gaudi NICs show "down" — is that bad?

### Symptom
```
$ /opt/habanalabs/qual/gaudi3/bin/manage_network_ifs.sh --status
accel0
3 ports down (5, 8, 9)
accel1
3 ports down (2, 3, 7)
...
```

### Root cause
On a single-server box, each Gaudi3 has 3 **external** ports (cables to other
boxes). With no cables, those ports stay down. The 21 **internal** ports between
the 8 Gaudis must be up.

### Fix
This is normal — no action needed. To confirm internal ports are up after a
fresh boot:

```bash
sudo /opt/habanalabs/qual/gaudi3/bin/manage_network_ifs.sh --up
```

`gaudi-tune.service` does this on every boot.

---

## 9. The `Habana 1.24` driver fails to build on kernel 6.17

### Symptom
```
[habanalabs] Error! Bad return status for module build on kernel: 6.17.0-22-generic
... cc1: all warnings being treated as errors
```

### Root cause
Habana 1.24.0 is certified against kernel 6.8.0 on Ubuntu 24.04. Newer kernels
(Ubuntu HWE rolled forward to 6.17) introduce APIs that break the driver's
`-Werror` build.

### Fix
Install kernel 6.8 alongside and pin GRUB to it. `install.sh` does this.

```bash
sudo apt install -y linux-image-6.8.0-110-generic linux-headers-6.8.0-110-generic
# Edit /etc/default/grub: GRUB_DEFAULT to point at the 6.8 menuentry, e.g.
#   GRUB_DEFAULT="gnulinux-advanced-<UUID>>gnulinux-6.8.0-110-generic-advanced-<UUID>"
sudo update-grub
sudo reboot
```
Don't `apt-mark hold linux-image-generic-hwe-24.04` — let it stay so security
updates flow; just make sure GRUB always picks 6.8.

---

## 10. `glibc.cpu.hwcaps=-AVX512F,...` doesn't help

We tried disabling AVX-512 via `GLIBC_TUNABLES` thinking the AVX-512 memset
in glibc was the culprit. **It wasn't.** The crash signature *was* in AVX-512
memset, but the underlying bug was the missing `iommu=pt`. Don't waste time
on this — fix #1 is the answer.

---

## 11. Where to look for habana logs

```bash
sudo docker exec <vllm-container> bash -c 'ls /var/log/habana_logs/'
sudo docker exec <vllm-container> bash -c 'cat /var/log/habana_logs/hcl.log'
journalctl -u docker --since '5 min ago'
sudo dmesg --since '5 min ago'
```
