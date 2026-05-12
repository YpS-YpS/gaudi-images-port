# PLAYBOOK — first boot, day-2 ops, scaling, recovery

## A. Day 0 — fresh OS to working vLLM

```
fresh Ubuntu 24.04 box with 8× Gaudi3
       │
       ▼
sudo bash install.sh
       │  (installs kernel 6.8, GRUB iommu params, Habana driver,
       │   ib_uverbs+habanalabs_ib, hugepages, MSR tunings, NICs,
       │   docker+habana runtime, gaudi-dash, vllm-launch)
       │
       │  if NEEDS_REBOOT:
       │      sudo reboot
       │      sudo bash install.sh    # 2nd run finishes Habana
       │
       ▼
sudo bash scripts/check.sh        ◀── all checks should be ✓
       │
       ▼
sudo /usr/local/bin/vllm-launch 32b-thinking
       │
       ▼  (~5-10 min warmup on first launch; ~2 min on cached restart)
sudo bash scripts/verify.sh smoke
       │
       ▼  ✅ live API on http://10.234.184.59:8000/v1
```

## B. Day-2 ops

### Start a model
```bash
vllm-launch 32b-thinking         # default 32B Thinking on Gaudi 0
vllm-launch list                  # all presets + ports
vllm-launch logs 32b-thinking     # tail logs
vllm-launch stop 32b-thinking     # tear down
vllm-launch stop-all              # kill every vllm-* container
```

### Run multiple models in parallel
Different presets use different `HABANA_VISIBLE_DEVICES` and different ports.
Open `bin/vllm-launch` and edit the preset to pick a different Gaudi:

```bash
# example — change 8b-thinking to use Gaudi 4 instead of 0:
[8b-thinking]="Qwen/Qwen3-VL-8B-Thinking-FP8|1|4|8003|32768"
```
Then start both: `vllm-launch 32b-thinking && vllm-launch 8b-thinking`.

### Add a new model preset
Edit `bin/vllm-launch` — append to the `PRESETS` array:
```bash
[my-llama]="meta-llama/Llama-3.3-70B-Instruct|4|0,1,2,3|8006|32768"
#         model                              TP devices  port maxlen
```
Make sure: TP×devices match (TP=4 needs 4 device IDs), pick a unique port,
and `max_model_len` fits in your KV-cache budget.

### Watch live load
```bash
gaudi-dash      # tmux 3-pane: btop + htop + hl-smi -l 1
                # Ctrl-b d to detach, gaudi-dash kill to teardown
hl-smi -l 1     # hl-smi alone, refreshing every second
glances -w      # web dashboard on http://<host>:61208
```

### Health re-check at any time
```bash
sudo bash scripts/check.sh        # green/red checklist
sudo bash scripts/verify.sh       # readiness + hl_qual + API smoke
```

### Re-apply runtime tunings (e.g. after a hot kernel param change)
```bash
sudo bash scripts/fix.sh          # MSRs + hugepages + NICs + ib modules
                                   # NO reboot, NO package install
```

## C. Scaling up (32B → 235B)

The 235B-A22B-Thinking-FP8 model needs **TP ≥ 4**. Two presets ship for it:

```
235b-tp4   4 Gaudis (0-3)   port 8004   max-model-len 16384
235b-tp8   8 Gaudis (0-7)   port 8005   max-model-len 32768
```

Steps:

```bash
# 1. Make sure the 235B model is on disk (~222 GB)
ls -d ~/hf-cache/hub/models--Qwen--Qwen3-VL-235B-A22B-Thinking-FP8/
# If missing:
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Qwen/Qwen3-VL-235B-A22B-Thinking-FP8

# 2. Stop any single-Gaudi server occupying card 0
vllm-launch stop-all

# 3. Launch
vllm-launch 235b-tp4   # or 235b-tp8 if you want maximum throughput

# 4. Wait ~10-20 min for first warmup (model loads onto 4 Gaudis + warmup)
# 5. Smoke-test
sudo bash scripts/verify.sh smoke
```

⚠️ Heads-up: if you hit a graph-compile crash with 235B (similar to 32B-Thinking
needed `--enforce-eager`), the launcher already has it baked in. If you remove
`--enforce-eager` to chase higher throughput, watch for the crash.

## D. Reverting / recovering

### Roll back GRUB to before bring-up
```bash
sudo cp /etc/default/grub.bak.gaudi-setup /etc/default/grub
sudo update-grub
sudo reboot
```

### Boot the old kernel manually (one-time)
At GRUB menu (5 s timeout) → "Advanced options for Ubuntu" → pick the older
kernel (e.g. 6.17.0-22-generic).

### Disable boot-time tunings
```bash
sudo systemctl disable gaudi-tune.service
sudo rm -f /etc/sysctl.d/99-habana-hugepages.conf /etc/modules-load.d/habana-modules.conf
sudo sysctl -w vm.nr_hugepages=0
```

### Stop everything and free Gaudis
```bash
vllm-launch stop-all
hl-smi   # all 8 cards should report ~672 MiB (idle baseline)
```

### Hard reset a single Gaudi (rarely needed)
```bash
sudo hl-smi -r -i 0000:19:00.0    # reset Gaudi at the given PCI address
```

## E. Operational signals

| Signal | What it means | Action |
|---|---|---|
| `hl-smi` shows util > 0% | active workload running | normal |
| `hl-smi` shows ~131072 MiB used on a card | model + KV cache loaded | normal during serving |
| `hl-smi` red memory bar | > 90% HBM (we run with `--gpu-memory-utilization 0.9`) | normal during serving |
| `dmesg \| grep 'Bad pagetable'` non-zero | IOMMU corruption is back | run `scripts/check.sh`; verify `iommu=pt intel_iommu=on` in `/proc/cmdline` |
| Container in `restarting` state forever | model warmup loop crashed | `docker logs <name>` to find the EngineCore traceback; check TROUBLESHOOTING.md |
| `pip` / `apt` calls hang | proxy not configured | check `/etc/apt/apt.conf.d/*proxy*` and docker drop-in |

## F. Where the docs are

| | |
|---|---|
| What's installed where | `docs/CHECKPOINT.md` |
| Pre-flight requirements | `docs/REQUIREMENTS.md` |
| Things that broke + how to fix | `docs/TROUBLESHOOTING.md` |
| This file | `docs/PLAYBOOK.md` |
| Habana official | https://docs.habana.ai/en/latest/Installation_Guide/ |
| vLLM-Gaudi plugin | https://github.com/vllm-project/vllm-gaudi |
