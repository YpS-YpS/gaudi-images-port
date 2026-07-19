# 10 — new-box runbook (bring up a fresh Gaudi 3 → serving vLLM)

The exact, battle-tested sequence for standing up a new Gaudi 3 box from the
`gaudi-setup` repo. Distilled from real bring-ups of **box2** (`jf01sval0068`,
was stuck on the wrong kernel) and **box3** (`g3-dell02`, base already correct).
Do the probe first — how much of this you run depends on what's already there.

> Rule of thumb: a box that already has the `habanalabs-*` 1.24.0 packages on
> **kernel 6.8** and a loaded driver only needs Steps 5–9 (docker + models). A
> box on the wrong kernel or with the driver unbuilt needs Steps 3–4 too.

---

## 0. Inputs you need before starting
- **SSH**: host, user, auth (password or key). Whether the user has `sudo`.
- **Which model(s)** to serve (Gemma 4 31B ≈30 GB/1 card; MiniMax M2 ≈215 GB/4 cards).
- Site **proxy** (Intel = `http://proxy-dmz.intel.com:912`). Internet usually works
  both direct and via proxy — test both in the probe.

## 1. Connect (password auth, no sshpass needed)
`sshpass`/`paramiko` are often unavailable. Modern OpenSSH can feed a password
via `SSH_ASKPASS` + `SSH_ASKPASS_REQUIRE=force`. Store the password `600` and
build wrappers (see the "remote wrappers" snippet at the bottom). For repeat use,
install an SSH **key** and retire the password.

## 2. Probe (READ-ONLY — change nothing yet)
```bash
hostname; . /etc/os-release; echo "$PRETTY_NAME $(uname -r)"   # want Ubuntu 24.04 + kernel 6.8.x
lspci -nn | grep -ci '1da3:'                                    # expect 8 (Gaudi 3)
hl-smi --query-aip=index,name,memory.total --format=csv,noheader # 8 cards? or "driver not loaded"
lsmod | grep -cE 'habana|ib_uverbs'                             # expect 7 when healthy
dpkg -l | grep habanalabs                                       # 1.24.0-1007 packages present?
dkms status                                                     # "installed" for the RUNNING kernel?
cat /proc/cmdline                                               # has iommu=pt ?
docker info 2>/dev/null | grep -i runtimes                     # "habana" registered?
df -h $HOME | tail -1                                           # >250 GB free for weights
env | grep -i proxy; grep -i proxy /etc/environment            # proxy + no_proxy sane?
# internet:
for u in https://vault.habana.ai/ https://huggingface.co/; do curl -so/dev/null -w "%{http_code} $u\n" --max-time 12 "$u"; done
```
Decide from the results which of the steps below you still need.

## 3. Driver / kernel (only if `hl-smi` shows no cards)
Habana 1.24 **requires kernel 6.8** (`6.8.0-110`/`-111`…). Newer kernels (6.17,
6.11…) have **no matching habana modules** → `hl-smi` sees nothing. Fix:
```bash
# a) build the DKMS driver for 6.8 (headers must be present: dpkg -l linux-headers-6.8.0-*):
sudo dkms install habanalabs/1.24.0-1007 -k 6.8.0-110-generic
# if habanalabs-dkms is stuck half-configured (iF): reset then reconfigure:
sudo dkms remove habanalabs/1.24.0-1007 --all && sudo dpkg --configure -a
# b) pin GRUB to 6.8 + iommu, then SYNC grub.cfg (the step people forget):
#    /etc/default/grub: GRUB_DEFAULT=<...6.8 menuentry...>
#    GRUB_CMDLINE_LINUX_DEFAULT="quiet splash iommu=pt intel_iommu=on"
sudo update-grub          # <-- editing /etc/default/grub WITHOUT this = still boots old kernel
sudo reboot               # remote reboot: poll `ping` then ssh until it returns on 6.8
```
After reboot confirm: `uname -r` = 6.8.x, `lsmod|grep -cE 'habana|ib_uverbs'`=7,
`hl-smi` shows 8 cards. `linux-modules-extra-<kernel>` must be installed (gives
`ib_uverbs`, required by `habanalabs_ib` for HCL — even single-card vLLM inits HCL).

## 4. System tuning (only pieces the box is missing)
```bash
# hugepages (Habana formula: cores*110MB*2). Skip if HugePages_Total already ~13k-25k:
NR=$(( (110*1024*$(nproc)*2)/$(awk '/Hugepagesize/{print $2}' /proc/meminfo) + 1 ))
sudo sysctl -w vm.nr_hugepages=$NR; echo "vm.nr_hugepages=$NR" | sudo tee /etc/sysctl.d/99-habana-hugepages.conf
# module autoload + MSR/NIC tune service: `sudo bash install.sh` handles these idempotently,
# OR cherry-pick (see install.sh steps 6-10). Single-card serving does NOT need the scale-out NICs.
```

## 5. Docker (habana runtime + proxy + group)
```bash
sudo usermod -aG docker <user>                 # then a FRESH ssh session has docker access
ls /usr/bin/habana-container-runtime           # from habanalabs-container-runtime pkg
# /etc/docker/daemon.json must contain the habana runtime:
#   {"runtimes":{"habana":{"path":"/usr/bin/habana-container-runtime","runtimeArgs":[]}}}
# proxy drop-in so `docker pull` works (NO_PROXY keeps localhost/RFC1918 direct):
sudo install -d /etc/systemd/system/docker.service.d
#   http-proxy.conf: HTTP_PROXY/HTTPS_PROXY=proxy-dmz.intel.com:912,
#   NO_PROXY=localhost,127.0.0.1,::1,.intel.com,10.0.0.0/8,192.168.0.0/16,172.16.0.0/12
sudo systemctl daemon-reload && sudo systemctl restart docker
docker info | grep -i runtimes                 # expect: habana ... runc
```
NOTE: this box likely uses **docker.io** (not docker-ce). `install.sh`'s
`apt-get install docker.io` is then a no-op — safe.

## 6. Repo + per-user paths
```bash
rsync -a --exclude '.git/' --exclude '__pycache__/' <local>/gaudi-setup/ <user>@<host>:/home/<user>/gaudi-setup/
# vllm-launch hardcodes HF_CACHE=/home/satyajit-gaudi/hf-cache — REPOINT it per user:
sed -i 's#^HF_CACHE=.*#HF_CACHE=/home/<user>/hf-cache#' ~/gaudi-setup/bin/vllm-launch
mkdir -p /home/<user>/hf-cache
# put the tools on PATH:
sudo install -m755 ~/gaudi-setup/bin/{vllm-launch,gaudi-dash,hl-top,hl-top-mini,hl-power,launchpad} /usr/local/bin/
```

## 7. Images (base pull + patched Gemma image)
```bash
docker pull vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/vllm-0.17.1-ptfork-2.10.0:1.24.0-1007  # Qwen
docker pull vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/vllm-0.19.0-ptfork-2.10.0:1.24.0-1007  # Gemma/MiniMax base
cd ~/gaudi-setup/dockerfiles
docker build --build-arg HTTP_PROXY=$http_proxy --build-arg HTTPS_PROXY=$http_proxy \
  --build-arg NO_PROXY=localhost,127.0.0.1,.intel.com \
  -t gaudi-vllm-gemma4:0.19.0 -f Dockerfile.gemma4 .   # 5-patch stack; ~5 min
```
The build now pins torch via `dockerfiles/torch-constraints.txt` — REQUIRED, or an
unpinned `accelerate>=…` pulls CUDA torch and the image dies (see gotcha below).

## 8. Weights (pre-download in parallel with the image build)
```bash
python3 -m pip install --user --break-system-packages huggingface_hub hf_transfer   # if pip missing: sudo apt install python3-pip
HF_HOME=/home/<user>/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 \
  python3 -c 'from huggingface_hub import snapshot_download; snapshot_download("RedHatAI/gemma-4-31b-it-FP8-Dynamic")'
# MiniMax: snapshot_download("MiniMaxAI/MiniMax-M2")   # ~215 GB, CPU-bound to process at load
```

## 9. Launch + smoke
```bash
vllm-launch gemma4-31b                          # Gemma → card 2, :8004
vllm-launch minimax-m2                          # MiniMax → cards 4-7 TP=4, :8006
# wait for readiness (proxy MUST be bypassed — see gotcha):
until curl -fs --noproxy '*' http://localhost:8004/v1/models >/dev/null; do sleep 5; done
env http_proxy= https_proxy= no_proxy='*' bash scripts/smoke.sh http://localhost:8004 gemma4-31b 4000
```

---

## Gotchas learned this session (all cost real time)

1. **Wrong kernel = no cards.** Box booted 6.17; DKMS driver only exists for 6.8.
   Someone had edited `/etc/default/grub` but never ran `update-grub`, so it kept
   booting 6.17. Fix = build DKMS for 6.8 + `update-grub` + reboot.
2. **`Dockerfile.gemma4` torch drift.** Unpinned `accelerate>=1.10.0` now resolves
   to a version that drags in CUDA `torch 2.12`, clobbering the HPU torch fork →
   `"neither HPU fork nor CPU upstream"` → build dies at patch 1. Fixed with
   `torch-constraints.txt` + `pip -c`. Refresh pins after a base-image bump.
3. **Host `/etc/environment` proxy breaks localhost.** `curl localhost:8004`,
   `smoke.sh`, and `hl-top-mini`'s endpoint probe route through the corp proxy and
   fail — server is fine, probe lies. Use `--noproxy '*'`; `hl-top-mini` now bypasses
   the proxy in code. Also add `no_proxy=127.0.0.1,localhost,...` to `/etc/environment`.
4. **MiniMax TP math.** TP must divide `num_kv_heads=8` → only {1,2,4,8}. And FP8
   block-128: at TP=8 the gate/up shard=192 is not ÷128 → fails. **Max usable = TP=4.**
   "All 8 cards" needs 2× TP=4 replicas, not one TP=8. Steady HBM ≈100 GB/card, <128.
5. **MiniMax cold-load looks frozen.** After "Starting to load model" it processes
   MoE weights on **CPU** (700%+ CPU, HBM near 0) for minutes before HBM fills. Not a
   crash — don't kill it; watch `docker stats` CPU% and HBM climb.
6. **`gaudi-dash` panes.** btop hard-requires **80×24**; the old side-by-side split
   halved the width (<80) so btop blanked. Now full-width **stacked** with
   percentage heights so btop keeps ≥24 rows across resizes. Needs a terminal ≥~40 rows.
7. **HBM "leak" that isn't.** `--gpu-memory-utilization 0.9/0.8` pre-reserves the
   pool up front; high `hl-smi` numbers at idle are normal, not a leak.
8. **Reasoning eats the token budget.** Gemma/MiniMax emit a reasoning channel
   counted in `completion_tokens`; too-small `max_tokens` → empty `content`,
   `finish_reason=length`. Tool calls ≈70–280 out tokens; hard reasoning ≈1–8k.

## Remote wrappers (password auth over SSH, reusable)
```sh
printf '%s' '<password>' > /tmp/.sp; chmod 600 /tmp/.sp
printf '#!/bin/sh\ncat /tmp/.sp\n' > /tmp/.ap.sh; chmod 700 /tmp/.ap.sh
# rssh: run a remote command
#   env DISPLAY=:0 SSH_ASKPASS=/tmp/.ap.sh SSH_ASKPASS_REQUIRE=force setsid -w \
#     ssh -o StrictHostKeyChecking=accept-new -o NumberOfPasswordPrompts=1 <user>@<host> "$@"
# rsudo: remote root — pipe the password to `sudo -S -p ''` on stdin (SSH auth uses askpass, so stdin is free)
# rcp:  rsync -e '<the same ssh line>'
```

## Reference: the two boxes done this way
- **box2** `jf01sval0068` / 10.234.185.79 (user `user`) — needed the full kernel
  fix (6.17→6.8 reboot). See [[project_gaudi_box2]] memory.
- **box3** `g3-dell02` / 10.138.190.230 (user `ace`, India) — base already correct
  on 6.8, so just Steps 5–9. Runs Gemma 4 (:8004) + MiniMax M2 (:8006). See
  [[project_gaudi_box3]] memory.
