#!/usr/bin/env bash
# gaudi-readiness.sh — print a green/red checklist of Habana docs requirements.
# Run anytime; safe / read-only.
set -u

G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'; B='\033[0;34m'; N='\033[0m'
ok()   { printf "${G}✓${N} %s\n" "$*"; }
bad()  { printf "${R}✗${N} %s\n" "$*"; }
warn() { printf "${Y}⚠${N} %s\n" "$*"; }
hdr()  { printf "\n${B}── %s ──${N}\n" "$*"; }

hdr "kernel & boot params"
k=$(uname -r); echo "kernel: $k"
[[ "$k" =~ ^6\.8\.0- ]] && ok "kernel 6.8.0 (Habana 1.24 supported)" || bad "kernel $k not in Habana support matrix"
cm=$(cat /proc/cmdline)
grep -q 'intel_iommu=on' <<<"$cm" && ok "intel_iommu=on present"     || bad "intel_iommu=on MISSING (Habana required for kernel 6.8)"
grep -q 'iommu=pt'      <<<"$cm" && ok "iommu=pt present"            || bad "iommu=pt MISSING (Habana required for kernel 6.8)"

hdr "hardware visibility"
n=$(lspci -d 1da3: -nn 2>/dev/null | wc -l)
[[ "$n" == 8 ]] && ok "8 Habana PCIe devices visible" || bad "expected 8, got $n"
n=$(ls /sys/class/accel/ 2>/dev/null | grep -c '^accel[0-9]$')
[[ "$n" == 8 ]] && ok "8 /sys/class/accel/accelN devices" || bad "expected 8 accelN, got $n"
hl-smi -L >/dev/null 2>&1 && ok "hl-smi reports devices" || bad "hl-smi failed"

hdr "driver modules"
for m in habanalabs habanalabs_en habanalabs_cn habanalabs_compat habanalabs_ib ib_uverbs ib_core; do
  lsmod | awk '{print $1}' | grep -qx "$m" && ok "$m loaded" || bad "$m NOT loaded"
done

hdr "userspace packages"
for p in habanalabs-dkms habanalabs-firmware habanalabs-firmware-tools \
         habanalabs-graph habanalabs-rdma-core habanalabs-thunk \
         habanalabs-tools habanalabs-container-runtime habanalabs-qual; do
  dpkg -l "$p" 2>/dev/null | grep -q '^ii' && ok "$p installed" || warn "$p NOT installed"
done

hdr "performance tunings (Sapphire Rapids)"
need_msr=0
for c in $(awk -F: '/^processor/{print $2}' /proc/cpuinfo); do
  v=$(sudo rdmsr -p "$c" 0x1b0 2>/dev/null); [[ "$v" == "0" ]] || { need_msr=1; break; }
done
[[ $need_msr == 0 ]] && ok "MSR 0x1b0 = 0 on all cores (energy_perf_bias=performance)" \
                     || bad "MSR 0x1b0 not 0 on all cores — run sudo wrmsr -a 0x1b0 0x0"
need_msr=0
for c in $(awk -F: '/^processor/{print $2}' /proc/cpuinfo); do
  v=$(sudo rdmsr -p "$c" 0x774 2>/dev/null); [[ "$v" == "2708" ]] || { need_msr=1; break; }
done
[[ $need_msr == 0 ]] && ok "MSR 0x774 = 0x2708 on all cores (HWP max)" \
                     || bad "MSR 0x774 not 0x2708 — run sudo wrmsr -a 0x774 0x2708"

hdr "transparent hugepages"
thp=$(awk -F'[][]' '{print $2}' /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)
echo "THP setting: $thp"

hdr "hugepages (Habana installer formula)"
hp_total=$(awk '/^HugePages_Total:/ {print $2}' /proc/meminfo)
hp_size=$(awk '/^Hugepagesize:/ {print $2}' /proc/meminfo)
hp_gb=$(awk -v t="$hp_total" -v s="$hp_size" 'BEGIN{printf "%.1f", t*s/1024/1024}')
echo "HugePages_Total: $hp_total  ×  ${hp_size}kB  =  ${hp_gb} GB"
n_cores=$(awk -F: '/^processor/{n++} END{print n}' /proc/cpuinfo)
expected=$(( (110 * 1024 * n_cores * 2) / hp_size + 1 ))
[[ "$hp_total" -ge $((expected * 9 / 10)) ]] && ok "hugepages within Habana installer expected (~$expected)" \
                                              || bad "hugepages=$hp_total, Habana installer would set ~$expected"
[[ -f /etc/sysctl.d/99-habana-hugepages.conf ]] && ok "hugepages persisted (/etc/sysctl.d/99-habana-hugepages.conf)" \
                                                 || warn "hugepages not persisted — won't survive reboot"

hdr "Habana NIC scale-out interfaces"
NIC=/opt/habanalabs/qual/gaudi3/bin/manage_network_ifs.sh
if [[ -x "$NIC" ]]; then
  down=$(sudo "$NIC" --status 2>&1 | grep -c 'down')
  if [[ "$down" -le 8 ]]; then
    ok "internal NICs up (only ${down} card(s) report external ports down — expected on standalone box)"
  else
    bad "$down lines reporting 'down' — run sudo $NIC --up"
  fi
else
  warn "manage_network_ifs.sh not found"
fi

hdr "boot-time persistence services"
systemctl is-enabled gaudi-tune.service >/dev/null 2>&1 && ok "gaudi-tune.service enabled (MSRs + NICs on boot)" \
                                                        || warn "gaudi-tune.service not enabled"

hdr "docker + habana runtime"
docker info 2>/dev/null | grep -q 'Runtimes:.*habana' && ok "docker habana runtime registered" || bad "docker habana runtime missing"

hdr "vLLM image cached"
docker images vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/vllm-0.17.1-ptfork-2.10.0:1.24.0-1007 --format '{{.Size}}' 2>/dev/null | grep -q . \
  && ok "ptfork vllm image present" || warn "ptfork vllm image NOT present"

hdr "models on disk"
[[ -d /home/satyajit-gaudi/hf-cache/hub/models--Qwen--Qwen3-VL-32B-Thinking-FP8 ]] && ok "32B-Thinking-FP8" || warn "32B-Thinking-FP8 missing"
[[ -d /home/satyajit-gaudi/hf-cache/hub/models--Qwen--Qwen3-VL-235B-A22B-Thinking-FP8 ]] && ok "235B-A22B-Thinking-FP8" || warn "235B not on disk"

hdr "kernel oops history (since boot)"
n=$(sudo dmesg 2>/dev/null | grep -c 'Bad pagetable')
[[ "$n" == 0 ]] && ok "no Bad pagetable events since boot" || bad "$n Bad pagetable events in dmesg"

hdr "disk + memory"
df -h / | tail -1 | awk '{printf "  / : %s used / %s total (%s avail)\n",$3,$2,$4}'
free -h | awk '/^Mem:/ {printf "  RAM: %s used / %s total\n",$3,$2}'

echo
