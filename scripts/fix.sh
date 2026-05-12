#!/usr/bin/env bash
# fix.sh — re-apply runtime tunings without a reboot.
# Use after: a fresh boot where gaudi-tune.service didn't run, OR after manual
# changes to MSRs/hugepages/NICs that drifted the system off the documented config.
#
# Does NOT change GRUB or install packages — for those, use install.sh (and reboot).
#
# Usage:
#   sudo ./fix.sh

set -eu
[[ $EUID -eq 0 ]] || { echo "run as root: sudo $0"; exit 1; }

GRN='\033[0;32m'; NC='\033[0m'
ok() { printf "${GRN}[✓]${NC} %s\n" "$*"; }

# 1. MSR perf tunings (Sapphire Rapids)
modprobe msr 2>/dev/null
wrmsr -a 0x1b0 0x0    && ok "MSR 0x1b0 = 0    (energy_perf_bias=performance)"
wrmsr -a 0x774 0x2708 && ok "MSR 0x774 = 0x2708 (HWP max)"

# 2. Hugepages (recompute for current core count)
HP_SIZE=$(awk '/^Hugepagesize:/ {print $2}' /proc/meminfo)
N_CORES=$(awk -F: '/^processor/{n++} END{print n}' /proc/cpuinfo)
NR_HP=$(( (110 * 1024 * N_CORES * 2) / HP_SIZE + 1 ))
sysctl -w vm.nr_hugepages=$NR_HP >/dev/null
ok "hugepages = $NR_HP × ${HP_SIZE}kB ≈ $((NR_HP*HP_SIZE/1024/1024)) GB"

# 3. Bring up Gaudi internal NICs (24 internal interfaces; 3 external stay down)
NIC=/opt/habanalabs/qual/gaudi3/bin/manage_network_ifs.sh
if [[ -x "$NIC" ]]; then
  "$NIC" --up >/dev/null 2>&1 || true
  ok "Gaudi NICs toggled up"
fi

# 4. Make sure habanalabs_ib is loaded (depends on ib_uverbs)
modprobe ib_uverbs 2>/dev/null    && ok "ib_uverbs loaded"      || echo "  ib_uverbs unavailable — install linux-modules-extra-$(uname -r)"
modprobe habanalabs_ib 2>/dev/null && ok "habanalabs_ib loaded" || echo "  habanalabs_ib failed to load"

# 5. (Optional) THP — Habana docs are silent on this, default Ubuntu 'madvise' is fine.
#    Set to 'never' only if you have evidence of THP-related bugs.

echo
echo "(GRUB cmdline changes & package installs require install.sh + reboot.)"
