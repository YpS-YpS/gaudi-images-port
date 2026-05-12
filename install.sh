#!/usr/bin/env bash
# install.sh — bring an Ubuntu 24.04 Gaudi3 box from fresh OS to vLLM-ready.
#
# What it does (all idempotent — safe to re-run):
#   1. Verify Ubuntu 24.04, root, kernel.
#   2. Add Habana apt repo + GPG key.
#   3. Install monitoring tools (btop, htop, glances, iotop, iftop, nethogs, ncdu, tmux).
#   4. If running on the wrong kernel: install kernel 6.8.0 + headers, pin GRUB to it.
#   5. Add iommu=pt intel_iommu=on to GRUB (required by Habana on kernel 6.8).
#   6. Install Habana driver + userspace (hl-smi, hl_qual, container runtime).
#   7. Install ib_uverbs + persist habanalabs_ib module load (else HCL ibv init fails).
#   8. Configure 24640 hugepages (Habana installer formula: cores*110MB*2).
#   9. Apply Sapphire Rapids MSR perf tunings (energy_perf_bias=0, HWP max).
#  10. Bring up internal Gaudi NIC scale-out interfaces (24 internal ports).
#  11. Install gaudi-tune.service for boot-time persistence of MSRs + NIC bring-up.
#  12. Install Docker + wire up habana container runtime + Intel proxy (if env set).
#  13. Drop /usr/local/bin/{gaudi-dash, vllm-launch} for serving + dashboard.
#
# Usage:
#   sudo ./install.sh
#
# After first run on a wrong-kernel system: reboot into kernel 6.8 then re-run.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"

#=== UI helpers ============================================================
RED='\033[0;31m'; GRN='\033[0;32m'; YEL='\033[1;33m'; BLU='\033[0;34m'; NC='\033[0m'
log()  { printf "${BLU}[*]${NC} %s\n"  "$*"; }
ok()   { printf "${GRN}[✓]${NC} %s\n" "$*"; }
warn() { printf "${YEL}[!]${NC} %s\n" "$*"; }
err()  { printf "${RED}[✗]${NC} %s\n" "$*" >&2; }

#=== Preflight =============================================================
[[ $EUID -eq 0 ]] || { err "Run as root: sudo ./install.sh"; exit 1; }

. /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
  warn "Targets Ubuntu 24.04. Detected ${ID:-?} ${VERSION_ID:-?}. Continuing anyway."
fi

# Habana 1.24.0 requires kernel 6.8 on Ubuntu 24.04.
HABANA_KERNEL_PKG=linux-image-6.8.0-110-generic
HABANA_HEADER_PKG=linux-headers-6.8.0-110-generic
HABANA_KERNEL_VER=6.8.0-110-generic
KEYRING=/usr/share/keyrings/habana-artifactory.gpg

#=== 1. Habana apt repo ====================================================
log "Configuring Habana apt repo..."
existing=$(grep -lr 'vault.habana.ai' /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null | head -1 || true)
if [[ -n "$existing" ]]; then
  ok "Habana repo already configured: $existing"
else
  apt-get install -y --no-install-recommends curl gnupg ca-certificates >/dev/null
  curl -fsSL https://vault.habana.ai/artifactory/api/gpg/key/public \
    | gpg --dearmor -o "$KEYRING"
  chmod 644 "$KEYRING"
  echo "deb [signed-by=$KEYRING] https://vault.habana.ai/artifactory/debian noble main" \
    > /etc/apt/sources.list.d/habanalabs.list
  ok "Habana repo configured"
fi
apt-get update -qq

#=== 2. Monitoring tools ===================================================
log "Installing monitoring tools..."
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  btop htop glances iotop iftop nethogs ncdu tmux git curl \
  msr-tools \
  >/dev/null
ok "Monitoring tools installed"

#=== 3. Kernel 6.8 + GRUB pin (if not already on 6.8) ======================
RUNNING_KERNEL=$(uname -r)
NEEDS_REBOOT=0
if [[ "$RUNNING_KERNEL" =~ ^6\.8\.0- ]]; then
  ok "Already on kernel 6.8 ($RUNNING_KERNEL)"
else
  warn "On kernel $RUNNING_KERNEL — need 6.8 for Habana 1.24"
  log "Installing $HABANA_KERNEL_PKG..."
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    "$HABANA_KERNEL_PKG" "$HABANA_HEADER_PKG" || true
  NEEDS_REBOOT=1
fi

# Always ensure linux-modules-extra is installed for ib_uverbs (req'd by habanalabs_ib)
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  "linux-modules-extra-$RUNNING_KERNEL" >/dev/null 2>&1 || \
  warn "linux-modules-extra-$RUNNING_KERNEL not installable now (try after reboot to 6.8)"

#=== 4. GRUB: iommu params + pin to 6.8 ===================================
log "Configuring GRUB..."
cp -n /etc/default/grub /etc/default/grub.bak.gaudi-setup 2>/dev/null || true
update-grub >/dev/null 2>&1 || true

# Add iommu=pt intel_iommu=on to GRUB_CMDLINE_LINUX_DEFAULT if missing
if ! grep -E '^GRUB_CMDLINE_LINUX_DEFAULT=' /etc/default/grub | grep -q 'iommu=pt'; then
  sed -i 's|^\(GRUB_CMDLINE_LINUX_DEFAULT="[^"]*\)"|\1 iommu=pt intel_iommu=on"|' /etc/default/grub
  ok "Added iommu=pt intel_iommu=on to GRUB cmdline"
  NEEDS_REBOOT=1
else
  ok "GRUB already has iommu params"
fi

# Pin to 6.8 if available in grub.cfg
uuid=$(grep -oE "gnulinux-${HABANA_KERNEL_VER}-advanced-[a-f0-9-]+" /boot/grub/grub.cfg 2>/dev/null | head -1)
sub=$(grep -oE "gnulinux-advanced-[a-f0-9-]+" /boot/grub/grub.cfg 2>/dev/null | head -1)
if [[ -n "$uuid" && -n "$sub" ]]; then
  default="${sub}>${uuid}"
  if ! grep -q "GRUB_DEFAULT=\"${default}\"" /etc/default/grub; then
    sed -i \
      -e "s|^GRUB_DEFAULT=.*|GRUB_DEFAULT=\"${default}\"|" \
      -e 's|^GRUB_TIMEOUT_STYLE=.*|GRUB_TIMEOUT_STYLE=menu|' \
      -e 's|^GRUB_TIMEOUT=.*|GRUB_TIMEOUT=5|' \
      /etc/default/grub
    ok "GRUB pinned to $HABANA_KERNEL_VER (5s visible menu, fall back via menu)"
    NEEDS_REBOOT=1
  fi
fi
update-grub >/dev/null 2>&1 || true

#=== 5. Habana driver + userspace ==========================================
log "Installing Habana driver + userspace..."
# habanalabs-installer.sh path (Habana's preferred), but apt-direct works too.
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  habanalabs-dkms habanalabs-firmware habanalabs-firmware-tools \
  habanalabs-graph habanalabs-rdma-core habanalabs-thunk \
  habanalabs-tools habanalabs-container-runtime habanalabs-qual \
  || warn "apt install reported errors (will retry below)"

# If habanalabs-dkms is in 'iF' state, the previous postinst tried dkms add against
# an unsupported kernel and failed. Reset the dkms tree and reconfigure.
if dpkg -l habanalabs-dkms 2>/dev/null | awk '/habanalabs-dkms/{print $1}' | grep -q '^iF$'; then
  warn "habanalabs-dkms in iF state — resetting dkms tree"
  ver=$(dpkg-query -W -f '${Version}' habanalabs-dkms)
  dkms remove "habanalabs/${ver}" --all 2>/dev/null || true
  dpkg --configure -a || warn "(dpkg --configure -a still has errors — investigate after reboot)"
fi

#=== 6. ib_uverbs + habanalabs_ib module ===================================
# habanalabs_ib needs ib_uverbs (in linux-modules-extra) for HCL ibv init.
# Without this, vLLM-Gaudi fails with: "g_ibv.init() == hcclSuccess failed".
log "Loading + persisting habanalabs_ib (needs ib_uverbs)..."
modprobe ib_uverbs 2>/dev/null || warn "ib_uverbs not loadable (linux-modules-extra missing? — install after kernel 6.8 reboot)"
modprobe habanalabs_ib 2>/dev/null && ok "habanalabs_ib loaded" \
  || warn "habanalabs_ib failed to load (will succeed after reboot once ib_uverbs is present)"
install -m 644 -d /etc/modules-load.d
cat > /etc/modules-load.d/habana-modules.conf <<'EOF'
# habanalabs_ib is required for HCL collective comm.
# ib_uverbs is its dep, ships in linux-modules-extra-<kernel>.
ib_uverbs
habanalabs_ib
EOF
ok "module autoload persisted in /etc/modules-load.d/habana-modules.conf"

#=== 7. Hugepages (Habana installer formula) ==============================
HP_SIZE=$(awk '/^Hugepagesize:/ {print $2}' /proc/meminfo)
N_CORES=$(awk -F: '/^processor/{n++} END{print n}' /proc/cpuinfo)
NR_HP=$(( (110 * 1024 * N_CORES * 2) / HP_SIZE + 1 ))
log "Setting vm.nr_hugepages=$NR_HP (~$((NR_HP*HP_SIZE/1024/1024)) GB)..."
sysctl -w "vm.nr_hugepages=$NR_HP" >/dev/null
echo "vm.nr_hugepages=$NR_HP" > /etc/sysctl.d/99-habana-hugepages.conf
ok "hugepages persisted in /etc/sysctl.d/99-habana-hugepages.conf"

#=== 8. Sapphire Rapids MSR perf tunings ==================================
log "Applying Sapphire Rapids MSR perf tunings..."
modprobe msr
wrmsr -a 0x1b0 0x0     2>/dev/null && ok "MSR 0x1b0 = 0    (energy_perf_bias=performance)" || warn "wrmsr 0x1b0 failed"
wrmsr -a 0x774 0x2708  2>/dev/null && ok "MSR 0x774 = 0x2708 (HWP max)"                      || warn "wrmsr 0x774 failed"

#=== 9. Bring up internal Gaudi NICs (Habana docs require this on every load) ===
NIC=/opt/habanalabs/qual/gaudi3/bin/manage_network_ifs.sh
if [[ -x "$NIC" ]]; then
  log "Bringing up Gaudi internal NIC scale-out interfaces..."
  "$NIC" --up >/dev/null 2>&1 || warn "manage_network_ifs.sh --up exited non-zero"
  ok "Gaudi NICs toggled up (3 external ports per card stay down on standalone box)"
else
  warn "manage_network_ifs.sh not found — skipping (script will retry after reboot)"
fi

#=== 10. systemd service for boot-time persistence (MSRs + NICs) ==========
log "Installing gaudi-tune.service..."
install -m 644 "$SCRIPT_DIR/systemd/gaudi-tune.service" /etc/systemd/system/gaudi-tune.service 2>/dev/null \
  || cat > /etc/systemd/system/gaudi-tune.service <<'EOF'
[Unit]
Description=Gaudi performance tuning + scale-out NIC bring-up
After=multi-user.target
ConditionVirtualization=!container

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStartPre=/usr/sbin/modprobe msr
ExecStart=/usr/sbin/wrmsr -a 0x1b0 0x0
ExecStart=/usr/sbin/wrmsr -a 0x774 0x2708
ExecStart=/bin/bash -c '/opt/habanalabs/qual/gaudi3/bin/manage_network_ifs.sh --up || true'

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable gaudi-tune.service >/dev/null
ok "gaudi-tune.service enabled (re-applies MSRs + NICs every boot)"

#=== 11. Docker + Habana runtime + Intel proxy ============================
log "Installing Docker + Habana runtime..."
DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io >/dev/null
systemctl enable --now docker >/dev/null
usermod -aG docker "$SUDO_USER" 2>/dev/null || true

# Wire habana runtime into daemon.json
if ! grep -q '"habana"' /etc/docker/daemon.json 2>/dev/null; then
  cat > /etc/docker/daemon.json <<'EOF'
{
  "runtimes": {
    "habana": {
      "path": "/usr/bin/habana-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF
  ok "/etc/docker/daemon.json updated"
fi

# Intel corporate proxy → systemd drop-in (only if we detect we're behind one)
if [[ -n "${HTTP_PROXY:-}${http_proxy:-}" ]] || grep -q 'proxy-dmz.intel.com' /etc/environment 2>/dev/null; then
  PROXY="${HTTP_PROXY:-${http_proxy:-http://proxy-dmz.intel.com:912}}"
  install -m 644 -d /etc/systemd/system/docker.service.d
  cat > /etc/systemd/system/docker.service.d/http-proxy.conf <<EOF
[Service]
Environment="HTTP_PROXY=$PROXY"
Environment="HTTPS_PROXY=$PROXY"
Environment="NO_PROXY=localhost,127.0.0.1,::1,.intel.com,10.0.0.0/8,192.168.0.0/16,172.16.0.0/12"
EOF
  systemctl daemon-reload
  ok "docker proxy drop-in installed: $PROXY"
fi
systemctl restart docker
sleep 1

#=== 12. Drop dashboard + launcher into /usr/local/bin ====================
[[ -x "$SCRIPT_DIR/bin/gaudi-dash" ]]   && install -m 755 "$SCRIPT_DIR/bin/gaudi-dash"   /usr/local/bin/gaudi-dash   && ok "/usr/local/bin/gaudi-dash"
[[ -x "$SCRIPT_DIR/bin/vllm-launch" ]]  && install -m 755 "$SCRIPT_DIR/bin/vllm-launch"  /usr/local/bin/vllm-launch  && ok "/usr/local/bin/vllm-launch"

#=== Done ==================================================================
echo
ok "Setup complete."
echo
if (( NEEDS_REBOOT )); then
  printf "${YEL}=== REBOOT REQUIRED ===${NC}\n"
  echo  "Kernel and/or GRUB cmdline changed. After reboot, re-run this script to"
  echo  "finish the parts that need the new kernel."
  echo
  echo  "    sudo reboot"
  echo  "    # come back up:"
  echo  "    sudo $0"
else
  echo  "Try it out:"
  echo  "    bash $SCRIPT_DIR/scripts/check.sh    # readiness check"
  echo  "    hl-smi                                 # 8 Gaudis"
  echo  "    gaudi-dash                             # tmux dashboard"
  echo  "    vllm-launch list                       # available presets"
  echo  "    vllm-launch 32b-thinking               # start serving Qwen3-VL-32B"
fi
