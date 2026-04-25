#!/usr/bin/env bash
# install.sh — bring an Ubuntu 24.04 Gaudi machine to a working state.
#
# What this does (idempotent — safe to re-run):
#   1. Verify Ubuntu 24.04 (warn otherwise).
#   2. Add the Habana (Intel Gaudi Software) apt repo + GPG key.
#   3. Install monitoring tools: btop, htop, glances, iotop, iftop, nethogs, ncdu, tmux.
#   4. Install Habana userspace tools (hl-smi etc.) so monitoring works.
#   5. If the running kernel isn't supported by the Habana driver, install kernel 6.8
#      and pin GRUB to boot it. (Gaudi SW 1.24.0 supports kernel 6.8 on Ubuntu 24.04.)
#   6. Install /usr/local/bin/gaudi-dash — multi-pane tmux dashboard
#      (btop + htop + hl-smi -l 1).
#
# Usage:
#   sudo ./install.sh
#
# After first run, if a kernel was installed, you'll be told to reboot.

set -euo pipefail

#=== ID & color ============================================================
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()    { printf "${BLUE}[*]${NC} %s\n"  "$*"; }
ok()     { printf "${GREEN}[✓]${NC} %s\n" "$*"; }
warn()   { printf "${YELLOW}[!]${NC} %s\n" "$*"; }
err()    { printf "${RED}[✗]${NC} %s\n"   "$*" >&2; }

#=== Preflight =============================================================
if [[ $EUID -ne 0 ]]; then
  err "Run as root: sudo ./install.sh"
  exit 1
fi

. /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
  warn "This script targets Ubuntu 24.04 (noble). Detected: ${ID:-?} ${VERSION_ID:-?}"
  warn "Continuing, but kernel 6.8 + Habana 1.24 expectations may not hold."
fi

KEYRING=/usr/share/keyrings/habana-artifactory.gpg
SOURCES=/etc/apt/sources.list.d/habanalabs.list
HABANA_KERNEL_PKG=linux-image-6.8.0-110-generic
HABANA_HEADER_PKG=linux-headers-6.8.0-110-generic
HABANA_KERNEL_VER=6.8.0-110-generic
DASHBOARD=/usr/local/bin/gaudi-dash

#=== 1. Habana apt repo ====================================================
log "Configuring Habana apt repo..."
existing_repo_file=$(grep -lr 'vault.habana.ai' /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null | head -1 || true)
if [[ -n "$existing_repo_file" ]]; then
  ok "Habana repo already configured ($existing_repo_file) — leaving alone"
else
  if [[ ! -f $KEYRING ]]; then
    apt-get install -y --no-install-recommends curl gnupg ca-certificates >/dev/null
    curl -fsSL https://vault.habana.ai/artifactory/api/gpg/key/public \
      | gpg --dearmor -o "$KEYRING"
    chmod 644 "$KEYRING"
    ok "GPG key written to $KEYRING"
  fi
  echo "deb [signed-by=$KEYRING] https://vault.habana.ai/artifactory/debian noble main" > "$SOURCES"
  ok "Wrote $SOURCES"
fi

log "apt-get update..."
apt-get update -qq

#=== 2. Monitoring tools ===================================================
log "Installing monitoring tools..."
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  btop htop glances iotop iftop nethogs ncdu tmux

ok "Monitoring tools installed"

#=== 3. Kernel 6.8 if not running it ======================================
RUNNING_KERNEL=$(uname -r)
NEEDS_REBOOT=0
if [[ "$RUNNING_KERNEL" =~ ^6\.8\.0- ]]; then
  ok "Already on kernel 6.8 ($RUNNING_KERNEL) — supported by Habana driver"
else
  warn "Running kernel is $RUNNING_KERNEL — Habana driver needs 6.8 on Ubuntu 24.04"
  log  "Installing $HABANA_KERNEL_PKG and headers..."
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    "$HABANA_KERNEL_PKG" "$HABANA_HEADER_PKG" || true
  # 'apt' may exit nonzero if habanalabs-dkms postinst failed against current kernel.
  # The kernel itself installs fine; we'll fix dkms after reboot.

  log "Pinning GRUB to boot $HABANA_KERNEL_VER..."
  cp -n /etc/default/grub /etc/default/grub.bak.before-6.8-pin
  # find the menuentry id for the new kernel from grub.cfg
  update-grub >/dev/null 2>&1 || true
  uuid=$(grep -oE "gnulinux-${HABANA_KERNEL_VER}-advanced-[a-f0-9-]+" /boot/grub/grub.cfg | head -1)
  submenu_uuid=$(grep -oE "gnulinux-advanced-[a-f0-9-]+" /boot/grub/grub.cfg | head -1)
  if [[ -n "$uuid" && -n "$submenu_uuid" ]]; then
    new_default="${submenu_uuid}>${uuid}"
    sed -i \
      -e "s|^GRUB_DEFAULT=.*|GRUB_DEFAULT=\"${new_default}\"|" \
      -e 's|^GRUB_TIMEOUT_STYLE=.*|GRUB_TIMEOUT_STYLE=menu|' \
      -e 's|^GRUB_TIMEOUT=.*|GRUB_TIMEOUT=5|' \
      /etc/default/grub
    update-grub
    ok "GRUB pinned to $HABANA_KERNEL_VER (5s visible menu, fall back to old kernel via menu)"
    NEEDS_REBOOT=1
  else
    err "Could not find menu entry for $HABANA_KERNEL_VER in /boot/grub/grub.cfg"
    err "Pin GRUB manually before rebooting."
  fi
fi

#=== 4. Habana userspace tools (only attempt if 6.8 running, else after reboot) ====
if [[ "$RUNNING_KERNEL" =~ ^6\.8\.0- ]]; then
  log "Ensuring Habana driver dkms state is clean..."
  # If a previous postinst left package half-installed, clear the dkms tree and reconfigure.
  if dpkg -l habanalabs-dkms 2>/dev/null | awk '/habanalabs-dkms/ {print $1}' | grep -q '^iF$'; then
    warn "habanalabs-dkms in 'iF' state; resetting dkms tree..."
    dkms remove habanalabs/$(dpkg-query -W -f '${Version}' habanalabs-dkms) --all 2>/dev/null || true
    dpkg --configure -a || warn "(dpkg --configure -a still has errors — investigate after run)"
  fi

  log "Installing Habana userspace (hl-smi etc.)..."
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    habanalabs-tools habanalabs-thunk habanalabs-firmware-tools habanalabs-graph \
    habanalabs-rdma-core habanalabs-container-runtime || true

  if command -v hl-smi >/dev/null && hl-smi >/dev/null 2>&1; then
    cards=$(hl-smi -L 2>/dev/null | grep -c 'AIP' || true)
    ok "hl-smi works — $cards Gaudi accelerator(s) visible"
  else
    warn "hl-smi not yet functional. After the reboot, run: sudo dpkg --configure -a"
  fi
else
  warn "Skipping Habana userspace install — needs kernel 6.8. Re-run this script after reboot."
fi

#=== 5. gaudi-dash =========================================================
log "Installing gaudi-dash to $DASHBOARD..."
install -m 755 "$(dirname "$(realpath "$0")")/bin/gaudi-dash" "$DASHBOARD"
ok "gaudi-dash installed"

#=== Done ==================================================================
echo
ok "Setup complete."
echo
if [[ $NEEDS_REBOOT -eq 1 ]]; then
  printf "${YELLOW}=== REBOOT REQUIRED ===${NC}\n"
  echo  "Kernel 6.8 was installed and pinned. Reboot, then run this script again to"
  echo  "finish the Habana userspace install:"
  echo
  echo  "    sudo reboot"
  echo  "    # after coming back up:"
  echo  "    sudo $(realpath "$0")"
else
  echo  "Try it out:"
  echo
  echo  "    hl-smi -l 1     # live single-pane Gaudi monitor"
  echo  "    gaudi-dash      # multi-pane CPU/RAM/Gaudi dashboard"
fi
