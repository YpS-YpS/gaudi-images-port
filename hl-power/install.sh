#!/usr/bin/env bash
# Install hl-power systemd user units and start them.
set -euo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SYSTEMD_USER_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$SYSTEMD_USER_DIR"

install -m 0644 "$HERE/systemd/hl-power-logger.service" "$SYSTEMD_USER_DIR/"
install -m 0644 "$HERE/systemd/hl-power-server.service" "$SYSTEMD_USER_DIR/"

systemctl --user daemon-reload
systemctl --user enable --now hl-power-logger.service hl-power-server.service

echo
echo "Installed and started:"
systemctl --user --no-pager status hl-power-logger.service | head -3
systemctl --user --no-pager status hl-power-server.service | head -3
echo
echo "Dashboard: http://$(hostname -I | awk '{print $1}'):9090"
echo
echo "Note: services run while you are logged in. To keep them up across"
echo "reboots even when you are NOT logged in, run once (needs sudo):"
echo "  sudo loginctl enable-linger $USER"
