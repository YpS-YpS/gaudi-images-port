# gaudi-setup

One-shot installer that brings an **Ubuntu 24.04** Intel Gaudi machine to a working,
monitored state.

What you get after running it:

- Habana (Intel Gaudi Software) driver `1.24.0` working — `hl-smi` lists your accelerators
- Kernel `6.8.0-110-generic` installed and pinned in GRUB (the kernel the Habana driver actually supports on Ubuntu 24.04 — newer kernels break the driver build)
- A 5-second visible boot menu so you can fall back to your old kernel if anything goes wrong
- Monitoring tools: `btop`, `htop`, `glances`, `iotop`, `iftop`, `nethogs`, `ncdu`, `tmux`
- A `gaudi-dash` command — a tmux multi-pane dashboard (CPU/RAM + processes + 8× Gaudi)

## Requirements

- Ubuntu 24.04 (noble)
- Root / sudo
- An Intel Gaudi accelerator (Gaudi2 or Gaudi3). The script runs without one, but
  the Habana userspace install is only useful with hardware present.

## Install

```bash
git clone https://github.com/<your-user>/gaudi-setup.git
cd gaudi-setup
sudo ./install.sh
```

If you're not currently on kernel 6.8, the script will install it, pin GRUB to it,
and tell you to reboot. After reboot, run the script **a second time** to finish the
Habana userspace install:

```bash
sudo reboot
# ...comes back up on kernel 6.8...
cd gaudi-setup && sudo ./install.sh
```

Re-running the script is safe — it skips anything already done.

## Use

```bash
hl-smi              # one-shot table of all accelerators
hl-smi -l 1         # live updating, 1-second refresh
gaudi-dash          # full multi-pane dashboard (Ctrl-b d to detach, gaudi-dash kill to tear down)
glances -w          # web dashboard (already installed as a service on :61208)
```

Inside `gaudi-dash`:

| Keys | Action |
|---|---|
| `Ctrl-b d` | Detach (session keeps running) |
| `Ctrl-b z` | Zoom current pane |
| `Ctrl-b ←/→/↑/↓` | Move between panes |
| `gaudi-dash` (re-run) | Re-attach |
| `gaudi-dash kill` | Stop session |

## Why kernel 6.8?

The Habana 1.24.0 driver only compiles cleanly against **kernel 6.8** on Ubuntu 24.04
(per Intel's Gaudi Software support matrix). Ubuntu HWE will happily roll you forward
to kernel 6.17, at which point the driver `dkms` build fails (warnings-as-errors against
newer kernel APIs), the `habanalabs` module never loads, and `hl-smi` is missing.

If you've ended up on a newer kernel, this script installs 6.8 alongside, pins GRUB to
boot it, and leaves the newer kernel as a fallback in the GRUB advanced menu.

## What this script does

1. Adds the Habana apt repo (`vault.habana.ai/.../debian noble main`) with GPG key
   pinned to `/usr/share/keyrings/habana-artifactory.gpg`.
2. `apt install` the monitoring tools.
3. If the running kernel isn't 6.8.x, installs `linux-image-6.8.0-110-generic`
   + headers, edits `/etc/default/grub` to default-boot the 6.8 menuentry,
   sets `GRUB_TIMEOUT=5`, runs `update-grub`. Backs up the original at
   `/etc/default/grub.bak.before-6.8-pin`.
4. If running on 6.8, repairs any broken `habanalabs-dkms` state, then installs
   the Habana userspace packages so `hl-smi` works.
5. Installs `/usr/local/bin/gaudi-dash`.

## Reverting

To go back to your old kernel:
```bash
sudo cp /etc/default/grub.bak.before-6.8-pin /etc/default/grub
sudo update-grub
sudo reboot
```

To remove the dashboard helper: `sudo rm /usr/local/bin/gaudi-dash`.

## License

MIT
