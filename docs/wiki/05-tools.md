# 05 — tools

Every script in `bin/` and `scripts/`, what each does, when to use it.

## `bin/vllm-launch` — the workhorse launcher

[Source](../../bin/vllm-launch). One-line launcher backed by a preset table.

```bash
vllm-launch list                      # show all presets
vllm-launch <preset>                  # start one
vllm-launch stop <preset>             # stop one
vllm-launch stop-all                  # stop every vllm-* container
vllm-launch logs <preset>             # follow logs

# env overrides (added this session)
LAUNCH_DEVICES="0,1" vllm-launch <preset>     # custom cards (TP auto-derives)
LAUNCH_PORT=8500   vllm-launch <preset>       # custom port
```

Preset rows are at line 20-39. Each row is pipe-separated:
`<preset>=<model_id>|<TP>|<cards>|<port>|<max-len>|[optional-image]`

To add a new preset: insert a row, optionally add a per-preset parser branch
at line 60-95.

## `bin/launchpad` — interactive picker (new this session)

[Source](../../bin/launchpad). Wraps `vllm-launch` with a TUI flow:

1. Banner + card matrix (per-Gaudi HBM bars, container ownership)
2. Numbered menu of every preset, with ★ for recommended and ⚠ for broken
3. Prompts for custom card list (validates that count = preset's TP)
4. Prompts for port override + checks for existing port conflicts
5. Launches + waits for `/v1/models` to come up (timeout 1500s)
6. Fires the 17×89 smoke test
7. Prints a launch report

```bash
./bin/launchpad                        # interactive
./bin/launchpad state                  # just show the state matrix and exit
./bin/launchpad stop <preset>          # quick stop wrapper
./bin/launchpad stop-all               # quick stop-all wrapper
```

Use this when:
- You don't remember preset names
- You want to run something on non-default cards
- You want the launch verified before walking away

## `bin/gaudi-dash` — tmux dashboard

[Source](../../bin/gaudi-dash). Opens a tmux session with three panes:
- top: `btop` (CPU/RAM/network)
- middle: `htop` (process list)
- bottom: `hl-smi --loop` (per-card HBM/utilization)

Use this when you want a live view during a long-running operation.

## `scripts/check.sh` — readiness checklist

[Source](../../scripts/check.sh). Green/red verification:
- Kernel pin
- GRUB has iommu=pt
- All 7 Habana modules loaded
- 24640 hugepages present
- gaudi-tune.service enabled
- Docker daemon running with habana runtime
- HBM accessible

Run after every reboot, or when something feels off:

```bash
sudo bash scripts/check.sh
```

## `scripts/fix.sh` — re-apply runtime tunings

[Source](../../scripts/fix.sh). Idempotent re-application of:
- MSRs
- NIC bring-up
- hugepages
- modprobe of any missing modules

No reboot required for most. Use when `check.sh` shows red lines but you don't
want a full `install.sh` re-run.

## `scripts/verify.sh` — full readiness + smoke

[Source](../../scripts/verify.sh). `check.sh` + `hl_qual` GEMM benchmark + API
smoke for the default endpoint. Slower than `check.sh`. Use:
- After a fresh `install.sh` to verify everything green
- When you suspect a hardware issue (`hl_qual` will fail)

## `scripts/smoke.sh` — full API smoke for one endpoint

[Source](../../scripts/smoke.sh).

```bash
scripts/smoke.sh http://localhost:8000 qwen3-vl-32b-thinking
```

Tests in order:
1. `/v1/models` reachable
2. Plain text generation
3. Vision (inline PNG)
4. Single tool call
5. Parallel tools × 4 cities (same tool, 4 cities)
6. Parallel mixed × 4 different tools (weather + flights + exchange + hotel)

Output is colored ✓/✗ per stage with timing and token counts.

## `scripts/quality-bench.sh` — cross-endpoint quality bench

[Source](../../scripts/quality-bench.sh). Added this session for the L1-Dep test.

```bash
scripts/quality-bench.sh                                          # default L1-Dep prompt
scripts/quality-bench.sh "your prompt here"                       # custom prompt
scripts/quality-bench.sh "your prompt" 2500                       # custom prompt + max_tokens
```

Fires the same prompt at every running endpoint in parallel, prints all
responses side-by-side with timing. Used for subjective "which model gives a
sharper answer to this technical question" comparisons.

The default prompt is *"What is L1 Dep in microarchitecture and how does it
affect gaming performance?"* — a deliberately ambiguous test (the technical
reading is L1 load-to-use latency or L1 cache dependencies). Gemma 4 nails
the load-to-use interpretation; MiniMax M2 invents that "Dep is common
shorthand for Data" (factually wrong).

## `bootstrap.sh` (root of repo) — one-click bring-up

[Source](../../bootstrap.sh). The end-to-end installer:

```bash
sudo ./bootstrap.sh --with-gemma4   # default Qwen 32B-Thinking + Gemma 4 31B
sudo ./bootstrap.sh                  # just Qwen, no Gemma image build
```

Calls `install.sh` → `docker pull` → builds Gemma image (if `--with-gemma4`) →
downloads models → launches. Reboots once if the kernel needs swapping.

Don't use on a box you care about without reading `install.sh` first — it
modifies `/etc/`, GRUB, kernel modules, MSRs, hugepages.

## `install.sh` — idempotent base bring-up

[Source](../../install.sh). What it does:
1. Install kernel 6.8 + headers (pinned), reboot if newer kernel was active
2. Install Habana driver 1.24.0 + all 7 kernel modules
3. Set hugepages, MSRs, GRUB flags
4. Bring up scale-out NICs
5. Install Docker + Habana container runtime
6. Drop proxy systemd files
7. Enable `gaudi-tune.service` for boot persistence

Idempotent — safe to re-run. Use as a recovery option if something's broken at
the OS layer.

## Tool flow — typical day-2 ops

```
                 launchpad
                    │
                    ├─→ vllm-launch <preset>           (or LAUNCH_DEVICES=... vllm-launch ...)
                    │           │
                    │           ↓
                    │    docker run -d --runtime=habana ...
                    │
                    └─→ smoke.sh after ready
                            │
                            ↓
                       launch report
```

For health: `gaudi-dash` (live) or `launchpad state` (snapshot).
For comparison: `quality-bench.sh`.
For repair: `check.sh` → `fix.sh` → `install.sh` (escalating).
