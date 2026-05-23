# START HERE — orientation for a new session

> If you (a future Claude or a new human) are picking up this box, read this
> file first. It's short and points at everything else you need.

## In one paragraph

This is a Dell PowerEdge XE9680 with 8× Intel Gaudi 3 (HL-325) accelerators,
serving multiple vLLM endpoints. The repo at `/home/satyajit-gaudi/gaudi-setup/`
(GitHub: [YpS-YpS/gaudi-images-port](https://github.com/YpS-YpS/gaudi-images-port))
contains the bring-up scripts, derived docker images, presets, and operational
tools. Anything you do with this box should go through that repo.

## Who's working with you

- **Satyajit Bhuyan** — Intel/Habana, GitHub [@YpS-YpS](https://github.com/YpS-YpS)
- Sudo: ask the user — **do not embed credentials in pipes**.
  Use `! sudo ...` prefix (which prompts interactively) or ask the user to run
  the command themselves and paste the output.

## Three things to do FIRST

```bash
# 1. What's running right now?
docker ps --filter name=vllm- --format '{{.Names}}\t{{.Status}}'

# 2. Per-card HBM state
hl-smi --query-aip=index,memory.used,memory.free --format=csv,noheader

# 3. Recent commits — what was the last thing changed?
cd /home/satyajit-gaudi/gaudi-setup && git log --oneline -10
```

Or just run the dashboard once:

```bash
/home/satyajit-gaudi/gaudi-setup/bin/launchpad state
```

That prints the card matrix + running endpoints in one screen.

## What works today

| Model | Port | Cards | Image | Status |
|---|---|---|---|---|
| Qwen3-VL-32B-Thinking | 8000 | Gaudi 0 | vllm-0.17.1-ptfork | ✓ ~49 tok/s |
| Qwen3-VL-32B-Instruct | 8001 | Gaudi 1 | vllm-0.17.1-ptfork | ✓ |
| Gemma 4 31B (Anthropic API) | 8004 | Gaudi 2 | gaudi-vllm-gemma4:0.19.0 | ✓ 5 patches applied |
| Gemma 4 31B (2nd instance) | 8005 | Gaudi 3 | same | ✓ |
| MiniMax M2 (TP=4) | 8006 | Gaudis 4-7 | same | ✓ — preferred over M2.7 |
| MiniMax M2.7 (TP=4) | 8006 | Gaudis 4-7 | same | ✓ but slow cold-load, user prefers M2 |
| **gpt-oss-120b / 20b** | — | — | — | **✗ loads but incoherent — see [GPT-OSS.md](../GPT-OSS.md)** |

> Note: running state changes — verify with `docker ps` before quoting this table.

## The 5 things to know before doing anything

1. **The two vLLM images** —
   - `vllm-0.17.1-ptfork...` (stock Habana) for Qwen models
   - `gaudi-vllm-gemma4:0.19.0` (derived, 5 patches) for Gemma 4 + MiniMax + gpt-oss
   - Built from [`dockerfiles/Dockerfile.gemma4`](../../dockerfiles/Dockerfile.gemma4)

2. **`bin/vllm-launch <preset>` is how everything starts.** Presets are in
   the `PRESETS` array at line 20-39 of that file. Don't run `docker run`
   directly unless you're testing a one-off.

3. **`bin/launchpad`** is the interactive picker — card matrix, menu, custom-card
   override, smoke test, report. Use this for any non-trivial bring-up.

4. **`--gpu-memory-utilization 0.9`** pre-allocates 90% of HBM upfront.
   That's why hl-smi shows 90%+ used on active cards — it's not a leak.

5. **gpt-oss is broken** on this stack. Don't suggest it. The bug is in
   Habana's closed-source HPU MoE kernel, not patchable from user-space.
   Full analysis in [docs/GPT-OSS.md](../GPT-OSS.md).

## Where to go next, by question

| You're trying to … | Read |
|---|---|
| Understand the hardware/OS layer | [01-box.md](01-box.md) |
| Know what models exist and how to launch | [02-models.md](02-models.md) → also root [README.md](../../README.md) |
| Patch a new model | [03-patches-and-overrides.md](03-patches-and-overrides.md) → then [docs/GEMMA4.md](../GEMMA4.md) for a worked example |
| Calculate TP / HBM for a new model | [04-constraints.md](04-constraints.md) |
| Find a script or know what one does | [05-tools.md](05-tools.md) |
| Debug a symptom | [06-debugging-playbook.md](06-debugging-playbook.md) |
| Avoid a known trap | [07-gotchas.md](07-gotchas.md) |
| Know what's been tried before (and didn't work) | [08-history.md](08-history.md) |

## User preferences (saved in `~/.claude/memory/`)

These memories are auto-loaded by Claude every conversation — don't re-discover them:

- `user_identity.md` — Satyajit, Intel/Habana, the box specs
- `feedback_dashboard_style.md` — loves btop-style fancy TUIs (rounded frames,
  gradient bars). Match that aesthetic in any new tool.
- `feedback_dont_say_no_first.md` — when docs say "unsupported," enumerate
  paths A-D before saying no.
- `project_gaudi_setup.md` — canonical box state.
- `project_gemma4_patches.md` — the patch stack.
- `feedback_minimax_m2_preferred.md` — **default to MiniMax M2, not M2.7**,
  unless user explicitly asks for M2.7.

## Conventions

- **Don't run** `docker run` directly when a `vllm-launch` preset already covers
  the case. Add a preset instead.
- **Don't introduce** `--enforce-eager` casually — the older 0.17.1-ptupstream
  image needed it for a graph capture bug; the current 0.17.1-ptfork doesn't.
- **Don't** suggest gpt-oss as a working option.
- **Do** preserve the `VLLM_SKIP_WARMUP=true` setting — without it warmup is
  10-15 min per launch.
- **Do** use `--restart unless-stopped` for FP8 MoE models — defrag-OOM is real,
  auto-recover is the cheapest fix.

## Why this wiki exists

Every previous session of work on this box has rediscovered a few of the same
gotchas (the HBM "leak" that isn't a leak, the M2.7 silent 30-min CPU load,
the TP=6 invalid-divisor math). The wiki is a memory across sessions.

If you discover something new during your session and it generalizes, **add it
here** before the session ends — that's the deal.
