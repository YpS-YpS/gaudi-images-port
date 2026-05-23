# Wiki — the Gaudi 3 vLLM serving box

A knowledge base for this specific box: what's running, what's been tried, what
works, what doesn't, the math behind the constraints, and the load-bearing
hacks. Designed to be the **single onboarding entry point** for any future
session (human or LLM) picking up this work.

If you're new to this box, **read [START-HERE.md](START-HERE.md) first.**

## Tree

```
docs/wiki/
├── README.md                      ← you are here (the index)
├── START-HERE.md                  ← 5-minute orientation; read this first
├── 01-box.md                      ← hardware, OS, driver, persistence layer
├── 02-models.md                   ← what's served, what each preset does
├── 03-patches-and-overrides.md    ← Gemma 4 patches, MiniMax HPU class, env overrides
├── 04-constraints.md              ← TP math, HBM math, conflict pairs
├── 05-tools.md                    ← every script in bin/ and scripts/ — what each does
├── 06-debugging-playbook.md       ← symptom → cause → fix
├── 07-gotchas.md                  ← non-obvious surprises that cost time
├── 08-history.md                  ← chronological "what we tried, what we learned"
└── 09-future-roadmap.md           ← open-weight agentic leaderboard + 5-box fleet plan
```

## How this fits with the rest of the docs

| Layer | What it contains | When to read |
|---|---|---|
| **wiki/** (here) | Cross-cutting knowledge, debugging playbook, history | Onboarding a new session |
| **docs/GEMMA4.md** | Deep-dive: the 5 Gemma 4 patches with code | When patching anything Gemma-shaped |
| **docs/MINIMAX.md** | Deep-dive: MiniMax M2 + M2.7 full walkthrough | When touching MiniMax |
| **docs/GPT-OSS.md** | Deep-dive: gpt-oss failure analysis + upstream report | If anyone asks "why not gpt-oss" |
| **docs/MODELS.md** | Per-preset reference table + perf notes | Quick lookup of what runs where |
| **docs/index.html** | Browser-friendly dashboard | When sharing with non-developers |
| **README.md** (root) | Launch cheat sheet | Day-to-day launching |
| **~/.claude/memory/** | Compact summaries Claude auto-loads | Background context |

The wiki here adds **the connective tissue** — the things that span multiple docs, the lessons learned, and the things that don't fit cleanly in any of the existing files.
