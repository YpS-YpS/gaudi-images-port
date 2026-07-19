# 07 — gotchas

Non-obvious surprises that have cost time. Each one explained so you don't
re-discover them.

## The `of-00130` shard filename lies

`MiniMaxAI/MiniMax-M2.7` ships safetensors named
`model-00000-of-00130.safetensors` through `model-00124-of-00130.safetensors`
— **but only 125 shards exist**. The denominator "00130" is a holdover from
training-time planning, not the actual count.

If `huggingface-cli download` exits with 125 files and you trust the
filename, you'll waste time hunting for the "missing" 5 shards (00125-00129).
They don't exist. **The authoritative shard count is `model.safetensors.index.json`'s
`weight_map`.**

## hl-smi reports pool reservation, not active bytes

When you see `Gaudi 2: 130 GiB used / 1 GiB free`, that does NOT mean the
container is using 130 GiB right now. vLLM pre-allocates the pool based on
`--gpu-memory-utilization 0.9` (+ graph reserve). The actual working set at
idle is much smaller.

Don't panic at the "1 GiB free" number. The driver manages the pool
internally — it doesn't ask for more from the OS.

## MiniMax HpuMiniMaxM2ForCausalLM override warning is expected

```
WARNING [registry.py:915] Model architecture MiniMaxM2ForCausalLM is already
registered, and will be overwritten by the new model class
vllm_gaudi.models.minimax_m2:HpuMiniMaxM2ForCausalLM.
```

This fires on every MiniMax launch. It means the HPU-optimized class is
active. **Don't try to silence it.** It's the signal that the HPU path is wired
up correctly.

## VLLM_SKIP_WARMUP=true means first request is slow

Saves ~10 min on launch, but the first inference request after startup pays
the graph-capture cost (~30-60 s). If you smoke-test immediately and it looks
slow, that's expected. Subsequent requests are fast.

If you need consistent first-token latency in production, remove
`VLLM_SKIP_WARMUP=true` from the preset and let warmup run at launch instead.

## "Application startup complete" ≠ "ready for fast inference"

vLLM logs `Application startup complete` and `Uvicorn running on http://...`
as soon as the FastAPI server can accept connections. But the FIRST inference
request still needs graph capture (~30-60 s without warmup).

Smoke tests right after startup-complete will show the first-request overhead.
Run twice; the second one is the real latency.

## MiniMax M2.7 cold-load is 30 min of silence

Don't kill the container during the first 30 min. CPU pinned at 99% on every
worker, HBM at 0, no log lines after "Starting to load model" — looks frozen,
isn't. The 256-expert × 62-layer × TP=4 weight processing is Python-side and
silent.

Verify progress with `docker stats <container>` — watch the BLOCK I/O column
climb. When it stops climbing AND HBM is still 0, then something is wrong.

## HBM doesn't release immediately on container stop

`docker rm -f` returns instantly, but `hl-smi` still shows the previous
allocation for ~10 seconds. The driver releases lazily.

If the next `docker run` is too fast, it can race the lazy release and fail
to claim cards. Sleep 10-15 seconds between stop and start, OR rely on the
fact that `vllm-launch` does `docker rm -f` followed by `docker run` — the
race is rare.

## `--enforce-eager` is NOT needed on vllm-0.17.1-ptfork

Older `vllm-0.17.1-ptupstream` had a graph capture bug that required
`--enforce-eager`. The current `ptfork` image fixes it. Don't add
`--enforce-eager` unless you have a specific reason — it slows down inference
significantly.

(History: this was part of why we switched to ptfork in the first place. See
[08-history.md](08-history.md).)

## `PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false` is REQUIRED for Qwen

Without it, the recipe cache gets poisoned on Qwen graph capture, causing
`ValidateSyncInputTensors` crashes. This was a hard-fought fix. It's baked
into every preset's docker run — don't remove it.

## `NO_PROXY` must include localhost

Intel proxy is set for HF downloads. Without `NO_PROXY` including
`localhost,127.0.0.1`, local API calls (curl to vLLM endpoints) get routed
through the corp proxy and time out. Baked into every preset.

## Sudo with embedded password blocked

```bash
echo '<password>' | sudo -S <cmd>    # auto-mode classifier rejects this
```

Use `! sudo <cmd>` prefix (interactive) instead, or ask the user to run.
**Never embed the password literal in any command, doc, or comment.**

## Auto-mode blocks running just-written scripts

If Claude writes a new script and tries to execute it, the auto-mode
classifier may block on grounds that the content is unverified. Workaround:
`bash -n script.sh` to syntax-check, then ask the user to be the first to run.

## `metadata.total_size` in `safetensors.index.json` is misleading

For FP8 models, this field reports the **dequantized BF16 size**, not the
on-disk size. For MiniMax M2.7 it says ~480 GB even though the actual disk
footprint is ~215 GiB FP8. Don't size your HBM budget off this number — use
the actual `du -sh` of the snapshot directory.

## vLLM accepts 0.9 GPU utilization but errors on 0.95+

The HPU graph reserve carves an additional 30% of *free* (not total). At GPU
util 0.95, free is too small for graph reserve, and launch fails. Stick to
0.9 or 0.85.

## Two Gemma 4 instances on different cards = same container image, same patches

The `gemma4-31b` preset uses Gaudi 2, port 8004. The `gemma4-31b-b` preset (if
you add it) would use Gaudi 3, port 8005 — same image, same patches, just a
different `HABANA_VISIBLE_DEVICES` and `--port`. The HF cache is shared so
weights are loaded only once.

## The auto-tightened MiniMax M2.7 knobs

```bash
--gpu-memory-utilization 0.85    # vs M2's 0.9
--max-num-seqs 16                # vs M2's 32
HABANA_PGM_LRU_MAX=60000         # extra recipe cache
```

These are applied automatically when the preset is `minimax-m2.7` (see
`bin/vllm-launch` line 100-110). The reason: M2.7's 256 experts + 3 MTP
modules push per-card HBM closer to 128 GB; tighter knobs leave more defrag
headroom. Don't apply these to M2 — it doesn't need them and you'd waste
capacity.

## User prefers M2 over M2.7

Tested 2026-05-13 with the L1-Dep prompt. M2.7 invented facts ("Dep is common
shorthand for Data" — wrong; Dep = Dependency). M2 didn't have this test on
record. Saved as a memory: `feedback_minimax_m2_preferred.md`.

**Default to MiniMax M2 unless the user asks for M2.7 specifically.**

## Dockerfile.gemma4 unpinned deps pull CUDA torch (version drift)

The Gemma 4 image's second `pip install` upgrades `accelerate>=1.10.0`,
`compressed-tensors>=0.12.0`, etc. with no version ceiling. The image built
fine in May 2026, but by **June 2026** those `>=` ranges resolved to newer
releases (`accelerate 1.14.0`) whose dependency tree pulled **upstream CUDA
`torch 2.12.1+cu130`**, replacing the Habana HPU torch fork. The build then
dies at the first patch:

```
AssertionError: Current PyTorch version 2.12.1+cu130 is detected as
neither HPU fork nor CPU upstream.
... RuntimeError: Failed to load the backend extension: device_backend
```

(the `import vllm_gaudi` in patch 1's verification triggers `habana_frameworks.torch`).

**Fix (committed):** `dockerfiles/torch-constraints.txt` pins the HPU torch +
CPU torchvision/torchaudio, and `Dockerfile.gemma4` passes
`-c /tmp/torch-constraints.txt` to the deps-resolving install. pip then keeps
the HPU torch and back-tracks to a torch-2.10-compatible `accelerate`.
After a base-image bump, refresh the pins:
`docker run --rm --entrypoint pip <base> show torch torchvision torchaudio`.

## Host `/etc/environment` proxy breaks localhost curl AND smoke.sh

Separate from the in-container `NO_PROXY` note above: on some boxes the
**host** has `http_proxy`/`https_proxy` exported globally in
`/etc/environment`. Every host-side `curl http://localhost:8004/...` (incl.
`scripts/smoke.sh`, which doesn't pass `--noproxy`) then routes through the
Intel proxy and fails — the server is fine, the probe is lying.

Tells: `docker logs` shows `Application startup complete`, the container is
`running`, but `curl localhost:<port>/v1/models` returns nothing. Confirm with
`env | grep -i proxy`.

**Fix:** run host probes with the proxy cleared, e.g.
`curl --noproxy '*' http://localhost:8004/v1/models`, or for smoke:
`env http_proxy= https_proxy= no_proxy='*' bash scripts/smoke.sh http://localhost:8004 gemma4-31b 4000`.

## TP must divide num_kv_heads

The most common "but the math says 4 cards is fine" failure. For MiniMax M2:
- num_kv_heads = 8 → valid TP ∈ {1, 2, 4, 8}
- TP=3, 5, 6, 7 all fail at load time

This is in [04-constraints.md](04-constraints.md), but it's worth re-stating
because every TP-choice debate runs into this.
