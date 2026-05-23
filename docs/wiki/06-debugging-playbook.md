# 06 — debugging playbook

Symptom → likely cause → fix. Organized so you can grep for the symptom you're
seeing.

## Container alive but HBM is 0 after many minutes

**Symptom:** `docker ps` shows the vllm container up for 10-30 min, but
`hl-smi` shows the target Gaudis at near-zero HBM. No new log lines after the
initial "Starting to load model" line. CPU is pinned at 99% on all worker
processes.

**Likely cause:** This is normal for **large-MoE models on cold-load**.
Workers are iterating safetensors shards from disk into CPU RAM and processing
the expert weights — Python-side loops, no log emissions, all silent.

- MiniMax M2 (128 experts): ~15 min silent phase before HBM takeoff
- MiniMax M2.7 (256 experts): **~30 min silent phase**, longer than M2
- 235B-A22B: ~15 min

**Verification it's making progress:**

```bash
docker stats --no-stream <container>     # watch BLOCK I/O column climb
```

If BLOCK I/O is increasing (e.g. 50 GB → 100 GB → 150 GB), workers ARE reading
shards. Wait it out — don't kill.

**Verification it's stuck:**

If BLOCK I/O is flat for >5 min after initially climbing, that's a stall. Then
check workers with `py-spy` if available, or kill and retry.

## Incoherent output from a model

**Symptom:** `/v1/models` works, requests return, but output is repetitive
loops or off-topic text. `content` is None for chat-completions on gpt-oss.

**Cause:** Specific to **gpt-oss family on this stack** — bug in the HPU MoE
kernel or MXFP4 dequantization. Documented in [docs/GPT-OSS.md](../GPT-OSS.md).

**Fix:** No fix possible from user-space. Use Gemma 4 (Anthropic shape) or
MiniMax M2 (parallel tools + reasoning) instead.

If this happens on a model OTHER than gpt-oss, that's new — investigate:
- Did you mess with parsers (`--tool-call-parser`, `--reasoning-parser`)?
- Are you using the wrong served-name in the request?
- Is sampling broken (e.g. temperature=0 with a model that needs >0)?

## "head_size 512 not supported"

**Cause:** Gemma 4's global-attention layers use head_dim=512. vLLM's HPU
attention backend doesn't allow-list 512 in the base image.

**Fix:** This is patch ② of the 5 Gemma 4 patches. Already applied in
`gaudi-vllm-gemma4:0.19.0`. If you see this error, you're using the wrong
image — make sure the preset's `image` field points at the derived image.

## "Failed to import from vllm._C"

**Cause:** Benign. `vllm._C` is the CUDA extension; HPU doesn't have it. This
warning fires on every HPU vLLM launch.

**Fix:** Ignore. Don't filter your error-monitor on `Failed to import` — too
broad, will spam.

## hl-smi shows 90-99% HBM used on idle endpoint

**Cause:** Not a leak. vLLM **pre-allocates** HBM at startup:
- `--gpu-memory-utilization 0.9` → driver claims 90% of HBM upfront
- `VLLM_GRAPH_RESERVED_MEM=0.3` → carves another slice from the rest

The reported number is the **pool reservation**, not actively-used bytes.

**Fix:** No action needed. If you want a smaller reservation:
- Lower `--gpu-memory-utilization` to 0.85 (costs you a few concurrent sequences)
- Restart the container weekly to reclaim accumulated fragmentation (we saw 35 GiB reclaimed on a 5-day-old container)

## defrag-OOM / "out of memory" mid-inference

**Symptom:** After hours/days of inference, requests start failing with
out-of-memory errors. `hl-smi` shows the same pool reservation, but
allocations fail.

**Cause:** Internal HBM fragmentation. vLLM's pool reservation can't be
coalesced because allocations from varied sequence shapes left fragmented
gaps.

**Fix (in order):**
1. Auto-recovery: `--restart unless-stopped` (set on all FP8 MoE presets) — the container restarts and reclaims.
2. Lower `--gpu-memory-utilization` to 0.85 — more slack for the allocator.
3. Larger recipe cache: `HABANA_PGM_LRU_MAX=60000`.
4. Manual restart: `vllm-launch stop <preset> && vllm-launch <preset>`.

## "Bad pagetable" / kernel oops

**Cause:** Forgot `iommu=pt intel_iommu=on` in GRUB. Habana driver maps device
memory in a way that requires these.

**Fix:** Edit `/etc/default/grub`, add the flags, `sudo update-grub`, reboot.
`install.sh` does this idempotently if you run it.

## Shard count mismatch (downloaded N, model says M)

**Symptom:** `ls models--*/snapshots/*/*.safetensors | wc -l` returns a number
lower than the "of-NNN" suffix in the filenames.

**Example:** MiniMax M2.7 — filenames say `of-00130` but only 125 shards exist.

**Cause:** Filename suffix is a holdover from training/planning. The
authoritative source is `model.safetensors.index.json` — check `weight_map`
for the actual referenced shards.

**Fix:** Don't try to "complete" the download by force-pulling the missing
shards. Verify with:

```bash
python3 -c "
import json, re
data = json.load(open('models--<ORG>--<MODEL>/snapshots/<HASH>/model.safetensors.index.json'))
shards = set(data['weight_map'].values())
indices = sorted({int(re.search(r'model-(\d+)-of', s).group(1)) for s in shards})
print(f'shards referenced: {len(shards)}, range: {indices[0]} → {indices[-1]}')
"
```

## TP=N invalid

**Symptom:** vLLM refuses to start: "num_key_value_heads (X) must be divisible by tensor_parallel_size (N)"

**Cause:** TP=N doesn't divide num_kv_heads (or num_attention_heads, or num_local_experts).

**Fix:** See [04-constraints.md](04-constraints.md) for the TP-validity check
one-liner. Pick a valid TP. For MiniMax that's {1, 2, 4, 8}.

## First inference is super slow

**Cause:** `VLLM_SKIP_WARMUP=true` is set on all presets — saves 5-15 min at
launch but the FIRST inference request pays the graph-capture cost (~30-60 s).

**Fix:** Either accept it, or remove `VLLM_SKIP_WARMUP=true` from the preset's
docker run block (you'll then pay the warmup at launch time, not at first
request).

## HBM doesn't release immediately after `docker rm -f`

**Symptom:** Stop a container, `hl-smi` still shows the HBM as used for ~10 seconds.

**Cause:** Driver releases lazily.

**Fix:** Wait 10-20 s. The next container's launch will reclaim the cards.

## Container restart "stuck" — won't come up after stop

**Symptom:** `vllm-launch stop <preset>` succeeds, but the new `vllm-launch <preset>` either hangs or fails to claim cards.

**Cause:** Usually one of:
- The cards are still held by the previous driver session (rare, wait 30s)
- The container name still exists (`docker ps -a` shows it) — `docker rm -f vllm-<preset>` to clear
- The HF cache is being modified by another process — check `docker ps` for any download containers

## Open WebUI doesn't show a new endpoint

**Cause:** Open WebUI persists model configs in its SQLite DB.

**Fix:** See [docs/PLAYBOOK.md § Open WebUI](../PLAYBOOK.md#open-webui) for the
SQLite patch to add new endpoints.

## Sudo with embedded password rejected

**Symptom:** `echo '<pw>' | sudo -S <cmd>` → "Permission denied by auto-mode classifier"

**Cause:** Claude Code's auto-mode classifier blocks pipes with embedded
credentials.

**Fix:** Either run the command via `! sudo <cmd>` prefix (interactive prompt
in the chat), or ask the user to run it themselves and paste output. **Never
embed the password literal in any command or doc.**
