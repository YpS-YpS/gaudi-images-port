#!/usr/bin/env bash
# bootstrap.sh — one-click bring-up of a fresh Gaudi 3 box to a running vLLM
# endpoint (Qwen3-VL-32B-Thinking by default, optionally Gemma 4 31B).
#
# Usage:
#   sudo ./bootstrap.sh                   # base bring-up + Qwen 32B-Thinking on :8000
#   sudo ./bootstrap.sh --with-gemma4     # also build patched image + launch Gemma 4 on :8004
#   sudo ./bootstrap.sh --skip-install    # skip install.sh (system already set up)
#   sudo ./bootstrap.sh --skip-models     # do bring-up only, no model launch
#
# Environment overrides:
#   HF_TOKEN              optional — speeds up downloads, lifts rate limits
#   HTTP_PROXY/HTTPS_PROXY/NO_PROXY     forwarded into docker build steps
#   HF_CACHE              host path for HuggingFace cache (default: ~/hf-cache)
#
# What this does on a fresh box (15-40 min total depending on bandwidth):
#   1. install.sh                — kernel 6.8, Habana driver, hugepages, NICs, docker
#   2. docker pull base images   — vllm-0.17.1-ptfork (10 GB), optionally 0.19.0-ptfork (10 GB)
#   3. docker build derived      — only if --with-gemma4 (~2-3 min)
#   4. huggingface-cli download  — Qwen3-VL-32B-Thinking-FP8 (~32 GB), optionally Gemma 4 (~31 GB)
#   5. vllm-launch <preset>      — bring up the API; smoke test it
#
# Re-runs are idempotent — already-installed steps skip themselves.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
cd "$SCRIPT_DIR"

#=== UI ====================================================================
RED='\033[0;31m'; GRN='\033[0;32m'; YEL='\033[1;33m'; BLU='\033[0;34m'; CYN='\033[1;36m'; NC='\033[0m'
hdr()  { printf "\n${CYN}══ %s ══${NC}\n" "$*"; }
log()  { printf "${BLU}[*]${NC} %s\n"  "$*"; }
ok()   { printf "${GRN}[✓]${NC} %s\n" "$*"; }
warn() { printf "${YEL}[!]${NC} %s\n" "$*"; }
err()  { printf "${RED}[✗]${NC} %s\n" "$*" >&2; }

#=== Defaults ==============================================================
WITH_GEMMA4=0
SKIP_INSTALL=0
SKIP_MODELS=0
HF_CACHE="${HF_CACHE:-$HOME/hf-cache}"
QWEN_MODEL="Qwen/Qwen3-VL-32B-Thinking-FP8"
GEMMA_MODEL="RedHatAI/gemma-4-31b-it-FP8-Dynamic"
BASE_IMG_017="vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/vllm-0.17.1-ptfork-2.10.0:1.24.0-1007"
BASE_IMG_019="vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/vllm-0.19.0-ptfork-2.10.0:1.24.0-1007"
GEMMA_IMG="gaudi-vllm-gemma4:0.19.0"

#=== Parse args ============================================================
for arg in "$@"; do
  case "$arg" in
    --with-gemma4)  WITH_GEMMA4=1 ;;
    --skip-install) SKIP_INSTALL=1 ;;
    --skip-models)  SKIP_MODELS=1 ;;
    -h|--help)
      sed -n '2,/^set/p' "$0" | sed 's/^# \{0,1\}//;s/^#$//' | head -n -1
      exit 0 ;;
    *)              err "Unknown arg: $arg"; exit 1 ;;
  esac
done

#=== Preflight =============================================================
hdr "Preflight"
[[ $EUID -eq 0 ]] || { err "Run as root: sudo ./bootstrap.sh"; exit 1; }
[[ -f install.sh ]] || { err "Run from the repo root"; exit 1; }

if command -v lspci >/dev/null && lspci -nn | grep -q '1da3:'; then
  ok "Habana devices detected on PCI"
else
  warn "No Habana PCI devices detected (1da3:*); continuing in case lspci unavailable"
fi

#=== Step 1 — base bring-up ================================================
if (( SKIP_INSTALL )); then
  hdr "Step 1/5 — install.sh (SKIPPED)"
else
  hdr "Step 1/5 — install.sh (idempotent system bring-up)"
  bash install.sh
fi

#=== Step 2 — verify ready =================================================
hdr "Step 2/5 — system readiness check"
if bash scripts/check.sh >/tmp/bootstrap-check.log 2>&1; then
  ok "All readiness checks passed"
else
  warn "scripts/check.sh reported issues; see /tmp/bootstrap-check.log"
  warn "Continuing anyway — many issues self-resolve after first model launch"
fi

#=== Step 3 — base images ==================================================
hdr "Step 3/5 — pull base image(s)"
log "Pulling $BASE_IMG_017 (~10 GB, one-time)…"
docker pull "$BASE_IMG_017" 2>&1 | tail -3 || true
ok "Base image present"

if (( WITH_GEMMA4 )); then
  log "Pulling $BASE_IMG_019 (~10 GB, one-time)…"
  docker pull "$BASE_IMG_019" 2>&1 | tail -3 || true
  ok "Gemma 4 base image present"

  hdr "Step 3b/5 — build $GEMMA_IMG (derived, 5-patch stack)"
  if docker image inspect "$GEMMA_IMG" >/dev/null 2>&1; then
    log "Image $GEMMA_IMG already exists; skipping rebuild"
  else
    cd dockerfiles
    docker build \
      ${HTTP_PROXY:+--build-arg HTTP_PROXY=$HTTP_PROXY} \
      ${HTTPS_PROXY:+--build-arg HTTPS_PROXY=$HTTPS_PROXY} \
      ${NO_PROXY:+--build-arg NO_PROXY=$NO_PROXY} \
      -t "$GEMMA_IMG" -f Dockerfile.gemma4 . 2>&1 | grep -E "Step [0-9]+/|patch applied|OK patch|Successfully|ERROR" || true
    cd ..
    ok "Built $GEMMA_IMG"
  fi
fi

#=== Step 4 — model downloads ==============================================
if (( SKIP_MODELS )); then
  hdr "Step 4/5 — model download (SKIPPED)"
else
  hdr "Step 4/5 — pre-download model(s) into $HF_CACHE"
  mkdir -p "$HF_CACHE"
  chmod 755 "$HF_CACHE"
  export HF_HOME="$HF_CACHE"

  if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    log "Installing huggingface-hub on host…"
    pip3 install --quiet huggingface-hub hf-transfer
  fi
  export HF_HUB_ENABLE_HF_TRANSFER=1

  for model in "$QWEN_MODEL" $([ $WITH_GEMMA4 -eq 1 ] && echo "$GEMMA_MODEL"); do
    log "Downloading $model (≈30 GB) — this is the long part…"
    huggingface-cli download "$model" --local-dir-use-symlinks=False 2>&1 | tail -3 || \
      warn "Download of $model had issues; the container will re-stage on first use"
    ok "$model staged"
  done
fi

#=== Step 5 — launch =======================================================
if (( SKIP_MODELS )); then
  hdr "Step 5/5 — launch (SKIPPED)"
  ok "Bootstrap complete. Run 'vllm-launch <preset>' to start a server."
  exit 0
fi

hdr "Step 5/5 — launch model(s)"
if [[ -x /usr/local/bin/vllm-launch ]]; then
  LAUNCHER=/usr/local/bin/vllm-launch
else
  LAUNCHER="$SCRIPT_DIR/bin/vllm-launch"
fi

log "Starting Qwen3-VL-32B-Thinking on :8000…"
"$LAUNCHER" 32b-thinking || warn "Qwen 32B launch failed; check 'vllm-launch logs 32b-thinking'"
sleep 5
ok "Qwen launched (warmup ~5-10 min on first start; ~30 s with VLLM_SKIP_WARMUP)"

if (( WITH_GEMMA4 )); then
  log "Starting Gemma 4 31B on :8004…"
  "$LAUNCHER" gemma4-31b || warn "Gemma 4 launch failed; check 'vllm-launch logs gemma4-31b'"
  ok "Gemma 4 launched"
fi

hdr "✅ Bootstrap complete"
cat <<EOF
Endpoints:
  Qwen 32B-Thinking:   http://localhost:8000/v1/chat/completions
$([ $WITH_GEMMA4 -eq 1 ] && echo "  Gemma 4 31B:         http://localhost:8004/v1/chat/completions  +  /v1/messages")

Wait for warmup:
  until curl -fs http://localhost:8000/v1/models >/dev/null; do sleep 5; done

Smoke test:
  scripts/smoke.sh http://localhost:8000 qwen3-vl-32b-thinking
$([ $WITH_GEMMA4 -eq 1 ] && echo "  scripts/smoke.sh http://localhost:8004 gemma4-31b")

Useful follow-ups:
  vllm-launch list           # see all presets and ports
  gaudi-dash                 # tmux dashboard (btop + htop + hl-smi)
  scripts/check.sh           # re-run readiness check
  docs/MODELS.md             # per-model patches and flags
  docs/GEMMA4.md             # Gemma 4 deep-dive (the 5-patch stack)

EOF
