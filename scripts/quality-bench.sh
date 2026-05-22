#!/usr/bin/env bash
# quality-bench.sh — cross-endpoint domain-knowledge benchmark.
#
# Fires the same question at every running vLLM endpoint on this box and prints
# the responses side-by-side so you can compare quality. Designed for "subjective
# good answer" tests like the user's L1-Dep-and-gaming question, not for
# automated pass/fail scoring.
#
# Usage:
#   ./quality-bench.sh                                            # default L1-Dep prompt
#   ./quality-bench.sh "explain prefetching"                       # custom prompt
#   ./quality-bench.sh "what is L1 Dep..." 1500                    # custom prompt + max_tokens
set -u

PROMPT="${1:-What is L1 Dep in microarchitecture and how does it affect gaming performance?}"
MAXTOK="${2:-1200}"

C='\033[1;36m'; G='\033[0;32m'; B='\033[1;34m'; D='\033[0;37m'; N='\033[0m'

# Endpoints to test (host:port  served-name)
declare -a ENDPOINTS=(
  "localhost:8000  qwen3-vl-32b-thinking   Qwen3-VL-32B-Thinking"
  "localhost:8001  qwen3-vl-32b-instruct   Qwen3-VL-32B-Instruct"
  "localhost:8004  gemma4-31b              Gemma-4-31B-Instruct"
  "localhost:8006  minimax-m2              MiniMax-M2"
)

OUTDIR=$(mktemp -d)
trap 'rm -rf "$OUTDIR"' EXIT

# Fire all endpoints in parallel
PIDS=()
for ep in "${ENDPOINTS[@]}"; do
  read -r HOSTPORT MODEL LABEL <<<"$ep"
  (
    START=$(date +%s.%N)
    RESP=$(curl -sX POST "http://${HOSTPORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      --max-time 600 \
      -d "$(jq -n --arg m "$MODEL" --arg p "$PROMPT" --argjson mt "$MAXTOK" \
        '{model:$m, max_tokens:$mt, messages:[{role:"user",content:$p}]}')")
    END=$(date +%s.%N)
    ELAPSED=$(awk "BEGIN {printf \"%.1f\", $END - $START}")
    echo "$ELAPSED" >"$OUTDIR/$LABEL.time"
    echo "$RESP"    >"$OUTDIR/$LABEL.json"
  ) &
  PIDS+=($!)
done

# Show a small spinner while waiting
echo -ne "${B}firing 4 endpoints in parallel${N} "
for p in "${PIDS[@]}"; do wait "$p"; echo -ne "${G}.${N}"; done
echo ""

# Render results
echo ""
echo -e "${C}══════════ PROMPT ══════════${N}"
echo "$PROMPT"
echo ""

for ep in "${ENDPOINTS[@]}"; do
  read -r HOSTPORT MODEL LABEL <<<"$ep"
  ELAPSED=$(cat "$OUTDIR/$LABEL.time")
  echo -e "${C}══════════ $LABEL  (${HOSTPORT}, ${ELAPSED}s) ══════════${N}"
  python3 - "$OUTDIR/$LABEL.json" <<'PY'
import sys, json, textwrap
data = json.load(open(sys.argv[1]))
if "error" in data:
    print("  ERROR:", data["error"]); sys.exit()
c = data["choices"][0]
m = c["message"]
fr = c.get("finish_reason")
content = (m.get("content") or "").strip()
reasoning = (m.get("reasoning_content") or "").strip()
u = data.get("usage") or {}
print(f"  finish={fr}  tokens(p/c/t)={u.get('prompt_tokens')}/{u.get('completion_tokens')}/{u.get('total_tokens')}")
if reasoning:
    print()
    print("  --- reasoning ---")
    print(textwrap.fill(reasoning, 100, initial_indent="  ", subsequent_indent="  "))
if content:
    print()
    print("  --- answer ---")
    print(textwrap.fill(content, 100, initial_indent="  ", subsequent_indent="  "))
PY
  echo ""
done

echo -e "${B}done${N} — raw JSON in $OUTDIR (gone on exit)"
