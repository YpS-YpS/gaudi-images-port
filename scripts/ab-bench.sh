#!/usr/bin/env bash
# ab-bench.sh — head-to-head perf bench between two OpenAI-compat endpoints.
#
# Designed for the SGLang vs vLLM Qwen3.6-27B A/B test, but generic over any two
# endpoints serving the same model.
#
# Usage:
#   ./ab-bench.sh                                                # SGLang vs vLLM, default Qwen3.6-27B endpoints
#   ./ab-bench.sh "your prompt" 1000                             # custom prompt + max_tokens
#   ./ab-bench.sh "..." 1000 http://localhost:30000  http://localhost:8007  qwen3.6-27b
set -u

PROMPT="${1:-Explain in 100 words: why does prefix caching matter for chat workloads?}"
MAXTOK="${2:-500}"
A_URL="${3:-http://localhost:30000}"
B_URL="${4:-http://localhost:8007}"
MODEL="${5:-qwen3.6-27b}"

A_LABEL="SGLang (${A_URL##*/}, Gaudi 0)"
B_LABEL="vLLM   (${B_URL##*/}, Gaudi 1)"

C='\033[1;36m'; G='\033[0;32m'; R='\033[0;31m'; B='\033[1;34m'; D='\033[0;37m'; N='\033[0m'

OUTDIR=$(mktemp -d)
trap 'rm -rf "$OUTDIR"' EXIT

fire_one() {
  local label=$1 url=$2 outfile=$3 streaming=$4
  local body
  if [[ "$streaming" == "stream" ]]; then
    body=$(jq -n --arg m "$MODEL" --arg p "$PROMPT" --argjson mt "$MAXTOK" \
      '{model:$m, max_tokens:$mt, stream:true, temperature:0, messages:[{role:"user",content:$p}]}')
  else
    body=$(jq -n --arg m "$MODEL" --arg p "$PROMPT" --argjson mt "$MAXTOK" \
      '{model:$m, max_tokens:$mt, stream:false, temperature:0, messages:[{role:"user",content:$p}]}')
  fi
  local t0 t_first t_done
  t0=$(date +%s.%N)
  if [[ "$streaming" == "stream" ]]; then
    # Streaming run: capture every chunk + time of first chunk
    curl -sN -X POST "$url/v1/chat/completions" \
      -H 'Content-Type: application/json' -d "$body" \
      --max-time 600 \
      | while IFS= read -r line; do
          [[ -z "$line" ]] && continue
          if [[ -z "${SAW_FIRST:-}" ]]; then
            SAW_FIRST=1
            t_first=$(date +%s.%N)
            echo "FIRST $t_first" >>"$outfile.timing"
          fi
          echo "$line" >>"$outfile.stream"
        done
    t_done=$(date +%s.%N)
    echo "DONE $t_done" >>"$outfile.timing"
  else
    # Non-streaming: just total wall time + json body
    curl -s -X POST "$url/v1/chat/completions" \
      -H 'Content-Type: application/json' -d "$body" \
      --max-time 600 >"$outfile.json"
    t_done=$(date +%s.%N)
    echo "DONE $t_done" >>"$outfile.timing"
  fi
  echo "T0 $t0" >>"$outfile.timing"
}

run_round() {
  local mode=$1
  echo -e "\n${C}══════════ ROUND: $mode ══════════${N}"
  : >"$OUTDIR/A.timing"; : >"$OUTDIR/A.stream"; : >"$OUTDIR/A.json"
  : >"$OUTDIR/B.timing"; : >"$OUTDIR/B.stream"; : >"$OUTDIR/B.json"
  fire_one "A" "$A_URL" "$OUTDIR/A" "$mode" &
  PID_A=$!
  fire_one "B" "$B_URL" "$OUTDIR/B" "$mode" &
  PID_B=$!
  wait $PID_A; wait $PID_B
  for who in A:"$A_LABEL" B:"$B_LABEL"; do
    side=${who%%:*}; label=${who#*:}
    local t0 tfirst tdone
    t0=$(awk '/^T0/ {print $2}' "$OUTDIR/$side.timing")
    tfirst=$(awk '/^FIRST/ {print $2}' "$OUTDIR/$side.timing")
    tdone=$(awk '/^DONE/ {print $2}' "$OUTDIR/$side.timing")
    if [[ "$mode" == "stream" ]]; then
      ttft=$(awk -v a="$t0" -v b="$tfirst" 'BEGIN{printf "%.3f", b-a}')
      tot=$(awk -v a="$t0" -v b="$tdone" 'BEGIN{printf "%.3f", b-a}')
      # Parse tokens from stream — sglang and vllm both emit OpenAI-compat data: chunks
      ntok=$(grep -c '"delta"' "$OUTDIR/$side.stream" 2>/dev/null || echo 0)
      ttok_per_s=$(awk -v t="$tot" -v n="$ntok" -v f="$ttft" 'BEGIN{
        d=t-f; if (d>0) printf "%.1f", n/d; else printf "—"}')
      printf "  ${B}%s${N}\n    TTFT %ss   total %ss   stream-tok≈%s   decode tok/s≈%s\n" \
        "$label" "$ttft" "$tot" "$ntok" "$ttok_per_s"
    else
      tot=$(awk -v a="$t0" -v b="$tdone" 'BEGIN{printf "%.3f", b-a}')
      # Parse usage from non-stream json
      python3 - "$OUTDIR/$side.json" "$label" "$tot" <<'PY'
import sys, json
try: r=json.load(open(sys.argv[1]))
except: print(f"    parse error"); sys.exit()
u=r.get("usage") or {}
ch=(r.get("choices") or [{}])[0]
fr=ch.get("finish_reason")
ct=(ch.get("message") or {}).get("content") or ""
ntok=u.get("completion_tokens") or 0
ptok=u.get("prompt_tokens") or 0
tot=float(sys.argv[3])
tps=(ntok/tot) if tot>0 else 0
print(f"  \033[1;34m{sys.argv[2]}\033[0m")
print(f"    total {tot}s   prompt-tok={ptok}   completion-tok={ntok}   end-to-end tok/s≈{tps:.1f}   finish={fr}")
print(f"    answer: {ct[:160]}{'…' if len(ct)>160 else ''}")
PY
    fi
  done
}

echo -e "${C}PROMPT:${N} $PROMPT"
echo -e "${C}MAX TOKENS:${N} $MAXTOK"
echo -e "${C}A:${N} $A_LABEL  ${C}B:${N} $B_LABEL"

run_round "nostream"
run_round "stream"

echo ""
echo -e "${G}done${N} — raw artifacts in $OUTDIR (gone on exit)"
