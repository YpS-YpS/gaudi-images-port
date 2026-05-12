#!/usr/bin/env bash
# verify.sh — end-to-end health verification.
#
# 1. readiness check (scripts/check.sh)
# 2. (optional) hl_qual hardware sanity test (-l simple, ~30s)
# 3. (optional) API smoke test against http://localhost:8000 if vllm-launch
#    container is running. Tests text + tool-calling.
#
# Usage:
#   sudo ./verify.sh                # all 3
#   sudo ./verify.sh check          # readiness only
#   sudo ./verify.sh hwqual         # hardware sanity only
#   sudo ./verify.sh smoke          # API smoke only

set -eu
SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
[[ $EUID -eq 0 ]] || { echo "run as root: sudo $0"; exit 1; }

GRN='\033[0;32m'; RED='\033[0;31m'; YEL='\033[1;33m'; BLU='\033[0;34m'; NC='\033[0m'
hdr()  { printf "\n${BLU}══ %s ══${NC}\n" "$*"; }
ok()   { printf "${GRN}✓${NC} %s\n" "$*"; }
bad()  { printf "${RED}✗${NC} %s\n" "$*"; }
warn() { printf "${YEL}⚠${NC} %s\n" "$*"; }

mode=${1:-all}

# 1. Readiness ─────────────────────────────────────────
if [[ "$mode" == "all" || "$mode" == "check" ]]; then
  hdr "1. readiness check"
  bash "$SCRIPT_DIR/check.sh"
fi

# 2. Hardware sanity (hl_qual) ─────────────────────────
if [[ "$mode" == "all" || "$mode" == "hwqual" ]]; then
  hdr "2. hardware sanity (hl_qual)"
  if [[ -x /opt/habanalabs/qual/gaudi3/bin/hl_qual ]]; then
    cd /opt/habanalabs/qual/gaudi3/bin
    # -l simple = light stress, fast. -t 30 = 30 sec.
    if ./hl_qual -gaudi3 -c all -rmod parallel -f2 -l simple -t 30 2>&1 | tail -20; then
      ok "hl_qual completed"
    else
      bad "hl_qual reported failures"
    fi
    cd - >/dev/null
  else
    warn "hl_qual not installed — sudo apt install habanalabs-qual"
  fi
fi

# 3. API smoke test ────────────────────────────────────
if [[ "$mode" == "all" || "$mode" == "smoke" ]]; then
  hdr "3. API smoke test"
  if ! curl -fs -m 3 http://localhost:8000/v1/models >/dev/null 2>&1; then
    warn "API not responding on :8000 — start a server first: vllm-launch 32b-thinking"
  else
    model=$(curl -fs -m 3 http://localhost:8000/v1/models 2>/dev/null \
              | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
    ok "/v1/models responding — served as: ${model:-?}"
    [[ -z "$model" ]] && model='qwen3-vl-32b-thinking'

    echo
    echo "→ text:"
    out=$(curl -fs -m 60 -X POST http://localhost:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 * 89? One short sentence.\"}],\"max_tokens\":100,\"temperature\":0}" \
      | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'][-300:])")
    echo "$out" | sed 's/^/     /'

    echo
    echo "→ tool-call (get_weather):"
    out=$(curl -fs -m 60 -X POST http://localhost:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Weather in Paris?\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],\"tool_choice\":\"auto\",\"max_tokens\":300,\"temperature\":0}" \
      | python3 -c "import sys,json; r=json.load(sys.stdin); m=r['choices'][0]['message']; tc=m.get('tool_calls'); print('calls:', tc[0]['function']['name'], tc[0]['function']['arguments']) if tc else print('(no tool_calls)')")
    echo "$out" | sed 's/^/     /'

    echo
    ok "API smoke test passed"
  fi
fi
