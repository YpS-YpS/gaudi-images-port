#!/usr/bin/env bash
# smoke.sh — full API smoke test for a vllm-gaudi server.
# Verifies: text generation, vision, single tool call, parallel tool calls.
#
# Usage:
#   ./smoke.sh                                              # localhost:8000, qwen3-vl-32b-thinking
#   ./smoke.sh http://localhost:8001 qwen3-vl-32b-instruct  # different port + model
#   ./smoke.sh http://10.234.184.59:8000                    # over LAN
set -u

API="${1:-http://localhost:8000}"
MODEL="${2:-qwen3-vl-32b-thinking}"
MAXTOK="${3:-2000}"

C='\033[1;36m'; G='\033[0;32m'; R='\033[0;31m'; N='\033[0m'
hdr() { printf "\n${C}══ %s ══${N}\n" "$*"; }
ok()  { printf "${G}✓${N} %s\n" "$*"; }
bad() { printf "${R}✗${N} %s\n" "$*"; }

# Pretty-print a chat-completions response from stdin
PARSER='
import sys, json
try:
    r = json.load(sys.stdin)
except Exception as e:
    print(f"  parse error: {e}"); sys.exit(2)
if "error" in r:
    print("  ERROR:", r["error"]); sys.exit(1)
ch = r["choices"][0]
m = ch["message"]
fr = ch.get("finish_reason")
content = (m.get("content") or "").strip()
tcs = m.get("tool_calls") or []
u = r.get("usage") or {}
print("  finish_reason:", fr)
print("  tokens:", "prompt=" + str(u.get("prompt_tokens")), "completion=" + str(u.get("completion_tokens")), "total=" + str(u.get("total_tokens")))
if content:
    snippet = content if len(content) < 200 else "…" + content[-200:]
    print("  content:", snippet)
if tcs:
    print("  tool_calls:", len(tcs))
    for i, tc in enumerate(tcs):
        name = tc["function"]["name"]
        args = tc["function"]["arguments"]
        print("    [" + str(i) + "] " + name.ljust(22) + " args=" + args)
'

# Send body (with model + max_tokens injected) and pipe response through parser.
post() {
  local body=$1 max=${2:-$MAXTOK}
  body=$(printf '%s' "$body" | python3 -c "
import sys, json
b = json.load(sys.stdin)
b['model'] = '$MODEL'
b['max_tokens'] = $max
b.setdefault('temperature', 0)
print(json.dumps(b))
")
  local t0 t1
  t0=$(date +%s.%N)
  local resp
  resp=$(curl -s -m 300 -X POST "$API/v1/chat/completions" \
           -H 'Content-Type: application/json' -d "$body")
  t1=$(date +%s.%N)
  printf '%s' "$resp" | python3 -c "$PARSER"
  awk -v t0="$t0" -v t1="$t1" 'BEGIN{printf "  wall: %.2fs\n",t1-t0}'
}

# ── 1. /v1/models ──────────────────────────────────────────
hdr "1. /v1/models"
out=$(curl -fs -m 5 "$API/v1/models" 2>&1) || { bad "API not reachable at $API"; exit 1; }
served=$(echo "$out" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
ok "served: $served  (target: $MODEL)"

# ── 2. text ────────────────────────────────────────────────
hdr "2. plain text"
post '{"messages":[{"role":"user","content":"What is 17 * 89? One short sentence."}]}' 200

# ── 3. vision ──────────────────────────────────────────────
hdr "3. vision (inline base64 PNG: red square + blue circle)"
B64=$(python3 -c "
import base64, io
from PIL import Image, ImageDraw
img = Image.new('RGB', (200, 200), 'white')
d = ImageDraw.Draw(img)
d.rectangle([50,50,150,150], fill='red')
d.ellipse([60,60,140,140], fill='blue')
buf = io.BytesIO(); img.save(buf, 'PNG')
print(base64.b64encode(buf.getvalue()).decode(), end='')
" 2>/dev/null)
if [[ -z "$B64" ]]; then
  bad "PIL unavailable on host (skip vision)"
else
  vbody=$(python3 -c "
import json
b = {'messages':[{'role':'user','content':[
  {'type':'text','text':'What shapes and colors do you see?'},
  {'type':'image_url','image_url':{'url':'data:image/png;base64,$B64'}}
]}]}
print(json.dumps(b))
")
  post "$vbody" 400
fi

# ── 4. single tool call ────────────────────────────────────
hdr "4. single tool call (get_weather)"
post '{
  "messages":[{"role":"user","content":"What is the weather in Paris right now?"}],
  "tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
  "tool_choice":"auto"
}' 800

# ── 5. parallel — get_weather × 4 cities ───────────────────
hdr "5. parallel tool calls — get_weather × 4 cities"
post '{
  "messages":[{"role":"user","content":"What is the current weather in Paris, Tokyo, New York, and Mumbai? Call get_weather once for each city."}],
  "tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
  "tool_choice":"auto"
}' "$MAXTOK"

# ── 6. parallel — 4 different tools ────────────────────────
hdr "6. parallel tool calls — 4 different tools (trip planner)"
post '{
  "messages":[{"role":"user","content":"Plan a Paris trip next Friday: get the weather, find a flight from JFK to CDG, look up the EUR-USD exchange rate, and book a 2-night hotel near the Eiffel Tower. Call all needed tools in one go."}],
  "tools":[
    {"type":"function","function":{"name":"get_weather","description":"Current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},
    {"type":"function","function":{"name":"search_flights","description":"Find flights between two airports on a date","parameters":{"type":"object","properties":{"origin":{"type":"string"},"destination":{"type":"string"},"date":{"type":"string"}},"required":["origin","destination","date"]}}},
    {"type":"function","function":{"name":"get_exchange_rate","description":"Currency exchange rate","parameters":{"type":"object","properties":{"from_currency":{"type":"string"},"to_currency":{"type":"string"}},"required":["from_currency","to_currency"]}}},
    {"type":"function","function":{"name":"book_hotel","description":"Book a hotel","parameters":{"type":"object","properties":{"city":{"type":"string"},"area":{"type":"string"},"nights":{"type":"integer"}},"required":["city","area","nights"]}}}
  ],
  "tool_choice":"auto"
}' 4000   # 4× more headroom — mixed tools + Thinking variant ≈ long <think> + 4 emissions

hdr "Done"
