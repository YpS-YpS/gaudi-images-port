# API_ACCESS — how to talk to this box from anywhere

The Dell PowerEdge XE9680 with 8× Gaudi 3 has multiple OpenAI-compatible
endpoints exposed on the LAN, plus an Open WebUI front-end. This file is the
single reference for **how to call them from your other machines**.

## Host network identity

| | |
|---|---|
| LAN IP | `XXXXXXXXXXXX` (eno8303) |
| Hostname | `` (corp DNS may also resolve) |
| Network |  for outbound |
| Firewall | `ufw inactive`, `iptables INPUT=ACCEPT` — nothing blocks inbound |
| TLS | none — plain HTTP |
| Auth | none — vLLM is open (LAN-only context) |

## Live endpoints

```
┌──────┬─────────────────────────────────────────┬──────────┬─────────┐
│ PORT │ SERVED MODEL                             │ GAUDI(S) │ STEADY │
├──────┼─────────────────────────────────────────┼──────────┼─────────┤
│ 8    │ qwen3-vl-32b-thinking                    │ 0        │ ~48 t/s │
│ 8    │ qwen3-vl-32b-instruct                    │ 1        │ ~49 t/s │
│ 8    │ qwen3-vl-235b-tp4 (235B-A22B-Thinking)   │ 4,5,6,7  │ ~23 t/s │
│ 3    │ Open WebUI (chat front-end)              │ -        │ -       │
└──────┴─────────────────────────────────────────┴──────────┴─────────┘

All endpoints bind to 0.0.0.0 — reachable from any LAN client.
   http://localhost:<port>          from this box
   http://<ip>:<port>               from any LAN client
```

Every vLLM endpoint serves the standard OpenAI API surface:

```
GET  /v1/models                       list served model
POST /v1/chat/completions             main: text / vision / tool-calling
POST /v1/completions                  legacy
GET  /health                          liveness probe
GET  /metrics                         Prometheus metrics (no scrape yet)
GET  /docs                            interactive Swagger
```

## 🎯 Per-model access guide

Each model has its own endpoint, served-name, and characteristics.
Below: copy-paste ready snippets for each.

---

### 1. `qwen3-vl-32b-instruct`  →  port **8001**, Gaudi **1**

```
URL                    http://<ip>/v1
served model name      qwen3-vl-32b-instruct
HF ID                  Qwen/Qwen3-VL-32B-Instruct-FP8
type                   dense 32B, FP8, vision-language
reasoning trace        none — terse, direct answers
parallel tool calls    ✓ (validated 4 calls in one response)
vision                 ✓
steady tok/s           ~49
max context            16,384
best for               fast tool dispatch, scripted automation, quick Q&A
```

**curl — text:**
```bash
curl -s -X POST http://<ip>:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-vl-32b-instruct",
       "messages":[{"role":"user","content":"hello"}],
       "max_tokens":200}'
```

**curl — vision (drag any PNG into the data URL):**
```bash
B64=$(base64 -w0 game_screen.png)
curl -s -X POST http://<ip>:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"qwen3-vl-32b-instruct\",
       \"messages\":[{\"role\":\"user\",\"content\":[
         {\"type\":\"text\",\"text\":\"describe what's on screen\"},
         {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$B64\"}}
       ]}],\"max_tokens\":500}"
```

**curl — parallel tool calls:**
```bash
curl -s -X POST http://<ip>:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3-vl-32b-instruct",
    "messages":[{"role":"user","content":"weather in Paris and Tokyo?"}],
    "tools":[{"type":"function","function":{
      "name":"get_weather",
      "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
    "tool_choice":"auto","max_tokens":600}'
```

**Python:**
```python
from openai import OpenAI
client = OpenAI(base_url="http://<ip>:8001/v1", api_key="dummy")
resp = client.chat.completions.create(
    model="qwen3-vl-32b-instruct",
    messages=[{"role":"user","content":"hello"}],
    max_tokens=200,
)
print(resp.choices[0].message.content)
```

---

### 2. `qwen3-vl-32b-thinking`  →  port **8000**, Gaudi **0**

```
URL                    http://<ip>9:8000/v1
served model name      qwen3-vl-32b-thinking
HF ID                  Qwen/Qwen3-VL-32B-Thinking-FP8
type                   dense 32B, FP8, vision-language
reasoning trace        emits <think>...</think> before answer
parallel tool calls    ✓
vision                 ✓
steady tok/s           ~48
max context            16,384
best for               problems where you want CoT visible (debugging,
                       teachable explanations); same speed as Instruct but
                       reasoning eats some of your token budget
```

**curl — text (note the bigger max_tokens to fit `<think>`):**
```bash
curl -s -X POST http://<ip>9:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-vl-32b-thinking",
       "messages":[{"role":"user","content":"why is the sky blue?"}],
       "max_tokens":2000}'
```

The reply has the structure:
```
<think>
… reasoning trace …
</think>

The actual user-facing answer.
```

**Python — strip out the reasoning if you only want the answer:**
```python
import re
content = resp.choices[0].message.content or ""
answer = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
```

**Tool calls work identically** — but always pass `max_tokens >= 2000` so
the reasoning preamble doesn't cut off the tool call.

---

### 3. `qwen3-vl-235b-tp4`  →  port **8004**, Gaudis **4-7** (tensor-parallel ×4)

```
URL                    http://<ip>9:8004/v1
served model name      qwen3-vl-235b-tp4
HF ID                  Qwen/Qwen3-VL-235B-A22B-Thinking-FP8
type                   MoE 235B total / 22B active, FP8, vision-language
reasoning trace        emits <think>...</think>
parallel tool calls    ✓ (validated 4 mixed tools in one response)
vision                 ✓
steady tok/s           ~23 (TP=4 collective comm overhead)
max context            16,384
best for               hard games, complex UIs, when 32B gets it wrong;
                       frontier multimodal reasoning at the cost of speed
```

**curl is identical to 32B — only `model` and port change:**
```bash
curl -s -X POST http://<ip>9:8004/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-vl-235b-tp4",
       "messages":[{"role":"user","content":"hard problem here"}],
       "max_tokens":4000}'
```

`max_tokens 4000` recommended for tool-call workloads — Thinking-variant
reasoning + 235B's thoroughness can be lengthy before the tool call lands.

---

### Streaming (token-by-token via SSE) — works on all 3 endpoints

```python
stream = client.chat.completions.create(
    model="qwen3-vl-32b-instruct",   # or any other served model
    messages=[{"role":"user","content":"tell me a story"}],
    max_tokens=500,
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

```bash
# curl streaming
curl -N -X POST http://<ip>9:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-vl-32b-instruct",
       "messages":[{"role":"user","content":"hi"}],
       "max_tokens":100,"stream":true}'
```

---

### Switching model mid-conversation (Open WebUI does this for you)

The dropdown in Open WebUI at `http://<ip>9:3000` lists all 3
served names. Switch any time — context stays in the chat thread.

If you want to switch programmatically: just change `base_url` + `model`
in your client. Each endpoint is independent.

---

## When to pick which model

| Use case | Endpoint | Why |
|---|---|---|
| Fast tool-call dispatch (no reasoning trace) | `:8001` qwen3-vl-32b-instruct | dense, no `<think>`, fastest |
| Reasoning with chain-of-thought visible | `:8000` qwen3-vl-32b-thinking | same speed, emits `<think>...</think>` |
| Hard games / complex UIs / frontier vision | `:8004` qwen3-vl-235b-tp4 | best brain, slower (~23 t/s, TP=4) |
| Browser chat UI | `:3000` Open WebUI | model dropdown lets you pick any of the above |

## Quick curl smoke

```bash
# 1. list model
curl -s http://<ip>9:8001/v1/models | python3 -m json.tool

# 2. text completion
curl -s -X POST http://<ip>9:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-vl-32b-instruct",
       "messages":[{"role":"user","content":"What is 17 * 89?"}],
       "max_tokens":200}'

# 3. vision (base64 PNG inline)
B64=$(base64 -w0 your_image.png)
curl -s -X POST http://<ip>9:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-vl-32b-instruct",
       "messages":[{"role":"user","content":[
         {"type":"text","text":"Describe this image."},
         {"type":"image_url","image_url":{"url":"data:image/png;base64,'"$B64"'"}}
       ]}],"max_tokens":300}'

# 4. parallel tool calling
curl -s -X POST http://<ip>9:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3-vl-32b-instruct",
    "messages":[{"role":"user","content":"Weather in Paris and Tokyo?"}],
    "tools":[{"type":"function","function":{
      "name":"get_weather","description":"Weather for a city",
      "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
    "tool_choice":"auto","max_tokens":600}'
```

## Python (`openai` SDK)

```python
# pip install openai
from openai import OpenAI

client = OpenAI(
    base_url="http://<ip>9:8001/v1",   # pick endpoint
    api_key="dummy",                            # vLLM doesn't check; field is required
)

# Plain text
resp = client.chat.completions.create(
    model="qwen3-vl-32b-instruct",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=200,
)
print(resp.choices[0].message.content)

# Vision: image_url with public URL OR base64 data URL
import base64, pathlib
img_b64 = base64.b64encode(pathlib.Path("screenshot.png").read_bytes()).decode()
resp = client.chat.completions.create(
    model="qwen3-vl-32b-instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's on this screen?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ],
    }],
    max_tokens=500,
)

# Parallel tool calling (works on all 3 endpoints)
resp = client.chat.completions.create(
    model="qwen3-vl-32b-instruct",
    messages=[{"role": "user", "content": "Click Settings, then take a screenshot."}],
    tools=[
        {"type": "function", "function": {
            "name": "click",
            "description": "Click on screen at (x,y)",
            "parameters": {"type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"]}}},
        {"type": "function", "function": {
            "name": "screenshot",
            "description": "Take a screenshot",
            "parameters": {"type": "object", "properties": {}}}},
    ],
    tool_choice="auto",
    max_tokens=600,
)
for tc in resp.choices[0].message.tool_calls or []:
    print(tc.function.name, tc.function.arguments)
```

## Streaming (token-by-token via SSE)

```python
stream = client.chat.completions.create(
    model="qwen3-vl-32b-instruct",
    messages=[{"role": "user", "content": "Tell me a story."}],
    max_tokens=500,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
```

## Model variants — their behavior signatures

| Model | Reasoning trace | Speed | Best for |
|---|---|---|---|
| `qwen3-vl-32b-instruct` | none, terse | fast | tool dispatch, scripted automation |
| `qwen3-vl-32b-thinking` | emits `<think>…</think>` | same speed | debugging / explaining decisions |
| `qwen3-vl-235b-tp4` | emits `<think>` (Thinking) | ~half the 32B speed | hard problems, frontier quality |

Thinking variants emit a long `<think>...</think>` block before the final
answer. **Set `max_tokens >= 2000`** for the Thinking endpoints when expecting
tool calls, otherwise reasoning eats your token budget before the call lands.

## Open WebUI (browser front-end)

| | |
|---|---|
| URL | http://<ip>9:3000 |
| Auth | disabled (`WEBUI_AUTH=false`) — anyone on LAN can use |
| Backend | wired to all 3 vLLM endpoints; model dropdown auto-populates |
| Persistent | conversations + settings stored in docker volume `open-webui-data` |
| Features | chat, markdown/LaTeX, image upload (drag-and-drop), tool-call rendering, RAG, embeddings, conversation export |

To reach the WebUI from another machine, just open the URL in any browser
on the LAN. No client setup needed.

## Game-automation harness — minimal client config

```yaml
# game-auto/config.yaml
gaudi_api:
  endpoints:
    fast:        # for most games / quick decisions
      url:    http://<ip>9:8001/v1
      model:  qwen3-vl-32b-instruct
    reasoning:   # when you want the <think> trace
      url:    http://<ip>9:8000/v1
      model:  qwen3-vl-32b-thinking
    hard:        # frontier brain for stuck games
      url:    http://<ip>9:8004/v1
      model:  qwen3-vl-235b-tp4
  api_key: dummy

routing:
  default_endpoint: fast
  fallback_to_hard_after_failures: 3
```

```python
# game-auto/client.py
import yaml, base64, pathlib
from openai import OpenAI

cfg = yaml.safe_load(open("config.yaml"))

def make_client(role="fast"):
    e = cfg["gaudi_api"]["endpoints"][role]
    return OpenAI(base_url=e["url"], api_key=cfg["gaudi_api"]["api_key"]), e["model"]

def screenshot_to_data_url(path: str) -> str:
    return "data:image/png;base64," + base64.b64encode(
        pathlib.Path(path).read_bytes()
    ).decode()

def plan_step(screenshot_path: str, goal: str, role="fast"):
    client, model = make_client(role)
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Goal: {goal}\nWhat actions next?"},
                {"type": "image_url", "image_url": {"url": screenshot_to_data_url(screenshot_path)}},
            ],
        }],
        tools=[
            # define your game's primitive actions here
        ],
        tool_choice="auto",
        max_tokens=2000,
    )
    return resp.choices[0].message.tool_calls or []
```

## Health checks (paste into Prometheus / Datadog / your monitor)

```bash
# liveness
curl -fs http://<ip>9:8001/health    # 200 OK if healthy

# functional check (will exercise actual decode path)
curl -fs -m 10 -X POST http://<ip>9:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-vl-32b-instruct","messages":[{"role":"user","content":"ping"}],"max_tokens":3}' \
  | grep -q 'choices' && echo OK || echo FAIL

# Prometheus metrics endpoint
curl -fs http://<ip>9:8001/metrics
```

## Hardening checklist (not done yet — required if exposing beyond corp LAN)

- [ ] `--api-key <token>` on each vLLM (vLLM accepts a `Authorization: Bearer <token>` header)
- [ ] nginx/Caddy reverse proxy with TLS termination (`https://`) — Let's Encrypt or internal CA
- [ ] Rate limit at the proxy (e.g. 100 req/min per IP)
- [ ] log rotation for docker stdout (`--log-opt max-size=100m --log-opt max-file=5`)
- [ ] auto-restart policy back to `unless-stopped` once stable
- [ ] Open WebUI auth ON (`WEBUI_AUTH=true`) — admin signup on first visit
- [ ] Prometheus + Grafana scraping `/metrics`

## Stopping / starting / inspecting

```bash
# launcher provides preset-driven control
vllm-launch list                           # show all presets
vllm-launch 32b-instruct                   # start preset
vllm-launch logs 32b-instruct              # tail logs
vllm-launch stop 32b-instruct              # stop one
vllm-launch stop-all                       # nuclear

# raw docker
sudo docker ps --filter name=vllm-
sudo docker logs -f vllm-32b-instruct
sudo docker restart vllm-32b-instruct

# Open WebUI control
sudo docker logs -f open-webui
sudo docker restart open-webui
```

## File / repo locations on this box

| Path | What |
|---|---|
| `/home/satyajit-gaudi/gaudi-setup/` | This repo (install.sh, scripts/, docs/, bin/) |
| `/home/satyajit-gaudi/hf-cache/` | HuggingFace model cache (bound into vLLM containers) |
| `/etc/sysctl.d/99-habana-hugepages.conf` | hugepage persistence |
| `/etc/modules-load.d/habana-modules.conf` | ib_uverbs + habanalabs_ib autoload |
| `/etc/systemd/system/gaudi-tune.service` | MSR + NIC bring-up on every boot |
| `/etc/default/grub` | iommu=pt + intel_iommu=on cmdline |
| `/etc/docker/daemon.json` | habana container runtime registration |
| `/etc/systemd/system/docker.service.d/http-proxy.conf` | corp proxy for dockerd |
