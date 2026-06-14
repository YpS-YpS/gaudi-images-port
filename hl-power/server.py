#!/usr/bin/env python3
"""hl-power-server — read-only HTTP server for the hl-power dashboard.

Serves the static dashboard at `/`, an aggregated JSON time-series at
`/api/series?range=…`, and a small `/api/now` endpoint that returns the most
recent sample for live needle widgets. Reads the SQLite log written by
hl-power-logger; never writes.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).parent))
from common import open_db  # noqa: E402

HERE = Path(__file__).parent
STATIC_DIR = HERE / "static"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9090

# Default electricity rate for the cost projection tiles. Override via the
# `?rate=` query param, e.g. /api/series?range=1d&rate=0.18.
DEFAULT_RATE_PER_KWH = 0.12

# Per-AIP HBM capacity in MiB for HL-325; used to render mem % in the strip.
AIP_HBM_TOTAL_MIB = 131072

SPARKLINE_POINTS = 60

# Model → AIP mapping cache. We discover the mapping by inspecting docker
# containers whose env contains HABANA_VISIBLE_DEVICES / HABANA_VISIBLE_MODULES
# and whose cmd includes a --port arg. Cached for 30 s so each /api/series
# call doesn't shell out 4× to docker.
_AIP_MAP_CACHE = {"ts": 0.0, "ports": {}, "modules": {}}
AIP_MAP_TTL_SEC = 30.0


def _hl_smi_module_to_index() -> dict[int, int]:
    """Map physical module_id → driver index (these differ on multi-AIP boxes)."""
    try:
        out = subprocess.check_output(
            ["hl-smi", "-Q", "index,module_id", "-f", "csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=2,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}
    mapping = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                mapping[int(parts[1])] = int(parts[0])
            except ValueError:
                pass
    return mapping


_PORT_RE = re.compile(r"--port[=\s]+(\d+)")

def _parse_port_from_args(args: list[str]) -> int | None:
    # Containers using `bash -lc 'vllm serve … --port 8000 …'` collapse all
    # vllm flags into a single string arg, so we regex-search any string for
    # `--port N` or `--port=N` as well as the simple split form.
    for i, a in enumerate(args or []):
        if not isinstance(a, str):
            continue
        if a == "--port" and i + 1 < len(args):
            try: return int(args[i + 1])
            except ValueError: pass
        elif a.startswith("--port="):
            try: return int(a.split("=", 1)[1])
            except ValueError: pass
        m = _PORT_RE.search(a)
        if m:
            try: return int(m.group(1))
            except ValueError: pass
    return None


def _parse_aip_indices(env: dict[str, str], module_to_index: dict[int, int]) -> list[int]:
    raw = env.get("HABANA_VISIBLE_DEVICES")
    if raw:
        return sorted({int(x) for x in raw.split(",") if x.strip().lstrip("-").isdigit()})
    raw = env.get("HABANA_VISIBLE_MODULES")
    if raw:
        out = set()
        for x in raw.split(","):
            x = x.strip()
            if x.lstrip("-").isdigit():
                m = int(x)
                out.add(module_to_index.get(m, m))
        return sorted(out)
    return []


def _models_on_aips_for(models: list[dict]) -> list[dict]:
    """Combine docker-detected port→AIP mapping with the model list (from
    model_runs) into [{"id": model, "port": p, "aips": [...]}, ...]."""
    port_to_aips, _ = detect_model_aips()
    out = []
    for m in models:
        mid = m["id"]
        for port in m.get("ports", []) or []:
            if port in port_to_aips:
                out.append({"id": mid, "port": port, "aips": port_to_aips[port]})
                break
    return out


def detect_model_aips() -> tuple[dict[int, list[int]], dict[int, str]]:
    """Return (port_to_aip_indices, port_to_container_name) for vLLM containers."""
    now = time.time()
    if now - _AIP_MAP_CACHE["ts"] < AIP_MAP_TTL_SEC and _AIP_MAP_CACHE["ports"]:
        return _AIP_MAP_CACHE["ports"], _AIP_MAP_CACHE["modules"]
    try:
        names_out = subprocess.check_output(
            ["docker", "ps", "--format", "{{.Names}}"],
            text=True, stderr=subprocess.DEVNULL, timeout=2,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}, {}
    module_to_index = _hl_smi_module_to_index()
    ports: dict[int, list[int]] = {}
    names: dict[int, str] = {}
    for name in (n.strip() for n in names_out.splitlines() if n.strip()):
        if "vllm" not in name.lower():
            continue
        try:
            cfg_json = subprocess.check_output(
                ["docker", "inspect", name, "--format", "{{json .Config}}"],
                text=True, stderr=subprocess.DEVNULL, timeout=2,
            )
            cfg = json.loads(cfg_json)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError):
            continue
        env_list = cfg.get("Env") or []
        env = {}
        for entry in env_list:
            if "=" in entry:
                k, v = entry.split("=", 1)
                env[k] = v
        args = list((cfg.get("Entrypoint") or []) + (cfg.get("Cmd") or []))
        port = _parse_port_from_args(args)
        aips = _parse_aip_indices(env, module_to_index)
        if port is None or not aips:
            continue
        ports[port] = aips
        names[port] = name
    _AIP_MAP_CACHE["ts"] = now
    _AIP_MAP_CACHE["ports"] = ports
    _AIP_MAP_CACHE["modules"] = names
    return ports, names

# (window_seconds, source_table, bucket_seconds_for_plot)
RANGES = {
    "1h":  (3600,         "samples_1s",  60),
    "8h":  (8 * 3600,     "samples_1s",  300),
    "1d":  (86400,        "samples_1s",  600),
    "7d":  (7 * 86400,    "samples_10s", 3600),
    "30d": (30 * 86400,   "samples_10s", 14400),
    "1y":  (365 * 86400,  "samples_1m",  86400),
}

EWMA_ALPHA_BY_RANGE = {
    "1h": 0.20, "8h": 0.15, "1d": 0.10,
    "7d": 0.05, "30d": 0.05, "1y": 0.03,
}

CTYPES = {
    ".html": "text/html; charset=utf-8",
    ".js":   "application/javascript",
    ".css":  "text/css",
    ".svg":  "image/svg+xml",
    ".png":  "image/png",
    ".ico":  "image/x-icon",
    ".json": "application/json",
}


def get_series(rng: str, rate_per_kwh: float = DEFAULT_RATE_PER_KWH) -> dict:
    if rng not in RANGES:
        rng = "1h"
    seconds, table, bucket = RANGES[rng]
    now = int(time.time())
    since = now - seconds
    conn = open_db(readonly=True)
    try:
        if table == "samples_1s":
            sql = (
                f"SELECT (ts/{bucket})*{bucket} AS b, "
                f"AVG(total_w), MAX(total_w), MIN(total_w), "
                f"AVG(util_avg), MAX(temp_max) "
                f"FROM {table} WHERE ts >= ? GROUP BY b ORDER BY b"
            )
        else:
            sql = (
                f"SELECT (ts/{bucket})*{bucket} AS b, "
                f"AVG(total_w_avg), MAX(total_w_max), MIN(total_w_min), "
                f"AVG(util_avg), MAX(temp_max) "
                f"FROM {table} WHERE ts >= ? GROUP BY b ORDER BY b"
            )
        rows = conn.execute(sql, (since,)).fetchall()

        # In-query EWMA (computed in Python over the bucketed series)
        alpha = EWMA_ALPHA_BY_RANGE.get(rng, 0.1)
        ewma_val = None
        ewma_series = []
        for r in rows:
            avg_w = r[1]
            ewma_val = avg_w if ewma_val is None else alpha * avg_w + (1 - alpha) * ewma_val
            ewma_series.append([r[0] * 1000, round(ewma_val, 1)])

        power_avg = [[r[0] * 1000, round(r[1] or 0, 1)] for r in rows]
        power_max = [[r[0] * 1000, round(r[2] or 0, 1)] for r in rows]
        power_min = [[r[0] * 1000, round(r[3] or 0, 1)] for r in rows]
        util      = [[r[0] * 1000, round(r[4] or 0, 1)] for r in rows]
        temp      = [[r[0] * 1000, round(r[5] or 0, 1)] for r in rows]

        # Peak markers: any sample at a bucket where max is >5% above EWMA.
        peaks = []
        ewma_map = {ts: v for ts, v in ewma_series}
        for ts_ms, max_w in power_max:
            base = ewma_map.get(ts_ms)
            if base and max_w > base * 1.05:
                peaks.append([ts_ms, max_w])

        # Model active spans inside the window
        spans = conn.execute(
            "SELECT model_id, port, started_at, last_seen FROM model_runs "
            "WHERE last_seen >= ? ORDER BY started_at",
            (since,),
        ).fetchall()
        models_dict: dict[str, dict] = {}
        for mid, port, start, end in spans:
            entry = models_dict.setdefault(mid, {"id": mid, "ports": set(), "spans": []})
            entry["ports"].add(port)
            entry["spans"].append([max(start, since) * 1000, end * 1000])
        models = [
            {"id": e["id"], "ports": sorted(e["ports"]), "spans": e["spans"]}
            for e in models_dict.values()
        ]

        # Per-model request rate (counter delta per bucket)
        per_model: dict[str, list] = {}
        m_rows = conn.execute(
            f"SELECT (ts/{bucket})*{bucket} AS b, model_id, "
            f"MAX(request_success_total), MAX(prompt_tokens_total), MAX(generation_tokens_total) "
            f"FROM model_metrics WHERE ts >= ? "
            f"GROUP BY b, model_id ORDER BY b",
            (since,),
        ).fetchall()
        # Group by model, then compute delta per bucket
        per_model_buckets: dict[str, list] = {}
        for b, mid, req, ptok, gtok in m_rows:
            per_model_buckets.setdefault(mid, []).append(
                (b, req or 0, ptok or 0, gtok or 0)
            )
        for mid, series in per_model_buckets.items():
            out = []
            prev_req = prev_gtok = None
            for b, req, ptok, gtok in series:
                if prev_req is None:
                    out.append({"ts": b * 1000, "reqs": 0, "tok_s": 0})
                else:
                    out.append({
                        "ts": b * 1000,
                        "reqs": max(0, req - prev_req),
                        "tok_s": max(0, (gtok - prev_gtok) / max(1, bucket)),
                    })
                prev_req, prev_gtok = req, gtok
            per_model[mid] = out

        # Summary tiles
        today = time.strftime("%Y-%m-%d", time.localtime(now))
        yest = time.strftime("%Y-%m-%d", time.localtime(now - 86400))
        week_start = time.strftime("%Y-%m-%d", time.localtime(now - 7 * 86400))
        month_start = time.strftime("%Y-%m-%d", time.localtime(now - 30 * 86400))
        year_start = time.strftime("%Y-%m-%d", time.localtime(now - 365 * 86400))

        def kwh_sum(since_date: str) -> float:
            r = conn.execute(
                "SELECT SUM(kwh) FROM daily_kwh WHERE date >= ?", (since_date,)
            ).fetchone()
            return float(r[0] or 0.0)

        kwh_today_row = conn.execute(
            "SELECT kwh FROM daily_kwh WHERE date=?", (today,)
        ).fetchone()
        kwh_yest_row = conn.execute(
            "SELECT kwh FROM daily_kwh WHERE date=?", (yest,)
        ).fetchone()
        peak_row = conn.execute(
            "SELECT date, peak_w, peak_w_at FROM daily_kwh "
            "WHERE date >= ? ORDER BY peak_w DESC LIMIT 1",
            (month_start,),
        ).fetchone()

        # Latest 1s sample (live needle) + per-AIP arrays
        last_row = conn.execute(
            "SELECT ts, total_w, util_avg, temp_max, "
            "aip_powers, aip_utils, aip_temps, aip_mems "
            "FROM samples_1s ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        live = None
        per_aip_live: list = []
        if last_row:
            live = {
                "ts": last_row[0] * 1000,
                "total_w": round(last_row[1] or 0, 1),
                "util_avg": round(last_row[2] or 0, 1),
                "temp_max": round(last_row[3] or 0, 1),
            }
            try:
                p = json.loads(last_row[4] or "[]")
                u = json.loads(last_row[5] or "[]")
                t = json.loads(last_row[6] or "[]")
                m = json.loads(last_row[7] or "[]")
                for i in range(len(p)):
                    per_aip_live.append({
                        "idx": i,
                        "power": round(float(p[i]), 1),
                        "util":  round(float(u[i]) if i < len(u) else 0.0, 1),
                        "temp":  round(float(t[i]) if i < len(t) else 0.0, 1),
                        "mem_used_mib": round(float(m[i]) if i < len(m) else 0.0, 1),
                        "mem_total_mib": AIP_HBM_TOTAL_MIB,
                    })
            except (ValueError, TypeError):
                per_aip_live = []

        # Sparklines: last N raw 1s samples for total_w / util_avg / temp_max
        spark_rows = conn.execute(
            "SELECT ts, total_w, util_avg, temp_max FROM samples_1s "
            "ORDER BY ts DESC LIMIT ?", (SPARKLINE_POINTS,)
        ).fetchall()
        spark_rows = list(reversed(spark_rows))
        sparklines = {
            "total_w": [[r[0] * 1000, round(r[1] or 0, 1)] for r in spark_rows],
            "util":    [[r[0] * 1000, round(r[2] or 0, 1)] for r in spark_rows],
            "temp":    [[r[0] * 1000, round(r[3] or 0, 1)] for r in spark_rows],
        }

        # Hourly kWh for today + yesterday — used by the today/yesterday tile
        # sparklines so they reflect their actual day's profile, not a live signal.
        lt = time.localtime(now)
        today_start = int(time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1)))
        yest_start = today_start - 86400

        def _hourly_kwh(start: int, end: int) -> list[dict]:
            rows = conn.execute(
                "SELECT (ts/3600)*3600 AS h, AVG(total_w) "
                "FROM samples_1s WHERE ts >= ? AND ts < ? GROUP BY h ORDER BY h",
                (start, end),
            ).fetchall()
            return [
                {"ts": int(h) * 1000, "kwh": round((w or 0) / 1000.0, 4)}
                for h, w in rows
            ]

        today_hourly_kwh = _hourly_kwh(today_start, now)
        yest_hourly_kwh  = _hourly_kwh(yest_start, today_start)

        # 30-day daily kWh + daily peak watt (for bar chart + peak-tile sparkline)
        thirty_days_ago = time.strftime("%Y-%m-%d", time.localtime(now - 29 * 86400))
        daily_rows = conn.execute(
            "SELECT date, kwh, peak_w FROM daily_kwh WHERE date >= ? ORDER BY date",
            (thirty_days_ago,)
        ).fetchall()
        kwh_30d_daily = [
            {"date": d, "kwh": round(float(k or 0), 3)} for d, k, _ in daily_rows
        ]
        peak_w_30d_daily = [
            {"date": d, "peak_w": float(p or 0)} for d, _, p in daily_rows
        ]

        # Energy per token: integrate energy and tokens over the visible window.
        # Power-side: use samples_1s/10s/1m with appropriate bucket weight.
        if table == "samples_1s":
            energy_row = conn.execute(
                "SELECT SUM(total_w) / 3600.0 / 1000.0 FROM samples_1s WHERE ts >= ?",
                (since,)
            ).fetchone()
        else:
            # 10s buckets: each row covers 10s; 1m buckets: each row covers 60s
            bucket_s = 10 if table == "samples_10s" else 60
            energy_row = conn.execute(
                f"SELECT SUM(total_w_avg) * {bucket_s} / 3600.0 / 1000.0 FROM {table} WHERE ts >= ?",
                (since,)
            ).fetchone()
        kwh_window = float(energy_row[0] or 0.0)
        tokens_row = conn.execute(
            "SELECT MIN(generation_tokens_total), MAX(generation_tokens_total) "
            "FROM model_metrics WHERE ts >= ?",
            (since,)
        ).fetchone()
        tok_total = 0
        if tokens_row and tokens_row[0] is not None and tokens_row[1] is not None:
            tok_total = max(0, int(tokens_row[1]) - int(tokens_row[0]))
        joules_per_token = (
            (kwh_window * 3_600_000.0) / tok_total if tok_total > 0 else 0.0
        )

        # Cost projections
        cost = {
            "rate_per_kwh": rate_per_kwh,
            "today":  round((float(kwh_today_row[0]) if kwh_today_row else 0.0) * rate_per_kwh, 3),
            "yest":   round((float(kwh_yest_row[0])  if kwh_yest_row  else 0.0) * rate_per_kwh, 3),
            "kwh_7d":  round(kwh_sum(week_start)   * rate_per_kwh, 2),
            "kwh_30d": round(kwh_sum(month_start)  * rate_per_kwh, 2),
            "kwh_1y":  round(kwh_sum(year_start)   * rate_per_kwh, 2),
        }

        # Hour-of-day × day-of-week heatmap: average power per (dow, hour) bucket
        # over the visible window. SQLite strftime: %w = 0-6 (Sun=0), %H = 00-23.
        if table == "samples_1s":
            heatmap_rows = conn.execute(
                "SELECT CAST(strftime('%w', ts, 'unixepoch', 'localtime') AS INT) AS dow, "
                "       CAST(strftime('%H', ts, 'unixepoch', 'localtime') AS INT) AS hr, "
                "       AVG(total_w) FROM samples_1s WHERE ts >= ? GROUP BY dow, hr",
                (since,)
            ).fetchall()
        else:
            heatmap_rows = conn.execute(
                f"SELECT CAST(strftime('%w', ts, 'unixepoch', 'localtime') AS INT) AS dow, "
                f"       CAST(strftime('%H', ts, 'unixepoch', 'localtime') AS INT) AS hr, "
                f"       AVG(total_w_avg) FROM {table} WHERE ts >= ? GROUP BY dow, hr",
                (since,)
            ).fetchall()
        # 7 × 24 matrix; None where no data
        heatmap: list[list[float | None]] = [[None] * 24 for _ in range(7)]
        for dow, hr, w in heatmap_rows:
            if dow is None or hr is None:
                continue
            heatmap[int(dow)][int(hr)] = round(float(w or 0), 1)

        return {
            "range": rng,
            "now_ms": now * 1000,
            "since_ms": since * 1000,
            "live": live,
            "per_aip_live": per_aip_live,
            "sparklines": sparklines,
            "power_avg": power_avg,
            "power_max": power_max,
            "power_min": power_min,
            "ewma": ewma_series,
            "util": util,
            "temp": temp,
            "peaks": peaks,
            "models": models,
            "per_model_metrics": per_model,
            "kwh_30d_daily": kwh_30d_daily,
            "peak_w_30d_daily": peak_w_30d_daily,
            "today_hourly_kwh": today_hourly_kwh,
            "yest_hourly_kwh":  yest_hourly_kwh,
            "models_on_aips":   _models_on_aips_for(models),
            "joules_per_token": round(joules_per_token, 2),
            "kwh_window": round(kwh_window, 3),
            "tokens_window": tok_total,
            "heatmap": heatmap,
            "cost": cost,
            "summary": {
                "kwh_today": float(kwh_today_row[0]) if kwh_today_row else 0.0,
                "kwh_yest":  float(kwh_yest_row[0])  if kwh_yest_row  else 0.0,
                "kwh_7d":    kwh_sum(week_start),
                "kwh_30d":   kwh_sum(month_start),
                "kwh_1y":    kwh_sum(year_start),
                "peak_w":    float(peak_row[1]) if peak_row else 0.0,
                "peak_w_at": (peak_row[2] * 1000) if peak_row else None,
            },
        }
    finally:
        conn.close()


class Handler(BaseHTTPRequestHandler):
    server_version = "hl-power/1.0"

    def log_message(self, format: str, *args) -> None:  # noqa: D401
        pass  # quiet

    def _send_json(self, payload: dict, code: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, rel: str) -> None:
        # Block traversal: resolved path must stay inside STATIC_DIR.
        target = (STATIC_DIR / rel.lstrip("/")).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            self.send_error(404)
            return
        ctype = CTYPES.get(target.suffix.lower(), "application/octet-stream")
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        try:
            if url.path in ("/", "/index.html"):
                return self._send_static("index.html")
            if url.path.startswith("/static/"):
                return self._send_static(url.path[len("/static/"):])
            if url.path == "/api/series":
                qs = parse_qs(url.query)
                rng = qs.get("range", ["1h"])[0]
                try:
                    rate = float(qs.get("rate", [str(DEFAULT_RATE_PER_KWH)])[0])
                except ValueError:
                    rate = DEFAULT_RATE_PER_KWH
                return self._send_json(get_series(rng, rate_per_kwh=rate))
            if url.path == "/healthz":
                return self._send_json({"ok": True})
            self.send_error(404)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._send_json({"error": str(e)}, code=500)


def main() -> int:
    host = os.environ.get("HL_POWER_HOST", DEFAULT_HOST)
    port = int(os.environ.get("HL_POWER_PORT", DEFAULT_PORT))
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"hl-power-server listening on http://{host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
