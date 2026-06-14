#!/usr/bin/env python3
"""hl-power-logger — always-on logger for Gaudi power + model usage.

Samples `hl-smi` every 1 s, scans common ports for OpenAI-compatible model
servers every 15 s, scrapes their Prometheus `/metrics` for request/token
counters, and writes everything into a SQLite file at
`~/.local/share/hl-power/log.db`.

Old rows are continuously downsampled into coarser tiers so the DB never grows
unbounded:
  • last 7 days   → 1-second raw
  • next 30 days  → 10-second averages
  • next 12 months → 1-minute averages
"""
from __future__ import annotations

import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    DATA_DIR,
    MODEL_METRICS_KEEP,
    TIER_10S_SECONDS,
    TIER_1M_SECONDS,
    TIER_1S_SECONDS,
    ensure_schema,
    open_db,
)

SAMPLE_INTERVAL_SEC = 1.0
MODEL_SCAN_INTERVAL_SEC = 15.0
ROLLUP_INTERVAL_SEC = 300.0
PROBE_TIMEOUT_SEC = 0.5
MODEL_RUN_GAP_SEC = 60  # gap longer than this starts a new run row

MODEL_SCAN_PORTS = (
    list(range(8000, 8021))
    + list(range(30000, 30011))
    + [11434, 9000, 9001]
)

HL_SMI_FIELDS = [
    "index", "power.draw", "utilization.aip", "temperature.aip",
    "memory.used", "memory.total",
]

PROM_LINE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>\S+)'
)


def query_hl_smi() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["hl-smi", "-Q", ",".join(HL_SMI_FIELDS), "-f", "csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=3,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    aips: list[dict] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(HL_SMI_FIELDS):
            continue
        try:
            aips.append({
                "idx": int(parts[0]),
                "power": float(parts[1]),
                "util": float(parts[2]),
                "temp": float(parts[3]),
                "mem_used": float(parts[4]),
                "mem_total": float(parts[5]),
            })
        except ValueError:
            continue
    aips.sort(key=lambda x: x["idx"])
    return aips


def probe_model_port(port: int) -> list[str]:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=PROBE_TIMEOUT_SEC):
            pass
    except (OSError, socket.timeout):
        return []
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read())
        return [m.get("id", "?") for m in data.get("data", []) if m.get("id")]
    except Exception:
        return []


def parse_prom_text(text: str):
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = PROM_LINE.match(line)
        if not m:
            continue
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        labels: dict[str, str] = {}
        raw = m.group("labels")
        if raw:
            for kv in raw[1:-1].split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    labels[k.strip()] = v.strip().strip('"')
        yield m.group("name"), labels, value


def scrape_vllm_metrics(port: int) -> list[dict]:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/metrics", timeout=PROBE_TIMEOUT_SEC
        ) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return []
    by_model: dict[str, dict] = defaultdict(lambda: {
        "running": None, "waiting": None, "requests": None,
        "ptokens": None, "gtokens": None,
    })
    for name, labels, val in parse_prom_text(text):
        mid = labels.get("model_name") or labels.get("model") or "?"
        if name == "vllm:num_requests_running":
            by_model[mid]["running"] = val
        elif name == "vllm:num_requests_waiting":
            by_model[mid]["waiting"] = val
        elif name == "vllm:request_success_total":
            by_model[mid]["requests"] = val
        elif name == "vllm:prompt_tokens_total":
            by_model[mid]["ptokens"] = val
        elif name == "vllm:generation_tokens_total":
            by_model[mid]["gtokens"] = val
    return [
        {"model_id": mid, **vals}
        for mid, vals in by_model.items()
        if any(v is not None for v in vals.values())
    ]


def write_sample(conn: sqlite3.Connection, ts: int, aips: list[dict]) -> None:
    powers = [a["power"] for a in aips]
    utils = [a["util"] for a in aips]
    temps = [a["temp"] for a in aips]
    mems = [a["mem_used"] for a in aips]
    conn.execute(
        "INSERT OR REPLACE INTO samples_1s "
        "(ts, total_w, util_avg, util_max, temp_max, aip_powers, aip_utils, aip_temps, aip_mems) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            ts,
            sum(powers),
            sum(utils) / len(utils) if utils else 0.0,
            max(utils) if utils else 0.0,
            max(temps) if temps else 0.0,
            json.dumps(powers),
            json.dumps(utils),
            json.dumps(temps),
            json.dumps(mems),
        ),
    )


def write_model_observation(
    conn: sqlite3.Connection,
    ts: int,
    models_seen: list[tuple[str, int]],
    metrics: dict[int, list[dict]],
) -> None:
    # Extend or start a model_runs row per (model, port).
    for mid, port in models_seen:
        row = conn.execute(
            "SELECT started_at FROM model_runs WHERE model_id=? AND port=? "
            "AND last_seen >= ? ORDER BY started_at DESC LIMIT 1",
            (mid, port, ts - MODEL_RUN_GAP_SEC),
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE model_runs SET last_seen=? "
                "WHERE model_id=? AND port=? AND started_at=?",
                (ts, mid, port, row[0]),
            )
        else:
            conn.execute(
                "INSERT OR IGNORE INTO model_runs "
                "(model_id, port, started_at, last_seen) VALUES (?, ?, ?, ?)",
                (mid, port, ts, ts),
            )

    for port, items in metrics.items():
        for m in items:
            conn.execute(
                "INSERT OR REPLACE INTO model_metrics "
                "(ts, model_id, port, requests_running, requests_waiting, "
                " request_success_total, prompt_tokens_total, generation_tokens_total) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ts, m["model_id"], port,
                    m.get("running"),
                    m.get("waiting"),
                    int(m["requests"]) if m.get("requests") is not None else None,
                    int(m["ptokens"]) if m.get("ptokens") is not None else None,
                    int(m["gtokens"]) if m.get("gtokens") is not None else None,
                ),
            )


def update_daily(conn: sqlite3.Connection, ts: int, total_w: float, dt: float) -> None:
    """Integrate power × dt into today's kWh bucket + track per-day peak watt."""
    if dt <= 0 or dt > 60:
        return
    today = time.strftime("%Y-%m-%d", time.localtime(ts))
    kwh_inc = total_w * dt / 3600.0 / 1000.0
    row = conn.execute(
        "SELECT kwh, peak_w FROM daily_kwh WHERE date=?", (today,)
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO daily_kwh (date, kwh, peak_w, peak_w_at) VALUES (?, ?, ?, ?)",
            (today, kwh_inc, total_w, ts),
        )
        return
    new_kwh = (row[0] or 0.0) + kwh_inc
    if total_w > (row[1] or 0.0):
        conn.execute(
            "UPDATE daily_kwh SET kwh=?, peak_w=?, peak_w_at=? WHERE date=?",
            (new_kwh, total_w, ts, today),
        )
    else:
        conn.execute("UPDATE daily_kwh SET kwh=? WHERE date=?", (new_kwh, today))


def rollup(conn: sqlite3.Connection, now: int) -> None:
    """Tiered downsampling: 1s→10s→1m→drop, and prune model_metrics."""
    cutoff_1s = now - TIER_1S_SECONDS
    conn.execute("""
        INSERT OR REPLACE INTO samples_10s
            (ts, total_w_avg, total_w_max, total_w_min, util_avg, temp_max)
        SELECT (ts / 10) * 10,
               AVG(total_w), MAX(total_w), MIN(total_w),
               AVG(util_avg), MAX(temp_max)
        FROM samples_1s WHERE ts < ?
        GROUP BY (ts / 10) * 10
    """, (cutoff_1s,))
    conn.execute("DELETE FROM samples_1s WHERE ts < ?", (cutoff_1s,))

    cutoff_10s = now - TIER_10S_SECONDS
    conn.execute("""
        INSERT OR REPLACE INTO samples_1m
            (ts, total_w_avg, total_w_max, total_w_min, util_avg, temp_max)
        SELECT (ts / 60) * 60,
               AVG(total_w_avg), MAX(total_w_max), MIN(total_w_min),
               AVG(util_avg), MAX(temp_max)
        FROM samples_10s WHERE ts < ?
        GROUP BY (ts / 60) * 60
    """, (cutoff_10s,))
    conn.execute("DELETE FROM samples_10s WHERE ts < ?", (cutoff_10s,))

    conn.execute("DELETE FROM samples_1m WHERE ts < ?", (now - TIER_1M_SECONDS,))
    conn.execute("DELETE FROM model_metrics WHERE ts < ?", (now - MODEL_METRICS_KEEP,))


def model_scanner_loop(stop_event: threading.Event) -> None:
    """Runs in its own thread + opens its own connection — sqlite3 connections
    are per-thread by default."""
    conn = open_db()
    ensure_schema(conn)
    while not stop_event.is_set():
        now = int(time.time())
        seen: list[tuple[str, int]] = []
        metrics: dict[int, list[dict]] = {}
        for p in MODEL_SCAN_PORTS:
            ids = probe_model_port(p)
            if ids:
                for mid in ids:
                    seen.append((mid, p))
                m = scrape_vllm_metrics(p)
                if m:
                    metrics[p] = m
        try:
            write_model_observation(conn, now, seen, metrics)
            conn.commit()
        except sqlite3.OperationalError:
            pass
        if stop_event.wait(MODEL_SCAN_INTERVAL_SEC):
            return


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = open_db()
    ensure_schema(conn)

    stop_event = threading.Event()
    scanner = threading.Thread(
        target=model_scanner_loop, args=(stop_event,), daemon=True
    )
    scanner.start()

    last_ts = 0
    last_rollup_at = 0.0
    try:
        while True:
            now = time.time()
            ts = int(now)
            if ts <= last_ts:
                time.sleep(0.2)
                continue
            aips = query_hl_smi()
            if aips:
                write_sample(conn, ts, aips)
                total_w = sum(a["power"] for a in aips)
                dt = (ts - last_ts) if last_ts else 1.0
                update_daily(conn, ts, total_w, dt)
                conn.commit()
            last_ts = ts

            if now - last_rollup_at >= ROLLUP_INTERVAL_SEC:
                try:
                    rollup(conn, ts)
                    conn.commit()
                except sqlite3.OperationalError:
                    pass
                last_rollup_at = now

            # Sleep until next whole-second tick, minus a small drift correction
            time.sleep(max(0.05, SAMPLE_INTERVAL_SEC - (time.time() - now)))
    except KeyboardInterrupt:
        return 0
    finally:
        stop_event.set()
        conn.close()


if __name__ == "__main__":
    sys.exit(main() or 0)
