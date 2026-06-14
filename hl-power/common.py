"""Shared paths, schema, and small helpers for hl-power-logger + hl-power-server."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DATA_DIR = Path(
    os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
) / "hl-power"
DB_PATH = DATA_DIR / "log.db"

# Tiered retention (seconds).
TIER_1S_SECONDS = 7 * 86400         # keep 1Hz raw for 7 days
TIER_10S_SECONDS = 30 * 86400       # then 10s averages for 30 days
TIER_1M_SECONDS = 365 * 86400       # then 1m averages for 12 months
MODEL_METRICS_KEEP = 30 * 86400     # keep prom snapshots for 30 days

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS samples_1s (
    ts INTEGER PRIMARY KEY,
    total_w REAL,
    util_avg REAL,
    util_max REAL,
    temp_max REAL,
    aip_powers TEXT,
    aip_utils TEXT,
    aip_temps TEXT,
    aip_mems TEXT
);

CREATE TABLE IF NOT EXISTS samples_10s (
    ts INTEGER PRIMARY KEY,
    total_w_avg REAL,
    total_w_max REAL,
    total_w_min REAL,
    util_avg REAL,
    temp_max REAL
);

CREATE TABLE IF NOT EXISTS samples_1m (
    ts INTEGER PRIMARY KEY,
    total_w_avg REAL,
    total_w_max REAL,
    total_w_min REAL,
    util_avg REAL,
    temp_max REAL
);

CREATE TABLE IF NOT EXISTS model_runs (
    model_id TEXT NOT NULL,
    port INTEGER NOT NULL,
    started_at INTEGER NOT NULL,
    last_seen INTEGER NOT NULL,
    PRIMARY KEY (model_id, port, started_at)
);
CREATE INDEX IF NOT EXISTS idx_model_runs_last_seen ON model_runs(last_seen);

CREATE TABLE IF NOT EXISTS model_metrics (
    ts INTEGER NOT NULL,
    model_id TEXT NOT NULL,
    port INTEGER NOT NULL,
    requests_running REAL,
    requests_waiting REAL,
    request_success_total INTEGER,
    prompt_tokens_total INTEGER,
    generation_tokens_total INTEGER,
    PRIMARY KEY (ts, model_id, port)
);
CREATE INDEX IF NOT EXISTS idx_model_metrics_ts ON model_metrics(ts);

CREATE TABLE IF NOT EXISTS daily_kwh (
    date TEXT PRIMARY KEY,
    kwh REAL,
    peak_w REAL,
    peak_w_at INTEGER
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def open_db(readonly: bool = False) -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if readonly:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5.0)
    else:
        conn = sqlite3.connect(str(DB_PATH), timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()
