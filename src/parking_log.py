"""Optional local SQLite log of observed plates."""

import sqlite3
from datetime import datetime, timezone


def open_log(path):
    connection = sqlite3.connect(path)
    connection.execute("""CREATE TABLE IF NOT EXISTS plate_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        observed_at_utc TEXT NOT NULL,
        plate_text TEXT NOT NULL,
        detector_confidence REAL NOT NULL
    )""")
    connection.commit()
    return connection


def log_detection(connection, cleaned_text, confidence):
    if not cleaned_text:
        return
    connection.execute(
        "INSERT INTO plate_events (observed_at_utc, plate_text, detector_confidence) VALUES (?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), cleaned_text, float(confidence)),
    )
    connection.commit()
