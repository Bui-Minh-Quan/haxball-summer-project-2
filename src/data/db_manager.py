import sqlite3
from typing import Any
from config.match_config import PlayerStats


class DatabaseManager:

    def __init__(self, db_path: str = "haxball.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS players (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    speed_stat REAL DEFAULT 3200.0,
                    mass_stat REAL DEFAULT 2.0,
                    kick_stat REAL DEFAULT 1000.0,
                    is_bot INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS match_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mode TEXT NOT NULL,
                    red_score INTEGER NOT NULL,
                    blue_score INTEGER NOT NULL,
                    duration REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)

    def save_player(self, stats: PlayerStats, is_bot: bool = False):
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO players (name, speed_stat, mass_stat, kick_stat, is_bot)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    speed_stat=excluded.speed_stat,
                    mass_stat=excluded.mass_stat,
                    kick_stat=excluded.kick_stat
            """, (stats.name, stats.accel, stats.mass, stats.kick_strength, int(is_bot)))

    def load_player(self, name: str) -> PlayerStats | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM players WHERE name = ?", (name,)).fetchone()
            if row:
                return PlayerStats(
                    name=row["name"],
                    accel=row["speed_stat"],
                    mass=row["mass_stat"],
                    kick_strength=row["kick_stat"],
                )
        return None