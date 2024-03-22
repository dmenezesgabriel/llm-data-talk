import sqlite3

from src.config import get_config

config = get_config()


def get_database_connection() -> sqlite3.Connection:
    return sqlite3.connect(config.DATABASE_URI, check_same_thread=False)
