import sqlite3

from src.config import get_config

config = get_config()


def get_database_connection():
    return sqlite3.connect(config.DATABASE_URI)
