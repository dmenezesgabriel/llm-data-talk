import sqlite3

import streamlit as st

from src.config import get_config

config = get_config()


@st.cache_resource
def get_database_connection():
    return sqlite3.connect(config.DATABASE_URI)
