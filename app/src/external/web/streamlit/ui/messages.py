from typing import Callable

import pandas as pd
import streamlit as st
from src.common.utils.database import get_database_connection
from src.config import get_config

config = get_config()


def format_user_message(message_content) -> None:
    st.markdown(message_content)


def format_assistant_message(message_content) -> None:
    chart_spec = message_content["chart"]

    conn = get_database_connection()
    chart = chart_spec["result"]
    sql = chart_spec["intermediates"]["sql"]
    df = pd.read_sql_query(sql, conn)

    tab_titles = ["SQL", "Table", "Chart", "Text"]
    tab_sql, tab_table, tab_chart, tab_text = st.tabs(tab_titles)

    with tab_sql:
        st.code(sql, language="sql")

    with tab_table:
        st.dataframe(df)

    with tab_chart:
        with st.expander("spec"):
            st.write(chart_spec)

        st.vega_lite_chart(df, chart)

    with tab_text:
        st.write("Not Implemented")


def message_formatter_factory(role: str) -> Callable:
    messages = {
        "user": format_user_message,
        "assistant": format_assistant_message,
    }
    return messages[role]


def render_messages() -> None:
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            message_formatter_factory(role)(content)
