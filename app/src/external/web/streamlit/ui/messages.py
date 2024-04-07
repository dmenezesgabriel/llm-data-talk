from typing import Callable

import pandas as pd
import streamlit as st
from src.common.utils.database import get_database_connection
from src.config import get_config
from src.external.web.streamlit.ui.dynamic_dataframe_filters import (
    dynamic_dataframe_filter,
)

config = get_config()


def format_user_message(message_content) -> None:
    st.markdown(message_content)


def format_assistant_message(message_content) -> None:
    chart_spec = message_content["chart"]

    conn = get_database_connection()
    chart = chart_spec["chart_spec"]
    sql = chart_spec["sql_query"]
    df = pd.read_sql_query(sql, conn)

    tab_titles = ["SQL", "Table", "Chart", "Text", "Steps"]
    tab_sql, tab_table, tab_chart, tab_text, tab_steps = st.tabs(tab_titles)

    with tab_sql:
        st.code(sql, language="sql")

    with tab_table:
        dynamic_filters_table = dynamic_dataframe_filter(df, key="table")
        st.dataframe(dynamic_filters_table)

    with tab_chart:
        dynamic_filters_table_chart = dynamic_dataframe_filter(df, key="chart")
        st.vega_lite_chart(dynamic_filters_table_chart, chart)

    with tab_text:
        st.write("Not Implemented")

    with tab_steps:
        with st.expander("spec"):
            st.write(chart_spec)


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
