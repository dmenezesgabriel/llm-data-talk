from typing import Callable

import pandas as pd  # type: ignore
import streamlit as st
from src.common.utils.database import get_database_connection
from src.common.utils.dataframe import interpolate_template
from src.config import get_config
from src.external.web.streamlit.ui.dynamic_dataframe_filters import (
    dynamic_dataframe_filter_ui,
)

config = get_config()


def format_user_message(message_content, message_index) -> None:
    st.markdown(message_content)


def format_assistant_message(message_content, message_index) -> None:
    chart_spec = message_content["chart"]

    conn = get_database_connection()
    chart = chart_spec["chart_spec"]
    sql = chart_spec["sql_query"]
    template = chart_spec["text_response"]
    df = pd.read_sql_query(sql, conn)

    tab_titles = ["SQL", "Table", "Chart", "Text", "Steps"]
    tab_sql, tab_table, tab_chart, tab_text, tab_steps = st.tabs(tab_titles)

    with tab_sql:
        st.code(sql, language="sql")

    with tab_table:
        dynamic_filters_table = dynamic_dataframe_filter_ui(
            df, key=f"table_{message_index}"
        )
        st.dataframe(dynamic_filters_table)

    with tab_chart:
        dynamic_filters_table_chart = dynamic_dataframe_filter_ui(
            df, key=f"chart_{message_index}"
        )
        st.vega_lite_chart(dynamic_filters_table_chart, chart)

    with tab_text:
        st.markdown(interpolate_template(template, df))

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
    for index, message in enumerate(st.session_state.messages):
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            message_formatter_factory(role)(content, index)
