from typing import Callable

import pandas as pd
import streamlit as st

from src.config import get_config
from src.external.web.streamlit.utils import get_database_connection

config = get_config()


def format_user_message(message_content) -> None:
    st.markdown(message_content)


def format_assistant_message(message_content) -> None:
    sql_code = message_content["sql"]
    chart_spec = message_content["chart"]

    tab_titles = ["SQL", "Table", "Chart"]
    tab_sql, tab_table, tab_chart = st.tabs(tab_titles)

    with tab_sql:
        st.code(sql_code, language="sql")

    with tab_table:
        conn = get_database_connection()
        df = pd.read_sql_query(sql_code, conn)
        st.dataframe(df)

    with tab_chart:
        with st.expander("chart spec"):
            st.write(chart_spec)
        st.vega_lite_chart(data=df, spec=chart_spec)


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
