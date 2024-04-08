from typing import Any, Callable, Dict

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


def _render_sql_code(sql: str) -> None:
    st.code(sql, language="sql")


def _render_table(df: pd.DataFrame, message_index: int) -> None:
    dynamic_filters_table = dynamic_dataframe_filter_ui(
        df, key=f"table_{message_index}"
    )
    st.dataframe(dynamic_filters_table)


def _render_chart(
    df: pd.DataFrame,
    chart: Dict[str, Any],
    message_index: int,
) -> None:
    dynamic_filters_table_chart = dynamic_dataframe_filter_ui(
        df, key=f"chart_{message_index}"
    )
    st.vega_lite_chart(dynamic_filters_table_chart, chart)


def _render_text(message_content: str) -> None:
    st.markdown(message_content)


def _response_factory(
    response_format, sql, df, chart, message_content, message_index
):
    response_formats = {
        "SQL String": lambda: _render_sql_code(sql),
        "Table": lambda: _render_table(df, message_index),
        "Chart": lambda: _render_chart(df, chart, message_index),
        "Text": lambda: _render_text(message_content),
    }
    return response_formats[response_format]()


def format_assistant_message(message_content, message_index) -> None:
    chart_spec = message_content["chart"]

    conn = get_database_connection()
    chart = chart_spec["chart_spec"]
    sql = chart_spec["sql_query"]
    template = chart_spec["text_response"]
    response_format = chart_spec["response_format"]

    df = pd.read_sql_query(sql, conn)

    index_factor_modifier = 33.33

    _response_factory(
        response_format=response_format,
        sql=sql,
        df=df,
        chart=chart,
        message_content=interpolate_template(template, df),
        message_index=message_index / index_factor_modifier,
    )

    with st.expander("Another analysis options"):
        tab_titles = ["SQL", "Table", "Chart", "Text", "Steps"]
        tab_sql, tab_table, tab_chart, tab_text, tab_steps = st.tabs(
            tab_titles
        )

        with tab_sql:
            _render_sql_code(sql=sql)

        with tab_table:
            _render_table(df=df, message_index=message_index)

        with tab_chart:
            _render_chart(df=df, chart=chart, message_index=message_index)

        with tab_text:
            _render_text(
                message_content=interpolate_template(template, df),
            )

        with tab_steps:
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
