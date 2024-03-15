import sqlite3

import pandas as pd
import streamlit as st

from src.communication.controllers.llm import LLMController
from src.config import get_config
from src.external.llm.helpers.text import TextHelper
from src.external.llm.repository.open_ai import OpenAiRepository

config = get_config()
open_ai_repository = OpenAiRepository(api_key=config.OPENAI_API_KEY)
llm_controller = LLMController(open_ai_repository)


def setup():
    st.write("Waiting for vector store")
    with open("./data/schema.sql") as f:
        schema = f.read()

    text_chunks = TextHelper.get_text_chunks(schema)
    st.session_state.vector_store = llm_controller.create_vector_store(
        text_chunks
    )
    st.session_state.is_loaded = True


def render_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                sql_code = message["content"]["sql"]
                chart_spec = message["content"]["chart"]
                tab_titles = ["SQL", "Table", "Chart"]
                tab_sql, tab_table, tab_chart = st.tabs(tab_titles)
                with tab_sql:
                    st.code(sql_code, language="sql")
                with tab_table:
                    conn = sqlite3.connect(config.DATABASE_URI)
                    df = pd.read_sql_query(sql_code, conn)
                    st.dataframe(df)
                with tab_chart:
                    with st.expander("chart spec"):
                        st.write(chart_spec)
                    st.vega_lite_chart(data=df, spec=chart_spec)
            else:
                st.markdown(message["content"])


def handle_user_input(user_question):
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )
    sql_response = llm_controller.get_sql(
        user_question, st.session_state.vector_store.as_retriever()
    )
    chart_response = llm_controller.get_chart(
        user_question, st.session_state.vector_store.as_retriever()
    )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": {"sql": sql_response, "chart": chart_response},
        }
    )


def main() -> None:
    st.set_page_config("Chat with your data", page_icon=":chart")
    st.header("Chat with your data")

    if "is_loaded" not in st.session_state:
        st.session_state.is_loaded = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.is_loaded:
        with st.status("loading..."):
            setup()

    if user_question := st.chat_input("You: "):
        handle_user_input(user_question)

    if st.session_state.messages:
        render_messages()
