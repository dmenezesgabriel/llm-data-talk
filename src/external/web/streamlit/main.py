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


def handle_user_input(user_question):
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )
    response = llm_controller.get_sql(
        user_question, st.session_state.vector_store.as_retriever()
    )
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                tab_sql, tab_table = st.tabs(["SQL", "Table"])
                with tab_sql:
                    st.code(message["content"])
                with tab_table:
                    sqlite_uri = "./data/Chinook.db"
                    conn = sqlite3.connect(sqlite_uri)
                    df = pd.read_sql_query(message["content"], conn)
                    st.dataframe(df)
            else:
                st.markdown(message["content"])


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
