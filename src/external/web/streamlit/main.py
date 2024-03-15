import sqlite3

import streamlit as st

from src.communication.controllers.llm import LLMController
from src.config import get_config
from src.external.llm.helpers.text import TextHelper
from src.external.llm.repository.open_ai import OpenAiRepository

config = get_config()
open_ai_repository = OpenAiRepository(api_key=config.OPENAI_API_KEY)
llm_controller = LLMController(open_ai_repository)


def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state["chat_history"] = response["chat_history"]
    for index, message in enumerate(response["chat_history"]):
        if index % 2 == 0:
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(message.content)


def setup():
    st.write("Waiting for vector store")
    with open("./data/schema.sql") as f:
        schema = f.read()
    text_chunks = TextHelper.get_text_chunks(schema)
    vector_store = vector_store = llm_controller.create_vector_store(
        text_chunks
    )

    st.write("Waiting for conversational chain")
    st.session_state.conversation = llm_controller.create_conversational_chain(
        vector_store
    )
    st.session_state.is_loaded = True


def main() -> None:
    st.set_page_config("Chat with your data", page_icon=":chart")
    st.header("Chat with your data")

    sqlite_uri = "./data/Chinook.db"
    conn = sqlite3.connect(sqlite_uri)

    if "is_loaded" not in st.session_state:
        st.session_state.is_loaded = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if not st.session_state.is_loaded:
        with st.status("loading..."):
            setup()

    if user_question := st.chat_input("You: "):
        handle_user_input(user_question)
