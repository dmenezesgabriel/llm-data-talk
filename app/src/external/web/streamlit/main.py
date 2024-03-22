import logging

import streamlit as st
from src.common.utils.performance import log_time
from src.communication.controllers.llm import LLMController
from src.config import get_config
from src.external.llm.langchain.helpers.text import TextHelper
from src.external.llm.langchain.repository.open_ai import OpenAiRepository
from src.external.web.streamlit.ui.messages import render_messages

config = get_config()

open_ai_repository = OpenAiRepository(api_key=config.OPENAI_API_KEY)
llm_controller = LLMController(open_ai_repository)

logger = logging.getLogger()


def setup_session_state() -> None:
    if "is_loaded" not in st.session_state:
        st.session_state.is_loaded = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "messages" not in st.session_state:
        st.session_state.messages = []


@log_time
def setup_vector_store() -> None:
    with open("./data/schema.sql") as f:
        schema = f.read()

    text_chunks = TextHelper.get_text_chunks(schema)
    st.session_state.vector_store = llm_controller.create_vector_store(
        text_chunks
    )
    st.session_state.is_loaded = True


def handle_user_input(user_question: str) -> None:
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )
    retriever = st.session_state.vector_store.as_retriever()

    sql_response = llm_controller.get_sql(
        {"question": user_question}, retriever
    )
    chart_response = llm_controller.get_chart(
        {"question": user_question}, retriever
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

    setup_session_state()

    if not st.session_state.is_loaded:
        logger.info("setup vector store")
        with st.status("loading..."):
            st.write("Waiting for vector store")
            setup_vector_store()

    if user_question := st.chat_input("You: "):
        handle_user_input(user_question)

    if st.session_state.messages:
        render_messages()
