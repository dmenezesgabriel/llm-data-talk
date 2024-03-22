from typing import Any, Dict, cast

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.common.interfaces.llm_repository import LLMRepositoryInterface
from src.external.llm.chains import get_chart_chain, get_sql_chain


class OpenAiRepository(LLMRepositoryInterface):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._llm = ChatOpenAI(api_key=self._api_key)

    def get_sql(self, _input: Dict[str, Any], retriever) -> str:
        sql_chain = get_sql_chain(self._llm, retriever)
        return sql_chain.invoke(input=_input)

    def get_chart(self, _input: Dict[str, Any], retriever) -> Dict[str, Any]:
        sql_chain = get_chart_chain(self._llm, retriever)
        return sql_chain.invoke(input=_input)

    def create_vector_store(self, text_chunks):
        embeddings = OpenAIEmbeddings(openai_api_key=self._api_key)
        return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


if __name__ == "__main__":
    from src.config import get_config
    from src.external.llm.helpers.text import TextHelper

    with open("./data/schema.sql") as f:
        schema = f.read()

    config = get_config()
    repository = OpenAiRepository(api_key=config.OPENAI_API_KEY)
    text_chunks = TextHelper.get_text_chunks(schema)
    vector_store = repository.create_vector_store(text_chunks=text_chunks)

    sql_result = repository.get_chart(
        _input={"question": "What are the top 10 artists by sales"},
        retriever=vector_store.as_retriever(),
    )
    print(sql_result)
