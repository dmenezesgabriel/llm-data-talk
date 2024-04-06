from typing import Any, Dict

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.common.interfaces.llm_repository import LLMRepositoryInterface
from src.external.llm.langchain.chains.sql_generation import SQLGenerationChain
from src.external.llm.langchain.graphs.chart import ChartGraph


class OpenAiRepository(LLMRepositoryInterface):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._llm = ChatOpenAI(api_key=self._api_key)  # type: ignore

    def get_sql(self, _input: Dict[str, Any], retriever: Any) -> str:
        sql_chain = SQLGenerationChain(llm=self._llm, retriever=retriever)
        return sql_chain.chain().invoke(input=_input)

    def get_chart(
        self, _input: Dict[str, Any], retriever: Any, conn: Any
    ) -> Dict[str, Any]:
        chart_spec = ChartGraph(self._llm, retriever, conn)
        return chart_spec.graph().invoke(input=_input)

    def create_vector_store(self, text_chunks):
        embeddings = OpenAIEmbeddings(api_key=self._api_key)
        return Chroma.from_texts(texts=text_chunks, embedding=embeddings)
