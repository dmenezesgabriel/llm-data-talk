import logging
from typing import Any, Dict

from langchain_community.callbacks import get_openai_callback
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.common.interfaces.llm_repository import LLMRepositoryInterface
from src.config import get_config
from src.external.llm.langchain.chains.sql_generation import SQLGenerationChain
from src.external.llm.langchain.graphs.chart import ChartGraph

config = get_config()

logger = logging.getLogger()


class OpenAiRepository(LLMRepositoryInterface):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._llm = ChatOpenAI(api_key=self._api_key)  # type: ignore

    def get_sql(self, _input: Dict[str, Any], retriever: Any) -> str:
        sql_chain = SQLGenerationChain(llm=self._llm, retriever=retriever)
        with get_openai_callback() as cb:
            result = sql_chain.chain().invoke(input=_input)
            logger.info(f"Used a total of {cb.total_tokens} tokens")
        return result

    def get_chart(
        self, _input: Dict[str, Any], retriever: Any, conn: Any
    ) -> Dict[str, Any]:
        chart_spec = ChartGraph(self._llm, retriever, conn)
        with get_openai_callback() as cb:
            result = chart_spec.graph().invoke(input=_input)
            logger.info(f"Used a total of {cb.total_tokens} tokens")
        return result

    def create_vector_store(self, text_chunks):
        embeddings = OpenAIEmbeddings(api_key=self._api_key)
        return Chroma.from_texts(texts=text_chunks, embedding=embeddings)
