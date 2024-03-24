from operator import itemgetter
from typing import Any, Dict

from langchain.chains.llm import LLMChain
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from src.common.utils.dataframe import query_to_pandas_schema
from src.common.utils.performance import log_time
from src.external.llm.langchain.templates import (chart_spec,
                                                  entity_extraction,
                                                  sql_template)


class BaseChain:
    def __init__(self, llm: Any, retriever: BaseRetriever) -> None:
        self._llm = llm
        self._retriever = retriever
        self._intermediates: Dict[str, Any] = {}

    def _save_intermediates(self, value: Any, **kwargs: Any) -> Any:
        key = kwargs.get("key")
        self._intermediates[key] = value
        return value

    def _post_process(self, value: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": value, "intermediates": self._intermediates}


class SQLChain(BaseChain):
    def __init__(self, llm: Any, retriever: BaseRetriever) -> None:
        super().__init__(llm, retriever)

    @log_time
    def chain(self) -> LLMChain:
        prompt = PromptTemplate(
            template=sql_template,
            input_variables=["context", "question"],
        )

        return (
            {
                "context": itemgetter("question") | self._retriever,
                "question": itemgetter("question"),
            }
            | prompt
            | self._llm
            | StrOutputParser()
        )


class SQLEntityExtractionChain(BaseChain):
    def __init__(self, llm: Any, retriever: BaseRetriever) -> None:
        super().__init__(llm, retriever)

    @log_time
    def chain(self) -> LLMChain:
        prompt = PromptTemplate(
            template=entity_extraction,
            input_variables=["query", "question"],
        )

        context_sql_chain = SQLChain(self._llm, self._retriever).chain()

        return (
            {
                "query": lambda x: context_sql_chain,
                "question": itemgetter("question"),
            }
            | prompt
            | self._llm
            | JsonOutputParser()
        )


class ChartChain(BaseChain):
    def __init__(self, llm: Any, retriever: BaseRetriever, conn: Any) -> None:
        super().__init__(llm, retriever)
        self._conn = conn

    @log_time
    def chain(self) -> LLMChain:
        prompt = PromptTemplate(
            template=chart_spec,
            input_variables=["query", "question"],
        )
        context_sql_chain = SQLChain(self._llm, self._retriever).chain()

        def _query_to_pandas_schema(sql: str) -> str:
            return query_to_pandas_schema(sql, self._conn)

        return (
            {
                "schema": lambda _: context_sql_chain
                | RunnableLambda(self._save_intermediates).bind(key="sql")
                | RunnableLambda(func=_query_to_pandas_schema),
                "question": itemgetter("question"),
            }
            | prompt
            | self._llm
            | JsonOutputParser()
            | RunnableLambda(self._post_process)
        )
