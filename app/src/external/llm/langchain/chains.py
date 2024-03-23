from operator import itemgetter
from typing import Any, Dict

from langchain.chains.llm import LLMChain
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from src.common.utils.dataframe import query_to_pandas_schema
from src.common.utils.performance import log_time
from src.external.llm.langchain.templates import (
    chart_spec,
    entity_extraction,
    sql_template,
)


class SQLChain:
    def __init__(self, llm: Any, retriever: BaseRetriever):
        self._llm = llm
        self._retriever = retriever

    @log_time
    def chain(self):
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


class SQLEntityExtractionChain:

    def __init__(self, llm: Any, retriever: BaseRetriever):
        self._llm = llm
        self._retriever = retriever

    @log_time
    def chain(self):
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


class ChartChain:
    def __init__(self, llm: Any, retriever: BaseRetriever, conn: Any):
        self._llm = llm
        self._retriever = retriever
        self._conn = conn

    @log_time
    def chain(self):
        prompt = PromptTemplate(
            template=chart_spec,
            input_variables=["query", "question"],
        )
        context_sql_chain = SQLChain(self._llm, self._retriever).chain()

        def _query_to_pandas_schema(sql: str) -> str:
            return query_to_pandas_schema(sql, self._conn)

        def _result(x):
            return {"x": x}

        return (
            {
                "sql": context_sql_chain,
                "schema": lambda x: context_sql_chain
                | RunnableLambda(func=_query_to_pandas_schema),
                "question": itemgetter("question"),
            }
            | prompt
            | self._llm
            | JsonOutputParser()
            | RunnableLambda(_result)
        )
