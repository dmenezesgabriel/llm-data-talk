from operator import itemgetter
from textwrap import dedent
from typing import Any, Dict, Optional

from langchain.chains.llm import LLMChain
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from src.common.utils.dataframe import query_to_pandas_schema
from src.common.utils.performance import log_time
from src.external.llm.langchain.models.router import ResponseTypeRouteQuery
from src.external.llm.langchain.templates import (
    chart_template,
    intent_extraction_template,
    sql_entity_extraction_template,
    sql_template,
)


class BaseChain:
    def __init__(
        self, llm: Any, retriever: Optional[BaseRetriever] = None
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._intermediates: Dict[str, Any] = {}

    def _save_intermediates(self, value: Any, **kwargs: Any) -> Any:
        key = kwargs.get("key")
        if key is None:
            raise ValueError("Key must be provided to save intermediates.")
        self._intermediates[key] = value
        return value

    def _post_process(self, value: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": value, "intermediates": self._intermediates}


class ResponseTypeRouteChain(BaseChain):

    def __init__(self, llm) -> None:
        super().__init__(llm)

    @log_time
    def chain(self) -> LLMChain:
        structured_llm = self._llm.with_structured_output(
            ResponseTypeRouteQuery
        )
        system = dedent(
            """
            You are an expert at routing a user question to the appropriate
            response type. Based on the analysis the question is
            referring to, route to the relevant response type.
            """
        )

        prompt = ChatPromptTemplate.from_messages(
            [("system", system), ("human", "{question}")]
        )

        return prompt | structured_llm


class UserIntentChain(BaseChain):
    def __init__(self, llm: Any, retriever) -> None:
        super().__init__(llm, retriever)

    @log_time
    def chain(self) -> LLMChain:
        prompt = PromptTemplate(
            template=intent_extraction_template,
            input_variables=["context", "question"],
        )

        return (
            {
                "question": itemgetter("question"),
            }
            | prompt
            | self._llm
            | JsonOutputParser()
        )


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
            template=sql_entity_extraction_template,
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
            template=chart_template,
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
