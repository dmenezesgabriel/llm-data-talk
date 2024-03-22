from operator import itemgetter
from typing import Any

from langchain.chains.llm import LLMChain
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from src.common.utils.dataframe import query_to_pandas_schema
from src.common.utils.performance import log_time
from src.external.llm.templates import (
    chart_spec,
    entity_extraction,
    sql_template,
)


@log_time
def get_sql_chain(llm: Any, retriever: BaseRetriever) -> LLMChain:
    prompt = PromptTemplate(
        template=sql_template,
        input_variables=["context", "question"],
    )

    return (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


@log_time
def get_entity_extraction_chain(
    llm: Any, retriever: BaseRetriever
) -> LLMChain:
    prompt = PromptTemplate(
        template=entity_extraction,
        input_variables=["query", "question"],
    )

    context_sql_chain = get_sql_chain(llm, retriever)

    return (
        {
            "query": lambda x: context_sql_chain,
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | JsonOutputParser()
    )


@log_time
def get_chart_chain(llm: Any, retriever: BaseRetriever, conn: Any) -> LLMChain:

    prompt = PromptTemplate(
        template=chart_spec,
        input_variables=["query", "question"],
    )
    context_sql_chain = get_sql_chain(llm, retriever)

    def _query_to_pandas_schema(sql: str) -> str:
        return query_to_pandas_schema(sql, conn)

    return (
        {
            "schema": lambda x: context_sql_chain
            | RunnableLambda(func=_query_to_pandas_schema),
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | JsonOutputParser()
    )
