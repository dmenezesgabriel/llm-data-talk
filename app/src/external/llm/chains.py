from operator import itemgetter
from typing import Any

from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from src.common.utils.performance import log_time
from src.external.llm.templates import (
    chart_spec,
    entity_extraction,
    sql_template,
)


@log_time
def get_sql_chain(llm: Any, retriever: BaseRetriever) -> LLMChain:
    prompt = ChatPromptTemplate.from_template(sql_template)

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
    prompt = ChatPromptTemplate.from_template(template=entity_extraction)

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
def get_chart_chain(llm: Any, retriever: BaseRetriever) -> LLMChain:
    prompt = ChatPromptTemplate.from_template(template=chart_spec)

    context_sql_chain = get_sql_chain(llm, retriever)

    return (
        {
            "query": lambda x: context_sql_chain,
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | JsonOutputParser()
    )
