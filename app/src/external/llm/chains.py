from operator import itemgetter
from typing import Any

from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from src.common.utils.performance import log_time
from src.external.llm.templates import sql_template, vega_spec_template


@log_time
def get_sql_chain(llm: Any, retriever: BaseRetriever) -> LLMChain:
    prompt = ChatPromptTemplate.from_template(sql_template)

    return (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )


@log_time
def get_vega_chain(llm: Any, retriever: BaseRetriever) -> LLMChain:
    prompt = ChatPromptTemplate.from_template(template=vega_spec_template)

    context_sql_chain = get_sql_chain(llm, retriever)

    return (
        {
            "query": lambda x: context_sql_chain,
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | prompt
        | llm.bind(
            stop=["\nVega-Lite Spec:"],
        )
        | JsonOutputParser()
    )
