from typing import Any

from langchain.chains.llm import LLMChain
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough

from src.external.llm.templates import sql_template, vega_spec_template

functions = [
    {
        "name": "vega_spec",
        "description": "Vega lite json chart spec",
        "parameters": {
            "type": "object",
            "properties": {
                "mark": {"type": "object", "description": "vega lite mark"},
                "encoding": {
                    "type": "object",
                    "description": "vega lite encoding",
                },
            },
            "required": ["mark", "encoding"],
        },
    }
]


def get_sql_chain(llm: Any, retriever: BaseRetriever) -> LLMChain:
    prompt = ChatPromptTemplate.from_template(sql_template)

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )


def get_vega_chain(llm: Any, retriever: BaseRetriever) -> LLMChain:
    prompt = ChatPromptTemplate.from_template(template=vega_spec_template)

    context_sql_chain = get_sql_chain(llm, retriever)

    return (
        {
            "query": lambda x: context_sql_chain,
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm.bind(
            stop=["\nVega-Lite Spec:"],
            function_call={"name": "vega_spec"},
            functions=functions,
        )
        | JsonOutputFunctionsParser()
        # | JsonOutputParser()
    )