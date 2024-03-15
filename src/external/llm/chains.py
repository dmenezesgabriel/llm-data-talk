from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.external.llm.templates import sql_template, vega_spec_template


def get_sql_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template(sql_template)

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )


def get_vega_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template(template=vega_spec_template)

    context_sql_chain = get_sql_chain(llm, retriever)

    return (
        {
            "query": lambda x: context_sql_chain,
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm.bind(stop=["\nVega-Lite Spec:"])
        | JsonOutputParser()
    )
