from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.external.llm.templates import sql_template


def get_sql_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template(sql_template)

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )
