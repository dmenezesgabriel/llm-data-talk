from operator import itemgetter
from typing import Any, Union

from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.retrievers import BaseRetriever
from src.common.utils.performance import log_time
from src.external.llm.templates import (
    chart_spec,
    entity_extraction,
    sql_template,
)


class ChartSpec(BaseModel):
    chart_type: Union[str, None] = Field(description="Chart type")
    x_axis: Union[str, None] = Field(description="X axis")
    y_axis: Union[str, None] = Field(description="Y axis")


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
def get_chart_chain(llm: Any, retriever: BaseRetriever) -> LLMChain:
    parser = PydanticOutputParser(pydantic_object=ChartSpec)

    prompt = PromptTemplate(
        template=chart_spec,
        input_variables=["query", "question"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )
    context_sql_chain = get_sql_chain(llm, retriever)

    return (
        {
            "query": lambda x: context_sql_chain,
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | parser
    )
