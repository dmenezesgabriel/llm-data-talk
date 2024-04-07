from operator import itemgetter
from typing import Any

from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from src.common.utils.performance import log_time
from src.external.llm.langchain.chains.base import BaseChain
from src.external.llm.langchain.templates import (
    NATURAL_LANGUAGE_ANALYSIS_GENERATION_TEMPLATE,
)


class TextAnalysisGenerationChain(BaseChain):

    def __init__(self, llm: Any, retriever: BaseRetriever) -> None:
        super().__init__(llm, retriever)

    @log_time
    def chain(self) -> LLMChain:
        prompt = PromptTemplate(
            template=NATURAL_LANGUAGE_ANALYSIS_GENERATION_TEMPLATE,
            input_variables=["schema", "question"],
        )

        return (
            {
                "context": itemgetter("question") | self._retriever,
                "question": itemgetter("question"),
                "schema": itemgetter("schema"),
            }
            | prompt
            | self._llm
            | StrOutputParser()
        )


if __name__ == "__main__":
    from langchain_community.vectorstores import Chroma
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from src.common.utils.database import get_database_connection
    from src.config import get_config
    from src.external.llm.langchain.helpers.text import TextHelper

    conn = get_database_connection()
    with open("./data/schema.sql") as f:
        schema = f.read()

    config = get_config()
    text_chunks = TextHelper.get_text_chunks(schema)
    llm = ChatOpenAI(api_key=config.OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY)
    vector_store = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    retriever = vector_store.as_retriever()

    # ======================================================================= #
    text_response = TextAnalysisGenerationChain(llm=llm, retriever=retriever)
    text_response_result = text_response.chain().invoke(
        input={
            "question": ("who are the top 5 artists by sales?"),
            "schema": (
                "|            | 0       |"
                "|:-----------|:--------|"
                "| Name       | object  |"
                "| TotalSales | float64 |",
            ),
        }
    )
    print(50 * "=")
    print(text_response_result)
    print(50 * "=")
