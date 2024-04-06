from operator import itemgetter
from typing import Any

from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from src.common.utils.performance import log_time
from src.external.llm.langchain.chains.base import BaseChain
from src.external.llm.langchain.templates import (
    COMPOUND_QUESTION_EXTRACTOR_TEMPLATE,
)


class CompoundQuestionExtractorChain(BaseChain):
    def __init__(self, llm: Any, retriever: BaseRetriever) -> None:
        super().__init__(llm, retriever)

    @log_time
    def chain(self) -> LLMChain:
        prompt = PromptTemplate(
            template=COMPOUND_QUESTION_EXTRACTOR_TEMPLATE,
            input_variables=["context", "question"],
        )

        return (
            {
                "context": itemgetter("question") | self._retriever,
                "question": itemgetter("question"),
            }
            | prompt
            | self._llm
            | JsonOutputParser()
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
    sql_chain = CompoundQuestionExtractorChain(llm=llm, retriever=retriever)
    for question in [
        "what are the top 20 artists by sales?",
        "Plot a chart with the top 10 artists by sales with different colors"
        "for each artist",
        "which is the artist with most sales?",
        "What is the artist with the most sales and the one with less sales? "
        "Also give me the difference in percentage of sales between the two",
    ]:
        sql_chain_result = sql_chain.chain().invoke(
            input={"question": question}
        )
        print(50 * "=")
        print(sql_chain_result)
        print(50 * "=")
