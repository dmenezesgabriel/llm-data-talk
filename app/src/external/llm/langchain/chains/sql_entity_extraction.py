from operator import itemgetter
from typing import Any

from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from src.common.utils.performance import log_time
from src.external.llm.langchain.chains.base import BaseChain
from src.external.llm.langchain.chains.sql_generation import SQLGenerationChain
from src.external.llm.langchain.templates import SQL_ENTITY_EXTRACTION_TEMPLATE


class SQLEntityExtractionChain(BaseChain):
    def __init__(self, llm: Any, retriever: BaseRetriever) -> None:
        super().__init__(llm, retriever)

    @log_time
    def chain(self) -> LLMChain:
        prompt = PromptTemplate(
            template=SQL_ENTITY_EXTRACTION_TEMPLATE,
            input_variables=["query", "question"],
        )

        context_sql_chain = SQLGenerationChain(
            self._llm, self._retriever
        ).chain()

        return (
            {
                "query": lambda x: context_sql_chain,
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
    entity_extraction = SQLEntityExtractionChain(llm, retriever)
    entity_extraction_result = entity_extraction.chain().invoke(
        input={"question": "what are the total iron maiden artist sales?"}
    )
    print(50 * "=")
    print(entity_extraction_result)
    print(50 * "=")
