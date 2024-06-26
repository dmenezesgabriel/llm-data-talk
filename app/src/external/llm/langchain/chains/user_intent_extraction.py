from operator import itemgetter
from typing import Any

from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from src.common.utils.performance import log_time
from src.external.llm.langchain.chains.base import BaseChain
from src.external.llm.langchain.templates import (
    USER_INTENT_EXTRACTION_TEMPLATE,
)


class UserIntentExtractionChain(BaseChain):
    def __init__(self, llm: Any) -> None:
        super().__init__(llm)

    @log_time
    def chain(self) -> LLMChain:
        prompt = PromptTemplate(
            template=USER_INTENT_EXTRACTION_TEMPLATE,
            input_variables=["context", "question"],
        )

        return (
            {
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
    user_intent = UserIntentExtractionChain(llm=llm)
    user_intent_result = user_intent.chain().invoke(
        input={"question": "calculate the total iron maiden artist sales"}
    )
    print(50 * "=")
    print(user_intent_result)
    print(50 * "=")
