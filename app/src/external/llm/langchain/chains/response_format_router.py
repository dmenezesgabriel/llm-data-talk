from textwrap import dedent

from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from src.common.utils.performance import log_time
from src.external.llm.langchain.chains.base import BaseChain
from src.external.llm.langchain.models import ResponseTypeRouteQuery


class ResponseFormatRouteChain(BaseChain):

    def __init__(self, llm) -> None:
        super().__init__(llm)

    @log_time
    def chain(self) -> LLMChain:
        structured_llm = self._llm.with_structured_output(
            ResponseTypeRouteQuery
        )
        system = dedent(
            """
            You are an expert at routing a user question to the appropriate
            response type. Based on the analysis the question is
            referring to, route to the relevant response type.
            """
        )

        prompt = ChatPromptTemplate.from_messages(
            [("system", system), ("human", "{question}")]
        )

        return prompt | structured_llm


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
    response_formate_route_chain = ResponseFormatRouteChain(llm=llm)
    response_formate_route_chain_result = (
        response_formate_route_chain.chain().invoke(
            input={"question": "what is the top 10 artists by sales?"}
        )
    )
    print(50 * "=")
    print(response_formate_route_chain_result)
    print(50 * "=")
