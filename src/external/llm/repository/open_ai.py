from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.external.llm.chains import get_sql_chain


class OpenAiRepository:
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._llm = ChatOpenAI(api_key=self._api_key)

    def get_sql(self, user_question: str, retriever) -> str:
        sql_chain = get_sql_chain(self._llm, retriever)
        return sql_chain.invoke(user_question)

    def create_vector_store(self, text_chunks):
        embeddings = OpenAIEmbeddings(openai_api_key=self._api_key)
        return FAISS.from_texts(texts=text_chunks, embedding=embeddings)
