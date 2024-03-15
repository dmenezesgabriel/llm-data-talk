from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class OpenAiRepository:
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._llm = ChatOpenAI(api_key=self._api_key)

    def get_sql(self, user_question: str, retriever) -> str:
        template = """Write a SQL query that would answer the user's question,
        based on the context:
        {context}

        Question: {question}
        SQL Query:"""

        prompt = ChatPromptTemplate.from_template(template)

        context_sql_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self._llm.bind(stop=["\nSQLResult:"])
            | StrOutputParser()
        )

        return context_sql_chain.invoke(user_question)

    def create_vector_store(self, text_chunks):
        embeddings = OpenAIEmbeddings(openai_api_key=self._api_key)
        return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    def create_conversational_chain(self, vector_store):
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        return ConversationalRetrievalChain.from_llm(
            llm=self._llm, retriever=vector_store.as_retriever(), memory=memory
        )
