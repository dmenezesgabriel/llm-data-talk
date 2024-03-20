if __name__ == "__main__":
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from src.config import get_config
    from src.external.llm.chains import get_sql_chain

    config = get_config()
    llm = ChatOpenAI(api_key=config.OPENAI_API_KEY)
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )

    with open("data/schema.sql") as f:
        schema = f.read()

    text_chunks = text_splitter.split_text(schema)
    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY),
    )
    chain = get_sql_chain(llm, vector_store.as_retriever())
    chain.invoke(
        {"question": "how many albuns there are?", "conversation_history": []}
    )
