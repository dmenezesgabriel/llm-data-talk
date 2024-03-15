class LLMUseCases:
    @staticmethod
    def get_sql(user_question: str, retriever, llm_gateway) -> str:
        llm_gateway.get_sql(user_question=user_question, retriever=retriever)

    @staticmethod
    def create_vector_store(text_chunks: list[str], llm_gateway):
        return llm_gateway.create_vector_store(text_chunks=text_chunks)

    @staticmethod
    def create_conversational_chain(vector_store, llm_gateway):
        return llm_gateway.create_conversational_chain(
            vector_store=vector_store
        )
