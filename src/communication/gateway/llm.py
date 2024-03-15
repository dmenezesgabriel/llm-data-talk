class LLMGateway:
    def __init__(self, llm_repository):
        self._llm_repository = llm_repository

    def get_sql(self, user_question: str, retriever) -> str:
        return self._llm_repository.get_sql(user_question, retriever)

    def create_vector_store(self, text_chunks):
        return self._llm_repository.create_vector_store(text_chunks)
