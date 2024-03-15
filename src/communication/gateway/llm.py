class LLMGateway:
    def __init__(
        self, llm_repository, vector_store=None, conversational_chain=None
    ):
        self._llm_repository = llm_repository
        self._vector_store = vector_store
        self._conversational_chain = conversational_chain

    @property
    def vector_store(self):
        return self._vector_store

    @vector_store.setter
    def vector_store(self, vector_store):
        self._vector_store = vector_store

    @property
    def conversational_chain(self):
        return self._conversational_chain

    @conversational_chain.setter
    def conversational_chain(self, conversational_chain):
        self._conversational_chain = conversational_chain

    def get_sql(self, user_question: str, retriever) -> str:
        return self._llm_repository.get_sql(user_question, retriever)

    def create_vector_store(self, text_chunks):
        return self._llm_repository.create_vector_store(text_chunks)

    def create_conversational_chain(self, vector_store):
        return self._llm_repository.create_conversational_chain(vector_store)
