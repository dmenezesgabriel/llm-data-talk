from typing import Any, Dict, List

from src.common.interfaces.llm_repository import LLMRepository


class LLMGateway:

    def __init__(self, llm_repository: LLMRepository) -> None:
        self._llm_repository = llm_repository

    def get_sql(self, user_question: str, retriever: Any) -> str:
        return self._llm_repository.get_sql(user_question, retriever)

    def get_chart(self, user_question: str, retriever: Any) -> Dict[str, Any]:
        return self._llm_repository.get_chart(user_question, retriever)

    def create_vector_store(self, text_chunks: List[str]) -> Any:
        return self._llm_repository.create_vector_store(text_chunks)
