from typing import Any, Dict, List

from src.common.interfaces.llm_gateway import LLMGatewayInterface
from src.common.interfaces.llm_repository import LLMRepositoryInterface


class LLMGateway(LLMGatewayInterface):

    def __init__(self, llm_repository: LLMRepositoryInterface) -> None:
        self._llm_repository = llm_repository

    def get_sql(
        self, user_question: str, conversation_history: list, retriever: Any
    ) -> str:
        return self._llm_repository.get_sql(
            user_question=user_question,
            conversation_history=conversation_history,
            retriever=retriever,
        )

    def get_chart(
        self, user_question: str, conversation_history: list, retriever: Any
    ) -> Dict[str, Any]:
        return self._llm_repository.get_chart(
            user_question=user_question,
            conversation_history=conversation_history,
            retriever=retriever,
        )

    def create_vector_store(self, text_chunks: List[str]) -> Any:
        return self._llm_repository.create_vector_store(
            text_chunks=text_chunks
        )
