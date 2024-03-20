from abc import ABC, abstractmethod
from typing import Any, Dict


class LLMGatewayInterface(ABC):

    @abstractmethod
    def get_sql(
        self, user_question: str, conversation_history: list, retriever
    ) -> str:
        raise NotImplementedError()

    @abstractmethod
    def get_chart(
        self, user_question: str, conversation_history: list, retriever
    ) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def create_vector_store(self, text_chunks) -> Any:
        raise NotImplementedError()
