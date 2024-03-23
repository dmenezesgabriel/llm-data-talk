from abc import ABC, abstractmethod
from typing import Any, Dict


class LLMRepositoryInterface(ABC):

    @abstractmethod
    def get_sql(self, user_question: Dict[str, Any], retriever: Any) -> str:
        raise NotImplementedError()

    @abstractmethod
    def get_chart(
        self, user_question: Dict[str, Any], retriever: Any, conn: Any
    ) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def create_vector_store(self, text_chunks) -> Any:
        raise NotImplementedError()
