from typing import Any, Dict

from src.common.interfaces.llm_repository import LLMRepositoryInterface
from src.communication.gateway.llm import LLMGateway
from src.core.use_cases.llm import LLMUseCases


class LLMController:

    def __init__(self, llm_repository: LLMRepositoryInterface) -> None:
        self.llm_repository = llm_repository

    def get_sql(self, user_question: str, retriever: Any) -> str:
        llm_gateway = LLMGateway(self.llm_repository)
        return LLMUseCases.get_sql(user_question, retriever, llm_gateway)

    def get_chart(self, user_question: str, retriever: Any) -> Dict[str, Any]:
        llm_gateway = LLMGateway(self.llm_repository)
        return LLMUseCases.get_chart(user_question, retriever, llm_gateway)

    def create_vector_store(self, text_chunks: list[str]) -> Any:
        llm_gateway = LLMGateway(self.llm_repository)
        return LLMUseCases.create_vector_store(text_chunks, llm_gateway)
