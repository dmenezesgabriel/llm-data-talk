from typing import Any, Dict

from src.common.interfaces.llm_repository import LLMRepositoryInterface
from src.communication.gateway.llm import LLMGateway
from src.core.use_cases.llm import LLMUseCases


class LLMController:

    def __init__(self, llm_repository: LLMRepositoryInterface) -> None:
        self.llm_repository = llm_repository

    def get_sql(self, _input: Dict[str, Any], retriever: Any) -> str:
        llm_gateway = LLMGateway(self.llm_repository)
        return LLMUseCases.get_sql(
            _input=_input, retriever=retriever, llm_gateway=llm_gateway
        )

    def get_chart(
        self, _input: Dict[str, Any], retriever: Any, conn: Any
    ) -> Dict[str, Any]:
        llm_gateway = LLMGateway(self.llm_repository)
        return LLMUseCases.get_chart(
            _input=_input,
            retriever=retriever,
            conn=conn,
            llm_gateway=llm_gateway,
        )

    def create_vector_store(self, text_chunks: list[str]) -> Any:
        llm_gateway = LLMGateway(self.llm_repository)
        return LLMUseCases.create_vector_store(text_chunks, llm_gateway)
