from typing import Any, Dict, List

from src.common.interfaces.llm_gateway import LLMGatewayInterface
from src.common.interfaces.llm_repository import LLMRepositoryInterface


class LLMGateway(LLMGatewayInterface):

    def __init__(self, llm_repository: LLMRepositoryInterface) -> None:
        self._llm_repository = llm_repository

    def get_sql(self, _input: Dict[str, Any], retriever: Any) -> str:
        return self._llm_repository.get_sql(_input=_input, retriever=retriever)

    def get_chart(
        self, _input: Dict[str, Any], retriever: Any, conn: Any
    ) -> Dict[str, Any]:
        return self._llm_repository.get_chart(
            _input=_input, retriever=retriever, conn=conn
        )

    def create_vector_store(self, text_chunks: List[str]) -> Any:
        return self._llm_repository.create_vector_store(text_chunks)
