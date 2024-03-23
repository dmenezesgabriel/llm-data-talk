from typing import Any, Dict

from src.common.interfaces.llm_gateway import LLMGatewayInterface


class LLMUseCases:
    @staticmethod
    def get_sql(
        _input: Dict[str, Any],
        retriever: Any,
        llm_gateway: LLMGatewayInterface,
    ) -> str:
        return llm_gateway.get_sql(_input=_input, retriever=retriever)

    def get_chart(
        _input: Dict[str, Any],
        retriever: Any,
        conn: Any,
        llm_gateway: LLMGatewayInterface,
    ) -> Dict[str, Any]:
        return llm_gateway.get_chart(
            _input=_input, retriever=retriever, conn=conn
        )

    @staticmethod
    def create_vector_store(
        text_chunks: list[str], llm_gateway: LLMGatewayInterface
    ) -> Any:
        return llm_gateway.create_vector_store(text_chunks=text_chunks)
