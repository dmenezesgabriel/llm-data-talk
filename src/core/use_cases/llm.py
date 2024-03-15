from typing import Any, Dict

from src.common.interfaces.llm_gateway import LLMGatewayInterface


class LLMUseCases:
    @staticmethod
    def get_sql(
        user_question: str, retriever: Any, llm_gateway: LLMGatewayInterface
    ) -> str:
        return llm_gateway.get_sql(
            user_question=user_question, retriever=retriever
        )

    def get_chart(
        user_question: str, retriever: Any, llm_gateway: LLMGatewayInterface
    ) -> Dict[str, Any]:
        return llm_gateway.get_chart(
            user_question=user_question, retriever=retriever
        )

    @staticmethod
    def create_vector_store(
        text_chunks: list[str], llm_gateway: LLMGatewayInterface
    ) -> Any:
        return llm_gateway.create_vector_store(text_chunks=text_chunks)
