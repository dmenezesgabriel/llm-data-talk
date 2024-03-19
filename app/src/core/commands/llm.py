from typing import Any, Dict


class LLMGetSQLCommand:
    def __init__(self, user_question: str, retriever, llm_controller) -> None:
        self._user_question = user_question
        self._retriever = retriever
        self._llm_controller = llm_controller
        self._cached_result = None

    def execute(self, cached_result=True) -> str:
        if cached_result and self._cached_result:
            return self._cached_result
        self._cached_result = self._llm_controller.get_sql(
            self._user_question, self._retriever
        )
        return self._cached_result


class LLMGetChartCommand:
    def __init__(self, user_question: str, retriever, llm_controller) -> None:
        self._user_question = user_question
        self._retriever = retriever
        self._llm_controller = llm_controller
        self._cached_result = None

    def execute(self, cached_result=True) -> Dict[str, Any]:
        if cached_result and self._cached_result:
            return self._cached_result
        self._cached_result = self._llm_controller.get_chart(
            self._user_question, self._retriever
        )
        return self._cached_result
