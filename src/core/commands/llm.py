from typing import Any, Dict


class LLMGetSQLCommand:
    def __init__(self, user_question: str, retriever, llm_controller) -> None:
        self.user_question = user_question
        self.retriever = retriever
        self.llm_controller = llm_controller

    def execute(self) -> str:
        return self.llm_controller.get_sql(self.user_question, self.retriever)


class LLMGetChartCommand:
    def __init__(self, user_question: str, retriever, llm_controller) -> None:
        self.user_question = user_question
        self.retriever = retriever
        self.llm_controller = llm_controller

    def execute(self) -> Dict[str, Any]:
        return self.llm_controller.get_chart(
            self.user_question, self.retriever
        )
