from operator import itemgetter
from typing import Any

from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from src.common.utils.performance import log_time
from src.external.llm.langchain.chains.base import BaseChain
from src.external.llm.langchain.templates import CHART_GENERATION_TEMPLATE


class ChartGenerationChain(BaseChain):
    def __init__(self, llm: Any) -> None:
        super().__init__(llm)

    @log_time
    def chain(self) -> LLMChain:
        prompt = PromptTemplate(
            template=CHART_GENERATION_TEMPLATE,
            input_variables=["schema", "question"],
        )

        return (
            {
                "schema": itemgetter("schema"),
                "question": itemgetter("question"),
            }
            | prompt
            | self._llm
            | JsonOutputParser()
        )
