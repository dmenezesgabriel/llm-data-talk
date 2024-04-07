from typing import Any, Dict, List

from langgraph.graph import END, StateGraph
from src.external.llm.langchain.chains.compound_question_extractor import (
    CompoundQuestionExtractorChain,
)
from src.external.llm.langchain.chains.sql_generation import SQLGenerationChain
from typing_extensions import TypedDict


class Analysis(TypedDict):
    question: str
    decomposed_question: str
    sql_query: str


class ChartGraphState(TypedDict):
    question: str
    analysis_list: List[Analysis]


class AnalyticsReport:
    def __init__(self, llm: Any, retriever: Any) -> None:
        self._llm = llm
        self._retriever = retriever
        self.workflow = StateGraph(ChartGraphState)

    def _decompose_question(self, state):
        question = state["question"]

        compound_question_extractor = CompoundQuestionExtractorChain(
            llm=self._llm
        )

        decomposed_questions = compound_question_extractor.chain().invoke(
            input={"question": question}
        )

        return {
            "analysis_list": [
                {"decomposed_question": decomposed_question["question"]}
                for decomposed_question in decomposed_questions
            ],
        }

    def _generate_sql(self, state):
        question = state["question"]
        analysis_list = state["analysis_list"]

        sql_chain = SQLGenerationChain(
            llm=self._llm, retriever=self._retriever
        )

        new_analysis_list = [
            {
                "question": analysis["decomposed_question"],
                "sql_query": sql_chain.chain().invoke(
                    input={"question": analysis["decomposed_question"]}
                ),
            }
            for analysis in analysis_list
        ]

        return {
            "question": question,
            "analysis_list": new_analysis_list,
        }

    def graph(self):
        self.workflow.add_node("decompose_question", self._decompose_question)
        self.workflow.add_node("generate_sql", self._generate_sql)

        self.workflow.set_entry_point("decompose_question")
        self.workflow.add_edge("decompose_question", "generate_sql")
        self.workflow.add_edge("generate_sql", END)

        return self.workflow.compile()


if __name__ == "__main__":
    from langchain_community.vectorstores import Chroma
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from src.common.utils.database import get_database_connection
    from src.config import get_config
    from src.external.llm.langchain.helpers.text import TextHelper

    conn = get_database_connection()
    with open("./data/schema.sql") as f:
        schema = f.read()

    config = get_config()
    text_chunks = TextHelper.get_text_chunks(schema)
    llm = ChatOpenAI(api_key=config.OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY)
    vector_store = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    retriever = vector_store.as_retriever()

    # ======================================================================= #

    _analytics_report = AnalyticsReport(llm=llm, retriever=retriever)
    analytics_report = _analytics_report.graph()
    for question in [
        "what are the top 20 artists by sales?",
        "Plot a chart with the top 10 artists by sales with different colors"
        "for each artist",
        "which is the artist with most sales?",
        "What is the artist with the most sales and the one with less sales? "
        "Also give me the difference in percentage of sales between the two",
    ]:
        analytics_report_result = analytics_report.invoke(
            input={"question": question}
        )

        print(50 * "=")
        print(analytics_report_result)
        print(50 * "=")
