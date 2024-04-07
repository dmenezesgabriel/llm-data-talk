from typing import Any, Dict

from langgraph.graph import END, StateGraph
from src.common.utils.dataframe import query_to_pandas_schema
from src.external.llm.langchain.chains.chart_generation import (
    ChartGenerationChain,
)
from src.external.llm.langchain.chains.response_format_router import (
    ResponseFormatRouteChain,
)
from src.external.llm.langchain.chains.sql_generation import SQLGenerationChain
from src.external.llm.langchain.chains.text_analysis_generation import (
    TextAnalysisGenerationChain,
)
from typing_extensions import TypedDict


class ChartGraphState(TypedDict):
    question: str
    response_format: str
    sql_query: str
    dataframe_schema: str
    chart_spec: Dict[str, Any]
    text_response: str


class ChartGraph:
    def __init__(self, llm: Any, retriever: Any, conn: Any):
        self._llm = llm
        self._retriever = retriever
        self._conn = conn
        self.workflow = StateGraph(ChartGraphState)

    def _choose_best_response_format(self, state):
        question = state["question"]

        response_format_chain = ResponseFormatRouteChain(llm=self._llm)
        response_format = response_format_chain.chain().invoke(
            input={
                "question": question,
            }
        )
        return {
            "question": question,
            "response_format": response_format.response_format,
        }

    def _generate_sql(self, state):
        question = state["question"]
        response_format = state["response_format"]

        sql_chain = SQLGenerationChain(
            llm=self._llm, retriever=self._retriever
        )
        sql_query = sql_chain.chain().invoke(input={"question": question})
        return {
            "question": question,
            "sql_query": sql_query,
            "response_format": response_format,
        }

    def _sql_to_pandas_schema(self, state):
        question = state["question"]
        response_format = state["response_format"]
        sql_query = state["sql_query"]

        schema = query_to_pandas_schema(sql_query, self._conn)
        return {
            "question": question,
            "response_format": response_format,
            "sql_query": sql_query,
            "dataframe_schema": schema,
        }

    def _generate_chart(self, state):
        question = state["question"]
        response_format = state["response_format"]
        sql_query = state["sql_query"]
        dataframe_schema = state["dataframe_schema"]

        chart_chain = ChartGenerationChain(llm=self._llm)
        chart_spec = chart_chain.chain().invoke(
            input={"question": question, "schema": dataframe_schema}
        )
        return {
            "question": question,
            "response_format": response_format,
            "sql_query": sql_query,
            "dataframe_schema": dataframe_schema,
            "chart_spec": chart_spec,
        }

    def _generate_text(self, state):
        question = state["question"]
        response_format = state["response_format"]
        sql_query = state["sql_query"]
        dataframe_schema = state["dataframe_schema"]
        chart_spec = state["chart_spec"]

        text_chain = TextAnalysisGenerationChain(
            llm=self._llm, retriever=self._retriever
        )
        template = text_chain.chain().invoke(
            input={"question": question, "schema": dataframe_schema}
        )
        return {
            "question": question,
            "response_format": response_format,
            "sql_query": sql_query,
            "dataframe_schema": dataframe_schema,
            "chart_spec": chart_spec,
            "text_response": template,
        }

    def graph(self):
        self.workflow.add_node(
            "choose_best_response_format", self._choose_best_response_format
        )
        self.workflow.add_node("generate_sql", self._generate_sql)
        self.workflow.add_node(
            "sql_to_pandas_schema", self._sql_to_pandas_schema
        )
        self.workflow.add_node("generate_chart", self._generate_chart)
        self.workflow.add_node("generate_text", self._generate_text)

        self.workflow.set_entry_point("choose_best_response_format")
        self.workflow.add_edge("choose_best_response_format", "generate_sql")
        self.workflow.add_edge("generate_sql", "sql_to_pandas_schema")
        self.workflow.add_edge("sql_to_pandas_schema", "generate_chart")
        self.workflow.add_edge("generate_chart", "generate_text")
        self.workflow.add_edge("generate_text", END)

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
    _chart_graph = ChartGraph(llm=llm, retriever=retriever, conn=conn)
    chart_graph = _chart_graph.graph()
    chart_inputs = {
        "question": ("What is the total sales for the artist Iron Maiden?")
    }
    chart_spec = chart_graph.invoke(chart_inputs)

    print(50 * "=")
    print(chart_spec)
    print(50 * "=")
