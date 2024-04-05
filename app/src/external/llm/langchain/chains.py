from operator import itemgetter
from textwrap import dedent
from typing import Any, Dict, Optional

from langchain.chains.llm import LLMChain
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from src.common.utils.dataframe import query_to_pandas_schema
from src.common.utils.performance import log_time
from src.external.llm.langchain.models import ResponseTypeRouteQuery
from src.external.llm.langchain.templates import (
    chart_template,
    intent_extraction_template,
    re_write_template,
    sql_entity_extraction_template,
    sql_template,
)


class BaseChain:
    def __init__(
        self, llm: Any, retriever: Optional[BaseRetriever] = None
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._intermediates: Dict[str, Any] = {}

    def _save_intermediates(self, value: Any, **kwargs: Any) -> Any:
        key = kwargs.get("key")
        if key is None:
            raise ValueError("Key must be provided to save intermediates.")
        self._intermediates[key] = value
        return value

    def _post_process(self, value: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": value, "intermediates": self._intermediates}


class ResponseFormatRouteChain(BaseChain):

    def __init__(self, llm) -> None:
        super().__init__(llm)

    @log_time
    def chain(self) -> LLMChain:
        structured_llm = self._llm.with_structured_output(
            ResponseTypeRouteQuery
        )
        system = dedent(
            """
            You are an expert at routing a user question to the appropriate
            response type. Based on the analysis the question is
            referring to, route to the relevant response type.
            """
        )

        prompt = ChatPromptTemplate.from_messages(
            [("system", system), ("human", "{question}")]
        )

        return prompt | structured_llm


class UserIntentChain(BaseChain):
    def __init__(self, llm: Any) -> None:
        super().__init__(llm)

    @log_time
    def chain(self) -> LLMChain:
        prompt = PromptTemplate(
            template=intent_extraction_template,
            input_variables=["context", "question"],
        )

        return (
            {
                "question": itemgetter("question"),
            }
            | prompt
            | self._llm
            | JsonOutputParser()
        )


class SQLChain(BaseChain):
    def __init__(self, llm: Any, retriever: BaseRetriever) -> None:
        super().__init__(llm, retriever)

    @log_time
    def chain(self) -> LLMChain:
        prompt = PromptTemplate(
            template=sql_template,
            input_variables=["context", "question"],
        )

        return (
            {
                "context": itemgetter("question") | self._retriever,
                "question": itemgetter("question"),
            }
            | prompt
            | self._llm
            | StrOutputParser()
        )


class PromptReWriterChain(BaseChain):
    def __init__(self, llm: Any, retriever: BaseRetriever) -> None:
        super().__init__(llm, retriever)

    @log_time
    def chain(self) -> LLMChain:
        prompt = PromptTemplate(
            template=re_write_template,
            input_variables=["context", "question"],
        )

        return (
            {
                "context": itemgetter("question") | self._retriever,
                "question": itemgetter("question"),
            }
            | prompt
            | self._llm
            | StrOutputParser()
        )


class SQLEntityExtractionChain(BaseChain):
    def __init__(self, llm: Any, retriever: BaseRetriever) -> None:
        super().__init__(llm, retriever)

    @log_time
    def chain(self) -> LLMChain:
        prompt = PromptTemplate(
            template=sql_entity_extraction_template,
            input_variables=["query", "question"],
        )

        context_sql_chain = SQLChain(self._llm, self._retriever).chain()

        return (
            {
                "query": lambda x: context_sql_chain,
                "question": itemgetter("question"),
            }
            | prompt
            | self._llm
            | JsonOutputParser()
        )


class ChartChain(BaseChain):
    def __init__(self, llm: Any) -> None:
        super().__init__(llm)

    def chain(self) -> LLMChain:
        prompt = PromptTemplate(
            template=chart_template,
            input_variables=["query", "question"],
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


class StatefulChartChain(BaseChain):
    def __init__(self, llm: Any, retriever: BaseRetriever, conn: Any) -> None:
        super().__init__(llm, retriever)
        self._conn = conn

    def _query_to_pandas_schema(self, sql: str) -> str:
        return query_to_pandas_schema(sql, self._conn)

    @log_time
    def chain(self) -> LLMChain:
        prompt = PromptTemplate(
            template=chart_template,
            input_variables=["query", "question"],
        )
        context_sql_chain = SQLChain(self._llm, self._retriever).chain()

        return (
            {
                "schema": lambda _: context_sql_chain
                | RunnableLambda(self._save_intermediates).bind(key="sql")
                | RunnableLambda(func=self._query_to_pandas_schema),
                "question": itemgetter("question"),
            }
            | prompt
            | self._llm
            | JsonOutputParser()
            | RunnableLambda(self._post_process)
        )


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
    response_formate_route_chain = ResponseFormatRouteChain(llm=llm)
    response_formate_route_chain_result = (
        response_formate_route_chain.chain().invoke(
            input={"question": "what is the top 10 artists by sales?"}
        )
    )
    print(50 * "=")
    print(response_formate_route_chain_result)
    print(50 * "=")
    # ======================================================================= #
    user_intent = UserIntentChain(llm=llm)
    user_intent_result = user_intent.chain().invoke(
        input={"question": "calculate the total iron maiden artist sales"}
    )
    print(50 * "=")
    print(user_intent_result)
    print(50 * "=")
    # ======================================================================= #
    prompt_rewriter = PromptReWriterChain(llm=llm, retriever=retriever)
    prompt_rewriter_result = prompt_rewriter.chain().invoke(
        input={"question": "what are the total iron maiden artist sales?"}
    )
    print(50 * "=")
    print(prompt_rewriter_result)
    print(50 * "=")
    # ======================================================================= #
    sql_chain = SQLChain(llm=llm, retriever=retriever)
    sql_chain_result = sql_chain.chain().invoke(
        input={
            "question": (
                "What is the total sales figure for the artist Iron Maiden?"
            )
        }
    )
    print(50 * "=")
    print(sql_chain_result)
    print(50 * "=")
    # ======================================================================= #
    entity_extraction = SQLEntityExtractionChain(llm, retriever)
    entity_extraction_result = entity_extraction.chain().invoke(
        input={"question": "what are the total iron maiden artist sales?"}
    )
    print(50 * "=")
    print(entity_extraction_result)
    print(50 * "=")
    # ======================================================================= #
    chart_spec = StatefulChartChain(llm, retriever, conn)
    chart_spec_result = chart_spec.chain().invoke(
        input={"question": "what are the total iron maiden artist sales?"}
    )
    print(50 * "=")
    print(chart_spec_result)
    print(50 * "=")
