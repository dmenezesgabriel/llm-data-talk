from textwrap import dedent
from typing import Literal

from langchain_core.pydantic_v1 import BaseModel, Field


class ResponseTypeRouteQuery(BaseModel):
    response_type: Literal["chart", "sql_query_string", "text", "table"] = (
        Field(
            ...,
            description=dedent(
                """
                Given a user question choose which response type would be most
                relevant for answering their question.
                """
            ),
        )
    )
