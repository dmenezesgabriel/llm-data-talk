from textwrap import dedent
from typing import List, Literal, Union

from langchain_core.pydantic_v1 import BaseModel, Field


class ResponseTypeRouteQuery(BaseModel):
    response_format: Literal["Chart", "SQL String", "Text", "Table"] = Field(
        ...,
        description=dedent(
            """
                Given a user question choose which response type would be most
                relevant for answering their question.
                """
        ),
    )


class ExtractedQueryWhereClause(BaseModel):
    clause: str = Field(..., description="Extracted where clause")
    table_name: str = Field(..., description="Extracted table name")
    column_name: str = Field(..., description="Extracted column name")
    operator: str = Field(..., description="Extracted operator")
    value: Union[str, List[str]] = Field(..., description="Extracted value")


class ExtractedQueryWhereClauseList(BaseModel):
    where_clauses: List[ExtractedQueryWhereClause]
