from textwrap import dedent
from typing import List, Literal

from langchain_core.pydantic_v1 import BaseModel, Field


class ResponseTypeRouteQuery(BaseModel):
    response_format: Literal["chart", "sql_query_string", "text", "table"] = (
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


class AnalyticsEngineeringTaskPlan(BaseModel):
    question: str = Field(..., description="Business question to be answered")
    final_response_format: Literal["chart", "text", "table"] = Field(
        ..., description="Task response format"
    )


class AnalyticsEngineeringTaskPlanList(BaseModel):
    tasks: List[AnalyticsEngineeringTaskPlan]
