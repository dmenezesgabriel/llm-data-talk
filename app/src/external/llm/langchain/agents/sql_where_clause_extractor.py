import re
from operator import itemgetter
from typing import Any, List

from langchain.agents import AgentExecutor, tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import (
    OpenAIToolsAgentOutputParser,
)
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.external.llm.langchain.templates import (
    SYSTEM_QUERY_CLAUSE_EXTRACTOR_TEMPLATE,
)


@tool
def regex_find_all(pattern: str, string: str) -> List["str"]:
    """Returns all occurrences of a regex pattern in a string."""
    return re.findall(pattern, string)


class SQLWhereClauseExtractorAgent:
    def __init__(self, llm: Any, tools: List[Any]):
        self._llm = llm
        self._tools = tools
        self._agent = None

    def agent(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_QUERY_CLAUSE_EXTRACTOR_TEMPLATE),
                ("human", "{query}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = (
            {
                "query": itemgetter("query"),
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | self._llm.bind_tools(self._tools)
            | OpenAIToolsAgentOutputParser()
        )

        return AgentExecutor(agent=agent, tools=self._tools)


if __name__ == "__main__":
    import json

    from langchain_openai import ChatOpenAI
    from src.config import get_config

    config = get_config()
    llm = ChatOpenAI(api_key=config.OPENAI_API_KEY)
    agent_executor = SQLWhereClauseExtractorAgent(
        llm=llm, tools=[regex_find_all]
    )
    result = agent_executor.agent().invoke(
        {
            "query": (
                """ \
            SELECT * FROM table
            WHERE name = 'Iron Maiden'
            AND artist = 'Bruce Dickinson'
            AND genre IN ('Grunge', 'Heavy Metal')
            """
            )
        }
    )
    print(json.loads(result["output"]))
