from textwrap import dedent

sql_template = dedent(
    """
    Write a SQL query that would answer the user's question, based on the\n
    context below.\n
    The queries should never use table name abbreviations, e.g. "t" instead\n
    of "table".\n

    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

    SQL Query:
    """
)

entity_extraction = dedent(
    """
    You need to extract entities from the user query in specified format.\n
    Extracted entities always should have valid json format, if you don't\n
    find any entities then respond with empty list.

    Entity Format: List of dict, where each dict having format as:\n
    {{
        "entity":"{{entity key}}",
        "attribute":"{{entity attribute key}}",
        "operator":"{{valid sql operator}}",
        "value":"{{value of entity}}"
    }}

    <sql-query>
    {query}
    </sql-query>

    <question>
    {question}
    </question>

    Entities:
    """
)

chart_spec = dedent(
    """
    You need to return a chart specification for the user's question and\n
    DQL sql snippet.\n
    Chart specification should always have valid json format, if you don't\n
    find any chart specification then respond with empty dict.\n

    {{
        "chart_type": "bar",
        "x": "{{DQL query column name for x axis}}",
        "y": "{{DQL query column name for y axis}}"
    }}

    <question>
    {question}
    </question>

    <sql-query>
    {query}
    </sql-query>

    Json:
    """
)
