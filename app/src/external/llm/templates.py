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
    You need to extract entities from the user query `WHERE` and `HAVING`
    clauses in specified format.\n
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
    Your task is to generate chart configuration for the given dataset and user
    question.\n
    Responses should be in JSON format compliant with the vega-lite
    specification, but `data` field must be excluded.\n

    <question>
    {question}
    </question>

    <dataset>
    {schema}
    </dataset>

    Json:
    """
)
