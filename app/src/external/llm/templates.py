from textwrap import dedent

sql_template = dedent(
    """
    Write a SQL query that would answer the user's question,
    based on the context:

    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

    SQL Query:
    """
)
chart_spec = dedent(
    """
    Based on the context below, question and
    sql query, return the following chart specification:\n

    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

    <sql-query>
    {query}
    </sql-query>

    ```json
    {{"chart_type": "bar", "x": "column_name", y: "column_name"}}
    ```
    - where "chart_type" can by any type of chart in "bar", "line", "pie",\n
    "scatter", "area", "boxplot"
    - where "x" and "y" are the names of columns in the sql-query\n

    Json:
    """
)
