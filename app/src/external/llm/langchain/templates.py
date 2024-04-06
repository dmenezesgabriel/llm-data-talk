from textwrap import dedent

ANALYTICS_ENGINING_PLANNER_TEMPLATE = dedent(
    """
You are an analytics engineering specialist and \
will lead a team of analytics engineers to execute analysis that best \
answers the user's questions based on the context below.
The analysis will always start with an SQL query and end with one of the \
following options: chart, table or text.
You must identify, separate and list complete analysis tasks. There is no \
problem to have only one task if it already answer the user questions.
The response format must be the format that best answers the user \
question being one of the three following options: chart, table or \
text.
The question must be clear, objective, concise, very specific and be \
optimized for document retrieval in a vector database.
Return a valid JSON format response without permeable or comments.

Tasks list format: List of dict where each dict having format as:

{{
    "question":"{{business question to be answered}}",
    "final_response_format":"{{task response format}}",
}},
{{
    "question":"{{business question to be answered}}",
    "final_response_format":"{{task response format}}",
}}


<context>
{context}
</context>

<question>
{question}
</question>

Tasks:
"""
)

USER_INTENT_EXTRACTION_TEMPLATE = dedent(
    """
You need to extract intents from the user question in specified format.
Extracted intents always should have valid json format, if you don't \
find any intents then respond with empty list.

Intent Format: List of dict, where each dict having format as:

{{
    "intent":"{{question intent}}",
    "entity":"{{
        "entity_type": {{question intent target entity type}},
        "entity_name": {{question intent target entity name}},
        }}",
    "action":"{{
        "action_subject": {{question intent action subject}},
        "action_details": {{question intent action details}},
        }}",
}}

<question>
{question}
</question>

Intents:
"""
)


SQL_GENERATION_TEMPLATE = dedent(
    """
Write a SQL query that would answer the user's question, based on the \
context below.
The queries should never use table name abbreviations, e.g. "t" instead \
of "table".

<context>
{context}
</context>

<question>
{question}
</question>

SQL Query:
"""
)

SQL_ENTITY_EXTRACTION_TEMPLATE = dedent(
    """
You need to extract entities from the user query `WHERE` and `HAVING` \
clauses in specified format.
Extracted entities always should have valid json format, if you don't \
find any entities then respond with empty list.

Entity Format: List of dict, where each dict having format as:

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

CHART_GENERATION_TEMPLATE = dedent(
    """
Your task is to generate chart configuration for the given dataset and user \
question.
Responses should be in JSON format compliant with the vega-lite \
specification, but `data` field must be excluded.

<question>
{question}
</question>

<dataset>
{schema}
</dataset>

Json:
"""
)

if __name__ == "__main__":
    print(20 * "=")
    print(USER_INTENT_EXTRACTION_TEMPLATE)
    print(20 * "=")
    print(SQL_GENERATION_TEMPLATE)
    print(20 * "=")
    print(SQL_ENTITY_EXTRACTION_TEMPLATE)
    print(20 * "=")
    print(CHART_GENERATION_TEMPLATE)
