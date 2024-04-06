from textwrap import dedent

COMPOUND_QUESTION_EXTRACTOR_TEMPLATE = dedent(
    """
You are an expert in data analysis and text interpretation. Identify \
if the question is a double-barreled question. Then extract each \
question exactly as it was written to a valid JSON format in the below \
structure.
Questions with a single action should not be splitted.

[
    {{"question":"{{question to be answered}}"}},
    {{"question":"{{question to be answered}}"}}
]

<question>
{question}
</question>
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
