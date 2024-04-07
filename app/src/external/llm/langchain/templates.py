from textwrap import dedent

COMPOUND_QUESTION_EXTRACTOR_TEMPLATE = dedent(
    """
You are an expert in data analysis and text interpretation. Identify \
if the question is a double-barreled question. Then extract each \
question exactly as it was written to a valid JSON format in the below \
structure.
Add the missing subjects if needed.
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

SYSTEM_QUERY_CLAUSE_EXTRACTOR_TEMPLATE = dedent(
    """
You are an data engineer specialist and need to extract the where \
clauses and each statement from the clause in user query then return \
them as a valid JSON format.
You must keep the order of the where clauses and statements.
First Clause will always be "WHERE" clause.

expected response format:
[
    {{
        "clause":"{{clause}}",
        "table_name: "{{table_name}}",
        "column_name": "{{column_name}}",
        "operator":"{{operator}}",
        "value":"{{value}}"
    }},
    {{
        "clause":"{{clause}}",
        "table_name: "{{table_name}}",
        "column_name": "{{column_name}}",
        "operator":"{{operator}}",
        "value": ["{{value}}", "{{value}}"]
    }}
]

Examples:
[
    {{
        "clause":"WHERE",
        "table_name": "TBL_ARTISTS",
        "column_name": "name",
        "operator":"=",
        "value":"Iron Maiden"
    }},
    {{
        "clause": "AND",
        "table_name": "TBL_BANDS",
        "column_name": "genre",
        "operator": "IN",
        "value": ["Grunge", "Heavy Metal"]
    }}
[

<query>
{query}
</query>
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

NATURAL_LANGUAGE_ANALYSIS_GENERATION_TEMPLATE = dedent(
    """
Based in the python pandas dataframe schema below, write a natural language \
string template that answer the user's question replacing the placeholders \
with text interpolation by iterating over the dataframe.

The function that will be used to iterate over the dataframe and interpolate \
the text is the following:

```py
def interpolate_template(template: str, df) -> str:
    placeholders = {{}}
    for row_index, row in df.iterrows():
        cols = df.columns
        for col_index, col in enumerate(cols):
            f"placeholder_row_{{row_index}}_col_{{col_index}}"
            placeholders[f"placeholder_row_{{row_index}}_col_{{col_index}}"] = row[
                col
            ]
    return template.format(**placeholders)
```

Example:

question: Among the top artists by album sales, which is the first?

schema:
|              | 0      |
|:-------------|:-------|
| Artist       | object |
| Album        | object |
| Sales        | int64  |
| Release Year | int64  |

expected response:

Among the top artists by album sales, the first is {{placeholder_row_0_col_0}}.
Their best-selling album, {{placeholder_row_0_col_1}}, released in \
{{placeholder_row_0_col_3}}, has achieved remarkable success worldwide.

<question>
{context}
</question>

<question>
{question}
</question>

<dataframe-schema>
{schema}
</dataframe-schema>

Expected response:
"""
)

if __name__ == "__main__":
    print(20 * "=")
    print(USER_INTENT_EXTRACTION_TEMPLATE)
    print(20 * "=")
    print(SQL_GENERATION_TEMPLATE)
    print(20 * "=")
    print(SYSTEM_QUERY_CLAUSE_EXTRACTOR_TEMPLATE)
    print(20 * "=")
    print(CHART_GENERATION_TEMPLATE)
