sql_template = """You are a data analyst at a company. You are interacting with
a user who is asking you questions about the company's database.
Based on the table schema, write a SQL query that would answer user's question.
Take the conversation history into account.

<context>
{context}
</context>

<conversation-history>
{conversation_history}
</conversation-history>

Write only the SQL query and nothing else. Do not wrap the SQL query in any
other text, not even backticks.

For example:
Question: which 3 artists have the most tracks?
SQL Query: SELECT ArtistId, Count(*) as track_count FROM Track GROUP BY
ArtistId ORDER BY track_count DESC LIMIT 3;
Question: Name 10 artists
SQL Query: SELECT Name FROM Artist ORDER BY Name LIMIT 10;

Your turn:

<question>
{question}
</question>

SQL Query:"""

vega_spec_template = """Based on the context below, question,
sql query generate a vega_lite spec, json only no text or comments.

- the vega_lite spec must not have the "data" key, only "mark" and "encoding".
- the vega_lite spec must have both "x" and "y" present in encoding dict.
- the vega_lite spec must contain only fields present on the sql query.

<context>
{context}
</context>

<conversation-history>
{conversation_history}
</conversation-history>

<question>
{question}
</question>

<sql-query>
{query}
</sql-query>

Vega-Lite Spec:"""
