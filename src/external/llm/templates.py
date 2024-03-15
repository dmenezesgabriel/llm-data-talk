sql_template = """Write a SQL query that would answer the user's question,
based on the context:
{context}

Question: {question}
SQL Query:"""

vega_spec_template = """Based on the context below, question,
sql query generate a vega_lite spec, json only no text or comments.

- the vega_lite spec must not have the "data" key, only "mark" and "encoding".
- the vega_lite spec must contain only fields present on the sql query.

{context}

Question: {question}
SQL Query: {query}"""
