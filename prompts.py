GENERATE_QUERIES_PROMPT = """
From the chat history provided and the query below, generate a series of {query_count} keyword-based queries that be used to retrieve the right information. They should be comma separated terms. This is being used for search on arXiv, so these have to be simple and direct queries.
Use the following format to generate the queries:

```json
{{
    "queries": [
        "query 1",
        "query 2",
        "query 3",
        ...
    ]
}}
```
"""

ANSWER_QUESTION_PROMPT = """
Based on the documents and the questions to answer, provide a detailed response to the user. Make sure that you include citations and references to the documents provided.
"""