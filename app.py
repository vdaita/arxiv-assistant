import arxiv

from typing import List
import pandas as pd
import numpy as np
from openai import AsyncOpenAI, OpenAI
import streamlit as st

import numpy as np
import json

# Load fasttext model from the web
from get_top_documents import get_topk_documents
from stqdm import stqdm

# Check if the fasttext model file exists locally
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

with st.expander("Model settings"):
    api_base = st.text_input("OpenAI-compatible API base URL", "https://api.openai.com/v1/")
    api_key = st.text_input("API key")
    model_name = st.text_input("Model name", "gpt-4o-mini")
    max_new_tokens = st.number_input("Max tokens per response", 2048)

with st.expander("Retrieval settings"):
    top_k = st.number_input("Per query, how many papers to send to LLM", value=5)
    max_papers_retrieved = st.number_input("Per query, how many papers to retrieve from Arxiv", value=500)
    max_num_queries = st.number_input("Number of queries to generate per request", value=4)

should_query = st.checkbox("Querying on")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.documents = []

def extract_queries_from_response(md_response: str):
    try:
        start = md_response.index("```json") + len("```json")
        end = md_response.index("```", start)
        json_block = md_response[start:end].strip()
        response_json = json.loads(json_block)
        queries = response_json.get("queries", [])
    except (json.JSONDecodeError, ValueError):
        queries = []
    return queries


def get_papers(query: str):
    search = arxiv.Search(
        query=query,
        max_results=max_papers_retrieved,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = []
    for result in search.results():
        title_abs = f"{result.title}\n[{result.pdf_url}]({result.pdf_url})\n{result.summary}"
        papers.append(title_abs)
    return papers

def retrieve_documents(query: str, client: OpenAI):
    # Ask API to generate JSON list of candidate keyword search queries {query: ["search query 1", "search query 2", ...]}
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": GENERATE_QUERIES_PROMPT.format(query_count=max_num_queries)},
        ] + st.session_state.messages,
        model=model_name,
        max_tokens=min(256, max_new_tokens),
        temperature=0.5,
        stream=False
    )
    # Extract JSON
    queries = extract_queries_from_response(response.choices[0].message.content)
    st.markdown("## Generated queries")
    for query in queries:
        st.markdown(f"- {query}")

    # For each search query, retrieve relevant papers from Arxiv
    all_papers = []
    for query in stqdm(queries, desc="Loading papers from Arxiv"):
        try:
            papers = get_papers(query)
            all_papers.extend(papers)
        except Exception as e:
            print(f"Error retrieving papers for query '{query}': {e}")
            st.error(f"Error retrieving papers for query '{query}': {e}")
    
    # De-deuplicate papers
    all_papers = list(set(all_papers))

    # Visual representation of the papers retrieved
    best_papers = get_topk_documents(all_papers, query, top_k)
    return best_papers

def answer_question(query: str):
    client = OpenAI(
        base_url=api_base,
        api_key=api_key
    )

    documents = st.session_state.documents
    if should_query:
        retrieved_documents = retrieve_documents(query, client)

        st.title("Retrieved documents")
        if len(retrieved_documents) == 0:
            st.write("No documents retrieved")
        else:
            for doc in retrieved_documents:
                with st.expander(doc.split("\n")[0]):
                    st.write(doc)

        documents += retrieved_documents
        st.session_state.documents = documents
    
    with st.chat_message("assistant"):
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": ANSWER_QUESTION_PROMPT}, {"role": "user", "content": "# Documents\n" + "\n".join(documents)}] + st.session_state.messages,
            model=model_name,
            max_tokens=max_new_tokens,
            temperature=0.5,
            stream=True
        )
        st.write_stream(response)

user_query = st.chat_input("What's your question?")
clear_chat_button = st.button("Clear chat")

if clear_chat_button:
    st.session_state.messages = []
    st.session_state.documents = []

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    answer_question(user_query)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])