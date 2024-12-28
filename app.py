import arxiv

from typing import List
import pandas as pd
import numpy as np
from openai import AsyncOpenAI, OpenAI
import streamlit as st

import numpy as np
import json
from prompts import GENERATE_QUERIES_PROMPT, ANSWER_QUESTION_PROMPT

# Load fasttext model from the web
from get_top_documents import get_topk_documents_word2vec, get_fasttext_model
import tiktoken
from stqdm import stqdm

enc = tiktoken.encoding_for_model("gpt-4o-mini")
fasttext_model = get_fasttext_model()

with st.expander("Model settings"):
    api_base = st.text_input("OpenAI-compatible API base URL", "https://api.openai.com/v1/")
    api_key = st.text_input("API key")
    model_name = st.text_input("Model name", "gpt-4o-mini")
    max_input_tokens = st.number_input("Maximum input tokens", value=32000)
    max_new_tokens = st.number_input("Max tokens per response", value=2048)
    temperature = st.number_input("Temperature", value=0.5)

with st.expander("Retrieval settings"):
    top_k = st.number_input("Per query, how many papers to send to LLM", value=5)
    max_papers_retrieved = st.number_input("Per query, how many papers to retrieve from Arxiv", value=500)
    max_num_queries = st.number_input("Number of queries to generate per request", value=4)

input_token_count = 0
output_token_count = 0
input_token_count_placeholder = st.empty()
output_token_count_placeholder = st.empty()

should_query = st.checkbox("Querying on", value=True)

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
        title_abs = f"{result.title}  \n\n[{result.pdf_url}]({result.pdf_url})  \n\n{result.summary}  "
        papers.append(title_abs)
    return papers

def make_openai_request(system_prompt: str, documents: List[str], query: str, max_new_tokens: int = max_new_tokens, title="Final response", stream_and_show_output: bool = True) -> str:
    global input_token_count, output_token_count
    client = OpenAI(
        base_url=api_base,
        api_key=api_key
    )

    messages = [{"role": "system", "content": system_prompt}] + [{"role": "user", "content": "\n".join(doc for doc in documents) + f"\n{query}"}]
    
    request_tokens = sum([len(enc.encode(m["content"])) for m in messages])
    if request_tokens > max_input_tokens:
        remaining_tokens = max_input_tokens - len(enc.encode(system_prompt)) - len(enc.encode(query))
        constructed_documents = []
        
        current_documents = []
        current_documents_length = 0
        for doc_idx, doc in enumerate(documents):
            doc_tokens = len(enc.encode(doc))
            print(f"Document {doc_idx} has {doc_tokens} tokens")
            if current_documents_length + doc_tokens > remaining_tokens:
                constructed_documents.append(current_documents)
                current_documents = []
                current_documents_length = 0
            current_documents.append(doc)
            current_documents_length += doc_tokens
        
        if len(current_documents) > 0:
            constructed_documents.append(current_documents)

        for i in range(len(constructed_documents)):
            print("Document group length, ", i, ": ", len(constructed_documents[i]))

        documents = []
        documents_seen = 0
        for i in range(len(constructed_documents)):
            documents.append(
                make_openai_request(system_prompt, constructed_documents[i], query, title=f"Response for documents {documents_seen + 1} to {documents_seen + len(constructed_documents[i])}")
            )
            documents_seen += len(constructed_documents[i])

    input_token_count += request_tokens
    input_token_count_placeholder.metric(label="Input tokens (gpt-4o encoding)", value=input_token_count, delta=request_tokens)

    if stream_and_show_output:
        st.markdown(f"# {title}")
        response = st.write_stream(client.chat.completions.create(
            messages=messages,
            model=model_name,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stream=True
        ))
    else:
        response = client.chat.completions.create(
            messages=messages,
            model=model_name,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stream=False
        )
        response = response.choices[0].message.content

    response_length = len(enc.encode(response))

    output_token_count += response_length
    output_token_count_placeholder.metric(label="Output tokens (gpt-4o encoding)", value=output_token_count, delta=response_length)

    return response

def retrieve_documents(query: str, client: OpenAI):
    # Ask API to generate JSON list of candidate keyword search queries {query: ["search query 1", "search query 2", ...]}
    queries = extract_queries_from_response(
        make_openai_request(
            GENERATE_QUERIES_PROMPT.format(query_count=max_num_queries), 
            [], 
            query, 
            max_new_tokens=min(256, max_new_tokens),
            stream_and_show_output=False
        )
    )
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
    best_papers = get_topk_documents_word2vec(fasttext_model, all_papers, query, top_k)
    return best_papers

def answer_question(query: str):
    client = OpenAI(
        base_url=api_base,
        api_key=api_key
    )

    documents = st.session_state.documents
    if should_query:
        retrieved_documents = retrieve_documents(query, client)

        for i in range(len(retrieved_documents)):
            retrieved_documents[i] = f"Document {i + 1} {retrieved_documents[i]}"

        # TODO: sort the retrieved documents by their number

        st.title("Retrieved documents")
        if len(retrieved_documents) == 0:
            st.write("No documents retrieved")
        else:
            for doc in retrieved_documents:
                with st.expander(doc.split("\n")[0]):
                    st.write(f"### {doc}")

        documents += retrieved_documents
        st.session_state.documents = documents

    # TODO: add clustering for visualizations
    
    with st.chat_message("assistant"):
        st.markdown(
            make_openai_request(
                system_prompt=ANSWER_QUESTION_PROMPT,
                documents=documents,
                query=query
            )
        )

user_query = st.chat_input("What's your question?")

if user_query:
    st.session_state.messages = []
    st.session_state.documents = []
    st.session_state.messages.append({"role": "user", "content": user_query})
    answer_question(user_query)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])