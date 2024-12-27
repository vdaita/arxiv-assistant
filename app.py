import paperscraper
from typing import List
import pandas as pd
import numpy as np
from openai import AsyncOpenAI, OpenAI
import streamlit as st
import fitz
from io import BytesIO

from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Load fasttext model from the web
fasttext_model = KeyedVectors.load_word2vec_format("https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip")

GENERATE_QUERIES_PROMPT = """
From the chat history provided and the query below, generate a series of {query_count} keyword-based queries that be used to retrieve the right information.
Use the following format to generate the queries:

```json
{
    "queries": [
        "query 1",
        "query 2",
        "query 3",
        ...
    ]
}
```
"""

with st.expander("Model settings"):
    api_base = st.text_input("OpenAI-compatible API base URL", "https://api.openai.com")
    api_key = st.text_input("API key")
    model_name = st.text_input("Model name", "gpt-4o-mini")
    max_new_tokens = st.number_input("Max tokens per response", 2048)

with st.expander("Retrieval settings"):
    top_k = st.number_input("Per query, how many papers to send to LLM")
    max_papers_retrieved = st.number_input("Per query, how many papers to retrieve from Arxiv")
    max_num_queries = st.number_input("Number of queries to generate per request")

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
    papers = paperscraper.arxiv.get_papers(
        query,
        max_results=max_papers_retrieved,
        search_options={
            'sort_by': paperscraper.arxiv.arxiv.SortCriterion.SubmittedDate
        }
    )
    papers = pd.DataFrame(papers)
    # TODO: add the link to the paper
    print(papers.columns)
    papers["title_abs"] = papers["title"].str.cat(papers["abstract"], sep="\n")
    return papers["title_abs"].tolist()

def get_topk_documents(docs: List[str], query: str, top_k: int = 5):
    # Use tf-idf vectorizer over the data + query
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs + [query])
    vocab = vectorizer.get_feature_names_out()

    doc_embeddings = []

    # Create a weighted embedding using trained fasttext
    for doc_index, doc in enumerate(docs):
        doc_embedding = np.zeros(fasttext_model.vector_size)
        total_weight = 0
        for word, weight in zip(vocab, tfidf_matrix[doc_index]):
            if word in fasttext_model.key_to_index:
                doc_embedding += fasttext_model[word] * weight
                total_weight += weight
        if total_weight > 0:
            doc_embedding /= total_weight
        doc_embeddings.append(doc_embedding)

    # Calculate cosine similarity between the query and each document
    query_embedding = np.zeros(fasttext_model.vector_size)
    total_weight = 0
    for word, weight in zip(vocab, tfidf_matrix[-1]):
        if word in fasttext_model.key_to_index:
            query_embedding += fasttext_model[word] * weight
            total_weight += weight
    
    if total_weight > 0:
        query_embedding /= total_weight
    
    similarities = cosine_similarity([query_embedding], doc_embeddings)
    top_k_indices = np.argsort(similarities[0])[-top_k:]

    # TODO: visualize the documents
    return [docs[i] for i in top_k_indices]

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
    queries = extract_queries_from_response(response.choices[0].delta.content)

    # For each search query, retrieve relevant papers from Arxiv
    all_papers = []
    for query in queries:
        papers = get_papers(query)
        all_papers.extend(papers)
    
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
            messages=st.session_state.messages,
            model=model_name,
            max_tokens=max_new_tokens,
            temperature=0.5,
            stream=True
        )
        st.write_stream(response)

    

user_query = st.chat_input("What's your question?")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])