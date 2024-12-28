import os
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from gensim.models import KeyedVectors
from gensim.downloader import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from scipy.sparse import coo_matrix
import time

in_streamlit = "STREAMLIT_SERVER_RUNNING" in os.environ

if in_streamlit:
    from stqdm import stqdm
else:
    from tqdm import tqdm as stqdm
import streamlit as st

@st.cache_resource
def get_fasttext_model():
    start_time = time.perf_counter()
    glove_model = load("glove-wiki-gigaword-50")
    end_time = time.perf_counter()
    print(f"Loaded GloVe model in {end_time - start_time} seconds")
    if in_streamlit:
        st.write(f"Loaded GloVe model in {end_time - start_time} seconds")
    return glove_model

def get_topk_documents_word2vec(fasttext_model, docs: List[str], query: str, top_k: int = 5):
    fasttext_model_keys = set(fasttext_model.key_to_index.keys())
    # Use tf-idf vectorizer over the data + query
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs + [query])
    vocab = vectorizer.get_feature_names_out()

    # print(tfidf_matrix)

    doc_embeddings = [np.zeros(fasttext_model.vector_size) for _ in range(len(docs) + 1)]
    doc_weights = [0 for _ in range(len(docs) + 1)]

    tfidf_matrix = coo_matrix(tfidf_matrix)

    for row, col, data in stqdm(zip(tfidf_matrix.row, tfidf_matrix.col, tfidf_matrix.data), desc="Embedding documents using Fasttext and TF-IDF"):
        if vocab[col] in fasttext_model_keys:
            doc_embeddings[row] += fasttext_model[vocab[col]] * data
            doc_weights[row] += data

    for i in range(len(docs) + 1):
        doc_embeddings[i] = doc_embeddings[i] / doc_weights[i]
    query_embedding = doc_embeddings[-1]
    doc_embeddings = doc_embeddings[:-1]
    
    # Calculate cosine similarity between the query and each document
    similarities = cosine_similarity([query_embedding], doc_embeddings)
    top_k_indices = np.argsort(similarities[0])[-top_k:]

    # TODO: visualize the documents
    return [docs[i] for i in top_k_indices]

if __name__ == "__main__":
    get_topk_documents_word2vec(["hello world", "world hello"], "hello", 1)