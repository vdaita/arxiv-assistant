import paperscraper 

from sentence_transformers import SentenceTransformer, util
from typing import List 
import pandas as pd
import numpy as np
from pprint import pprint
from openai import AsyncOpenAI 
import chainlit as cl

# Initialize SentenceTransformer model
model = SentenceTransformer("msmarco-distilbert-dot-v5")
MAX_NUMBER_PAPERS_PER_CAT = 100 
categories = ["cat:cs.AI", "cat:cs.LG", "cat:cs.CL"]

# Loading data into papers dataframe
papers_df = []
for cat in categories:
    papers_df.append(paperscraper.arxiv.get_arxiv_papers(
        cat, max_results=MAX_NUMBER_PAPERS_PER_CAT,
        search_options={'sort_by': paperscraper.arxiv.arxiv.SortCriterion.SubmittedDate}))
papers_df = pd.concat(papers_df).drop_duplicates()  # Concatenate papers and remove duplicates
papers_df['title_abs'] = papers_df['title'].str.cat(papers_df['abstract'], sep = model.tokenizer.sep_token) # Creating single string for each paper with title + abstract

# Embed everything
corpus_embedding = model.encode(papers_df.title_abs.values)

# Initialize OpenAI client w/ Chainlit
client = AsyncOpenAI() 
cl.instrument_openai()
settings = {
    "model": "gpt-4o-mini", 
    "temperature": 0
}

# Default start chat function with prompt explaining paper search and how to make a SentenceTransformer query
@cl.on_chat_start
def start_chat():
    cl.user_session.set( 
        "message_history",
        [{"role": "system", "content": """You are a helpful assistant that can helps users find the papers that are most relevant to them. You can search for content by saying: Search "whatever your query is". The user will search and give you a reply which you can then use to glean insights."""}],
    )

# Search papers utility function w/ SentenceTransformer
def search_papers(query: str):
    # Compute query embedding using SentenceTransformer and semantic search
    query_embedding = model.encode(query)
    top5 = util.semantic_search(query_embedding, corpus_embedding, top_k=5)

    tool_output = ""
    # Iterate over top 5 results and construct output
    for entry_id, entry in enumerate(top5[0]):
        idx = entry['corpus_id']
        content = papers_df.iloc[idx].title_abs.replace('[SEP]', '\n\n')  # Get paper content
        tool_output += f"# Document {entry_id}\n{content}\n\n"  # Construct output string

    return tool_output

# Define function to handle messages
@cl.on_message
async def on_message(message: cl.Message):
    # Get message history and show the user a new message.
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})
    msg = cl.Message(content="")
    await msg.send()

    # Stream response using OpenAI
    stream = await client.chat.completions.create(
        messages=message_history,
        **settings,
        stream=True,
    )

    # Process response stream
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

    print("Message content: ", msg.content)

    # Model should send response 'Search "query string for sentence transformer"' if it wants to make a query - processing for that below
    # Not using tools right now - seems to adhere to the format well enough in a couple tests.
    # Check if search query is present in message
    if "Search" in str(msg.content):
        print("Processing search query")
        query = str(msg.content).split("Search")[1].split('"')[1]
        response = search_papers(query)
        await msg.stream_token("\n" + response)

    # Append response to message history and show user
    message_history.append({"role": "assistant", "content": msg.content}) 
    await msg.update()