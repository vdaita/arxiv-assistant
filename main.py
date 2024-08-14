import paperscraper  # Import paperscraper library
import paperscraper  # Redundant import, remove it

from sentence_transformers import SentenceTransformer, util  # Import SentenceTransformer and utility functions
from typing import List  # Import List type
import pandas as pd  # Import pandas library
import numpy as np  # Import NumPy library
from pprint import pprint  # Import pprint function
from openai import AsyncOpenAI  # Import OpenAI's AsyncAPI
import chainlit as cl  # Import Chainlit library

# Initialize SentenceTransformer model
model = SentenceTransformer("msmarco-distilbert-dot-v5")

# Define constants
MAX_NUMBER_PAPERS_PER_CAT = 100  # Maximum number of papers per category
categories = ["cat:cs.AI", "cat:cs.LG", "cat:cs.CL"]  # List of categories

# Initialize empty list to store papers dataframe
papers_df = []

# Iterate over categories and get papers using arxiv API
for cat in categories:
    papers_df.append(paperscraper.arxiv.get_arxiv_papers(
        cat, max_results=MAX_NUMBER_PAPERS_PER_CAT,  # Get papers by category with max 100 results
        search_options={'sort_by': paperscraper.arxiv.arxiv.SortCriterion.SubmittedDate}))  # Sort papers by submission date
papers_df = pd.concat(papers_df).drop_duplicates()  # Concatenate papers and remove duplicates

# Concatenate title and abstract using SentenceTransformer's tokenizer
papers_df['title_abs'] = papers_df['title'].str.cat(papers_df['abstract'], sep = model.tokenizer.sep_token)

# Embed title and abstract using SentenceTransformer
corpus_embedding = model.encode(papers_df.title_abs.values)

# Initialize OpenAI client
client = AsyncOpenAI()  # Initialize OpenAI client

# Instrument OpenAI client using Chainlit
cl.instrument_openai()

# Define settings for OpenAI
settings = {
    "model": "gpt-4o-mini",  # Use gpt-4o-mini model
    "temperature": 0,  # Set temperature to 0
}

# Define function to start chat
@cl.on_chat_start
def start_chat():
    cl.user_session.set(  # Set user session
        "message_history",
        [{"role": "system", "content": """You are a helpful assistant that can helps users find the papers that are most relevant to them. You can search for content by saying: Search "whatever your query is". The user will search and give you a reply which you can then use to glean insights."""}],
    )

# Define function to search papers
def search_papers(query: str):
    # Compute query embedding using SentenceTransformer
    query_embedding = model.encode(query)

    # Perform semantic search using SentenceTransformer
    top5 = util.semantic_search(query_embedding, corpus_embedding, top_k=5)

    # Initialize output string
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
    # Get message history
    message_history = cl.user_session.get("message_history")

    # Append message to message history
    message_history.append({"role": "user", "content": message.content})

    # Create response message
    msg = cl.Message(content="")

    # Send response message
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

    # Print response message
    print("Message content: ", msg.content)

    # Check if search query is present in message
    if "Search" in str(msg.content):
        print("Search is the message content")
        query = str(msg.content).split("Search")[1].split('"')[1]
        response = search_papers(query)  # Perform search
        await msg.stream_token("\n" + response)

    # Append response to message history
    message_history.append({"role": "assistant", "content": msg.content})

    # Update message
    await msg.update()