import paperscraper 

from sentence_transformers import SentenceTransformer, util
from typing import List 
import pandas as pd
import numpy as np
from pprint import pprint
from openai import AsyncOpenAI 
import chainlit as cl
import requests
import fitz  # PyMuPDF
from io import BytesIO

# Initialize SentenceTransformer model
model = SentenceTransformer("msmarco-distilbert-dot-v5")
MAX_NUMBER_PAPERS = 50 
TOP_K = 5
# categories = ["cat:cs.AI", "cat:cs.LG", "cat:cs.CL"]
corpus_embedding = []
papers_df = pd.DataFrame()

async def load_papers_from_arxiv(query: str):
    # TODO: find a cleaner way to do this
    global papers_df
    global corpus_embedding
    global MAX_NUMBER_PAPERS
    global TOP_K
    global model

    print("Searching for papers in Arxiv...")
    all_new_papers = []
    titles = set(papers_df["title"]) if "title" in papers_df.columns else set([])
    new_papers_cat = paperscraper.arxiv.get_arxiv_papers(
        query, max_results=MAX_NUMBER_PAPERS,
        search_options={'sort_by': paperscraper.arxiv.arxiv.SortCriterion.SubmittedDate}
    )
    new_papers_cat = pd.DataFrame(new_papers_cat)
    new_papers_cat = new_papers_cat[~(new_papers_cat["title"].isin(titles))]
    print("Loaded...")
    new_papers_cat['title_abs'] = new_papers_cat['title'].str.cat(new_papers_cat['abstract'], sep = model.tokenizer.sep_token) # Creating single string for each paper with title + abstract
    new_papers_embeddings = model.encode(new_papers_cat.title_abs.values, convert_to_tensor=True)
    print("Encoded new paper embeddings: ", new_papers_embeddings.shape, type(new_papers_embeddings))
    corpus_embedding.extend(new_papers_embeddings)
    papers_df = pd.concat([papers_df, new_papers_cat])
    print("Added the papers to the corpus...")

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
    query_embedding = model.encode(query, convert_to_tensor=True)
    top5 = util.semantic_search(query_embedding, corpus_embedding, top_k=TOP_K)

    tool_output = ""
    # Iterate over top 5 results and construct output
    for entry_id, entry in enumerate(top5[0]):
        idx = entry['corpus_id']
        content = papers_df.iloc[idx].title_abs.replace('[SEP]', '\n\n')  # Get paper content
        tool_output += f"# Document {entry_id}\n{content}\n\n"  # Construct output string

    return tool_output

def is_search_query(content: str):
    return "search" in content.lower() or "find" in content.lower()

def pdf_url_to_markdown(pdf_url):
    """
    Downloads a PDF from the given URL and converts its content to a Markdown string.

    Args:
        pdf_url (str): The URL of the PDF to be converted.

    Returns:
        str: The Markdown representation of the PDF content.
    """
    try:
        # Download the PDF
        response = requests.get(pdf_url)
        response.raise_for_status()  # Check for HTTP errors

        # Load PDF into PyMuPDF from bytes
        pdf_bytes = BytesIO(response.content)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        markdown_text = ""

        # Iterate through each page
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)  # Load page
            text = page.get_text("text")    # Extract plain text

            # Append to markdown with a header for each page
            markdown_text += f"\n\n# Page {page_num + 1}\n\n"
            markdown_text += text

        return markdown_text.strip()

    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {e}")
        return ""
    except fitz.fitz.FileDataError:
        print("Error: The file downloaded is not a valid PDF.")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return ""

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

    print("Message content: ", message.content)

    # Model should send response 'Search "query string for sentence transformer"' if it wants to make a query - processing for that below
    # Not using tools right now - seems to adhere to the format well enough in a couple tests.
    # Check if search query is present in message
    if is_search_query(message.content):
        print("Processing search query")

        overarching_prompt = 'In one quotation mark, write down general search queries that can narrow down the papers. Example: "large language models, inference"'
        # print("Message history: ", message_history)
        message_history.append({
            "role": "user",
            "content": overarching_prompt
        })
        overarching_topic = await client.chat.completions.create(
            messages=message_history,
            **settings
        )
        message_history.pop(-1)
        overarching_topic = overarching_topic.choices[0].message.content
        print("Overarching topic query response: ", overarching_topic)
        search_query = overarching_topic.split('"')[1].split('"')[0]
        await msg.stream_token(f"\nQuerying Arxiv: {search_query}")
        await load_papers_from_arxiv(search_query)
        await msg.stream_token(f"\nFinished loading papers from arxiv.")

        query = str(msg.content).split("Search")[1].split('"')[1]
        response = search_papers(query)
        await msg.stream_token("\n" + response)
    
    if "load pdf url" in message.content.lower():
        print("Loading PDF")
        url = message.content.lower().split(" ")[-1]
        text = pdf_url_to_markdown(url)
        await msg.stream_token(f"\n# Retrieved PDF Paper\n{text}")

    # Append response to message history and show user
    message_history.append({"role": "assistant", "content": msg.content}) 
    await msg.update()