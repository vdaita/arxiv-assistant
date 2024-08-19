# Arxiv Assistant

This project uses sentence-transformers and paperscraper to make it easy for you to identify and extract information about papers of your choosing. You can change the embeddings model, the number of papers retrieved and indexed per category, the number of papers processed after a search query, and the Arxiv categories indexed at the top of `main.py` after the imports. 

Check out this video demo:
[![YouTube Video](https://img.youtube.com/vi/YWFVgYMXDEI/0.jpg)](https://www.youtube.com/watch?v=YWFVgYMXDEI)

This program uses Chainlit and the OpenAI API.

First, install the dependencies:
`pip install -r requirements.txt`

Next, set the OpenAI environment variable in your .env file.

Then, you can run the web app by running: 
`chainlit run main.py`