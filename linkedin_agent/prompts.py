RAG_TEMPLATE = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:
"""

RESEARCH_TEMPLATE = """"
Role: You are a researcher specialising in finding relevent papers on Arxiv.

You will receive a research question and will need to return a prompt to search Arxiv.
Only return the search query, do not return any other information.

Research Question: {question}

Useful Search Query:
"""