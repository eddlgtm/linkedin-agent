from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

from indexer import get_documents, index, search_arxiv_for_papers
from prompts import RESEARCH_TEMPLATE
from retriever import graph

llm = init_chat_model("gpt-4o", model_provider="openai")

prompt = PromptTemplate.from_template(RESEARCH_TEMPLATE)

question = "Recent LLM-as-a-Judge research"

messages = prompt.invoke({"question": question})
response = llm.invoke(messages)
research_query = response.content

ids = search_arxiv_for_papers(research_query)

for id in ids:
    documents = get_documents(id)
    index(documents)

result = graph.invoke({"question": "What is the latest LLM-as-a-Judge research?"})
print(f"Answer: {result['answer']}")
