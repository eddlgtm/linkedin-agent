from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_community.retrievers import ArxivRetriever
from langchain_openai import OpenAIEmbeddings

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="research",
    embedding_function=embeddings,
    persist_directory="./research_db",
)

retriever = ArxivRetriever(
    load_max_docs=1,
    get_full_documents=True,
    doc_content_chars_max=None,
)

docs = retriever.invoke("2507.08392")

print(docs[0].page_content)
