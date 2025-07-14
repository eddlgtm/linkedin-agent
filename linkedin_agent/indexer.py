import arxiv
from langchain_chroma import Chroma
from langchain_community.retrievers import ArxivRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

client = arxiv.Client()

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
    continue_on_failure=True,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)


def search_arxiv_for_papers(search_query: str, max_results: int = 10):
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    results = list(client.results(search))
    urls = [r.entry_id for r in results]
    arxiv_ids = [url.split("/")[-1].split("v")[0] for url in urls]

    return arxiv_ids


def get_documents(arxiv_id: str) -> list[Document]:
    docs = retriever.invoke(arxiv_id)

    return docs


def index(documents: list[Document]) -> None:
    all_splits = text_splitter.split_documents(documents)
    vector_store.add_documents(documents=all_splits)
