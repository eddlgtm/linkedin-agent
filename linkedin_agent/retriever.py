from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict

from indexer import vector_store
from prompts import RAG_TEMPLATE

load_dotenv()


prompt = PromptTemplate.from_template(RAG_TEMPLATE)


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


llm = init_chat_model("gpt-4o", model_provider="openai")


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
