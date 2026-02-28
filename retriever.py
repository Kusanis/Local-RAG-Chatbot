from typing import List, Optional
import os
import ollama

from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.embeddings import Embeddings

import config


_vectorstore: Optional[Chroma] = None
_rag_chain = None


class OllamaEmbeddingsDirect(Embeddings):
    """Direct Ollama embeddings using the ollama library"""

    def __init__(self, model: str):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        # Handle dict input from langchain
        if isinstance(text, dict):
            text = text.get("question", str(text))
        result = ollama.embed(model=self.model, input=text)
        return result["embeddings"][0]


def get_embeddings() -> OllamaEmbeddingsDirect:
    return OllamaEmbeddingsDirect(config.EMBED_MODEL)


def get_llm():
    return ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.3,
    )


def create_vectorstore(chunks: List[Document]) -> Chroma:
    embeddings = get_embeddings()

    if chunks:
        print("Creating new vectorstore...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(config.CHROMA_DIR),
        )
        vectorstore.persist()
    elif os.path.exists(config.CHROMA_DIR) and os.listdir(config.CHROMA_DIR):
        print("Loading existing vectorstore...")
        vectorstore = Chroma(
            persist_directory=str(config.CHROMA_DIR),
            embedding_function=embeddings,
        )
    else:
        raise ValueError("No chunks provided and no existing vectorstore found.")

    return vectorstore


def get_retriever(vectorstore: Chroma):
    return vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_K})


def build_rag_chain(vectorstore: Chroma):
    llm = get_llm()
    retriever = get_retriever(vectorstore)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant that answers questions based on uploaded documents.
- Only answer based on the provided context
- If you don't know, say so clearly
- Be concise and direct""",
            ),
            (
                "human",
                """Context from documents:
{context}

Question: {input}""",
            ),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": lambda x: x["question"]}
        | prompt
        | llm
    )

    return rag_chain


def initialize_rag(chunks: Optional[List[Document]] = None):
    global _vectorstore, _rag_chain

    if chunks:
        _vectorstore = create_vectorstore(chunks)
        _rag_chain = build_rag_chain(_vectorstore)
    elif os.path.exists(config.CHROMA_DIR) and os.listdir(config.CHROMA_DIR):
        _vectorstore = create_vectorstore([])
        _rag_chain = build_rag_chain(_vectorstore)
    else:
        raise ValueError("No documents processed and no existing vectorstore found.")

    print("RAG chain ready.")


def get_rag_chain():
    if _rag_chain is None:
        raise ValueError("RAG not initialized. Call initialize_rag() first.")
    return _rag_chain


def get_vectorstore() -> Optional[Chroma]:
    return _vectorstore


def clear_vectorstore():
    global _vectorstore, _rag_chain

    if _vectorstore:
        _vectorstore.delete_collection()

    import shutil

    if os.path.exists(config.CHROMA_DIR):
        shutil.rmtree(config.CHROMA_DIR)
        os.makedirs(config.CHROMA_DIR, exist_ok=True)

    _vectorstore = None
    _rag_chain = None
    print("Vectorstore cleared.")
