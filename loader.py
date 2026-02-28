from pathlib import Path
from typing import List
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)
from langchain_core.documents import Document

import config


def get_file_extension(filepath: str) -> str:
    return Path(filepath).suffix.lower()


def load_document(filepath: str) -> List[Document]:
    ext = get_file_extension(filepath)

    if ext == ".pdf":
        loader = PyPDFLoader(filepath)
    elif ext == ".txt":
        loader = TextLoader(filepath, encoding="utf-8")
    elif ext == ".md":
        loader = TextLoader(filepath, encoding="utf-8")
    elif ext == ".csv":
        loader = CSVLoader(filepath, encoding="utf-8")
    elif ext in [".xlsx", ".xls"]:
        loader = UnstructuredExcelLoader(filepath, mode="elements")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return loader.load()


def load_documents_from_directory(directory: Path = None) -> List[Document]:
    if directory is None:
        directory = config.DATA_DIR

    all_docs = []
    supported_extensions = {".pdf", ".txt", ".md", ".csv", ".xlsx", ".xls"}

    for filepath in directory.rglob("*"):
        if filepath.is_file() and filepath.suffix.lower() in supported_extensions:
            try:
                docs = load_document(str(filepath))
                for doc in docs:
                    doc.metadata["source"] = filepath.name
                all_docs.extend(docs)
                print(f"Loaded: {filepath.name}")
            except Exception as e:
                print(f"Error loading {filepath.name}: {e}")

    return all_docs


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE * 6,  # approximate chars (words * 6)
        chunk_overlap=config.CHUNK_OVERLAP * 6,
        separators=["\n\n", "\n", " ", ""],
    )

    return splitter.split_documents(documents)


def process_documents() -> List[Document]:
    print(f"Loading documents from {config.DATA_DIR}...")
    documents = load_documents_from_directory(config.DATA_DIR)

    if not documents:
        print("No documents found.")
        return []

    print(f"Loaded {len(documents)} document pages. Splitting into chunks...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    return chunks
