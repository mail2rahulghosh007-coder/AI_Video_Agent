import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

def build_vector_store(transcript: str, chunk_size: int = 500, chunk_overlap: int = 50,
                        collection_name: str = COLLECTION_NAME) -> Chroma:
    # WHY collection_name and the reset below were added:
    # previously this always wrote to the SAME collection/persist_directory
    # without clearing it first, so every call (every video, every eval
    # experiment) kept ADDING chunks on top of whatever was already there.
    # Retrieval would then mix chunks from unrelated content, silently
    # corrupting results. Resetting first (or using a distinct collection
    # name per experiment) makes each run isolated and comparable.
    print("building vector store...")
    embeddings = get_embeddings()

    # Clear any existing collection with this name before adding new data.
    try:
        existing = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
        existing.delete_collection()
    except Exception:
        pass  # collection didn't exist yet -- fine

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(transcript)
    docs = [
        Document(page_content=chunk, metadata={'chunk_index': i})
        for i, chunk in enumerate(chunks)
    ]
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR
    )

    return vector_store

# load vector store
def load_vector_store() -> Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vector_store

# Retriever
def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={'k': k}
    )