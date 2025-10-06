import os
import json
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
from config import CHROMA_PATH, DATA_PATH
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

COLLECTION = "book_summaries"

def build_or_load_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    ef = embedding_functions.OpenAIEmbeddingFunction(
        # api_key=os.getenv("OPENAI_API_KEY"),
        api_key = OPENAI_API_KEY,
        model_name=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
    )
    try:
        col = client.get_collection(name=COLLECTION)
        # If the collection exists but without an embedding function, rebuild it
        if getattr(col, "embedding_function", None) is None:
            client.delete_collection(COLLECTION)
            col = client.create_collection(
                name=COLLECTION,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
    except Exception:
        col = client.create_collection(
            name=COLLECTION,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    return col

def upsert_books(col, items: List[Dict]):
    ids, docs, metas = [], [], []
    for i, item in enumerate(items):
        title = item["title"]
        summary = item["summary"]
        ids.append(f"book-{i}-{title}")
        # Put only what you have: title + summary
        docs.append(f"Title: {title}\nSummary: {summary}")
        metas.append({"title": title})  # keep minimal metadata
    col.upsert(ids=ids, documents=docs, metadatas=metas)

def ensure_index():
    col = build_or_load_collection()
    # If empty, build; otherwise reuse
    if col.count() == 0:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        upsert_books(col, data)
    return col

if __name__ == "__main__":
    ensure_index()
    print("Index ready at", CHROMA_PATH)
