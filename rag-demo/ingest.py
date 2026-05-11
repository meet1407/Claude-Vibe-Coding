"""
Ingestion pipeline: reads documents from ./data, chunks them,
and stores embeddings in ChromaDB.
"""

import os
import chromadb

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # overlap between consecutive chunks


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if c]


def load_documents(data_dir: str) -> list[dict]:
    docs = []
    for filename in os.listdir(data_dir):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            docs.append({
                "id": f"{filename}_{i}",
                "text": chunk,
                "source": filename,
            })
    return docs


def ingest():
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Drop and recreate so re-running ingest is idempotent
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    docs = load_documents(DATA_DIR)
    if not docs:
        print("No .txt files found in ./data — nothing ingested.")
        return

    collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[{"source": d["source"]} for d in docs],
    )

    print(f"Ingested {len(docs)} chunks from {DATA_DIR} into ChromaDB.")


if __name__ == "__main__":
    ingest()
