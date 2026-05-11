"""
Core RAG logic: retrieves relevant chunks from ChromaDB,
then passes them as context to Claude to generate an answer.
"""

import os
import anthropic
import chromadb
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "knowledge_base"
TOP_K = 3          # number of chunks to retrieve
MODEL = "claude-haiku-4-5-20251001"


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source", "unknown")
        context_blocks.append(f"[Source {i}: {source}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_blocks)

    return f"""You are a helpful assistant. Answer the user's question using ONLY the context provided below.
If the context does not contain enough information to answer, say "I don't have enough information to answer that."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def query(question: str) -> dict:
    # --- Retrieval ---
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_texts=[question],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    chunks = [
        {
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1 - results["distances"][0][i],  # cosine similarity
        }
        for i in range(len(results["documents"][0]))
    ]

    # --- Generation ---
    prompt = build_prompt(question, chunks)

    claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = claude.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = message.content[0].text

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "source": c["metadata"]["source"],
                "score": round(c["score"], 3),
                "excerpt": c["text"][:150] + "...",
            }
            for c in chunks
        ],
    }
