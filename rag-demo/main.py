"""
Interactive CLI for the RAG demo.
Run: python main.py
"""

import sys
import os
from ingest import ingest
from rag import query

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")


def print_divider():
    print("\n" + "=" * 60 + "\n")


def run():
    # Auto-ingest on first run
    if not os.path.exists(CHROMA_DIR):
        print("No vector DB found — running ingestion first...\n")
        ingest()

    print_divider()
    print("  RAG Demo — Ask questions about AI & Machine Learning")
    print("  Type 'quit' or 'exit' to stop | 'reingest' to reload docs")
    print_divider()

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if question.lower() == "reingest":
            ingest()
            continue

        print("\nSearching knowledge base...\n")
        result = query(question)

        print(f"Answer:\n{result['answer']}\n")
        print("Sources retrieved:")
        for s in result["sources"]:
            print(f"  [{s['score']:.3f}] {s['source']}")
            print(f"          {s['excerpt']}")
        print_divider()


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.")
        print("Copy .env.example to .env and add your key.")
        sys.exit(1)
    run()
