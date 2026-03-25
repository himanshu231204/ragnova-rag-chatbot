from pathlib import Path

from src.rag.search import RAGSearch


def main() -> None:
    if not Path("faiss_store/faiss.index").exists():
        print("FAISS index not found. Build index first.")
        return

    rag = RAGSearch(persist_dir="faiss_store")
    response = rag.search_and_summarize(
        "What is machine learning?",
        top_k=3,
        response_mode="short",
    )
    print("Smoke test passed. Response preview:\n")
    print(response[:600])


if __name__ == "__main__":
    main()
