import argparse

from src.rag.data_loader import load_all_documents
from src.rag.vectorstore import FaissVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild FAISS index from data directory.")
    parser.add_argument("--data-dir", default="data", help="Directory containing source docs")
    parser.add_argument("--persist-dir", default="faiss_store", help="FAISS storage directory")
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()

    docs = load_all_documents(args.data_dir)
    store = FaissVectorStore(
        persist_dir=args.persist_dir,
        embedding_model=args.embed_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    store.build_from_documents(docs)
    print("Index rebuild complete.")


if __name__ == "__main__":
    main()
