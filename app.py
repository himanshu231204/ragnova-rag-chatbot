from src.rag.data_loader import load_all_documents
from src.rag.vectorstore import FaissVectorStore
from src.rag.search import RAGSearch


if __name__ == "__main__":
    
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    #store.build_from_documents(docs)
    store.load()
    #print(store.query("what is machine learning?", top_k=3))
    rag_search = RAGSearch()
    query = "what is machine learning? list the machine learning algorithms."
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)