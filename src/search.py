import os
from typing import Iterator
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.3-70b-versatile",
        groq_api_key: str | None = None,
    ):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            try:
                from src.data_loader import load_all_documents
            except ModuleNotFoundError:
                from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(groq_api_key=api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def _build_prompt(self, query: str, context: str, response_mode: str) -> str:
        if response_mode == "detailed":
            format_instruction = (
                "Provide a detailed answer with clear sections, important points, "
                "and examples from context where relevant."
            )
        else:
            format_instruction = (
                "Provide a concise summary in about 400-500 words. "
                "Keep it clear and easy to read."
            )

        return (
            f"Answer the query based only on the provided context.\\n"
            f"Query: '{query}'\\n"
            f"Style: {format_instruction}\\n\\n"
            f"Context:\\n{context}\\n\\n"
            "Final Answer:"
        )

    def _get_context(self, query: str, top_k: int) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        return "\n\n".join(texts)

    def search_and_summarize(self, query: str, top_k: int = 5, response_mode: str = "short") -> str:
        context = self._get_context(query, top_k)
        if not context:
            return "No relevant documents found."

        prompt = self._build_prompt(query, context, response_mode)
        response = self.llm.invoke([prompt])
        return response.content

    def stream_search_and_summarize(
        self,
        query: str,
        top_k: int = 5,
        response_mode: str = "short",
    ) -> Iterator[str]:
        context = self._get_context(query, top_k)
        if not context:
            yield "No relevant documents found."
            return

        prompt = self._build_prompt(query, context, response_mode)
        for chunk in self.llm.stream([prompt]):
            text = getattr(chunk, "content", "")
            if text:
                yield text

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is logistic regression and how does it differ from linear regression?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)