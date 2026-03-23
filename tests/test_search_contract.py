import inspect

from src.search import RAGSearch


def test_search_and_summarize_accepts_response_mode() -> None:
    sig = inspect.signature(RAGSearch.search_and_summarize)
    assert "response_mode" in sig.parameters


def test_stream_method_exists_with_response_mode() -> None:
    sig = inspect.signature(RAGSearch.stream_search_and_summarize)
    assert "response_mode" in sig.parameters
    assert "top_k" in sig.parameters


def test_prompt_style_switches_by_mode() -> None:
    # Avoid full __init__ (which loads models) for a fast unit test.
    obj = object.__new__(RAGSearch)

    detailed_prompt = RAGSearch._build_prompt(obj, "What is ML?", "context", "detailed")
    short_prompt = RAGSearch._build_prompt(obj, "What is ML?", "context", "short")

    assert "detailed answer" in detailed_prompt
    assert "400-500 words" in short_prompt
