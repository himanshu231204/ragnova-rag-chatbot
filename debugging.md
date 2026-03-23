# Debugging Log

This file tracks debugging issues, fixes, and current status for this project.
We will keep appending updates here.

## 2026-03-23

### 1) Git ignore update for vector store artifacts
- Issue: FAISS store files should not be tracked by git.
- Action taken:
  - Added ignore entries in .gitignore:
    - faiss_store/
    - fassi_store/
- Status: Resolved for new/untracked files.

### 2) Running src.search in virtual environment
- Issue: Direct run failed with import error.
- Error: ModuleNotFoundError: No module named 'src'
- Action taken:
  - Used module execution instead of direct file execution.
  - Working command:
    - .\\ragenv\\Scripts\\python.exe -m src.search
- Status: Resolved.

### 3) PDF parsing failure during vector store build
- File involved:
  - data/Artificial Intelligence - A Modern Approach (3rd Edition).pdf
- Issue: PDF parser error from pypdf on one PDF.
- Action taken in src/data_loader.py:
  - Added robust PDF loader strategy:
    - Primary: PyMuPDFLoader
    - Fallback: PyPDFLoader
    - If both fail: skip file and continue
- Status: PDF loading path improved and no longer blocked at initial parse stage.

### 4) Current runtime bottleneck
- Issue: Long embedding run interrupted during chunk embedding.
- Observed state:
  - Documents loaded successfully.
  - Large chunk volume causes long processing time.
- Status: Open (performance/timeout/interrupt related), not a code-crash at PDF load step.

### 5) Streamlit application added
- Goal: Create a simple web UI to run the same RAG flow.
- Action taken:
  - Added streamlit_app.py at project root.
  - Added streamlit in requirements.txt.
  - Added import hardening in src/search.py for data loader when FAISS index is missing.
- Status: Implemented. Ready to run with Streamlit command.

### 6) Streamlit model + API key controls
- Goal: Let users switch models and use their own API key from UI.
- Action taken:
  - Updated streamlit_app.py sidebar:
    - Embedding model selector (preset + custom input)
    - LLM model selector (preset + custom input)
    - Custom GROQ API key toggle and password input
  - Updated src/search.py:
    - RAGSearch now accepts optional groq_api_key argument
    - Falls back to environment key only if custom key not provided
  - Validation:
    - Syntax check passed for streamlit_app.py and src/search.py
- Status: Implemented and validated.

### 7) Streamlit crash: unexpected keyword argument 'response_mode'
- Issue: Streamlit crashed with `TypeError: RAGSearch.search_and_summarize() got an unexpected keyword argument 'response_mode'`.
- Root cause:
  - Streamlit cache could hold a stale `RAGSearch` client instance created before the updated method signature.
- Action taken:
  - Added compatibility fallback in streamlit_app.py:
    - Try calling `search_and_summarize(..., response_mode=...)`
    - On `TypeError`, clear cached client and retry with compatible call
  - Added CI workflow at `.github/workflows/ci.yml` to track future changes:
    - Trigger on push, pull_request, and manual dispatch
    - Install dependencies
    - Run Python syntax checks
    - Enforce API contract that `search_and_summarize` includes `response_mode`
- Status: Fixed in code, and guarded with CI for regression tracking.

## Next updates
- Add each new issue in this format:
  - Issue
  - Root cause
  - Action taken
  - Final status
