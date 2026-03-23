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

### 8) Chat experience upgrade (streaming + persistent thread)
- Goal: Make UI behave like a real chatbot with live answer streaming.
- Action taken:
  - Updated `src/search.py`:
    - Added `stream_search_and_summarize(...)` for token streaming
    - Refactored prompt/context helpers for reusable logic
  - Updated `streamlit_app.py`:
    - Assistant response now streams live while generating
    - User + assistant messages persist in same chat thread
    - Added avatars for clearer chat roles
    - Improved empty-state layout and pushed input area lower visually
- Status: Implemented.

  ### 9) Query movement during streaming
  - Goal: When user submits a query, it should move to chat history first and then stream assistant reply in real-time below it.
  - Action taken:
    - Added `pending_query` state in `streamlit_app.py`
    - Flow now works in two phases:
      - Phase 1: append user message and trigger rerun
      - Phase 2: stream assistant response for `pending_query`, save answer, then rerun
  - Status: Implemented. Chat now behaves closer to ChatGPT interaction.

  ### 10) Separate test workflow (`test.yml`)
  - Goal: Run tests in a dedicated GitHub Actions workflow separate from CI checks.
  - Action taken:
    - Added `.github/workflows/test.yml`
      - Installs dependencies + pytest
      - Runs `pytest -q`
    - Added `tests/test_search_contract.py` with lightweight tests for:
      - `response_mode` argument presence
      - streaming method contract
      - prompt style switching
  - Status: Implemented.

### 11) CI/CD split and deployment workflow
- Goal: Keep CI and tests separated, and add a dedicated CD pipeline.
- Action taken:
  - Updated `.github/workflows/ci.yml`:
    - CI now handles syntax checks only (`py_compile`)
  - Kept tests in `.github/workflows/test.yml`:
    - pytest execution remains isolated in test workflow
  - Added `.github/workflows/cd.yml`:
    - Triggers on `main` push and manual dispatch
    - Runs pre-deploy smoke check
    - Triggers optional deployment webhook via `STREAMLIT_DEPLOY_WEBHOOK_URL` secret
- Status: Implemented.

### 12) Streamlit Cloud deployment readiness
- Goal: Prepare repository for direct deployment on Streamlit Community Cloud.
- Action taken:
  - Added `.streamlit/config.toml` (server + theme defaults)
  - Added `runtime.txt` (Python 3.11)
  - Added deployment guide section in `README.md` with:
    - Main file path (`streamlit_app.py`)
    - Required secret (`GROQ_API_KEY`)
    - notes about ephemeral filesystem
- Status: Ready for Streamlit Cloud deployment from GitHub.

### 13) GitHub test workflow import failure (`No module named 'src'`)
- Issue: `test.yml` failed in GitHub Actions while collecting tests due to missing `src` import path.
- Root cause:
  - CI test runner environment did not consistently include repo root in Python import path.
- Action taken:
  - Added `tests/conftest.py` to insert project root into `sys.path` for pytest runs
  - Updated `.github/workflows/test.yml` with `PYTHONPATH: .` for extra CI stability
- Validation:
  - Local pytest run passed: `3 passed`
- Status: Fixed.

### 14) Chatbot naming + UI branding refresh
- Goal: Improve bot identity and make UI feel more product-like.
- Action taken:
  - Added branding constants in `streamlit_app.py`:
    - `APP_NAME = "RAGNOVA"`
    - `APP_TAGLINE = "Your live AI knowledge copilot"`
  - Applied branding to:
    - page title
    - sidebar header/caption
    - hero and sub-hero text
    - chat input placeholder
- Status: Implemented.

### 15) Professional "About RAGNOVA" tab
- Goal: Provide a clear, professional explanation of system behavior and data usage.
- Action taken:
  - Added `About RAGNOVA` tab in `streamlit_app.py`
  - Included detailed sections for:
    - what RAGNOVA is
    - whether it is trained/fine-tuned on user data
    - data sources and supported file types
    - result generation flow (retrieval + generation)
    - output modes, reliability limits, and privacy/API usage
- Status: Implemented.

### 16) Developer tab professional UI redesign
- Goal: Make Developer tab look polished and portfolio-ready.
- Action taken:
  - Replaced badge-only layout with a professional card-based design
  - Added structured profile card (name, role, intro)
  - Added summary pills for project/domain/stack/interface
  - Added clean social action links for GitHub, LinkedIn, Twitter, and Email
  - Added dedicated CSS classes for spacing, typography, and hover states
- Status: Implemented.

### 17) Developer tab visual polish (colors + logos)
- Goal: Improve visual identity beyond black/white and use platform logos in cards.
- Action taken:
  - Added colorful social card variants for GitHub, LinkedIn, Twitter/X, and Email
  - Added logo icons inside each social card using platform icon sources
  - Improved card depth with better borders, gradients, and hover elevation
- Status: Implemented.

### 18) GitHub logo link + RAGNOVA settings UI refresh
- Goal: Improve repo link visibility and make settings area look more professional.
- Action taken:
  - Updated chat header repo link to include GitHub logo + label in one clickable control
  - Added sidebar gradient theme styling for better visual hierarchy
  - Added styled settings header card for `RAGNOVA Settings`
  - Improved sidebar control/button colors and contrast for polished look
- Status: Implemented.

## Next updates
- Add each new issue in this format:
  - Issue
  - Root cause
  - Action taken
  - Final status
