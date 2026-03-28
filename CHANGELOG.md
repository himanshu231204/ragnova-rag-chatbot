# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project follows Semantic Versioning.

## [Unreleased]

### Added
- **Analytics Dashboard**: New analytics tab with query tracking and visualizations
  - `src/backend/analytics.py` - QueryLogger class with SQLite storage
  - `src/ui/pages/analytics.py` - Analytics dashboard UI
  - KPI metrics: Total queries, avg response time, avg chunks retrieved
  - Charts: Queries over time, response time distribution, top-k usage, response mode usage
  - Popular query terms word frequency analysis
  - Recent queries table with export capability
  - Query logging integrated into chat flow

## [2.0.1] - 2026-03-25

### Added
- `app.py` added to `.gitignore` for local app configuration

### Changed
- Updated CI/CD workflow files to reference new modular structure
  - `.github/workflows/ci.yml`: Updated syntax check paths for src/rag/*, src/ui/*, src/backend/*
  - `.github/workflows/cd.yml`: Updated syntax check paths for src/rag/*, src/ui/*, src/backend/*

### Fixed
- Test import path: `tests/test_search_contract.py` now imports from `src.rag.search` instead of `src.search`
- CI workflow now correctly validates all refactored module files

## [2.0.0] - 2026-03-25

### Added
- **UI/UX Refactoring**: Restructured streamlit_app.py into modular components
  - `src/ui/` - Frontend UI package
    - `config.py` - Centralized configuration constants
    - `styles.py` - Theme management and CSS styling
    - `sidebar.py` - Sidebar settings and controls
    - `pages/` - Tab-specific UI components
      - `chat.py` - Chat interface and query handling
      - `about.py` - About RAGNOVA information
      - `developer.py` - Developer info section
  - `src/backend/` - Backend business logic package
    - `index_manager.py` - FAISS index operations and utilities
    - `rag_client.py` - RAG search client initialization
  - `src/rag/` - Core RAG components package (NEW)
    - `data_loader.py` - Multi-format document loading
    - `embedding.py` - Text chunking and embeddings
    - `vectorstore.py` - FAISS vector index management
    - `search.py` - RAG orchestration and LLM integration
- Fixed embedding dimension mismatch error with proper validation
- Fixed light mode theme CSS (text was white, now correctly dark)
- Removed embedding model and LLM model selectors (fixed to recommended defaults)

### Changed
- `streamlit_app.py` now serves as clean entry point (imports all UI/backend modules)
- Modular package structure improves maintainability and testability
- Constants moved to `src/ui/config.py`
- Styling logic moved to `src/ui/styles.py`
- Index operations moved to `src/backend/index_manager.py`
- RAG client management moved to `src/backend/rag_client.py`
- All RAG core components consolidated in `src/rag/` folder
  - Imports updated: `from src.rag.search import RAGSearch`
  - Removed duplicate files from `src/` (data_loader.py, embedding.py, search.py, vectorstore.py)

### Fixed
- Light mode CSS now displays dark text (was previously white on light background)
- Embedding model validation on index load prevents dimension mismatch errors
- Sidebar UI properly initialized during page load
- Project structure cleaned - removed duplicate RAG component files
