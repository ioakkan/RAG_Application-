# The Sci-Fi Concept Explorer   


## 📖 Overview

The **Sci-Fi Concept Explorer** is a command-line AI assistant designed
to help writers overcome writer's block by analyzing classic science
fiction stories.

The system:

-   Loads multiple `.txt` sci-fi stories from a directory
-   Splits them into semantically meaningful chunks
-   Builds a persistent **Chroma vector database**
-   Uses a **LangChain LCEL-based RAG pipeline**
-   Generates concise answers grounded strictly in retrieved passages

If the answer cannot be found in the provided texts, the system clearly
states so.

------------------------------------------------------------------------

# 🏗 Architecture

## RAG Pipeline Flow

    User Question
          ↓
    Chroma Retriever
          ↓
    MMR (Maximal Marginal Relevance)
          ↓
    Prompt Template (Grounded Creative Assistant)
          ↓
         LLM
          ↓
    Final Answer

------------------------------------------------------------------------

# ⚙️ Core Implementation Details

## 1️⃣ Document Ingestion & Processing

-   Loads `.txt` files from the `data/` directory
-   Uses intelligent text splitting
-   Applies chunk overlap to preserve narrative continuity
-   Prepares chunks for semantic embedding

This ensures optimal retrieval precision and context preservation.

------------------------------------------------------------------------

## 2️⃣ Vector Database (Chroma)

-   Persistent local Chroma database
-   Embeddings generated once and stored
-   Database reused for subsequent runs

This avoids repeated embedding generation and improves efficiency.

------------------------------------------------------------------------

## 3️⃣ RAG Chain (LangChain LCEL)

The retrieval-generation pipeline is built using **LangChain Expression
Language (LCEL)**.

Components:

-   Retriever (Chroma similarity search)
-   Custom prompt template
-   LLM
-   Output parser

The retriever fetches Top-K documents selected by maximal marginal relevance and passes them to the
LLM along with the user question.

------------------------------------------------------------------------

## 4️⃣ Prompt Engineering Strategy

The system enforces strict grounding:

    You are a creative assistant and literary analyst.
    Answer ONLY using the provided context.
    If the answer cannot be found, say:
    "I could not find relevant information in the provided texts."

This prevents hallucinations and ensures academic integrity.

------------------------------------------------------------------------

## 5️⃣ Application Structure

All logic is encapsulated inside a single class:

``` python
class SciFiExplorer:
```

Responsibilities:

-   Document loading
-   Vector database building/loading
-   Retriever setup
-   LCEL chain construction
-   Query handling
-   Logging system events

------------------------------------------------------------------------

# 📂 Project Structure

    .
    ├── app.py              # Main application
    ├── test_app.py         # Unit tests
    ├── data/               # Public-domain sci-fi stories (.txt)
    ├── pyproject.toml      # Project configuration
    ├── uv.lock             # Locked dependencies
    └── README.md

------------------------------------------------------------------------

# 📦 Dependency Management (uv)

This project uses **uv** for dependency management.

## Install Dependencies

``` bash
uv sync
```

This installs all dependencies defined in `pyproject.toml` and locked in
`uv.lock`.

------------------------------------------------------------------------

# ▶️ Running the Application

``` bash
uv run python app.py
```

Example CLI interaction:

    Ask a question:
    > What are examples of early alien encounters?

    Generating answer...

------------------------------------------------------------------------

# 🧪 Running Tests

Unit tests are located in `test_app.py`.

Run tests with:

``` bash
uv run pytest
```

The test suite verifies:

-   Documents ingestion
-   Documents Retrieval 

------------------------------------------------------------------------

# 🔍 Logging

Basic logging tracks:

-   Database creation
-   Embedding generation
-   Query receipt
-   Retrieved documents 

This improves transparency and debugging.

------------------------------------------------------------------------

# 🔐 Hallucination Prevention

The system reduces hallucination risk by:

-   Strict context-only prompt instructions
-   Controlled model randomness
-   No external knowledge injection

------------------------------------------------------------------------

# 📚 Data Sources

Public-domain texts sourced from:

-   Project Gutenberg
-   Internet Archive

------------------------------------------------------------------------



### Notice: With The first run of the project (creation of the vectorstore) the log folder is gonna be full because all the 5 text files are loaded and the word embdeddings-vectors are logged. (fix the size of logger files)


