# Policy-Document-QA-Agent

A **production-grade Document Question Answering (QA) system** built with **LangChain**, offering both:

* a **powerful CLI interface** for research, debugging, and batch evaluation, and
* a **Django REST API** suitable for integration into real applications.

The system indexes internal policy documents (PDF / TXT / DOCX), answers questions **strictly from retrieved evidence**, provides **verifiable citations**, produces **structured JSON outputs**, and **refuses to answer** when evidence is insufficient.

This project is designed to be:

* audit-friendly
* deterministic and reviewable
* safe for policy / compliance documents
* production-oriented (logging, evaluation, API hardening)

---

## Key Features

### Core RAG Capabilities

* Multi-format ingestion: **PDF, TXT, DOCX**
* Chunked indexing with **metadata preservation**
* Vector search with configurable **k**, **MMR**, and **fetch-k**
* **Strict refusal behavior** when evidence is insufficient
* Chunk-level **citations** (deduplicated and capped)
* Schema-validated **structured JSON output**
* Deterministic, non-hallucinatory answers

### Model Flexibility

* **Pluggable embeddings**

  * Google Gemini embeddings
  * Hugging Face embeddings
* **Pluggable LLMs**

  * Google Gemini models
  * Hugging Face chat models

### CLI Tooling

* Interactive CLI mode
* Retrieval debugging commands
* One-command workflows via `make.bat` / `Makefile`
* Built-in **evaluation suite** with pass/fail reporting

### REST API (Django + DRF)

* Versioned REST endpoints (`/v1/*`)
* API key authentication
* Centralized error handling
* File-based error logging with stack traces
* Warm-up endpoint to preload models
* Postman-friendly request/response design

---

## Project Structure

```
Policy-Document-QA-Agent/
├─ server/                     # Django REST API
│  ├─ api/
│  │  ├─ services/             # Bridges API ↔ core RAG logic
│  │  ├─ serializers.py
│  │  ├─ views.py
│  │  ├─ auth.py               # API key auth
│  │  ├─ safe.py               # Global exception wrapper
│  │  └─ utils.py              # Response helpers
│  ├─ docqa_api/
│  │  ├─ settings.py
│  │  ├─ urls.py
│  │  └─ wsgi.py
│  ├─ logs/
│  │  └─ docqa_api.log         # Error & traceback logs
│  └─ manage.py
│
├─ src/                        # Core RAG system (CLI)
│  ├─ main.py
│  ├─ data/                    # Documents to index
│  ├─ .index/                  # Persistent vector store
│  └─ docqa_agent/
│     ├─ ingest.py
│     ├─ chunking.py
│     ├─ vectorstore.py
│     ├─ retriever.py
│     ├─ rag.py
│     ├─ structured_rag.py
│     ├─ schema.py
│     ├─ eval.py
│     ├─ interactive.py
│     ├─ cli.py
│     └─ logging_setup.py
│
├─ .env
├─ .env.example
├─ requirements.txt
└─ README.md
```

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create `.env` from the example:

```bash
copy .env.example .env   # Windows
```

Provide **at least one provider**:

```env
# Google Gemini
GOOGLE_API_KEY=your_google_api_key_here

# Hugging Face
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here

# REST API security
DOCQA_API_KEY=your_api_key_here
```

---

## CLI Usage (Core System)

### Ingest & index documents

```bash
python -m main ingest --docs ./data
python -m main chunk --docs ./data
python -m main index --docs ./data --rebuild-index
```

### Ask questions (human-readable)

```bash
python -m main ask --k 6 --mmr --embedding hf --llm-model google \
  --question "What are the leave policies?"
```

### Structured JSON output

```bash
python -m main ask_json --k 6 --embedding hf --llm-model google \
  --question "What are the leave policies?"
```

### Interactive mode

```bash
python -m main run --k 15 --mmr --embedding hf --llm-model google
```

---

## Evaluation Suite (CLI)

Runs predefined test questions and verifies:

* refusals occur when expected
* citations exist when answerable
* JSON schema is always valid

```bash
python -m main eval --k 10 --embedding hf --llm-model google
```

Example output:

```
=== Evaluation Report ===
Passed: 5/5
```

---

## REST API (Django)

### Start the server

```bash
cd server
python manage.py runserver
```

---

### API Endpoints

#### Health check

```
GET /health/
```

#### Warm-up models (recommended)

```
POST /v1/warmup
```

Body:

```json
{"embedding":"google","llm_model":"google"}
```

#### Rebuild index

```
POST /v1/index
```

Body:

```json
{
  "docs_path": "../src/data",
  "rebuild": true,
  "embedding": "google"
}
```

#### Ask (human-readable)

```
POST /v1/ask
```

Body:

```json
{
  "question": "What are the leave policies?",
  "k": 6,
  "embedding": "google",
  "llm_model": "google"
}
```

#### Ask (structured JSON)

```
POST /v1/ask_json
```

Same body as above.

---

### Authentication

All `/v1/*` endpoints require:

```
X-API-Key: <DOCQA_API_KEY>
```

Configured via `.env`.

---

## Safety & Design Guarantees

* No hallucinated answers
* Answers only from retrieved context
* Citations tied to indexed chunks
* Always valid JSON output
* Exact refusal text enforced
* Evaluation catches regressions early
* Full error stack traces logged to file

---

## Why This Project Is Different

Most RAG demos:

* trust the model too much
* skip refusals
* produce messy outputs
* lack evaluation and logging

This system:

* treats LLMs as **untrusted components**
* validates inputs and outputs rigorously
* separates core logic from API layer
* behaves like a real internal policy QA system

---

## Use Cases

* Company policy QA
* HR / compliance document search
* Internal knowledge bases
* Regulated environments
* Reference implementation for safe RAG systems

---

## License

MIT
