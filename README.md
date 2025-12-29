# Policy-Document-QA-Agent

---

# 📄 Policy-Document-QA-Agent

A **production-grade CLI Document Question Answering (QA) system** built with  **LangChain** .
It indexes internal policy documents (PDF / TXT / DOCX), answers questions  **only from retrieved evidence** , provides  **citations** , supports  **strict structured JSON output** , and **refuses to answer** when evidence is insufficient.

This project is designed to be:

* audit-friendly
* deterministic
* reviewable by senior engineers
* suitable for real policy / compliance documents

---

## ✨ Key Features

* **Multi-format ingestion** : PDF, TXT, DOCX
* **Chunked indexing** with metadata preservation
* **Vector search** with configurable `k`, MMR, and fetch-k
* **Pluggable embeddings** :
  * Google Gemini embeddings
  * Hugging Face embeddings
* **Pluggable LLMs** :
  * Google Gemini
  * Hugging Face chat models
* **Strict refusal behavior** :
  * If evidence is weak → *“Insufficient evidence in the provided documents.”*
* **Citations** (chunk-level, deduped, capped)
* **Structured JSON output** (schema-validated, always valid)
* **Evaluation suite** with pass/fail reporting
* **Interactive CLI mode**
* **One-command workflow** via `make.bat` / `Makefile`

---

## 🗂️ Project Structure

```
Policy-Document-QA-Agent/
├─ src/
│  ├─ main.py
│  ├─ data/                  # Documents to index
│  ├─ .index/                # Persistent vector store (auto-generated)
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
├─ .env
├─ .env.example
├─ requirements.txt
├─ make.bat                  # Windows task runner
├─ Makefile                  # macOS/Linux task runner
└─ README.md
```

---

## 🔧 Setup

### 1️⃣ Create a virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Configure environment variables

Create `.env` from the example:

```bash
copy .env.example .env   # Windows
```

Fill in **at least one** provider:

```env
# Google Gemini
GOOGLE_API_KEY=your_google_api_key_here

# Hugging Face
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
```

---

## 🚀 Quick Start (Recommended)

### Windows (using `make.bat`)

```powershell
.\make.bat index-rebuild
.\make.bat run
```

### macOS / Linux

```bash
make index-rebuild
make run
```

---

## 🧠 Supported Models

### Embeddings (`--embedding`)

| Value      | Provider                 |
| ---------- | ------------------------ |
| `google` | Google Gemini Embeddings |
| `hf`     | Hugging Face embeddings  |

### LLMs (`--llm-model`)

| Value      | Provider                          |
| ---------- | --------------------------------- |
| `google` | Gemini (e.g.`gemini-2.5-flash`) |
| `hf`     | Hugging Face chat model           |

Example:

```bash
python -m main ask --embedding hf --llm-model google
```

---

## 🧪 Core CLI Commands

### Health check

```bash
python main.py health
```

### Show config

```bash
python main.py config
```

### Ingest documents

```bash
python -m main ingest --docs ./data
```

### Chunk documents

```bash
python -m main chunk --docs ./data
```

### Build / rebuild index

```bash
python -m main index --docs ./data --rebuild-index
```

### Reload existing index

```bash
python -m main index --docs ./data
```

---

## 🔍 Retrieval Debugging

### Similarity search

```bash
python -m main retrieve --docs ./data --k 5 --embedding hf --query "What are the leave policies?"
```

### MMR (diverse retrieval)

```bash
python -m main retrieve --docs ./data --k 5 --mmr --fetch-k 30 --embedding hf --query "What are the leave policies?"
```

---

## 💬 Ask Questions (Human-Readable)

### Answerable question

```bash
python -m main ask --docs ./data --k 6 --mmr --embedding hf --llm-model google --question "What are the leave policies?"
```

### Unanswerable question (refusal)

```bash
python -m main ask --docs ./data --k 6 --mmr --embedding hf --llm-model google --question "What is the capital of Japan?"
```

---

## 📦 Structured JSON Output (API-Ready)

### Answerable → JSON with citations

```bash
python -m main ask_json --docs ./data --k 6 --embedding hf --llm-model google --question "What are the leave policies?"
```

### Unanswerable → refusal JSON

```bash
python -m main ask_json --docs ./data --k 6 --embedding hf --llm-model google --question "What is the capital of Japan?"
```

### Save JSON to file

```bash
python -m main ask_json --docs ./data --out response.json
```

---

## 🖥️ Interactive Mode

```bash
python -m main run --docs ./data --k 15 --mmr --embedding hf --llm-model google
```

### Interactive commands

```
:help
:citations on | off
:save last.json
:exit
```

---

## 🧪 Evaluation Suite

Runs predefined test questions and checks:

* citations exist when answerable
* refusals happen when expected
* JSON schema is valid

```bash
python -m main eval --k 10 --embedding hf --llm-model google
```

Example output:

```
=== Evaluation Report ===
Passed: 5/5
```

---

## 🛑 Safety & Design Guarantees

* ❌ No hallucinated answers
* 📚 Answers **only** from retrieved context
* 🧾 Citations are **validated against retrieved chunks**
* 🧱 Structured output **always valid JSON**
* 🔒 Refusal text is exact and enforced
* 🧪 Eval catches regressions early

---

## 🧠 Why This Project Is Different

Most RAG demos:

* trust the model too much
* skip refusals
* produce messy outputs
* lack evaluation

This project:

* treats LLMs as **untrusted components**
* validates everything at boundaries
* behaves like a real internal policy QA system

---

## 📌 Use Cases

* Company policy QA
* HR / compliance document search
* Internal knowledge bases
* Regulated environments
* RAG system reference implementation

---

## 📜 License

MIT (or update as needed).
