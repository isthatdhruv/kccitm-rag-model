# KCCITM RAG AI Assistant

A self-improving RAG + LLM AI assistant for KCCITM institute. Helps faculty and administrators analyze student academic performance data using natural language queries.

The system combines:
- **Milvus** hybrid search (dense vectors + BM25 keyword matching)
- **Text-to-SQL** generation for structured queries
- **Query routing** to pick the best pipeline per question
- **Ollama** local LLM inference (Qwen 3) with optional OpenAI fast-track

---

## Architecture Overview

```
User Query
    │
    ▼
┌──────────┐     ┌─────────────────────────────────┐
│  Router   │────▶│  SQL  │  RAG  │  HYBRID         │
│ (GPT-4o / │     │       │       │ (both parallel)  │
│  Qwen 3)  │     └───┬───┴───┬───┴───┬─────────────┘
└──────────┘         │       │       │
                     ▼       ▼       ▼
              ┌─────────┐ ┌──────┐ ┌──────────────┐
              │  MySQL   │ │Milvus│ │ MySQL+Milvus │
              │ (SQL)    │ │(RAG) │ │  (merged)    │
              └────┬─────┘ └──┬───┘ └──────┬───────┘
                   │          │            │
                   ▼          ▼            ▼
              ┌──────────────────────────────────┐
              │   LLM Response Generation        │
              │   (Qwen 3 via Ollama)            │
              └──────────────────────────────────┘
                            │
                            ▼
                     Natural Language Answer
```

---

## Prerequisites

| Dependency | Version | Purpose |
|------------|---------|---------|
| Docker + Docker Compose | latest | MySQL + Milvus containers |
| Python | 3.11+ | Backend runtime |
| Ollama | latest | Local LLM inference |
| OpenAI API key | (optional) | Fast routing + SQL generation |

---

## Hosting on a New Machine (Step by Step)

### Step 1: Clone the Repository

```bash
git clone <repo-url> kccitm-rag-model
cd kccitm-rag-model
```

### Step 2: Start Infrastructure

```bash
docker compose up -d
```

This starts:
- **MySQL** on `127.0.0.1:3306` (database: `student_db`, user: `user`)
- **Milvus** on `localhost:19530` (vector database with BM25 support)

Wait ~15 seconds for both services to be ready:
```bash
docker compose ps   # both should show "running"
```

### Step 3: Install and Start Ollama

```bash
# Install Ollama (if not installed)
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull qwen3:8b           # Main generation model (5.2 GB)
ollama pull qwen3:1.7b         # Draft model for dev/routing (1.4 GB)
ollama pull nomic-embed-text   # Embedding model (274 MB)

# Start Ollama server (if not running as a service)
ollama serve
```

### Step 4: Set Up Python Environment

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 5: Configure Environment

```bash
cp ../.env.example ../.env   # or create manually
```

Create/edit `.env` in the project root:
```env
# MySQL
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_DB=student_db
MYSQL_USER=user
MYSQL_PASSWORD=qCsfeuECc3MW

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=student_results
MILVUS_FAQ_COLLECTION=faq

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:8b          # Use qwen3:1.7b for faster dev
OLLAMA_DRAFT_MODEL=qwen3:1.7b
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_EMBED_DIM=768

# OpenAI (optional — speeds up routing + SQL generation)
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o

# SQLite
SESSION_DB=data/sessions.db
CACHE_DB=data/cache.db
FEEDBACK_DB=data/feedback.db
PROMPTS_DB=data/prompts.db

# Auth
JWT_SECRET=change-me-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRY_HOURS=24

# RAG
RAG_TOP_K=10
RAG_RERANK_TOP_K=10
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
CACHE_SIMILARITY_THRESHOLD=0.92
CACHE_TTL_HOURS=24

# LLM
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=2048
LLM_NUM_CTX=32768
```

**For CPU-only machines**, use these for faster responses:
```env
OLLAMA_MODEL=qwen3:1.7b
LLM_MAX_TOKENS=512
LLM_NUM_CTX=4096
RAG_TOP_K=5
```

### Step 6: Run the Data Ingestion Pipeline

These must be run **in order**. Each step depends on the previous one.

```bash
cd backend
source .venv/bin/activate

# 1. ETL: Extract raw data from MySQL → normalize into 3 tables
python3 -m ingestion.etl
#    Creates: students, semester_results, subject_marks tables

# 2. Chunker: Convert student records → natural language chunks
python3 -m ingestion.chunker
#    Creates: data/chunks.jsonl

# 3. Embedder: Generate 768-dim dense vectors for each chunk
python3 -m ingestion.embedder
#    Creates: data/embeddings.jsonl, data/embeddings.npy, data/chunk_ids.json
#    NOTE: Requires Ollama running with nomic-embed-text

# 4. Milvus Indexer: Create collections + insert chunks with indexes
python3 -m ingestion.milvus_indexer
#    Creates: student_results collection (HNSW + BM25)
#             faq collection (empty, ready for Phase 9)

# 5. Initialize prompt templates in SQLite
python3 -m ingestion.init_prompts
#    Creates: data/prompts.db with system prompts

# 6. Validate everything works end-to-end
python3 -m ingestion.validate
#    Runs 9 checks: MySQL tables, Milvus search, SQLite DBs
```

### Step 7: Test the System

```bash
cd backend
source .venv/bin/activate

# Run automated tests for each component
python3 -m tests.test_llm_client          # LLM connectivity (7 tests)
python3 -m tests.test_sql_validator       # SQL safety (21 tests)
python3 -m tests.test_router              # Query routing (18 tests)
python3 -m tests.test_sql_pipeline        # SQL pipeline (10 tests)
python3 -m tests.test_orchestrator        # Full end-to-end (6 tests)

# Interactive chat mode — talk to the assistant
python3 -m tests.test_orchestrator --interactive
```

**Example queries to try in interactive mode:**
```
top 5 students by SGPA in semester 1          → SQL route
tell me about Aakash Singh                    → RAG route
KCS503 results                                → RAG (BM25 keyword match)
students struggling in programming            → RAG (semantic search)
analyze CSE batch performance across semesters → HYBRID route
what about semester 3                         → follow-up with history
```

---

## Project Structure

```
kccitm-rag-model/
├── docker-compose.yml              # MySQL + Milvus containers
├── mysql-custom.cnf                # MySQL UTF-8 config
├── .env                            # Environment variables (not in git)
├── .gitignore
├── README.md
├── frontend/                       # (Phase 10+)
└── backend/
    ├── requirements.txt
    ├── config.py                   # Pydantic settings (loads .env)
    ├── main.py                     # FastAPI entry point (Phase 7)
    ├── core/
    │   ├── llm_client.py           # Ollama async REST client
    │   ├── openai_client.py        # OpenAI client (optional fast-track)
    │   ├── router.py               # Query classifier (SQL/RAG/HYBRID)
    │   ├── sql_pipeline.py         # NL → SQL → execute → format
    │   ├── rag_pipeline.py         # Embed → Milvus search → LLM generate
    │   ├── context_builder.py      # Token budget manager
    │   └── orchestrator.py         # Master orchestrator (ties everything)
    ├── db/
    │   ├── mysql_client.py         # Async + sync MySQL helpers
    │   ├── milvus_client.py        # Milvus hybrid/dense/keyword search
    │   └── sqlite_client.py        # SQLite for sessions, cache, feedback
    ├── ingestion/
    │   ├── etl.py                  # Raw data → normalized MySQL tables
    │   ├── chunker.py              # Records → natural language chunks
    │   ├── embedder.py             # Chunks → 768-dim vectors (Ollama)
    │   ├── milvus_indexer.py       # Build Milvus collections + indexes
    │   ├── init_prompts.py         # Store system prompts in SQLite
    │   └── validate.py             # End-to-end validation (9 checks)
    ├── data/                       # Generated at runtime (not in git)
    │   ├── chunks.jsonl
    │   ├── embeddings.jsonl
    │   ├── sessions.db
    │   ├── cache.db
    │   ├── feedback.db
    │   └── prompts.db
    └── tests/
        ├── test_llm_client.py
        ├── test_router.py
        ├── test_sql_pipeline.py
        ├── test_sql_validator.py
        └── test_orchestrator.py
```

---

## Database Schema

### MySQL (student_db)

| Table | Key Columns |
|-------|-------------|
| `students` | `roll_no` (PK), `name`, `course`, `branch`, `gender` |
| `semester_results` | `roll_no` (FK), `semester` (1-8), `sgpa`, `result_status` |
| `subject_marks` | `roll_no` (FK), `semester`, `subject_code`, `subject_name`, `grade`, `internal_marks`, `external_marks` |

### Milvus (vector search)

| Collection | Indexes | Purpose |
|------------|---------|---------|
| `student_results` | HNSW (dense, cosine) + BM25 (sparse) | Hybrid semantic + keyword search |
| `faq` | HNSW + BM25 | FAQ matching (Phase 9) |

### SQLite (app state)

| Database | Tables |
|----------|--------|
| `sessions.db` | sessions, messages, users |
| `cache.db` | query_cache |
| `feedback.db` | feedback, implicit_signals, chunk_analytics |
| `prompts.db` | prompt_templates, prompt_evolution_log |

---

## Phase Progress

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Foundation (ETL, chunking, embeddings, Milvus indexing) | Done |
| 2 | LLM Client + Query Router | Done |
| 3 | SQL Pipeline (NL → SQL → execute → format) | Done |
| 4 | Basic RAG Pipeline + Orchestrator | Done |
| 5 | Advanced RAG (HyDE, re-ranking, compression) | Planned |
| 6 | Chat History + Sliding Window + Summarization | Planned |
| 7 | FastAPI Server + SSE Streaming | Planned |
| 8 | Feedback System + Adaptive Learning | Planned |
| 9 | Prompt Evolution + FAQ Auto-generation | Planned |
| 10 | Frontend (React/Next.js) | Planned |
| 11 | Auth + Multi-tenancy | Planned |
| 12 | Production Deployment | Planned |

---

## Troubleshooting

**Ollama not responding:**
```bash
ollama ps          # Check running models
ollama serve       # Start if not running
```

**Milvus collection not found:**
```bash
# Re-run the full ingestion pipeline (Step 6 above)
# Milvus loses data if Docker volume wasn't persisted
docker compose down && docker compose up -d
python3 -m ingestion.milvus_indexer
```

**Slow responses on CPU:**
Set these in `.env`:
```env
OLLAMA_MODEL=qwen3:1.7b
LLM_MAX_TOKENS=512
LLM_NUM_CTX=4096
RAG_TOP_K=5
```

**Missing data files (chunks.jsonl, embeddings.jsonl):**
```bash
# Re-run ingestion from Step 6, in order
python3 -m ingestion.etl
python3 -m ingestion.chunker
python3 -m ingestion.embedder
python3 -m ingestion.milvus_indexer
```
