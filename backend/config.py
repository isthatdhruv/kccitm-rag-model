"""Central configuration via pydantic-settings. All settings overridable via .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve paths relative to the backend/ directory
_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
    )

    # MySQL
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_DB: str = "kccitm"
    MYSQL_USER: str = "user"
    MYSQL_PASSWORD: str = "qCsfeuECc3MW"

    # Milvus
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "student_results"
    MILVUS_FAQ_COLLECTION: str = "faq"

    # OpenAI (used for fast structured tasks: SQL generation, routing)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"

    # Ollama
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3:8b"
    OLLAMA_DRAFT_MODEL: str = "qwen3:1.7b"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    OLLAMA_EMBED_DIM: int = 768

    # SQLite database paths (relative to backend/)
    SESSION_DB: str = "data/sessions.db"
    CACHE_DB: str = "data/cache.db"
    FEEDBACK_DB: str = "data/feedback.db"
    PROMPTS_DB: str = "data/prompts.db"

    # Auth
    JWT_SECRET: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24

    # RAG settings
    RAG_TOP_K: int = 10
    RAG_RERANK_TOP_K: int = 10
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CACHE_SIMILARITY_THRESHOLD: float = 0.92
    CACHE_TTL_HOURS: int = 24

    # Ingestion / Embedding
    EMBED_BATCH_SIZE: int = 50       # Texts per Ollama /api/embed call (CPU: 50, GPU: 200-500)
    EMBED_CONCURRENCY: int = 1       # Parallel batch calls (CPU: 1, GPU: 2-4)

    # LLM settings
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 2048
    LLM_NUM_CTX: int = 32768

    def db_path(self, relative: str) -> Path:
        """Resolve a relative DB path to an absolute path under backend/."""
        return _BACKEND_DIR / relative


settings = Settings()
