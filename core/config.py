from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import List

class Settings(BaseSettings):
    # 1. NO DEFAULT VALUE. 
    # If it can't find the .env file, the server will crash and tell us exactly why.
    google_api_key: str 

    # Database
    database_url: str = "postgresql://user:password@localhost:5432/talent_intel"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"

    # Security
    secret_key: str = "change_this_in_production"
    api_key_header: str = "X-API-Key"

    # App
    debug: bool = False
    log_level: str = "INFO"
    max_file_size_mb: int = 10
    allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # LLM model
    llm_model: str = "gemini-1.5-flash"
    llm_max_tokens: int = 4096

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"

    # 2. PYDANTIC V2 SYNTAX (This is what actually forces it to read the .env file)
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

@lru_cache()
def get_settings() -> Settings:
    return Settings()