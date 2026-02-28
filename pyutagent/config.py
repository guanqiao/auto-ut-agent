"""Configuration management for PyUT Agent."""

from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """Application settings."""
    
    # LLM Configuration
    llm_provider: str = "openai"
    llm_api_key: Optional[str] = None
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096
    
    # Coverage Configuration
    target_coverage: float = 0.8
    max_iterations: int = 10
    
    # Paths
    data_dir: Path = Path.home() / ".pyutagent"
    vector_db_path: Path = data_dir / "vector_db"
    memory_db_path: Path = data_dir / "memory.db"
    
    class Config:
        env_prefix = "PYUT_"
        env_file = ".env"


# Global settings instance
settings = Settings()


def get_data_dir() -> Path:
    """Get or create data directory."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings.data_dir
