from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field("gpt-4o", env="OPENAI_MODEL")
    TEMPERATURE: float = Field(0.3, env="TEMPERATURE")  # Slight creativity for recipes

    # Agent
    MAX_AGENT_ITERATIONS: int = Field(6, env="MAX_AGENT_ITERATIONS")
    MAX_MEMORY_MESSAGES: int = Field(6, env="MAX_MEMORY_MESSAGES")

    # App
    DEBUG: bool = Field(False, env="DEBUG")
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
