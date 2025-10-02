"""
Configuration module for the AI SaaS backend.
Reads configuration from environment variables and .env file.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class that reads from environment variables."""
    
    def __init__(self):
        self.OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
        self.TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")
        self.REDIS_URL: Optional[str] = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.EMBED_MODEL: str = os.getenv("EMBED_MODEL", "all-mpnet-base-v2")


# Global config instance
cfg = Config()
