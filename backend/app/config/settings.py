import os
from dotenv import load_dotenv
#from pydantic_settings import BaseSettings
from functools import lru_cache


# Load environment variables from .env file
load_dotenv()

class Settings():
    # JWT settings
    JWT_PRIVATE_KEY_PATH: str = os.getenv("JWT_PRIVATE_KEY_PATH")
    JWT_PUBLIC_KEY_PATH: str = os.getenv("JWT_PUBLIC_KEY_PATH")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "RS256")
    
    # Token expiration times
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30  # Token expiry for access tokens
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7  # Token expiry for refresh tokens
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    MONGO_URI: str = os.getenv("MONGO_URI")
    GCP_JSON: str = os.getenv("GCP_JSON")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY: str = os.getenv("PINECONES_API_KEY")

    # Read RSA keys from files
    @property
    def JWT_PRIVATE_KEY(self) -> str:
        with open(self.JWT_PRIVATE_KEY_PATH, 'r') as f:
            return f.read()

    @property
    def JWT_PUBLIC_KEY(self) -> str:
        with open(self.JWT_PUBLIC_KEY_PATH, 'r') as f:
            return f.read()

    class Config:
        env_file = ".env"

# Instantiate the settings object
settings = Settings()
