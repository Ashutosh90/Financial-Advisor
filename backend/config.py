"""
Configuration settings for Financial Advisor application
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # OpenAI Configuration
    openai_api_key: str
    
    # Database Configuration
    database_url: str = "sqlite:///./data/financial_advisor.db"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"
    
    # Model Configuration
    risk_model_path: str = "./models/risk_model.json"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Initialize settings
settings = Settings()
