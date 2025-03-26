"""
Configuration settings for the LLM Chatbot.
This file reads configuration from user_config.json if it exists.
"""

import os
import json
from pathlib import Path

# Path to user config file
CONFIG_FILE = Path("user_config.json")

# Default configurations
DEFAULT_CONFIG = {
    # General settings
    "default_model_type": "Ollama",
    "rtl_enabled": True,
    
    # Ollama settings
    "default_ollama_model": "",  # Will be populated from available models
    
    # Azure OpenAI settings
    "azure_deployment": "",
    "azure_api_version": "2023-06-01-preview",
    "azure_endpoint": "",
    "azure_api_key": "",
    
    # Google Gemini settings
    "gemini_model": "gemini-1.5-pro",
    "gemini_api_key": "",
    
    # UI Settings
    "app_title": "LLM Chatbot",
    "chat_history_title": "Chat History",
    "provider_info_prefix": "Current provider: "
}

def load_config():
    """
    Load configuration from the user config file if it exists,
    otherwise use the default configuration.
    """
    config = DEFAULT_CONFIG.copy()
    
    # If user config file exists, load and update values
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                user_config = json.load(f)
                config.update(user_config)
                print(f"Loaded configuration from {CONFIG_FILE}")
        except Exception as e:
            print(f"Error loading user config: {e}")
    else:
        print(f"No user config file found at {CONFIG_FILE}, using defaults")
    
    # Override with environment variables if they exist
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        config["azure_endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")
    if os.getenv("AZURE_OPENAI_API_KEY"):
        config["azure_api_key"] = os.getenv("AZURE_OPENAI_API_KEY")
    if os.getenv("GOOGLE_API_KEY"):
        config["gemini_api_key"] = os.getenv("GOOGLE_API_KEY")
    
    return config

def save_config(config):
    """
    Save the current configuration to the user config file.
    Sensitive data like API keys will be saved only if explicitly included.
    """
    # Create a copy of the config to save
    save_data = {}
    
    # Only save non-empty and non-sensitive values by default
    for key, value in config.items():
        if key in ["azure_api_key", "gemini_api_key"]:
            # Skip sensitive data unless it's explicitly set
            if value and value != DEFAULT_CONFIG[key]:
                save_data[key] = value
        elif value != DEFAULT_CONFIG[key]:
            # Only save values that differ from defaults
            save_data[key] = value
    
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(save_data, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False 