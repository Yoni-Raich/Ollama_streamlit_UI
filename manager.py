from typing import Dict, Generator
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, AIMessage
import requests

class OllamaManager:
    """Singleton manager for Ollama model instances"""
    _instances: Dict[str, ChatOllama] = {}

    def __new__(cls, model_name: str):
        """Get or create a ChatOllama instance for the specified model"""
        if model_name not in cls._instances:
            cls._instances[model_name] = ChatOllama(
                model=model_name,
                streaming=True,
                temperature=0.7,
                # Add other model parameters here
            )
        return cls._instances[model_name]

def get_installed_models() -> list:
    """Fetch list of installed Ollama models"""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        response.raise_for_status()
        return [model['name'] for model in response.json().get('models', [])]
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Ollama connection failed: {str(e)}")

def stream_llm_response(messages: list, model_name: str) -> Generator[str, None, None]:
    """Generate streaming response with model reuse"""
    chat = OllamaManager(model_name)  # Reuse existing instance
    
    # Convert message format for LangChain compatibility
    converted_messages = [
        HumanMessage(content=msg['content']) if msg['role'] == 'user' 
        else AIMessage(content=msg['content'])
        for msg in messages
    ]
    
    # Stream response chunks
    for chunk in chat.stream(converted_messages):
        yield chunk.content