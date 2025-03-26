
import asyncio
import requests
import subprocess
from langchain_ollama import ChatOllama

def get_installed_models():
    try:
        response = requests.get('http://localhost:11434/api/tags')
        response.raise_for_status() 
        data = response.json()
        models = data.get('models', [])
        model_names = [model['name'] for model in models]
        return model_names
    except requests.exceptions.RequestException as e:
        print(f'Error fetching models: {e}')
        return []


def pull_model(model_name):
    try:
        result = subprocess.run(['ollama', 'pull', model_name], check=True, capture_output=True, text=True)
        print(f'Successfully pulled model: {model_name}')
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f'Error pulling model: {model_name}')
        print(e.stderr)
        

