"""
Configuration editor for the LLM Chatbot.
This provides a simple UI to edit and save configuration settings.
"""

import streamlit as st
import json
from config import load_config, save_config, DEFAULT_CONFIG, CONFIG_FILE
from ollama_llm_manager import get_installed_models

def main():
    st.set_page_config(page_title="LLM Chatbot Config", page_icon="⚙️")
    st.title("LLM Chatbot Configuration")
    
    # Load current config
    config = load_config()
    
    # Create tabs for different config sections
    tabs = st.tabs(["General Settings", "Ollama Settings", "Azure OpenAI Settings", "Google Gemini Settings", "UI Settings"])
    
    with tabs[0]:
        st.header("General Settings")
        config["default_model_type"] = st.selectbox(
            "Default Model Provider", 
            options=["Ollama", "Azure OpenAI", "Google Gemini"],
            index=["Ollama", "Azure OpenAI", "Google Gemini"].index(config["default_model_type"])
        )
        
        config["rtl_enabled"] = st.toggle("Enable RTL by default", value=config["rtl_enabled"])
    
    with tabs[1]:
        st.header("Ollama Settings")
        try:
            ollama_models = get_installed_models()
            if ollama_models:
                default_index = ollama_models.index(config["default_ollama_model"]) if config["default_ollama_model"] in ollama_models else 0
                config["default_ollama_model"] = st.selectbox(
                    "Default Ollama Model", 
                    options=ollama_models,
                    index=default_index
                )
            else:
                st.warning("No Ollama models found. Make sure Ollama is running.")
        except Exception as e:
            st.error(f"Error connecting to Ollama: {str(e)}")
    
    with tabs[2]:
        st.header("Azure OpenAI Settings")
        config["azure_deployment"] = st.text_input(
            "Default Deployment Name", 
            value=config["azure_deployment"]
        )
        config["azure_api_version"] = st.text_input(
            "Default API Version", 
            value=config["azure_api_version"]
        )
        config["azure_endpoint"] = st.text_input(
            "Default Azure Endpoint", 
            value=config["azure_endpoint"]
        )
        
        # For sensitive data, we use a different approach
        new_api_key = st.text_input(
            "Azure API Key", 
            type="password",
            value=config["azure_api_key"] if config["azure_api_key"] else "",
            help="Leave empty to keep current value or use environment variable"
        )
        
        if new_api_key:
            config["azure_api_key"] = new_api_key
    
    with tabs[3]:
        st.header("Google Gemini Settings")
        config["gemini_model"] = st.text_input(
            "Default Gemini Model", 
            value=config["gemini_model"]
        )
        
        new_gemini_key = st.text_input(
            "Google API Key", 
            type="password",
            value=config["gemini_api_key"] if config["gemini_api_key"] else "",
            help="Leave empty to keep current value or use environment variable"
        )
        
        if new_gemini_key:
            config["gemini_api_key"] = new_gemini_key
    
    with tabs[4]:
        st.header("UI Settings")
        config["app_title"] = st.text_input("Application Title", value=config["app_title"])
        config["chat_history_title"] = st.text_input("Chat History Title", value=config["chat_history_title"]) 
        config["provider_info_prefix"] = st.text_input("Provider Info Prefix", value=config["provider_info_prefix"])
    
    # Save button
    if st.button("Save Configuration"):
        if save_config(config):
            st.success(f"Configuration saved to {CONFIG_FILE}")
        else:
            st.error("Failed to save configuration")
    
    # View current config
    with st.expander("View Current Configuration"):
        # Create a safe version of config to display (hide sensitive info)
        display_config = config.copy()
        if display_config["azure_api_key"]:
            display_config["azure_api_key"] = "********"
        if display_config["gemini_api_key"]:
            display_config["gemini_api_key"] = "********"
        
        st.code(json.dumps(display_config, indent=4))

if __name__ == "__main__":
    main() 