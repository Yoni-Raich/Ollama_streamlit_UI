import streamlit as st
import os
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from ollama_llm_manager import get_installed_models, pull_model

def stream_ollama_response(messages, model_name):
    """
    Generator function that yields each chunk of the Ollama response.
    """
    print(f"Using Ollama model: {model_name}")
    # Initialize the Ollama chat model with streaming enabled
    chat = ChatOllama(model=model_name, streaming=True)
    
    # Yield each chunk as it comes in
    for chunk in chat.stream(messages):
        yield chunk.content

def stream_azure_response(messages, deployment_name, api_version, endpoint):
    """
    Generator function that yields each chunk of the Azure OpenAI response.
    """
    print(f"Using Azure deployment: {deployment_name}")
    # Initialize the Azure chat model with streaming enabled
    chat = AzureChatOpenAI(
        azure_deployment=deployment_name,
        api_version=api_version,
        azure_endpoint=endpoint,
        streaming=True
    )
    
    # Format messages for Azure
    formatted_messages = []
    for msg in messages:
        if msg["role"] == "user":
            formatted_messages.append({"type": "human", "content": msg["content"]})
        else:
            formatted_messages.append({"type": "ai", "content": msg["content"]})
    
    # Yield each chunk as it comes in
    for chunk in chat.stream(formatted_messages):
        yield chunk.content

def stream_gemini_response(messages, model_name):
    """
    Generator function that yields each chunk of the Gemini response.
    """
    print(f"Using Gemini model: {model_name}")
    # Initialize the Gemini chat model with streaming enabled
    chat = ChatGoogleGenerativeAI(model=model_name, streaming=True)
    
    # Format messages for Gemini
    formatted_messages = []
    for msg in messages:
        if msg["role"] == "user":
            formatted_messages.append({"type": "human", "content": msg["content"]})
        else:
            formatted_messages.append({"type": "ai", "content": msg["content"]})
    
    # Yield each chunk as it comes in
    for chunk in chat.stream(formatted_messages):
        yield chunk.content

# Function to generate CSS for the app
def generate_css(rtl_enabled=True):
    base_css = """
    <style>
        .user-message {
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
        }
        .bot-message {
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
        }
    """
    
    if rtl_enabled:
        rtl_css = """
        .rtl {
            direction: rtl;
            text-align: right;
            font-family: 'Arial', sans-serif;
        }
        .user-message {
            float: right;
        }
        .bot-message {
            float: right;
        }
        .stTextInput>div>div>input {
            direction: rtl;
            text-align: right;
        }
        """
        return base_css + rtl_css + "</style>"
    else:
        ltr_css = """
        .ltr {
            direction: ltr;
            text-align: left;
            font-family: 'Arial', sans-serif;
        }
        .user-message {
            float: left;
        }
        .bot-message {
            float: left;
        }
        .stTextInput>div>div>input {
            direction: ltr;
            text-align: left;
        }
        """
        return base_css + ltr_css + "</style>"

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'model_type' not in st.session_state:
    st.session_state.model_type = "Ollama"
if 'ollama_model' not in st.session_state:
    st.session_state.ollama_model = None
if 'rtl_enabled' not in st.session_state:
    st.session_state.rtl_enabled = True
if 'azure_deployment' not in st.session_state:
    st.session_state.azure_deployment = ""
if 'azure_api_version' not in st.session_state:
    st.session_state.azure_api_version = "2023-06-01-preview"
if 'azure_endpoint' not in st.session_state:
    st.session_state.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
if 'azure_api_key' not in st.session_state:
    st.session_state.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = "gemini-1.5-pro"
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = os.getenv("GOOGLE_API_KEY", "")

# Main UI
st.header("LLM Chatbot")

# Model type selection
col1, col2 = st.columns([3, 1])
with col1:
    model_type = st.selectbox(
        "Select Model Provider",
        ["Ollama", "Azure OpenAI", "Google Gemini"],
        key="model_type_selector",
        on_change=lambda: setattr(st.session_state, 'model_type', st.session_state.model_type_selector)
    )
with col2:
    rtl_enabled = st.toggle("Enable RTL", value=st.session_state.rtl_enabled, key="rtl_toggle")
    st.session_state.rtl_enabled = rtl_enabled

# Apply CSS based on RTL setting
st.markdown(generate_css(rtl_enabled), unsafe_allow_html=True)

# Model-specific settings
if model_type == "Ollama":
    ollama_models = get_installed_models()
    st.session_state.ollama_model = st.selectbox("Select Ollama Model", ollama_models)

elif model_type == "Azure OpenAI":
    st.subheader("Azure OpenAI Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.azure_deployment = st.text_input(
            "Deployment Name", 
            value=st.session_state.azure_deployment,
            help="Enter your Azure OpenAI deployment name"
        )
        st.session_state.azure_api_version = st.text_input(
            "API Version", 
            value=st.session_state.azure_api_version,
            help="Azure OpenAI API version (e.g., 2023-06-01-preview)"
        )
    
    with col2:
        st.session_state.azure_endpoint = st.text_input(
            "Azure Endpoint", 
            value=st.session_state.azure_endpoint,
            help="Enter your Azure OpenAI endpoint URL"
        )
        st.session_state.azure_api_key = st.text_input(
            "API Key", 
            value=st.session_state.azure_api_key,
            type="password",
            help="Enter your Azure OpenAI API key"
        )
    
    # Update environment variables with manual inputs
    if st.session_state.azure_api_key:
        os.environ["AZURE_OPENAI_API_KEY"] = st.session_state.azure_api_key
    if st.session_state.azure_endpoint:
        os.environ["AZURE_OPENAI_ENDPOINT"] = st.session_state.azure_endpoint

elif model_type == "Google Gemini":
    st.subheader("Google Gemini Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.gemini_model = st.text_input(
            "Model Name", 
            value=st.session_state.gemini_model,
            help="Enter the Gemini model name (e.g., gemini-1.5-pro)"
        )
    
    with col2:
        st.session_state.gemini_api_key = st.text_input(
            "API Key", 
            value=st.session_state.gemini_api_key,
            type="password",
            help="Enter your Google API key"
        )
    
    # Update environment variable with manual input
    if st.session_state.gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = st.session_state.gemini_api_key

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Apply RTL/LTR styling with message-specific formatting
        div_class = "user-message" if msg["role"] == "user" else "bot-message"
        direction_class = "rtl" if st.session_state.rtl_enabled else "ltr"
        st.markdown(
            f'<div class="{div_class} {direction_class}">{msg["content"]}</div>', 
            unsafe_allow_html=True
        )

# User input handling
prompt = st.chat_input("Type your message here...")
if prompt:
    # Store and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        direction_class = "rtl" if st.session_state.rtl_enabled else "ltr"
        st.markdown(
            f'<div class="user-message {direction_class}">{prompt}</div>', 
            unsafe_allow_html=True
        )
        
    with st.chat_message("assistant"):
        # Create a placeholder to stream the assistant's response
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # Stream response based on the selected model type
            if model_type == "Ollama":
                if not st.session_state.ollama_model:
                    raise ValueError("No Ollama model selected")
                
                stream_generator = stream_ollama_response(
                    st.session_state.messages, 
                    st.session_state.ollama_model
                )
            
            elif model_type == "Azure OpenAI":
                if not (st.session_state.azure_deployment and 
                       st.session_state.azure_api_version and 
                       st.session_state.azure_endpoint and 
                       st.session_state.azure_api_key):
                    raise ValueError("Missing Azure OpenAI configuration")
                
                stream_generator = stream_azure_response(
                    st.session_state.messages,
                    st.session_state.azure_deployment,
                    st.session_state.azure_api_version,
                    st.session_state.azure_endpoint
                )
            
            elif model_type == "Google Gemini":
                if not (st.session_state.gemini_model and st.session_state.gemini_api_key):
                    raise ValueError("Missing Google Gemini configuration")
                
                stream_generator = stream_gemini_response(
                    st.session_state.messages,
                    st.session_state.gemini_model
                )
            
            # Stream the response chunks
            for chunk in stream_generator:
                full_response += chunk
                direction_class = "rtl" if st.session_state.rtl_enabled else "ltr"
                response_placeholder.markdown(
                    f'<div class="bot-message {direction_class}">{full_response}</div>',
                    unsafe_allow_html=True
                )
            
            # Save the complete response
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            response_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})