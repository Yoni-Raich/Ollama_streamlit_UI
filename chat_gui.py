import streamlit as st
import os
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
from ollama_llm_manager import get_installed_models, pull_model
from config import load_config

# Load configuration
config = load_config()

def stream_ollama_response(messages, model_name):
    """
    Generator function that yields each chunk of the Ollama response.
    """
    print(f"Using Ollama model: {model_name}")
    # Initialize the Ollama chat model with streaming enabled
    chat = ChatOllama(model=model_name, streaming=True)
    
    # Convert message format for LangChain compatibility
    converted_messages = [
        HumanMessage(content=msg['content']) if msg['role'] == 'user' 
        else AIMessage(content=msg['content'])
        for msg in messages
    ]
    
    # Yield each chunk as it comes in
    for chunk in chat.stream(converted_messages):
        yield chunk.content

def stream_azure_response(messages, deployment_name, api_version, endpoint, api_key):
    """
    Generator function that yields each chunk of the Azure OpenAI response.
    """
    print(f"Using Azure deployment: {deployment_name}")
    # Initialize the Azure chat model with streaming enabled
    chat = AzureChatOpenAI(
        azure_deployment=deployment_name,
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
        streaming=True
    )
    
    # Convert message format for LangChain compatibility
    converted_messages = [
        HumanMessage(content=msg['content']) if msg['role'] == 'user' 
        else AIMessage(content=msg['content'])
        for msg in messages
    ]
    
    # Yield each chunk as it comes in
    for chunk in chat.stream(converted_messages):
        yield chunk.content

def stream_gemini_response(messages, model_name, api_key):
    """
    Generator function that yields each chunk of the Gemini response.
    """
    print(f"Using Gemini model: {model_name}")
    # Initialize the Gemini chat model with streaming enabled
    chat = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        streaming=True,
        temperature=0.7
    )
    
    # Convert message format for LangChain compatibility
    converted_messages = [
        HumanMessage(content=msg['content']) if msg['role'] == 'user' 
        else AIMessage(content=msg['content'])
        for msg in messages
    ]
    
    # Yield each chunk as it comes in
    for chunk in chat.stream(converted_messages):
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
    st.session_state.model_type = config["default_model_type"]
if 'ollama_model' not in st.session_state:
    st.session_state.ollama_model = config["default_ollama_model"]
if 'rtl_enabled' not in st.session_state:
    st.session_state.rtl_enabled = config["rtl_enabled"]
if 'azure_deployment' not in st.session_state:
    st.session_state.azure_deployment = config["azure_deployment"]
if 'azure_api_version' not in st.session_state:
    st.session_state.azure_api_version = config["azure_api_version"]
if 'azure_endpoint' not in st.session_state:
    st.session_state.azure_endpoint = config["azure_endpoint"]
if 'azure_api_key' not in st.session_state:
    st.session_state.azure_api_key = config["azure_api_key"]
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = config["gemini_model"]
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = config["gemini_api_key"]

# Sidebar configuration
st.sidebar.header("Model Settings")

# Model type selection in sidebar
model_options = ["Ollama", "Azure OpenAI", "Google Gemini"]
model_index = model_options.index(st.session_state.model_type) if st.session_state.model_type in model_options else 0

model_type = st.sidebar.selectbox(
    "Select Model Provider",
    options=model_options,
    index=model_index
)
# Update session state model type
st.session_state.model_type = model_type

# RTL toggle in sidebar
rtl_enabled = st.sidebar.toggle("Enable RTL", value=st.session_state.rtl_enabled)
st.session_state.rtl_enabled = rtl_enabled

# Model-specific settings in sidebar
if st.session_state.model_type == "Ollama":
    st.sidebar.subheader("Ollama Settings")
    ollama_models = get_installed_models()
    st.session_state.ollama_model = st.sidebar.selectbox("Select Ollama Model", ollama_models)

elif st.session_state.model_type == "Azure OpenAI":
    st.sidebar.subheader("Azure OpenAI Settings")
    
    st.session_state.azure_deployment = st.sidebar.text_input(
        "Deployment Name", 
        value=st.session_state.azure_deployment,
        help="Enter your Azure OpenAI deployment name"
    )
    st.session_state.azure_api_version = st.sidebar.text_input(
        "API Version", 
        value=st.session_state.azure_api_version,
        help="Azure OpenAI API version (e.g., 2023-06-01-preview)"
    )
    st.session_state.azure_endpoint = st.sidebar.text_input(
        "Azure Endpoint", 
        value=st.session_state.azure_endpoint,
        help="Enter your Azure OpenAI endpoint URL"
    )
    st.session_state.azure_api_key = st.sidebar.text_input(
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
    if st.session_state.azure_api_version:
        os.environ["OPENAI_API_VERSION"] = st.session_state.azure_api_version

elif st.session_state.model_type == "Google Gemini":
    st.sidebar.subheader("Google Gemini Settings")
    
    st.session_state.gemini_model = st.sidebar.text_input(
        "Model Name", 
        value=st.session_state.gemini_model,
        help="Enter the Gemini model name (e.g., gemini-1.5-pro)"
    )
    st.session_state.gemini_api_key = st.sidebar.text_input(
        "API Key", 
        value=st.session_state.gemini_api_key,
        type="password",
        help="Enter your Google API key"
    )
    
    # Update environment variable with manual input
    if st.session_state.gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = st.session_state.gemini_api_key

# Main UI
st.header(config["app_title"])

# Apply CSS based on RTL setting
st.markdown(generate_css(rtl_enabled), unsafe_allow_html=True)

# Display current provider in main area
st.info(f"{config['provider_info_prefix']}{st.session_state.model_type}")

# Display chat messages
st.subheader(config["chat_history_title"])
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
            if st.session_state.model_type == "Ollama":
                if not st.session_state.ollama_model:
                    raise ValueError("No Ollama model selected")
                
                stream_generator = stream_ollama_response(
                    st.session_state.messages, 
                    st.session_state.ollama_model
                )
            
            elif st.session_state.model_type == "Azure OpenAI":
                if not (st.session_state.azure_deployment and 
                       st.session_state.azure_api_version and 
                       st.session_state.azure_endpoint and 
                       st.session_state.azure_api_key):
                    raise ValueError("Missing Azure OpenAI configuration")
                
                stream_generator = stream_azure_response(
                    st.session_state.messages,
                    st.session_state.azure_deployment,
                    st.session_state.azure_api_version,
                    st.session_state.azure_endpoint,
                    st.session_state.azure_api_key
                )
            
            elif st.session_state.model_type == "Google Gemini":
                if not (st.session_state.gemini_model and st.session_state.gemini_api_key):
                    raise ValueError("Missing Google Gemini configuration")
                
                stream_generator = stream_gemini_response(
                    st.session_state.messages,
                    st.session_state.gemini_model,
                    st.session_state.gemini_api_key
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