import streamlit as st
from manager import get_installed_models, stream_llm_response

# RTL styling configuration
RTL_STYLE = """
<style>
    .rtl {
        direction: rtl;
        text-align: right;
        font-family: 'Arial', sans-serif;
    }
    .user-message {
        padding: 10px;
        border-radius: 10px 0 10px 10px;
        margin: 5px 0;
        float: right;
        max-width: 80%;
    }
    .bot-message {
        padding: 10px;
        border-radius: 0 10px 10px 10px;
        margin: 5px 0;
        float: right;
        max-width: 80%;
    }
    .stTextInput>div>div>input {
        direction: rtl;
        text-align: right;
    }
</style>
"""

def initialize_session():
    """Initialize Streamlit session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = None

def display_chat_message(role: str, content: str):
    """Display a chat message with RTL styling"""
    div_class = "user-message" if role == "user" else "bot-message"
    with st.chat_message(role):
        st.markdown(
            f'<div class="{div_class} rtl">{content}</div>',
            unsafe_allow_html=True
        )

def main():
    # Configure page and apply styling
    st.set_page_config(page_title="Ollama Chatbot", page_icon="🧑‍💻")
    st.markdown(RTL_STYLE, unsafe_allow_html=True)
    st.header("Ollama Chatbot")
    
    initialize_session()
    
    # Model selection sidebar
    try:
        models = get_installed_models()
        selected_model = st.sidebar.selectbox(
            "Select LLM Model",
            models,
            key='model_selector'
        )
        st.session_state.llm_model = selected_model
    except ConnectionError as e:
        st.sidebar.error(f"Ollama connection error: {str(e)}")
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message['role'], message['content'])
    
    # Handle user input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to history and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                for chunk in stream_llm_response(
                    st.session_state.messages,
                    st.session_state.llm_model
                ):
                    full_response += chunk
                    response_placeholder.markdown(
                        f'<div class="bot-message rtl">{full_response}</div>',
                        unsafe_allow_html=True
                    )
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()