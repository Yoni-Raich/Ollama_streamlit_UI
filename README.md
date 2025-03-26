# LLM Chatbot

This is a Streamlit-based chat application that supports multiple LLM providers:
- Ollama (local models)
- Azure OpenAI
- Google Gemini

## Features

- Multi-provider support (Ollama, Azure OpenAI, Google Gemini)
- Streaming responses for all providers
- RTL/LTR toggle for language support
- Environment variable support with manual override
- Easy-to-use UI for configuration

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. If using Ollama, make sure it's installed and running on your system
4. Set up environment variables (optional):
   - For Azure OpenAI: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`
   - For Google Gemini: `GOOGLE_API_KEY`

## Usage

1. Run the app:
   ```
   streamlit run chat_gui.py
   ```

2. In the interface:
   - Select your LLM provider (Ollama, Azure OpenAI, or Google Gemini)
   - Configure the provider-specific settings
   - Toggle RTL mode on/off as needed
   - Start chatting!

## Provider Setup

### Ollama
- Select from available local models
- No API keys required

### Azure OpenAI
- Enter your deployment name, API version, endpoint, and API key
- Or set them as environment variables

### Google Gemini
- Enter your model name (default: gemini-1.5-pro)
- Enter your API key
- Or set GOOGLE_API_KEY as an environment variable

## RTL Support

The app includes a toggle for RTL (Right-to-Left) support for languages like Hebrew and Arabic. This affects the text direction and alignment in the chat interface. 