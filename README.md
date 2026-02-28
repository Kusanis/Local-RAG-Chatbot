# Local RAG Chatbot

A local Retrieval-Augmented Generation (RAG) chatbot powered by Ollama and Gradio. Chat with your documents privately and securely on your own machine.

## Features

- **Local LLM**: Uses Ollama for running LLMs locally
- **Document Processing**: Supports PDF, TXT, MD, CSV, and Excel files
- **Vector Search**: ChromaDB for efficient document retrieval
- **Gradio UI**: Simple and intuitive web interface
- **Privacy-First**: All processing happens locally on your machine

## Prerequisites

1. **Python 3.10+** installed
2. **Ollama** installed and running
3. **Required Ollama models** pulled:
   ```bash
   ollama pull qwen2.5:1.5b
   ollama pull nomic-embed-text
   ```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag_chatbot
   ```

2. **Create a virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/Mac
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start Ollama** (in a separate terminal)
   ```bash
   ollama serve
   ```

2. **Run the application**
   ```bash
   python app.py
   ```

3. **Open in browser**
   Navigate to `http://localhost:7860`

## How to Use

1. **Upload Documents**: Click the upload button and select your files (PDF, TXT, MD, CSV, XLSX)
2. **Process**: Click "Process Documents" to index your files
3. **Chat**: Ask questions about your documents in the chat box
4. **Clear**: Use "Clear Chat" to reset conversation or "Clear All" to remove indexed documents

## Configuration

Edit `config.py` to customize:
- LLM model (`LLM_MODEL`)
- Embedding model (`EMBED_MODEL`)
- Chunk size (`CHUNK_SIZE`)
- Number of retrieved documents (`RETRIEVER_K`)
- Ollama server URL (`OLLAMA_BASE_URL`)

## Default Configuration

- **LLM**: qwen2.5:1.5b
- **Embedding**: nomic-embed-text
- **Chunk Size**: 100
- **Retriever K**: 3

## Tech Stack

- [Gradio](https://www.gradio.app/) - Web UI
- [LangChain](https://langchain.com/) - RAG framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [pypdf](https://pypdf.readthedocs.io/) - PDF processing
- [pandas](https://pandas.pydata.org/) - Data handling

## License

MIT License
