# 📄 RAG Chatbot using LLaMA-3 and FAISS

## Description
This project implements a Retrieval-Augmented Generation (RAG) chatbot that allows users to interact with PDF documents in a conversational manner. The system processes PDF files by extracting their text, splitting the text into smaller chunks, and converting those chunks into embeddings using the SentenceTransformers model. These embeddings are indexed using FAISS for efficient similarity-based search. When a user asks a question, the chatbot retrieves the most relevant chunks from the document, sends the context to a LLaMA-3 model (hosted via the Groq API), and generates a coherent response based on the document content.

This solution combines natural language processing (NLP) techniques with retrieval-based systems for enhanced performance, making it ideal for document-based Q&A tasks.

## Features
- **PDF Upload**: Upload a PDF document through the Streamlit interface.
- **Text Extraction**: Extracts and processes text from the PDF document.
- **Chunking**: Splits the document into chunks for better search and retrieval.
- **FAISS Integration**: Uses FAISS to create an index for fast and efficient similarity search.
- **Contextual Responses**: Retrieves relevant chunks from the document and uses LLaMA-3 to generate accurate and context-aware responses.
- **Interactive Interface**: User-friendly Streamlit interface for asking questions based on the PDF content.

## Technologies Used
- **Streamlit**: For building the interactive web-based UI.
- **FAISS**: Facebook AI Similarity Search for fast and efficient vector similarity searches on document chunks.
- **SentenceTransformers**: Used to create embeddings of text chunks for similarity comparison.
- **Groq (LLaMA-3 API)**: For natural language generation based on the retrieved context from the document.
- **PyPDF2**: A Python library for extracting text content from PDF files.
- **Python**: The core programming language for the backend implementation.

## Setup and Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/Sriman1210/ChatBot.git
    cd rag-chatbot
    ```

2. **Create and activate a virtual environment**:

    If you're using `venv`:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**:

    Ensure you have `requirements.txt` in your repo with the following dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    Or manually install the required libraries:

    ```bash
    pip install faiss-cpu numpy torch sentence-transformers streamlit pypdf groq
    ```

4. **Groq API Setup**:

    You'll need a Groq API key to interact with the LLaMA-3 model. Replace `your_api_key_here` with your actual API key in the code.  
    You can sign up for access at Groq and get your API key.

5. **Run the application**:

    Once everything is set up, run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

    This will start the application in your web browser. You'll be able to upload a PDF and ask questions related to its contents.

## Usage

1. **Upload a PDF**: On the left sidebar of the Streamlit interface, click the "Choose a PDF file" button to upload your PDF.
2. **Ask a Question**: After the file is uploaded and processed, enter your question in the text input field labeled "Ask a question based on the uploaded document".
3. **Receive Answer**: The chatbot will retrieve relevant chunks of the document, generate an answer using LLaMA-3, and display the response.
