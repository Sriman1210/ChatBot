import faiss
import numpy as np
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from groq import Groq
import tempfile
import os

# Initialize Streamlit UI
st.title("📄 RAG Chatbot using LLaMA-3 and FAISS")
st.sidebar.header("🔍 Upload Your PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# Initialize Groq Client (Replace with your API key)
client = Groq(api_key="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")

# Load Sentence Transformer model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip()

# Function to split text into chunks
def chunk_text(text, chunk_size=512):
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Initialize FAISS index
index = None
chunks = []

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name
    
    # Extract and process PDF text
    text = extract_text_from_pdf(pdf_path)
    os.remove(pdf_path)  # Clean up the temporary file
    
    if not text:
        st.error("⚠️ PDF text extraction failed! Try another PDF or use OCR.")
    else:
        st.success("✅ PDF text extracted successfully!")
        chunks = chunk_text(text)
        
        if chunks:
            # Generate embeddings
            chunk_embeddings = embed_model.encode(chunks)
            chunk_embeddings = np.array(chunk_embeddings)
            
            # Create FAISS index
            dimension = chunk_embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(chunk_embeddings)
            
            st.success("✅ FAISS index created successfully!")

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, top_k=3):
    if index is None:
        return ["FAISS index not initialized. Please upload a PDF."]
    
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[idx] for idx in indices[0]]

# Function to generate response using LLaMA-3 (Groq API)
def generate_response(query):
    retrieved_texts = retrieve_relevant_chunks(query)
    context = "\n\n".join(retrieved_texts)
    
    prompt = f"Context:\n{context}\n\nUser Query: {query}\n\nResponse:"
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": "You are an AI assistant providing helpful responses."},
                  {"role": "user", "content": prompt}],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True
    )

    response_text = ""
    for chunk in completion:
        response_text += chunk.choices[0].delta.content or ""

    return response_text

# Streamlit UI for user input
if uploaded_file:
    query = st.text_input("💬 Ask a question based on the uploaded document:")
    
    if query:
        with st.spinner("⏳ Generating response..."):
            response = generate_response(query)
            st.subheader("🤖 Chatbot Response:")
            st.write(response)
