# Streamlit-based chatbot app to answer questions from PDF content using OpenAI + FAISS.

import streamlit as st
from modules.file_handler import save_uploaded_file, extract_text_from_pdf
from modules.embeddings import generate_embeddings, create_faiss_index
from modules.chatbot import get_response_from_context, retrieve_relevant_chunks
from modules.utils import chunk_text, load_env_vars
import os

# Load environment variables from .env
load_env_vars()

# Set Streamlit UI config and title
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("Chat with your PDF")

# Sidebar for file upload
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save uploaded file
    file_path = save_uploaded_file(uploaded_file)

    # Extract text from PDF
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(file_path)

    # Split text into chunks
    with st.spinner("Splitting text into chunks..."):
        text_chunks = chunk_text(pdf_text)

    # Generate embeddings and create FAISS index
    with st.spinner("Generating embeddings and building FAISS index..."):
        embeddings, valid_chunks = generate_embeddings(text_chunks)
        faiss_index = create_faiss_index(embeddings)

    st.success("PDF processed successfully! You can now ask questions.")

    # User query input
    user_query = st.text_input("Ask a question about the PDF content:")

    if user_query:
        # Retrieve relevant chunks and generate response
        with st.spinner("Generating response..."):
            relevant_chunks = retrieve_relevant_chunks(user_query, faiss_index, valid_chunks)
            response = get_response_from_context(user_query, "\n".join(relevant_chunks))

        # Display response
        st.subheader("Response:")
        st.write(response)
else:
    st.info("Please upload a PDF file to get started.")