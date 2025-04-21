from typing import List, Any
import openai  # no need to import OpenAIObject
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()




# Ensure the OpenAI API key is set
if not openai.api_key:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable or configure it in the code.")

def get_response_from_context(question: str, context: str) -> str:
    """
    Generates a response using OpenAI based on user question and context.

    Args:
        question (str): The user query.
        context (str): The retrieved context from documents.

    Returns:
        str: The chatbot response.
    """
    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or "gpt-4"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )
    
    return response.choices[0].message['content'].strip()

def get_embedding(text: str):
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']
    except Exception as e:
        raise ValueError(f"Error generating embedding: {e}")


def retrieve_relevant_chunks(question: str, index: Any, documents: List[str], k: int = 3) -> List[str]:
    """
    Searches FAISS index for top-k relevant document chunks.

    Args:
        question (str): The user query.
        index (Any): Loaded FAISS index.
        documents (List[str]): List of all embedded chunks.
        k (int): Number of top results to return.

    Returns:
        List[str]: Top-k most relevant text chunks.
    """
    # Convert the query to an embedding if index expects vector input
    # This assumes you have a function or embedding model set up for it
    question_vector = get_embedding(question)  # you need to define this function

    D, I = index.search(question_vector, k)
    relevant_chunks = [documents[i] for i in I[0]]

    return relevant_chunks
