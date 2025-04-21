# Utility functions for chunking, text cleaning, and loading environment variables

from typing import List
import os

def chunk_text(text: str, max_tokens: int = 500) -> List[str]:
    """
    Splits long text into smaller chunks.

    Args:
        text (str): The full text string.
        max_tokens (int): Max number of words or tokens per chunk.

    Returns:
        List[str]: List of text chunks.
    """
    if not text:
        return []

    # Split the text into words
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        # Check if adding the next word exceeds the max token limit
        if len(current_chunk) + len(word.split()) > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = []  # Reset for the next chunk
        current_chunk.append(word)

    # Add any remaining words as the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def clean_text(text: str) -> str:
    """
    Cleans extracted text by removing unwanted characters.

    Args:
        text (str): Raw text.

    Returns:
        str: Cleaned text.
    """
    if not text:
        return ""

    # Remove unwanted characters and extra spaces
    cleaned_text = ' '.join(text.split())
    cleaned_text = cleaned_text.replace('\n', ' ').replace('\r', '')
    return cleaned_text

def load_env_vars() -> None:
    """
    Loads environment variables from the .env file.

    Returns:
        None
    """
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Optionally, you can print the loaded environment variables for debugging
    # print("Environment variables loaded:")
    # for key in os.environ.keys():
    #     print(f"{key}: {os.environ[key]}")
