# Embedding generation and FAISS indexing logic

from typing import List, Tuple, Any
import faiss
import numpy as np

def generate_embeddings(chunks: List[str], embedding_dim: int = 1536) -> Tuple[List[List[float]], List[str]]:
    """
    Generates embeddings for each chunk using OpenAI or any model API.

    Args:
        chunks (List[str]): List of text chunks.
        embedding_dim (int): Expected embedding dimension.

    Returns:
        Tuple[List[List[float]], List[str]]: Validated embeddings and original text chunks.
    """
    # Example placeholder — replace with actual embedding logic
    embeddings = [[0.0] * embedding_dim for _ in chunks]

    # Validation: filter out inconsistent embeddings
    valid_embeddings = []
    valid_chunks = []
    for emb, chunk in zip(embeddings, chunks):
        if len(emb) == embedding_dim:
            valid_embeddings.append(emb)
            valid_chunks.append(chunk)
        else:
            print(f"Skipped chunk due to invalid embedding length: {len(emb)}")

    if not valid_embeddings:
        raise ValueError("No valid embeddings were generated.")

    return valid_embeddings, valid_chunks



def create_faiss_index(embeddings: List[List[float]]):
    """
    Create a FAISS index from the given embeddings.
    
    Args:
        embeddings (List[List[float]]): 2D list of float embeddings.

    Returns:
        faiss.Index: FAISS index built with the embeddings.
    """
    if not embeddings or not all(isinstance(e, list) for e in embeddings):
        raise ValueError("Embeddings must be a non-empty list of lists.")

    # Ensure all embeddings have the same dimension
    embedding_lengths = list(map(len, embeddings))
    if len(set(embedding_lengths)) != 1:
        raise ValueError(f"Inconsistent embedding dimensions found: {set(embedding_lengths)}")

    try:
        embeddings_np = np.array(embeddings, dtype='float32')
    except Exception as e:
        raise ValueError("Error converting embeddings to NumPy array.") from e

    # Create FAISS index
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    return index



def load_faiss_index(index_path: str) -> Any:
    """
    Loads a FAISS index from disk.

    Args:
        index_path (str): Path to the saved FAISS index.

    Returns:
        Any: Loaded FAISS index object.
    """
    index = faiss.read_index(index_path)  # Load the index from disk
    return index


