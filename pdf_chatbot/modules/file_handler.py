# Handles PDF file upload and text extraction

from typing import BinaryIO
import PyPDF2
import os

def extract_text_from_pdf(file: BinaryIO) -> str:
    """
    Extracts all text from a PDF file.

    Args:
        file (BinaryIO): A file-like object of the uploaded PDF.

    Returns:
        str: The extracted text from the PDF.
    """
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text() + '\n'
    return text

def save_uploaded_file(uploaded_file: BinaryIO) -> str:
    """
    Saves an uploaded file to the 'data/' directory.

    Args:
        uploaded_file (BinaryIO): File uploaded through Streamlit.

    Returns:
        str: Path to the saved file.
    """
    if not os.path.exists('data/'):
        os.makedirs('data/')
    
    file_path = os.path.join('data/', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


