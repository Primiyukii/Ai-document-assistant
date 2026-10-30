import os
import io
from typing import List, Tuple
from pypdf import PdfReader
from docx import Document as DocxDocument

def extract_text_from_file(file_content: bytes, filename: str) -> List[Tuple[int, str]]:
    """
    Extracts text page by page or section by section from PDF, DOCX, or TXT files.
    Returns a list of tuples: (page_number, text_content).
    """
    ext = filename.lower().split('.')[-1]
    pages_text = []

    if ext == 'pdf':
        try:
            reader = PdfReader(io.BytesIO(file_content))
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages_text.append((i + 1, text.strip()))
        except Exception as e:
            raise ValueError(f"Error parsing PDF file: {str(e)}")

    elif ext == 'docx':
        try:
            doc = DocxDocument(io.BytesIO(file_content))
            full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            # For docx without explicit pages, treat as page 1
            if full_text:
                pages_text.append((1, full_text))
        except Exception as e:
            raise ValueError(f"Error parsing DOCX file: {str(e)}")

    elif ext in ['txt', 'md']:
        try:
            text = file_content.decode('utf-8', errors='ignore')
            if text.strip():
                pages_text.append((1, text.strip()))
        except Exception as e:
            raise ValueError(f"Error parsing text file: {str(e)}")
    else:
        raise ValueError(f"Unsupported file format: .{ext}. Supported formats: .pdf, .docx, .txt, .md")

    return pages_text

def chunk_text(pages_text: List[Tuple[int, str]], chunk_size: int = 500, chunk_overlap: int = 50) -> List[dict]:
    """
    Splits extracted pages into overlapping text chunks.
    Returns list of dicts: {'page_number': int, 'chunk_index': int, 'content': str}
    """
    chunks = []

    for page_num, text in pages_text:
        # Simple character/word based sliding window chunking
        words = text.split()
        if not words:
            continue

        current_chunk_words = []
        current_length = 0
        chunk_idx = 0

        i = 0
        while i < len(words):
            word = words[i]
            current_chunk_words.append(word)
            current_length += len(word) + 1

            if current_length >= chunk_size or i == len(words) - 1:
                chunk_content = " ".join(current_chunk_words)
                chunks.append({
                    "page_number": page_num,
                    "chunk_index": chunk_idx,
                    "content": chunk_content
                })
                chunk_idx += 1

                # Handle overlap by stepping back
                if chunk_overlap > 0 and i < len(words) - 1:
                    overlap_words = []
                    overlap_len = 0
                    j = i
                    while j >= 0 and overlap_len < chunk_overlap:
                        overlap_words.insert(0, words[j])
                        overlap_len += len(words[j]) + 1
                        j -= 1
                    current_chunk_words = overlap_words
                    current_length = overlap_len
                else:
                    current_chunk_words = []
                    current_length = 0
            i += 1

    return chunks
