"""
Chunk — Read .txt and .pdf files, split into chunks, save as JSONL
===================================================================
Scans the /train directory recursively for .txt and .pdf files,
chunks the text, formats using the model's chat template, and
saves to /data/train.jsonl.
"""

import os
import sys
import json
import glob

import config
from model_registry import get_chat_template, format_training_example


def extract_text_from_pdf(filepath: str) -> str:
    """Extract all text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print(f"  [ERROR] pypdf not installed. Cannot read: {filepath}")
        print(f"  Fix: pip install pypdf>=4.0.0")
        return ""

    try:
        reader = PdfReader(filepath)
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception as e:
                print(f"  [WARNING] Could not read page {page_num + 1} of {filepath}: {e}")
        return "\n".join(text_parts)
    except Exception as e:
        print(f"  [ERROR] Failed to read PDF {filepath}: {e}")
        return ""


def extract_text_from_txt(filepath: str) -> str:
    """Read a plain text file with flexible encoding."""
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    print(f"  [WARNING] Could not decode {filepath} with any encoding")
    return ""


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split text into overlapping chunks by word count.
    chunk_size: number of words per chunk
    chunk_overlap: number of overlapping words between chunks
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - chunk_overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        if chunk.strip():
            chunks.append(chunk)
        # Stop if we've captured all words
        if i + chunk_size >= len(words):
            break

    return chunks


def run_chunking():
    """
    Main chunking pipeline:
    1. Scan /train for .txt and .pdf
    2. Extract text
    3. Chunk with overlap
    4. Format with chat template
    5. Save as JSONL
    """
    train_dir = config.TRAIN_DIR
    data_dir = config.DATA_DIR
    output_path = os.path.join(data_dir, "train.jsonl")

    # Create data directory
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 65)
    print("  Chunking Training Data")
    print("=" * 65)

    # Find all supported files
    txt_files = glob.glob(os.path.join(train_dir, "**", "*.txt"), recursive=True)
    pdf_files = glob.glob(os.path.join(train_dir, "**", "*.pdf"), recursive=True)
    all_files = txt_files + pdf_files

    if not all_files:
        print(f"\n  [WARNING] No .txt or .pdf files found in {os.path.abspath(train_dir)}")
        print(f"  Drop your training files into the /train folder and run again.")
        print()
        sys.exit(1)

    print(f"\n  Found {len(txt_files)} .txt and {len(pdf_files)} .pdf files")
    print(f"  Chunk size: {config.CHUNK_SIZE} words")
    print(f"  Chunk overlap: {config.CHUNK_OVERLAP} words")
    print()

    # Get chat template for the configured model
    template = get_chat_template(config.MODEL_NAME)

    all_chunks = []
    for filepath in sorted(all_files):
        rel_path = os.path.relpath(filepath, train_dir)
        print(f"  Processing: {rel_path}", end="")

        # Extract text
        if filepath.lower().endswith(".pdf"):
            text = extract_text_from_pdf(filepath)
        else:
            text = extract_text_from_txt(filepath)

        if not text.strip():
            print(" → skipped (empty)")
            continue

        # Chunk
        chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        print(f" → {len(chunks)} chunks")

        # Format each chunk as a training example
        for chunk in chunks:
            system_msg = (
                "You are a knowledgeable assistant. Use the following information "
                "to answer questions accurately and helpfully."
            )
            formatted = format_training_example(
                template=template,
                user_text="Based on the following information, provide a helpful and detailed response:\n\n" + chunk,
                assistant_text="Based on the provided information:\n\n" + chunk,
                system_text=system_msg,
            )
            all_chunks.append({"text": formatted})

    if not all_chunks:
        print("\n  [WARNING] No text could be extracted from any files.")
        print("  Check that your files contain readable text.")
        sys.exit(1)

    # Save JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n  {'─' * 55}")
    print(f"  Total files processed:  {len(all_files)}")
    print(f"  Total chunks created:   {len(all_chunks)}")
    print(f"  Output saved to:        {os.path.abspath(output_path)}")
    print(f"  {'─' * 55}")
    print()

    return output_path


if __name__ == "__main__":
    run_chunking()
