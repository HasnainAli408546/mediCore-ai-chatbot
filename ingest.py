import os
from langchain.text_splitter import CharacterTextSplitter
from app.vectorstore import ingest_chunks  # your module

DATA_DIR = "data/medical_docs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def load_and_split(filepath: str):
    """Read a file and split it into text chunks."""
    with open(filepath, encoding="utf8") as f:
        text = f.read()
    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)

def main():
    all_ids, all_texts, all_metas = [], [], []
    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)
        # Skip non-text files if desired, or integrate a PDF parser here
        chunks = load_and_split(path)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{filename}-{idx}"
            all_ids.append(chunk_id)
            all_texts.append(chunk)
            all_metas.append({"source": filename})

    # Ingest in batches (optional) or all at once
    ingest_chunks(ids=all_ids, texts=all_texts, metadatas=all_metas)
    print(f"Ingested {len(all_ids)} chunks from {DATA_DIR}")

if __name__ == "__main__":
    main()
