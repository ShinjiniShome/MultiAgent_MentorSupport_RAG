import os
import json
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
import chromadb


# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent
JSON_FILE = PROJECT_ROOT / "JsonChunksPapers" / "conflict_kb.json"
CHROMA_DIR = PROJECT_ROOT / "ChromaEmbeddings"


# Name of the Chroma collection that will store the conflict knowledge base chunks
CONFLICT_COLLECTION_NAME = "conflict_kb"

# Minimum length for a text chunk to be embedded
MIN_CHUNK_LENGTH = 50  # characters


def load_chunks() -> List[Dict[str, Any]]:
    """
    Load conflict_kb.json and return a list of chunk records.
    Each record is expected to contain:
      - id (str)
      - paper_id (str)
      - paper_title (str)
      - paper_authors (str)
      - year (int)
      - chunk_index (int)
      - chunk_text (str)
    """
    if not JSON_FILE.exists():
        raise RuntimeError(f"JSON file not found: {JSON_FILE}\n")

    with JSON_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise RuntimeError("Expected conflict_kb.json to contain a list of chunks.\n")

    valid_chunks: List[Dict[str, Any]] = []
    for item in data:
        text = (item.get("chunk_text") or "").strip()
        if not text:
            continue
        if len(text) < MIN_CHUNK_LENGTH:
            continue

        if "id" not in item:
            # Skip invalid records.
            print(f"[WARNING] Skipping record without 'id': {item}")
            continue

        valid_chunks.append(item)

    if not valid_chunks:
        raise RuntimeError("No valid chunks found in conflict_kb.json after filtering.\n")

    print(f"Loaded {len(valid_chunks)} valid chunks from {JSON_FILE}")
    return valid_chunks


def build_conflict_embeddings() -> None:
    """
    Read conflict_kb.json, create embeddings for each chunk_text, and
    store them in a persistent Chroma collection named 'conflict_kb'.
    """
    # Load environment variables
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Please set it in your .env file.\n")

    # Embedding model can be overridden via .env
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    client = OpenAI(api_key=api_key)

    # Prepare Chroma persistent client
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Create or get the conflict collection
    collection = chroma_client.get_or_create_collection(CONFLICT_COLLECTION_NAME)

    # Clear existing data so re-running this script does not duplicate entries
    print(f"Clearing existing data in Chroma collection '{CONFLICT_COLLECTION_NAME}'...\n")
    existing = collection.get()
    ids = existing.get("ids", [])
    if ids:
        collection.delete(ids=ids)
        print(f"Deleted {len(ids)} existing entries.\n")

    else:
        print("Collection was already empty.\n")
    

    chunks = load_chunks()

    print(f"Using embedding model: {embed_model}")
    print(f"Storing embeddings in: {CHROMA_DIR} (collection: {CONFLICT_COLLECTION_NAME})\n")

    # Embed and insert chunks
    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for item in chunks:
        chunk_id = item["id"]
        text = item["chunk_text"].strip()

        meta = {
            "paper_id": item.get("paper_id"),
            "paper_title": item.get("paper_title"),
            "paper_authors": item.get("paper_authors"),
            "year": item.get("year"),
            "chunk_index": item.get("chunk_index"),
        }

        ids.append(chunk_id)
        texts.append(text)
        metadatas.append(meta)

    # Embedding done in small batches
    batch_size = 32
    total = len(texts)
    print(f"Creating embeddings for {total} chunks...\n")

    all_embeddings: List[List[float]] = []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]

        response = client.embeddings.create(
            model=embed_model,
            input=batch_texts,
        )
        batch_embeddings = [d.embedding for d in response.data]
        all_embeddings.extend(batch_embeddings)

        print(f"  -> Embedded chunks {start} to {end - 1}")

    if len(all_embeddings) != total:
        raise RuntimeError(
            f"Embedding count mismatch: expected {total}, got {len(all_embeddings)}\n"
        )

    # Add all vectors to Chroma
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=all_embeddings,
    )

    print(f"\nDone. Stored {total} embedded chunks in collection '{CONFLICT_COLLECTION_NAME}'.")
    print(f"Chroma persistent directory: {CHROMA_DIR}\n")


if __name__ == "__main__":
    build_conflict_embeddings()
