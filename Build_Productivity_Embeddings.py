import os
import json
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_ROOT / "ChromaEmbeddings"

# Name of the Chroma collection that will store the productivity stat chunks
PRODUCTIVITY_COLLECTION_NAME = "productivity_kb"

# We use the productivity evaluation file as Knowledge Base here.
SYNTHETIC_DATA_DIR = PROJECT_ROOT / "SyntheticEmployeeData"
PRODUCTIVITY_EVAL_FILE = SYNTHETIC_DATA_DIR / "synthetic_productivity_evaluation.json"



def load_productivity_records() -> List[Dict[str, Any]]:
    """
    Load productivity evaluation records from the synthetic JSON file and
    convert each record into a text representation suitable for embedding.

    Expected structure of synthetic_productivity_evaluation.json:
    [
      {
        "day": "Day 1",
        "name": "Alice",
        "evaluations": [
          {"metric": "...", "value": 3.5, "status": "normal"},
          ...
        ],
        "notes": ["...", "..."]
      },
      ...
    ]
    """
    if not PRODUCTIVITY_EVAL_FILE.exists():
        raise RuntimeError(
            f"Productivity evaluation file not found: {PRODUCTIVITY_EVAL_FILE}\n"
            "Please generate it by running something like:\n"
            "  python generate_synthetic_productivity_data.py\n"
            "  python generate_synthetic_employeesurvey_data.py\n"
            "  python evaluation_of_productivity.py\n"
        )

    with open(PRODUCTIVITY_EVAL_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise RuntimeError(
            f"Expected a list of records in {PRODUCTIVITY_EVAL_FILE}, "
            f"got {type(data)} instead."
        )

    records: List[Dict[str, Any]] = []

    for idx, item in enumerate(data):
        day = item.get("day", "Unknown day")
        name = item.get("name", "Unknown employee")
        evaluations = item.get("evaluations", [])
        notes = item.get("notes", [])

        lines = [
            f"Day: {day}",
            f"Employee: {name}",
            "",
            "Productivity and well-being evaluations:"
        ]

        for ev in evaluations:
            metric = ev.get("metric", "Unknown metric")
            value = ev.get("value", "Unknown value")
            status = ev.get("status", "Unknown status")
            lines.append(f"- {metric}: {value} ({status})")

        if notes:
            lines.append("")
            lines.append("Notes / qualitative observations:")
            for n in notes:
                lines.append(f"- {n}")

        text = "\n".join(lines).strip()

        # Ensure the text is non-empty
        if not text.strip():
            continue


        rec_id = f"productivity_{idx:04d}_{day}_{name}".replace(" ", "_")

        records.append(
            {
                "id": rec_id,
                "text": text,
                "day": day,
                "employee_name": name,
            }
        )

    if not records:
        raise RuntimeError(
            f"No valid records were constructed from {PRODUCTIVITY_EVAL_FILE}. "
            "Please check the JSON structure."
        )

    print(f"Loaded {len(records)} productivity evaluation records for embedding.")
    return records



def build_productivity_embeddings() -> None:
    # Load environment variables
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Please set it in your .env file.\n"
        )

    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    # Create OpenAI client
    client = OpenAI(api_key=api_key)

    # Ensure Chroma directory exists
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Connect to Chroma
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Get or create the productivity collection
    collection = chroma_client.get_or_create_collection(PRODUCTIVITY_COLLECTION_NAME)

    # Load records from JSON
    records = load_productivity_records()

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for rec in records:
        ids.append(rec["id"])
        documents.append(rec["text"])
        metadatas.append(
            {
                "day": rec["day"],
                "employee_name": rec["employee_name"],
                "source": "synthetic_productivity_evaluation",
            }
        )

    # Create embeddings with OpenAI
    print(f"Creating embeddings for {len(documents)} productivity records...")
    response = client.embeddings.create(
        model=embed_model,
        input=documents,
    )

    embeddings = [item.embedding for item in response.data]

    if len(embeddings) != len(documents):
        raise RuntimeError(
            "Number of embeddings does not match number of documents. "
            "Check embedding response."
        )

    # Upsert into Chroma
    print("\nUpserting embeddings into Chroma collection 'productivity_kb'...")
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"\nDone. Stored embedded chunks in collection {PRODUCTIVITY_COLLECTION_NAME}")


if __name__ == "__main__":
    build_productivity_embeddings()
