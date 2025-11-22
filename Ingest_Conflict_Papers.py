import os
import json
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import detect

# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent
PDF_FOLDER = PROJECT_ROOT / "KnowledgeBase" / "Conflict_Papers"
OUTPUT_FOLDER = PROJECT_ROOT / "JsonChunksPapers"
OUTPUT_FILE = OUTPUT_FOLDER / "conflict_kb.json"


# METADATA
# Keys must match the PDF filenames in KnowledgeBase/Conflict_Papers (including .pdf)
PAPER_METADATA = {
    "conflict_paper1.pdf": {
        "paper_id": "paper1",
        "paper_title": "CONFLICT RESOLUTION STRATEGIES IN THE WORKPLACE: EMPIRICAL STUDY OF MANAGING INTERPERSONAL AND TEAM CONFLICTS",
        "paper_authors": "Vishal Singh Varma et al.",
        "year": 2023
    },
    "conflict_paper2.pdf": {
        "paper_id": "paper2",
        "paper_title": "The role of workplace mediation: a critical assessment",
        "paper_authors": "Anthony BENNETT",
        "year": 2014
    },
    "conflict_paper3.pdf": {
        "paper_id": "paper3",
        "paper_title": "Managing conflict in the modern workplace",
        "paper_authors": "CIPD Research Team",
        "year": 2020
    },
    "conflict_paper4.pdf": {
        "paper_id": "paper4",
        "paper_title": "WORKPLACE CONFLICT AND ROLE OF MEDIATION",
        "paper_authors": "Vasudha Dubey et al.",
        "year": 2023
    },
    "conflict_paper5.pdf": {
        "paper_id": "paper5",
        "paper_title": "Workplace Conflict Resolution Through Mediation and Arbitration",
        "paper_authors": "Alan D. Scheinkman",
        "year": 2023
    },
    "conflict_paper6.pdf": {
        "paper_id": "paper6",
        "paper_title": "Mediation: Understanding a Constructive Conflict Management Tool in the Workplace",
        "paper_authors": "Lourdes Munduate et al.",
        "year": 2022
    },
    "conflict_paper7.pdf": {
        "paper_id": "paper7",
        "paper_title": "Mediation for Management: Dealing with Conflict in the Workplace",
        "paper_authors": "Otto Federico von Feigenblatt",
        "year": 2021
    },
    "conflict_paper8.pdf": {
        "paper_id": "paper8",
        "paper_title": "Conflict Resolution in Team: Analyzing the of Conflicts and Best Skills for Resolution",
        "paper_authors": "Tawfig Khidir Ibnouf Adham",
        "year": 2023
    }
}
    



# CONFIG TEXT SPLITTER
# ~200–400 words per chunk.
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1200,   # characters
    chunk_overlap=200, # characters
    separators=["\n\n", "\n", ". ", " ", ""],
)


def ingest_conflict_papers() -> None:
    """
    Load all PDFs from KnowledgeBase/Conflict_Papers,
    split them into chunks, attach metadata, and write
    a single JSON file at JsonChunksPapers/conflict_kb.json.
    """
    if not PDF_FOLDER.exists():
        raise RuntimeError(f"PDF folder not found: {PDF_FOLDER}")

    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"No PDF files found in {PDF_FOLDER}")

    # Ensure metadata is defined for each PDF file
    missing_meta = [
        f.name for f in pdf_files if f.name not in PAPER_METADATA
    ]
    if missing_meta:
        raise RuntimeError(
            "Missing metadata for the following PDF filenames in PAPER_METADATA:\n"
            + "\n".join(missing_meta)
        )

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    all_chunks: List[Dict[str, Any]] = []
    total_chunks = 0

    print(f"Found {len(pdf_files)} PDF(s) in {PDF_FOLDER}.\n")

    for pdf_path in pdf_files:
        filename = pdf_path.name
        meta = PAPER_METADATA[filename]

        paper_id = meta["paper_id"]
        paper_title = meta["paper_title"]
        paper_authors = meta["paper_authors"]
        year = meta["year"]

        print(f"Processing: {filename}  (paper_id={paper_id})")

        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()  # One Document per page

        # Split pages into smaller chunks
        split_docs = TEXT_SPLITTER.split_documents(pages)

        paper_chunk_index = 0

        for doc in split_docs:
            text = doc.page_content or ""
            text = text.strip()

            # Skip empty or very short chunks
            if not text:
                continue
            if len(text) < 50:  # Mainly headings or noise skipped
                continue
            # Language filter (skip non-English text if papers contain any)
            try:
             if detect(text) != "en":
                continue
            except:
                continue

            chunk_id = f"conflict_{paper_id}_{paper_chunk_index:03d}"

            chunk_record = {
                "id": chunk_id,
                "paper_id": paper_id,
                "paper_title": paper_title,
                "paper_authors": paper_authors,
                "year": year,
                "chunk_index": paper_chunk_index,
                "chunk_text": text,
            }

            all_chunks.append(chunk_record)
            paper_chunk_index += 1
            total_chunks += 1

        print(f"  -> kept {paper_chunk_index} chunks from this paper.\n")

    if not all_chunks:
        raise RuntimeError("No valid chunks were created. Check splitter settings or PDFs.\n")

    # Write JSON output
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nDone... Wrote {total_chunks} chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    ingest_conflict_papers()