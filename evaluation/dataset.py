"""Labelled evaluation samples.

Each sample is two files with the same name in a samples folder:
    <id>.txt | .pdf | .docx   the document
    <id>.json                 {"document_type": "cv" | "cover_letter" | "other",
                               "gold": {...correct extraction...} or null}

evaluation/samples/  fictional documents, committed (reproducible by anyone)
evaluation/private/  real documents, same format, ignored by Git (personal data)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.schemas.cover_letter import CoverLetterDocument
from app.schemas.cv import CVDocument
from app.services.docx_service import extract_text_from_docx
from app.services.pdf_service import extract_text_from_pdf

ROOT = Path(__file__).parent
PUBLIC_DIR = ROOT / "samples"
PRIVATE_DIR = ROOT / "private"

SCHEMAS = {"cv": CVDocument, "cover_letter": CoverLetterDocument}
_READERS = {
    ".txt": lambda path: path.read_text(encoding="utf-8"),
    ".pdf": lambda path: extract_text_from_pdf(path.read_bytes(), filename=path.name)["text"],
    ".docx": lambda path: extract_text_from_docx(path.read_bytes(), filename=path.name)["text"],
}


@dataclass
class Sample:
    id: str
    document_type: str               # "cv", "cover_letter" or "other"
    text: str
    gold: dict[str, Any] | None      # normalised by the schema, ready to compare
    source: Path
    private: bool


def load_samples(include_private: bool = True) -> list[Sample]:
    folders = [(PUBLIC_DIR, False)] + ([(PRIVATE_DIR, True)] if include_private else [])
    samples = []
    for folder, private in folders:
        if not folder.is_dir():
            continue
        for label_file in sorted(folder.glob("*.json")):
            document = _find_document(label_file)
            samples.append(_load(label_file, document, private))
    return samples


def _find_document(label_file: Path) -> Path:
    for extension in _READERS:
        candidate = label_file.with_suffix(extension)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No .txt/.pdf/.docx document next to {label_file.name}")


def _load(label_file: Path, document: Path, private: bool) -> Sample:
    label = json.loads(label_file.read_text(encoding="utf-8"))
    document_type = label["document_type"]
    gold = label.get("gold")
    if gold is not None:
        # Same validation as model output: gold and prediction are normalised identically
        gold = SCHEMAS[document_type].model_validate(gold).model_dump(mode="json")
    return Sample(
        id=label_file.stem,
        document_type=document_type,
        text=_READERS[document.suffix](document),
        gold=gold,
        source=document,
        private=private,
    )