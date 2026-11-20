# Llama Document Intelligence

**Turn CVs and cover letters (PDF, DOCX) into validated, structured JSON with a local Llama model.**

A FastAPI service that extracts and cleans document text, then (in progress) uses a small quantised Llama model running locally through [Ollama](https://ollama.com) to fill strict Pydantic schemas. Everything runs on a laptop with 8 GB of RAM: no document ever leaves the machine.

> **Status:** All phases completed . See the [roadmap](#roadmap).

---

## Why this project

Most "AI document parsing" demos send a PDF to a large cloud model and print whatever comes back. In a real company that fails in quiet ways: messy PDF text confuses the model, outputs drift from the expected format, and nobody measures accuracy.

This project is built the other way round:

1. **Clean input first.** Robust PDF/DOCX extraction and text normalisation before any model sees the text.
2. **Strict output contract.** The LLM must return JSON matching Pydantic schemas; anything invalid is rejected, not silently accepted.
3. **Measured, not assumed.** An evaluation suite (per-field precision, recall, F1 and latency) to compare models objectively.
4. **Local and private.** A small model on local hardware, suitable for sensitive HR documents.

---

## Pipeline

```
 PDF / DOCX
     │
     ▼
 ┌──────────────────────┐
 │ 1. Extraction        │  PyMuPDF · python-docx (paragraphs + tables in order)
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 2. Text cleaning     │  Unicode, bullets, hyphenation, whitespace
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 3. Local LLM         │  Llama via Ollama, schema-constrained output   (in progress)
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 4. Validation        │  Pydantic: CVDocument / CoverLetterDocument
 └──────────┬───────────┘
            ▼
      Structured JSON
```

---

## Features

### Done
- **Upload API** — `POST /documents/upload` accepts `.pdf` and `.docx` (max 10 MB) with clear HTTP errors:

  | Case | Code |
  |---|---|
  | Unsupported file type | 415 |
  | Empty, corrupted or fake file | 400 |
  | File over 10 MB | 413 |
  | Password-protected PDF | 422 |

- **PDF extraction** — per-page text, character counts, metadata, `%PDF` header check, encrypted-file detection, empty-PDF flag (scanned documents).
- **DOCX extraction** — paragraphs *and* tables in document order, core metadata; same output shape as PDFs so later stages don't care about the format.
- **Text cleaning** — Unicode NFKC normalisation, typographic quotes and dashes, invisible characters, list bullets vs inline separators, words split across lines, lone bullets re-joined with their text.
- **Pydantic schemas**
  - `CVDocument`: contact, headline, summary, skill groups, experience, education, languages, certifications.
  - `CoverLetterDocument`: sender, company, position, key skills, motivation, availability.
  - Dates normalised from English, French and German (`"Aug 2026"`, `"août 2026"`, `"08/2026"` → `"2026-08"`; `"Present"`, `"heute"` → current role).
  - Validation: email and phone format, end date not before start date, case-insensitive skill de-duplication, unknown LLM keys ignored.
  - Automatic routing to the right schema via a `document_type` discriminator.
- **Typed API responses** — the endpoint validates its own output with a response model, documented in Swagger.
- **49 automated tests** (pytest) covering extraction, cleaning, schemas and the API.

### Planned
See the [roadmap](#roadmap).

---

## Tech stack

| Area | Tools |
|---|---|
| API | FastAPI, Uvicorn |
| Validation | Pydantic v2 |
| Extraction | PyMuPDF, python-docx |
| LLM (in progress) | Ollama, small quantised Llama models |
| Testing | pytest, FastAPI TestClient |
| Language | Python 3.13 |

---

## Project structure

```
llama-document-intelligence/
├── app/
│   ├── main.py                  # FastAPI app, /health
│   ├── api/routes.py            # POST /documents/upload
│   ├── schemas/
│   │   ├── common.py            # BaseSchema, date normalisation, ContactInfo
│   │   ├── cv.py                # CVDocument, Experience, Education, SkillGroup
│   │   ├── cover_letter.py      # CoverLetterDocument
│   │   └── document.py          # API response models, document-type routing
│   ├── services/
│   │   ├── pdf_service.py       # PDF extraction
│   │   ├── docx_service.py      # DOCX extraction
│   │   ├── text_cleaner.py      # Text normalisation
│   │   ├── exceptions.py        # Domain errors
│   │   └── llm_service.py       # Ollama client (Phase 3)
│   └── prompts/                 # LLM prompts (Phase 3)
├── tests/                       # 49 pytest tests
└── requirements.txt
```

---

## Getting started

**Requirements:** Python 3.11+ (developed on 3.13). Ollama is only needed from Phase 3.

```bash
git clone https://github.com/molka-a/llama-document-intelligence.git
cd llama-document-intelligence

python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

Run the API:

```bash
python -m uvicorn app.main:app --reload
```

Interactive documentation: <http://127.0.0.1:8000/docs>

---

## Usage

```bash
curl -X POST http://127.0.0.1:8000/documents/upload \
  -F "file=@my_cv.pdf;type=application/pdf"
```

Response (shortened):

```json
{
  "status": "success",
  "message": "PDF processed successfully",
  "document": {
    "filename": "my_cv.pdf",
    "file_type": "pdf",
    "page_count": 1,
    "text": "JANE DOE\nApplied AI Engineer\n...\nPROFESSIONAL EXPERIENCE\nAI Engineer - Acme, Berlin\nJan 2026 - Present\n- Built ...",
    "pages": [{ "page_number": 1, "text": "...", "char_count": 4518 }],
    "metadata": { "format": "PDF 1.4", "title": "..." },
    "is_empty": false
  }
}
```

Target output once the LLM layer is in place (validated by `CVDocument`):

```json
{
  "document_type": "cv",
  "contact": { "full_name": "Jane Doe", "email": "jane@example.com", "linkedin": "linkedin.com/in/jane-doe" },
  "experience": [
    { "title": "AI Engineer", "company": "Acme", "start_date": "2026-01", "end_date": null, "is_current": true,
      "achievements": ["Built ..."] }
  ],
  "skills": [{ "category": "Machine learning", "skills": ["PyTorch", "scikit-learn"] }]
}
```

---

## Testing

```bash
python -m pytest -v
```

```
49 passed
```

| Test file | Covers |
|---|---|
| `test_pdf_service.py` | valid, blank, fake, corrupted and encrypted PDFs |
| `test_docx_service.py` | paragraphs, tables in order, empty and invalid files |
| `test_text_cleaner.py` | whitespace, special characters, bullets, hyphenation |
| `test_schemas.py` | multilingual dates, validation rules, de-duplication, schema routing |
| `test_api.py` | upload endpoint and HTTP error codes |

Tests build their PDF and Word files in memory, so no sample documents are needed.

---

## Engineering notes

- **The Word bullet that deleted list items.** Word often writes bullets as `U+F0B7`, a private-use character. The cleaner's control-character filter removed it *before* the bullet rule could convert it, so every Word list item silently lost its marker. A unit test caught it; the fix was to handle bullets before filtering. Small, quiet input bugs like this are exactly what degrades LLM extraction later.
- **Bullets on their own line.** Some PDF generators draw the bullet as a separate text block, so extraction produced `-\nAchievement`. The cleaner re-joins them, so the model sees one achievement per line.
- **Lenient where it helps, strict where it matters.** Every schema field is optional (a CV without a phone number is still a CV), unknown keys from the model are ignored, but invalid emails, impossible dates and reversed date ranges are rejected.

---

## Roadmap

- [x] **Phase 1 — Extraction:** PDF and DOCX extraction, text cleaning, robust error handling
- [x] **Phase 2 — Data model:** Pydantic schemas for CVs and cover letters, typed API responses
- [x] **Phase 3 — Local LLM:** Ollama setup, model selection for 8 GB RAM, schema-constrained JSON output
- [x] **Phase 4 — Intelligent processing:** document classification, structured extraction endpoint, handling of missing or uncertain fields
- [x] **Phase 5 — Evaluation:** labelled sample set, per-field precision / recall / F1, latency, comparison of small models
- [x] **Phase 6 — Finalisation:** logging, error handling, Docker, CI

---

## Author

**Molka Amara** — Applied AI Engineer
[linkedin.com/in/molka-amara](https://linkedin.com/in/molka-amara) · [github.com/molka-a](https://github.com/molka-a)

## License

[MIT](LICENSE)
