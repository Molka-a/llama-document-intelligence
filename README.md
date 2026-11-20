# Llama Document Intelligence

[![CI](https://github.com/Molka-a/llama-document-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/Molka-a/llama-document-intelligence/actions/workflows/ci.yml)

**Turn CVs and cover letters (PDF, DOCX) into validated, structured JSON with a small Llama model running locally.**

A FastAPI service that extracts and cleans document text, classifies the document, and uses `llama3.2:3b` through [Ollama](https://ollama.com) to fill strict Pydantic schemas. Every result is checked against the source text, and the whole pipeline is measured on a labelled dataset. It runs on a laptop with 8 GB of RAM and no GPU: no document ever leaves the machine.

**98% field-level F1** on the labelled evaluation set, **100%** document classification. [Details below](#evaluation).

---

## Why this project

Most "AI document parsing" demos send a PDF to a large cloud model and print whatever comes back. In a real company that fails in quiet ways: messy PDF text confuses the model, outputs drift from the expected format, values are invented, and nobody measures accuracy.

This project is built the other way round:

1. **Clean input first.** Robust PDF/DOCX extraction and text normalisation before any model sees the text.
2. **Strict output contract.** The model's output is constrained to a JSON schema, then validated by Pydantic; invalid output is sent back to the model once for correction.
3. **LLM for understanding, code for normalisation.** Anything with a predictable pattern (dates, bullets, URLs, greetings) is fixed by deterministic code, which never makes those mistakes.
4. **Checked against the source.** Every extracted value is searched for in the document; values that appear nowhere are flagged as possibly invented.
5. **Measured, not assumed.** A labelled dataset and per-field precision, recall and F1. Every improvement below was found and verified by the evaluation.

---

## Pipeline

```
 PDF / DOCX
     │
     ▼
 1. Extraction        PyMuPDF · python-docx (paragraphs and tables in order)
     ▼
 2. Cleaning          Unicode, bullets, hyphenation, whitespace
     ▼
 3. Classification    keyword rules (EN/FR/DE), LLM fallback when unclear
     ▼
 4. LLM extraction    llama3.2:3b via Ollama, schema-constrained, streamed
     ▼
 5. Validation        Pydantic schemas + deterministic repairs
     ▼
 6. Quality review    missing fields, values not found in the document
     ▼
 Structured JSON  (+ completeness, uncertain fields, needs_review)
```

Extraction takes minutes on a CPU, so the API runs it as a **background job**: `POST` returns a job id at once, and the result is polled with `GET /jobs/{id}`.

---

## Features

- **API**
  - `POST /documents/upload`: text and metadata only, instant.
  - `POST /documents/extract`: full extraction as a background job (`202 Accepted` + job id). Optional `document_type=cv|cover_letter` skips classification.
  - `GET /jobs/{job_id}`: `queued` (with queue position) → `processing` → `completed` / `failed`.
  - `GET /health` (liveness) and `GET /health/llm` (is Ollama running, is the model installed).
- **Clear errors.** 415 wrong type, 400 empty or corrupted file, 413 over 10 MB, 422 encrypted PDF or no text (scan), 503 Ollama unavailable. Every response carries an `X-Request-ID`, and unexpected errors return JSON with that id while the traceback goes to the logs.
- **Extraction**
  - CV: contact, headline, summary, skill groups, experience (dates, achievements), education (thesis, grades), languages, certifications.
  - Cover letter: sender, recipient, company, position, job reference, date, key skills, motivation, availability.
  - Dates normalised from English, French and German (`"Aug 2026"`, `"août 2026"`, `"08/2026"` → `"2026-08"`; `"Present"`, `"heute"` → ongoing).
- **Reliability**
  - Output constrained to the JSON schema, with every field required so the model cannot skip one.
  - Streaming, a token limit against repetition loops, and one self-correction round on invalid JSON.
  - Deterministic repairs: skill lines split into groups, bullet markers removed, "(Freelance)" moved out of job titles, all-caps names, `"null"` strings, recipient read from the greeting, URLs checked against their field.
- **Quality review** on every result: completeness score, missing key fields, values not found in the document, `needs_review` flag.
- **Evaluation**: a labelled dataset, per-field metrics, model comparison, saved predictions for re-scoring without the LLM.
- **Engineering**: 170+ tests that need no Ollama (the model is replaced by fakes), structured logs, Docker, and GitHub Actions CI.

---

## Evaluation

5 fictional labelled documents (2 CVs with different layouts, 2 cover letters, 1 invoice), scored field by field against hand-written gold answers. The gold answers are themselves tested: every value must appear in its document.

**Hardware:** laptop CPU, 8 GB RAM, integrated GPU (not used). **Model:** `llama3.2:3b` (Q4, about 2 GB).

| Version | Precision | Recall | F1 | Classification |
|---|---|---|---|---|
| Baseline | 96% | 96% | 96% | 100% |
| + deterministic post-processing | **98%** | **98%** | **98%** | **100%** |

What the evaluation found, and how it was fixed:

| Problem found | Fix | Effect |
|---|---|---|
| `recipient_name` never filled (0/2) | Read it from the greeting line ("Dear Ms. Keller,") | 0% → 100% |
| Education city invented from the school name ("University of Hamburg" → "Hamburg") | Keep a city only if it is written on the degree line, outside the school name | 2 false positives removed |
| A GitHub URL put in the LinkedIn field | Each URL must match its field's domain | fixed |
| **The model copied the example from a field description** ("Go, Kubernetes" into a real headline) | No copyable examples in descriptions | fixed |

**Latency** (median 64 s per document): about 2–3.5 min for a CV, 25–65 s for a cover letter, under 10 s for the classifier fallback. Writing the JSON is the bottleneck: a few tokens per second on this CPU with schema-constrained output. Ollama's prompt cache makes every document after the first of its type noticeably faster.

Reproduce, or compare models:

```bash
python -m evaluation.run_eval --public-only
python -m evaluation.run_eval --public-only --models llama3.2:3b llama3.2:1b
python -m evaluation.run_eval --rescore evaluation/results/run_<date>.json   # no LLM needed
```

Real documents can be evaluated privately in `evaluation/private/` (same format, ignored by Git).

---

## Getting started

**Requirements:** Python 3.12+, and [Ollama](https://ollama.com/download) for the LLM features.

```bash
git clone https://github.com/Molka-a/llama-document-intelligence.git
cd llama-document-intelligence

python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux

python -m pip install -r requirements.txt
ollama pull llama3.2:3b

python -m uvicorn app.main:app --reload
```

Interactive documentation: <http://127.0.0.1:8000/docs>

### With Docker

```bash
docker compose up -d --build
docker compose exec ollama ollama pull llama3.2:3b     # first time only
```

The API is on <http://localhost:8000/docs>. If Ollama already runs on your machine, set `OLLAMA_HOST=http://host.docker.internal:11434` in `docker-compose.yml` and start only the `api` service.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server |
| `LLM_MODEL` | `llama3.2:3b` | Model used for extraction |
| `LLM_NUM_CTX` | `8192` | Context window (tokens) |
| `LLM_MAX_TOKENS` | `3072` | Output limit (stops repetition loops) |
| `LLM_TIMEOUT` | `600` | Max seconds between two streamed chunks |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`... |

---

## Usage

```bash
# 1. Start an extraction (returns immediately)
curl -X POST http://127.0.0.1:8000/documents/extract -F "file=@cv.pdf"
# {"job_id": "069b45...", "status": "queued", "status_url": "/jobs/069b45..."}

# 2. Poll until completed
curl http://127.0.0.1:8000/jobs/069b45...
```

Completed job (shortened):

```json
{
  "status": "completed",
  "duration_s": 131.4,
  "outcome": {
    "document_type": "cv",
    "classification": {"method": "rules", "cv_signals": 5, "letter_signals": 0},
    "result": {
      "contact": {"full_name": "Jane Doe", "email": "jane.doe@example.com", "location": "Berlin, Germany"},
      "experience": [{"title": "Data Engineer", "company": "Acme GmbH", "start_date": "2024-01",
                      "end_date": null, "is_current": true, "achievements": ["Built a streaming pipeline..."]}]
    },
    "review": {"completeness": 1.0, "missing_fields": [], "uncertain_fields": [], "needs_review": false},
    "llm": {"model": "llama3.2:3b", "prompt_version": "cv-v4", "attempts": 1, "output_tokens": 1180}
  }
}
```

The CLI runs the same pipeline on one file and shows the JSON while it is written:

```bash
python -m scripts.extract_cv path/to/cv.pdf
```

---

## Project structure

```
llama-document-intelligence/
├── app/
│   ├── main.py                  # FastAPI app, health checks, logging, error handlers
│   ├── core/logging.py          # log format and level
│   ├── api/routes.py            # /documents/upload, /documents/extract, /jobs/{id}
│   ├── schemas/                 # Pydantic: common, cv, cover_letter, document
│   ├── prompts/                 # versioned prompts (cv-v4, cl-v2, classify-v1)
│   └── services/
│       ├── pdf_service.py, docx_service.py, text_cleaner.py
│       ├── classifier.py        # rules + LLM fallback
│       ├── llm_service.py       # Ollama: schema-constrained, streamed, self-correcting
│       ├── extraction_service.py
│       ├── postprocess.py       # deterministic fixes found by the evaluation
│       ├── quality_service.py   # missing / ungrounded fields
│       └── job_service.py       # background worker
├── evaluation/                  # dataset, metrics, runner
├── scripts/extract_cv.py        # CLI
├── tests/                       # 170+ tests, no Ollama needed
├── Dockerfile, docker-compose.yml
└── .github/workflows/ci.yml     # tests on Python 3.12 and 3.13 + Docker build
```

---

## Engineering notes

- **The model only skips what it is allowed to skip.** Ollama turns the JSON schema into a grammar in which optional fields can be omitted, and the first full-CV run lost every date. Marking every field as required in the schema sent to the model (while Pydantic stays lenient) took dates from 0/14 to 14/14.
- **Schema descriptions are prompts.** Ollama constrains the output with the schema but the model never reads it, so the schema is also written into the prompt. That took contact extraction from 3/5 to 5/5, and the field order is chosen to follow a CV's reading order.
- **Examples in prompts leak.** A description containing `e.g. 'Backend Developer | Go, Kubernetes'` produced a real headline ending in "Go, Kubernetes". Invented examples are now kept out of fields the model copies.
- **Deterministic code where patterns are predictable.** Splitting skill lines, stripping bullets and reading the greeting are 100% reliable in code and cost nothing. They also apply to saved outputs, so the evaluation can be re-scored without calling the model.
- **The Word bullet that deleted list items.** Word writes bullets as `U+F0B7`, a private-use character that the control-character filter removed before the bullet rule could convert it. A unit test caught it.
- **A race condition found by a log line.** Adding a log call in the job queue made a test fail 4 times out of 5: the worker could start a job before `submit()` returned its state. The snapshot is now taken before the job is handed to the worker.
- **One job at a time, on purpose.** On an 8 GB CPU-only machine, two parallel generations each run at half speed and compete for memory.

## Limitations

- The evaluation set is small (5 documents). It shows the method and catches regressions, but it is not a benchmark.
- CPU latency is minutes per CV. A GPU, a smaller output schema or the 1B model would trade some accuracy for speed.
- Jobs are kept in memory and lost on restart (a database or Redis queue would replace this in production).
- Scanned PDFs (images) are detected but not processed: OCR is not included.
- The quality review catches invented values, not values copied from the wrong place in the document.

---

## Roadmap

- [x] **Phase 1 — Extraction:** PDF and DOCX extraction, text cleaning, robust error handling
- [x] **Phase 2 — Data model:** Pydantic schemas for CVs and cover letters, typed API responses
- [x] **Phase 3 — Local LLM:** Ollama, model selection for 8 GB RAM, schema-constrained JSON output
- [x] **Phase 4 — Intelligent processing:** classification, extraction endpoint with background jobs, missing and uncertain fields
- [x] **Phase 5 — Evaluation:** labelled dataset, per-field precision / recall / F1, latency, model comparison
- [x] **Phase 6 — Finalisation:** logging, error handling, Docker, CI

Possible next steps: OCR for scanned PDFs, a persistent job store, a GPU benchmark, and a larger evaluation set.

---

## Author

**Molka Amara** — Applied AI Engineer
[linkedin.com/in/molka-amara](https://linkedin.com/in/molka-amara) · [github.com/molka-a](https://github.com/molka-a)

## License

[MIT](LICENSE)
