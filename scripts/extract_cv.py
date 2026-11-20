"""Run the full pipeline on one CV file and save the result.

Usage (from the project root):
    python -m scripts.extract_cv path/to/cv.pdf
    python -m scripts.extract_cv path/to/cv.docx --model llama3.2:1b
    python -m scripts.extract_cv path/to/cv.pdf --quiet   # no live output
"""

import argparse
import json
import sys
from pathlib import Path

from app.services.docx_service import extract_text_from_docx
from app.services.extraction_service import extract_cv
from app.services.llm_service import DEFAULT_MODEL, LLMError
from app.services.pdf_service import extract_text_from_pdf

EXTRACTORS = {".pdf": extract_text_from_pdf, ".docx": extract_text_from_docx}
OUTPUT_DIR = Path("data/outputs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a structured CV with the local LLM.")
    parser.add_argument("path", type=Path, help="PDF or DOCX file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--quiet", action="store_true", help="Do not print the JSON while it is generated")
    args = parser.parse_args()

    if not args.path.is_file():
        raise SystemExit(f"File not found: {args.path}")

    extractor = EXTRACTORS.get(args.path.suffix.lower())
    if extractor is None:
        raise SystemExit(f"Unsupported file type: {args.path.suffix}")

    document = extractor(args.path.read_bytes(), filename=args.path.name)
    print(f"Text extracted: {len(document['text'])} characters")
    print(f"Sending to {args.model} (this can take a few minutes on CPU)...")
    if not args.quiet:
        print("The model first reads the prompt (about 1 minute), then writes the JSON below:\n")

    def show(piece: str) -> None:
        sys.stdout.write(piece)
        sys.stdout.flush()

    try:
        result = extract_cv(document["text"], model=args.model,
                            on_token=None if args.quiet else show)
    except LLMError as exc:
        raise SystemExit(f"\n\nExtraction failed: {exc}")
    cv = result.data

    print(f"\n\nDone in {result.duration_s} s | attempts: {result.attempts} | "
          f"tokens read: {result.prompt_tokens} | tokens written: {result.output_tokens}")
    print(f"Name:        {cv.contact.full_name}")
    print(f"Email:       {cv.contact.email}")
    print(f"Headline:    {cv.headline}")
    print(f"Experience:  {len(cv.experience)} entries")
    for job in cv.experience:
        end = "present" if job.is_current else job.end_date
        print(f"  - {job.title} @ {job.company} ({job.start_date} -> {end}), "
              f"{len(job.achievements)} achievements")
    print(f"Education:   {len(cv.education)} entries")
    print(f"Skills:      {len(cv.all_skills)} unique skills in {len(cv.skills)} groups")
    print(f"Languages:   {', '.join(f'{l.name} ({l.level})' for l in cv.languages)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"{args.path.stem}.{args.model.replace(':', '-')}.json"
    out_file.write_text(json.dumps({
        "source": args.path.name,
        "model": result.model,
        "attempts": result.attempts,
        "duration_s": result.duration_s,
        "prompt_tokens": result.prompt_tokens,
        "output_tokens": result.output_tokens,
        "cv": cv.model_dump(mode="json"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()