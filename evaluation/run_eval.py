"""Run the evaluation: every sample, every model, scored against the gold answers.

Usage (from the project root, Ollama running):
    python -m evaluation.run_eval                                  # default model, all samples
    python -m evaluation.run_eval --models llama3.2:3b llama3.2:1b # compare models
    python -m evaluation.run_eval --samples cv_01_en letter_02_short
    python -m evaluation.run_eval --public-only                    # numbers for the README
    python -m evaluation.run_eval --rescore evaluation/results/run_....json   # no LLM

Each run saves every prediction, so metrics can be recomputed later
(--rescore) without calling the model again. Results go to
evaluation/results/, which is ignored by Git (private samples may contain
personal data): copy the summary you want into the README.
"""

import argparse
import json
import statistics
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from app.prompts.document_prompts import CV_PROMPT_VERSION, COVER_LETTER_PROMPT_VERSION
from app.services.extraction_service import process_document
from app.services.llm_service import DEFAULT_MODEL
from evaluation.dataset import Sample, load_samples
from evaluation.metrics import Scorecard, score_document

RESULTS_DIR = Path(__file__).parent / "results"

Processor = Callable[..., dict[str, Any]]


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_model(samples: list[Sample], model: str, processor: Processor = process_document,
              log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """Process every sample with one model. Failures are recorded, never fatal."""
    records = []
    for i, sample in enumerate(samples, start=1):
        log(f"[{model}] {i}/{len(samples)} {sample.id} ...")
        start = time.perf_counter()
        try:
            outcome, error = processor(sample.text, "auto", model=model), None
        except Exception as exc:                          # keep going: a failure is a result too
            outcome, error = None, f"{type(exc).__name__}: {exc}"
        seconds = round(time.perf_counter() - start, 2)
        predicted = outcome["document_type"] if outcome else None
        log(f"    -> {predicted or 'FAILED'} in {seconds} s" + (f" ({error})" if error else ""))
        records.append({
            "sample": sample.id,
            "private": sample.private,
            "expected_type": sample.document_type,
            "predicted_type": predicted,
            "classification_method": (outcome or {}).get("classification", {}).get("method"),
            "seconds": seconds,
            "llm": (outcome or {}).get("llm"),
            "prediction": (outcome or {}).get("result"),
            "error": error,
        })
    return records


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_records(records: list[dict], samples: dict[str, Sample]) -> dict[str, Any]:
    """Aggregate metrics for one model's records."""
    overall, by_type = Scorecard(), {"cv": Scorecard(), "cover_letter": Scorecard()}
    correct_types = 0
    for record in records:
        sample = samples[record["sample"]]
        correct_types += record["predicted_type"] == sample.document_type
        if sample.gold is None:
            continue
        # wrong type or failure: nothing usable was extracted for this document
        prediction = record["prediction"] if record["predicted_type"] == sample.document_type else None
        card = score_document(sample.document_type, sample.gold, prediction)
        record["scores"] = card.to_dict()
        record["errors"] = card.errors
        record["f1"] = card.total.f1
        overall.merge(card)
        by_type[sample.document_type].merge(card)

    seconds = [r["seconds"] for r in records if not r["error"]]
    return {
        "documents": len(records),
        "failures": sum(1 for r in records if r["error"]),
        "classification_accuracy": correct_types / len(records) if records else None,
        "micro": _counts(overall.total),
        "by_type": {t: _counts(c.total) for t, c in by_type.items() if c.fields},
        "fields": overall.to_dict(),
        "latency_s": {
            "median": statistics.median(seconds) if seconds else None,
            "max": max(seconds) if seconds else None,
            "total": round(sum(seconds), 1),
        },
    }


def _counts(c) -> dict[str, Any]:
    return {"tp": c.tp, "fp": c.fp, "fn": c.fn,
            "precision": c.precision, "recall": c.recall, "f1": c.f1}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:.0f}%"


def render_report(run: dict[str, Any]) -> str:
    models = run["models"]
    lines = [
        f"# Evaluation report — {run['started_at'][:16].replace('T', ' ')}",
        "",
        f"Samples: {', '.join(run['samples'])}  ",
        f"Prompts: {run['prompt_versions']['cv']} (CV), {run['prompt_versions']['cover_letter']} (cover letter)",
        "",
        "## Summary",
        "",
        "| Model | Precision | Recall | F1 | CV F1 | Letter F1 | Classification | Median latency | Failures |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in models.items():
        s = m["summary"]
        lines.append(
            f"| {name} | {_pct(s['micro']['precision'])} | {_pct(s['micro']['recall'])} "
            f"| **{_pct(s['micro']['f1'])}** | {_pct(s['by_type'].get('cv', {}).get('f1'))} "
            f"| {_pct(s['by_type'].get('cover_letter', {}).get('f1'))} "
            f"| {_pct(s['classification_accuracy'])} | {s['latency_s']['median']} s | {s['failures']} |"
        )

    for name, m in models.items():
        lines += ["", f"## {name}: per field", "",
                  "| Field | TP | FP | FN | Precision | Recall | F1 |", "|---|---|---|---|---|---|---|"]
        for field, c in m["summary"]["fields"].items():
            lines.append(f"| {field} | {c['tp']} | {c['fp']} | {c['fn']} "
                         f"| {_pct(c['precision'])} | {_pct(c['recall'])} | {_pct(c['f1'])} |")
        lines += ["", f"## {name}: per document", "",
                  "| Sample | Expected | Predicted | Method | F1 | Time |", "|---|---|---|---|---|---|"]
        for r in m["records"]:
            lines.append(f"| {r['sample']} | {r['expected_type']} | {r['predicted_type'] or 'FAILED'} "
                         f"| {r['classification_method'] or '–'} | {_pct(r.get('f1'))} | {r['seconds']} s |")
        lines += ["", f"## {name}: errors (expected vs got)", ""]
        for r in m["records"]:
            if r.get("error"):
                lines.append(f"- **{r['sample']}**: FAILED — {r['error']}")
            for e in r.get("errors", []):
                lines.append(f"- **{r['sample']}** `{e['field']}`: expected {_show(e['expected'])}, "
                             f"got {_show(e['got'])}")
    return "\n".join(lines) + "\n"


def _show(value: Any) -> str:
    if value is None:
        return "*nothing*"
    text = str(value)
    return f"`{text[:80]}{'…' if len(text) > 80 else ''}`"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def evaluate(models: list[str], samples: list[Sample],
             processor: Processor = process_document) -> dict[str, Any]:
    by_id = {s.id: s for s in samples}
    run = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "samples": [s.id for s in samples],
        "prompt_versions": {"cv": CV_PROMPT_VERSION, "cover_letter": COVER_LETTER_PROMPT_VERSION},
        "models": {},
    }
    for model in models:
        records = run_model(samples, model, processor)
        run["models"][model] = {"records": records, "summary": score_records(records, by_id)}
    return run


def rescore(run: dict[str, Any], samples: list[Sample]) -> dict[str, Any]:
    """Recompute every metric from saved predictions (after changing metrics or gold)."""
    by_id = {s.id: s for s in samples}
    for model in run["models"].values():
        model["records"] = [r for r in model["records"] if r["sample"] in by_id]
        model["summary"] = score_records(model["records"], by_id)
    run["samples"] = sorted({r["sample"] for m in run["models"].values() for r in m["records"]})
    return run


def save(run: dict[str, Any]) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = run["started_at"].replace(":", "").replace("-", "")
    json_path = RESULTS_DIR / f"run_{stamp}.json"
    report_path = RESULTS_DIR / f"report_{stamp}.md"
    json_path.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(render_report(run), encoding="utf-8")
    return json_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate extraction quality against gold answers.")
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL])
    parser.add_argument("--samples", nargs="+", help="Only these sample ids")
    parser.add_argument("--public-only", action="store_true", help="Ignore evaluation/private/")
    parser.add_argument("--rescore", type=Path, help="Recompute metrics from a saved run (no LLM)")
    args = parser.parse_args()

    samples = load_samples(include_private=not args.public_only)
    if args.samples:
        samples = [s for s in samples if s.id in set(args.samples)]
    if not samples:
        raise SystemExit("No samples selected.")

    if args.rescore:
        run = rescore(json.loads(args.rescore.read_text(encoding="utf-8")), samples)
    else:
        print(f"{len(samples)} samples x {len(args.models)} model(s). On CPU, expect minutes per document.\n")
        run = evaluate(args.models, samples)

    json_path, report_path = save(run)
    print("\n" + render_report(run))
    print(f"Saved: {json_path}\n       {report_path}")


if __name__ == "__main__":
    main()