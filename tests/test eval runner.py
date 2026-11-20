"""Runner tests with fake models: no Ollama, runs in a second."""

import copy

from evaluation.dataset import load_samples
from evaluation.run_eval import evaluate, render_report, rescore

SAMPLES = load_samples(include_private=False)
BY_TEXT = {s.text: s for s in SAMPLES}


def perfect_model(text, requested_type, model):
    """Returns exactly the gold answer: every score must be 100 %."""
    sample = BY_TEXT[text]
    return {"document_type": sample.document_type,
            "classification": {"method": "rules"},
            "result": copy.deepcopy(sample.gold),
            "llm": {"model": model}}


def broken_model(text, requested_type, model):
    raise TimeoutError("Ollama sent nothing for 600 s")


def lazy_model(text, requested_type, model):
    """Right type, empty extraction."""
    sample = BY_TEXT[text]
    return {"document_type": sample.document_type, "classification": {"method": "rules"},
            "result": {} if sample.gold is not None else None}


def test_perfect_model_scores_100_percent():
    run = evaluate(["perfect"], SAMPLES, processor=perfect_model)
    summary = run["models"]["perfect"]["summary"]

    assert summary["micro"]["f1"] == 1.0
    assert summary["classification_accuracy"] == 1.0
    assert summary["failures"] == 0
    assert set(summary["by_type"]) == {"cv", "cover_letter"}


def test_failures_are_recorded_not_fatal():
    run = evaluate(["broken"], SAMPLES, processor=broken_model)
    summary = run["models"]["broken"]["summary"]

    assert summary["failures"] == len(SAMPLES)
    assert summary["micro"]["recall"] == 0.0
    assert "Ollama sent nothing" in run["models"]["broken"]["records"][0]["error"]


def test_models_are_compared_in_one_run():
    run = evaluate(["perfect", "lazy"], SAMPLES, processor=lambda t, k, model:
                   (perfect_model if model == "perfect" else lazy_model)(t, k, model))

    assert run["models"]["perfect"]["summary"]["micro"]["f1"] == 1.0
    assert run["models"]["lazy"]["summary"]["micro"]["recall"] == 0.0
    report = render_report(run)
    assert "| perfect |" in report and "| lazy |" in report
    assert "experience.start_date" in report


def test_rescore_needs_no_model():
    run = evaluate(["perfect"], SAMPLES, processor=perfect_model)
    run["models"]["perfect"]["summary"] = None            # forget the scores

    rescored = rescore(run, SAMPLES)

    assert rescored["models"]["perfect"]["summary"]["micro"]["f1"] == 1.0


def test_wrong_classification_loses_the_whole_document():
    def confused(text, requested_type, model):
        outcome = perfect_model(text, requested_type, model)
        if outcome["document_type"] == "cv":
            outcome["document_type"] = "cover_letter"
        return outcome

    summary = evaluate(["confused"], SAMPLES, processor=confused)["models"]["confused"]["summary"]

    assert summary["classification_accuracy"] < 1.0
    assert summary["by_type"]["cv"]["recall"] == 0.0
    assert summary["by_type"]["cover_letter"]["f1"] == 1.0