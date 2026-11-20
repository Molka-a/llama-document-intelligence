"""The labelled dataset must itself be correct, or every score is wrong."""

import pytest

from app.schemas.cover_letter import CoverLetterDocument
from app.schemas.cv import CVDocument
from app.services.classifier import count_signals
from app.services.quality_service import review
from evaluation.dataset import load_samples

SAMPLES = load_samples(include_private=False)
LABELLED = [s for s in SAMPLES if s.gold is not None]


def test_public_dataset_covers_every_type():
    types = {s.document_type for s in SAMPLES}
    assert types == {"cv", "cover_letter", "other"}
    assert not any(s.private for s in SAMPLES)


@pytest.mark.parametrize("sample", LABELLED, ids=lambda s: s.id)
def test_gold_values_are_written_in_the_document(sample):
    """A gold answer may only contain what the document says (no invented labels)."""
    schema = CVDocument if sample.document_type == "cv" else CoverLetterDocument
    report = review(schema.model_validate(sample.gold), sample.text)
    assert report.uncertain_fields == []


@pytest.mark.parametrize("sample", LABELLED, ids=lambda s: s.id)
def test_rules_classify_labelled_documents(sample):
    cv, letter = count_signals(sample.text)
    winner = "cv" if cv > letter else "cover_letter"
    assert winner == sample.document_type


def test_invoice_is_not_settled_by_rules():
    """The 'other' sample exists to exercise the LLM fallback of the classifier."""
    invoice = next(s for s in SAMPLES if s.document_type == "other")
    cv, letter = count_signals(invoice.text)
    assert cv < 2 and letter < 2