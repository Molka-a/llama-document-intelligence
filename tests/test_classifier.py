"""Classifier tests: keyword rules, and the LLM fallback (replaced by a fake)."""

from app.services import classifier
from app.services.classifier import classify_document, count_signals
from app.services.llm_service import LLMResult

CV_TEXT = """JANE DOE
Data Engineer | Python, Spark
PROFILE
Data engineer with five years of experience.
TECHNICAL SKILLS
Programming: Python, SQL
PROFESSIONAL EXPERIENCE
Data Engineer - Acme, Berlin
EDUCATION
MSc Computer Science
"""

LETTER_TEXT = """Jane Doe
Application: Data Engineer
Dear Hiring Team,
I am writing to apply for the Data Engineer role. My experience with Spark...
Kind regards,
Jane Doe
"""

FRENCH_LETTER = """Madame, Monsieur,
Je me permets de vous adresser ma candidature au poste de Data Engineer.
Cordialement,
Jeanne Dupont
"""


def fake_llm(document_type, reason="fake"):
    def _generate(system_prompt, user_prompt, schema, model):
        data = schema(document_type=document_type, reason=reason)
        return LLMResult(data=data, model=model, attempts=1, duration_s=1.0,
                         prompt_tokens=10, output_tokens=5)
    return _generate


def test_cv_headings_are_counted():
    cv, letter = count_signals(CV_TEXT)
    assert cv == 4 and letter == 0


def test_experience_in_a_sentence_is_not_a_heading():
    cv, _ = count_signals("My experience at Acme taught me a lot about education and skills.")
    assert cv == 0


def test_cv_is_classified_by_rules(monkeypatch):
    monkeypatch.setattr(classifier, "generate_structured", None)      # must not be called
    result = classify_document(CV_TEXT)
    assert result.document_type == "cv" and result.method == "rules"


def test_cover_letter_is_classified_by_rules(monkeypatch):
    monkeypatch.setattr(classifier, "generate_structured", None)
    result = classify_document(LETTER_TEXT)
    assert result.document_type == "cover_letter" and result.method == "rules"


def test_french_cover_letter(monkeypatch):
    monkeypatch.setattr(classifier, "generate_structured", None)
    assert classify_document(FRENCH_LETTER).document_type == "cover_letter"


def test_unclear_text_falls_back_to_llm(monkeypatch):
    monkeypatch.setattr(classifier, "generate_structured", fake_llm("other", "It is an invoice"))

    result = classify_document("INVOICE 2026-114\nAmount due: 1,200 EUR")

    assert result.document_type == "other"
    assert result.method == "llm"
    assert result.reason == "It is an invoice"
