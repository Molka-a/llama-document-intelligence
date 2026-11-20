import pytest
from pydantic import TypeAdapter, ValidationError

from app.schemas import CoverLetterDocument, CVDocument, StructuredDocument
from app.schemas.common import normalize_date

# Reference example: what the LLM should return for the sample CV
SAMPLE_CV = {
    "document_type": "cv",
    "contact": {
        "full_name": "Molka Amara",
        "email": "MolkaAmara220@gmail.com",
        "phone": "+212 6 45 30 55 15",
        "location": "Rabat, Morocco",
        "linkedin": "https://www.linkedin.com/in/molka-amara/",
        "github": "github.com/molka-a",
    },
    "headline": "Applied AI Engineer | LLM Systems, Evals & Data Pipelines | Python",
    "skills": [
        {"category": "Machine learning", "skills": ["PyTorch", "TensorFlow", "pytorch", "YOLOv8"]},
        {"category": "MLOps & cloud", "skills": ["Docker", "Azure (basics)", "PyTorch"]},
    ],
    "experience": [
        {
            "title": "Machine Learning Engineer",
            "company": "Healthaide",
            "location": "Remote",
            "employment_type": "Freelance",
            "start_date": "Aug 2026",
            "end_date": "Sep 2026",
            "achievements": ["Designed a self-hosted vision-LLM pipeline."],
        },
        {
            "title": "AI Engineer",
            "company": "Qomics",
            "location": "Tunis",
            "start_date": "Jan 2026",
            "end_date": "Present",
        },
    ],
    "education": [
        {"degree": "Master in Services Digitalization", "institution": "ESSECT",
         "start_date": "2025", "end_date": "Jul 2026"},
        {"degree": "Master in Business Analytics", "institution": "Esprit School of Business",
         "start_date": 2022, "end_date": 2024},
    ],
    "languages": [
        {"name": "Arabic", "level": "native"},
        {"name": "German", "level": "basic"},
    ],
    "confidence": 0.9,   # unknown key -> ignored
}


# --- Dates ---------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("Aug 2026", "2026-08"),
    ("août 2026", "2026-08"),
    ("Sept. 2025", "2025-09"),
    ("08/2026", "2026-08"),
    ("2026-08-15", "2026-08"),
    ("2025", "2025"),
    (2025, "2025"),
    ("Present", "present"),
    ("heute", "present"),
    ("", None),
    (None, None),
])
def test_normalize_date(raw, expected):
    assert normalize_date(raw) == expected


@pytest.mark.parametrize("raw", ["13/2025", "Foo 2025", "last summer"])
def test_normalize_date_rejects_garbage(raw):
    with pytest.raises(ValueError):
        normalize_date(raw)


# --- CV ------------------------------------------------------------------

def test_sample_cv_is_valid():
    cv = CVDocument.model_validate(SAMPLE_CV)

    assert cv.contact.full_name == "Molka Amara"
    assert cv.contact.email == "molkaamara220@gmail.com"          # lower-cased
    assert cv.contact.linkedin == "linkedin.com/in/molka-amara"   # prefix removed
    assert cv.experience[0].start_date == "2026-08"
    assert cv.education[0].end_date == "2026-07"
    assert cv.education[1].end_date == "2024"
    assert len(cv.experience) == 2 and len(cv.languages) == 2


def test_present_sets_is_current():
    job = CVDocument.model_validate(SAMPLE_CV).experience[1]
    assert job.is_current is True
    assert job.end_date is None


def test_skills_are_deduplicated():
    cv = CVDocument.model_validate(SAMPLE_CV)
    assert cv.skills[0].skills == ["PyTorch", "TensorFlow", "YOLOv8"]
    assert cv.all_skills == ["PyTorch", "TensorFlow", "YOLOv8", "Docker", "Azure (basics)"]


def test_end_before_start_rejected():
    data = {"experience": [{"title": "X", "start_date": "2026-05", "end_date": "2025-01"}]}
    with pytest.raises(ValidationError, match="before start_date"):
        CVDocument.model_validate(data)


def test_same_year_range_is_valid():
    data = {"education": [{"degree": "X", "start_date": "2025", "end_date": "2025"}]}
    assert CVDocument.model_validate(data).education[0].end_date == "2025"


def test_invalid_email_rejected():
    with pytest.raises(ValidationError, match="Invalid email"):
        CVDocument.model_validate({"contact": {"email": "not-an-email"}})


def test_invalid_phone_rejected():
    with pytest.raises(ValidationError, match="Invalid phone"):
        CVDocument.model_validate({"contact": {"phone": "call me"}})


def test_empty_cv_is_valid():
    """Missing information is allowed: every field has a default."""
    cv = CVDocument.model_validate({})
    assert cv.experience == [] and cv.contact.full_name is None


# --- Cover letter and type routing ---------------------------------------

def test_cover_letter_is_valid():
    letter = CoverLetterDocument.model_validate({
        "sender": {"full_name": "Molka Amara", "email": "molkaamara220@gmail.com"},
        "company": "Resonance",
        "position": "Graduate Applied AI Engineer",
        "key_skills": ["Python", "LLMs"],
    })
    assert letter.company == "Resonance"


def test_discriminator_picks_correct_class():
    adapter = TypeAdapter(StructuredDocument)

    assert isinstance(adapter.validate_python(SAMPLE_CV), CVDocument)
    assert isinstance(
        adapter.validate_python({"document_type": "cover_letter", "company": "X"}),
        CoverLetterDocument,
    )


def test_json_schema_can_be_generated():
    """Needed in Phase 3 to tell Ollama which JSON structure to return."""
    schema = CVDocument.model_json_schema()
    assert "experience" in schema["properties"]