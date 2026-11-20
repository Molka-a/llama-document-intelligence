"""Quality review: missing fields and values not found in the document."""

from app.schemas import CVDocument
from app.schemas.cover_letter import CoverLetterDocument
from app.services.quality_service import review

SOURCE = """JANE DOE
Data Engineer | Python, Spark
Berlin, Germany | +49 151 2345 6789 | jane.doe@example.com
PROFILE
Data engineer who builds reliable pipelines.
SKILLS
Data: Python, Spark, SQL
EXPERIENCE
Data Engineer - Acme GmbH, Berlin
Jan 2024 - Present
- Built a streaming pipeline processing 2M events per day.
EDUCATION
MSc Computer Science - TU Berlin
2021 - 2023
LANGUAGES
English (fluent)
"""

GOOD = {
    "contact": {"full_name": "JANE DOE", "email": "jane.doe@example.com",
                "phone": "+49 151 2345 6789", "location": "Berlin, Germany"},
    "headline": "Data Engineer | Python, Spark",
    "summary": "Data engineer who builds reliable pipelines.",
    "skills": [{"category": "Data", "skills": ["Python", "Spark", "SQL"]}],
    "experience": [{"title": "Data Engineer", "company": "Acme GmbH", "location": "Berlin",
                    "start_date": "Jan 2024", "end_date": "Present",
                    "achievements": ["Built a streaming pipeline processing 2M events per day."]}],
    "education": [{"degree": "MSc Computer Science", "institution": "TU Berlin",
                   "start_date": "2021", "end_date": "2023"}],
    "languages": [{"name": "English", "level": "fluent"}],
}


def test_faithful_extraction_needs_no_review():
    report = review(CVDocument.model_validate(GOOD), SOURCE)

    assert report.completeness == 1.0
    assert report.missing_fields == []
    assert report.uncertain_fields == []
    assert report.needs_review is False


def test_invented_values_are_flagged():
    data = {**GOOD, "skills": [{"category": "Data", "skills": ["Python", "Kubernetes"]}]}
    data["education"] = [{**GOOD["education"][0], "location": "Munich"}]

    report = review(CVDocument.model_validate(data), SOURCE)

    flagged = {u.field: u.value for u in report.uncertain_fields}
    assert flagged == {"skills[0].skills[1]": "Kubernetes", "education[0].location": "Munich"}
    assert report.needs_review is True


def test_invented_date_year_is_flagged():
    data = {**GOOD, "experience": [{**GOOD["experience"][0], "start_date": "Jan 2019"}]}
    report = review(CVDocument.model_validate(data), SOURCE)
    assert [u.field for u in report.uncertain_fields] == ["experience[0].start_date"]


def test_phone_matches_whatever_the_spacing():
    data = {**GOOD, "contact": {**GOOD["contact"], "phone": "+4915123456789"}}
    assert review(CVDocument.model_validate(data), SOURCE).uncertain_fields == []


def test_missing_fields_are_listed_with_their_path():
    data = {**GOOD, "contact": {"full_name": "Jane Doe"}, "summary": None,
            "experience": [{"title": "Data Engineer"}]}

    report = review(CVDocument.model_validate(data), SOURCE)

    assert "contact.email" in report.missing_fields
    assert "summary" in report.missing_fields
    assert "experience[0].company" in report.missing_fields
    assert "experience[0].start_date" in report.missing_fields
    assert report.completeness < 1.0


def test_low_completeness_needs_review():
    report = review(CVDocument.model_validate({"contact": {"full_name": "Jane Doe"}}), SOURCE)
    assert report.completeness < 0.7 and report.needs_review is True


def test_cover_letter_summary_is_not_grounded():
    """motivation_summary is written by the model on purpose: never flagged."""
    letter = CoverLetterDocument.model_validate({
        "sender": {"full_name": "Jane Doe"}, "company": "Acme GmbH", "position": "Data Engineer",
        "key_skills": ["Python"], "motivation_summary": "She wants to scale data platforms.",
    })
    report = review(letter, SOURCE)
    assert report.uncertain_fields == [] and report.missing_fields == []
