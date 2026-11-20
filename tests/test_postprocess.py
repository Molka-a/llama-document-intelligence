"""Corrections added after the first evaluation run."""

import pytest

from app.schemas import CVDocument
from app.schemas.cover_letter import CoverLetterDocument
from app.services.postprocess import fix_cover_letter, fix_cv, recipient_from_greeting


# --- Recipient from the greeting ---------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Jane Doe\n\nDear Ms. Keller,\nI am writing...", "Ms. Keller"),
    ("Dear Hiring Team,\n...", "Hiring Team"),
    ("Hello Resonance team!\n...", "Resonance team"),
    ("Sehr geehrte Frau Schmidt,\n...", "Frau Schmidt"),
    ("To whom it may concern,\n...", None),
    ("Dear Sir or Madam,\n...", None),
    ("No greeting in this text.", None),
])
def test_recipient_from_greeting(text, expected):
    assert recipient_from_greeting(text) == expected


def test_recipient_filled_only_when_missing():
    text = "Dear Ms. Keller,\n..."
    assert fix_cover_letter(CoverLetterDocument(), text).recipient_name == "Ms. Keller"
    kept = fix_cover_letter(CoverLetterDocument(recipient_name="Dr. Keller"), text)
    assert kept.recipient_name == "Dr. Keller"


# --- Education location ------------------------------------------------------

def cv_with(institution, degree, location):
    return CVDocument.model_validate(
        {"education": [{"degree": degree, "institution": institution, "location": location}]})


@pytest.mark.parametrize("line, institution, degree, location, kept", [
    # city only inside the school name: invented by the model
    ("MSc Computer Science - University of Hamburg", "University of Hamburg",
     "MSc Computer Science", "Hamburg", False),
    ("Politecnico di Milano - MSc Management Engineering", "Politecnico di Milano",
     "MSc Management Engineering", "Milan", False),
    # 'Tunis' is not 'Tunisia' (Molka's Esprit line)
    ("Master in Business Analytics - Esprit School of Business, Tunisia",
     "Esprit School of Business", "Master in Business Analytics", "Tunis", False),
    # city really written next to the school (Molka's ESSECT line)
    ("Master in Services Digitalization - ESSECT, Tunis, Tunisia",
     "ESSECT", "Master in Services Digitalization", "Tunis, Tunisia", True),
    ("BSc Physics | ETH Zurich | Zurich", "ETH Zurich", "BSc Physics", "Zurich", True),
])
def test_education_location_kept_only_if_written(line, institution, degree, location, kept):
    cv = fix_cv(cv_with(institution, degree, location), f"EDUCATION\n{line}\n2019 - 2021")
    assert (cv.education[0].location == location) is kept


# --- "null" written as a string -------------------------------------------------

def test_null_strings_become_real_nulls():
    letter = CoverLetterDocument.model_validate({
        "recipient_name": "null", "job_reference": "N/A",
        "sender": {"website": "null", "full_name": "Jane Doe"},
        "key_skills": ["Python", "null"],
    })
    assert letter.recipient_name is None and letter.job_reference is None
    assert letter.sender.website is None and letter.sender.full_name == "Jane Doe"
    assert letter.key_skills == ["Python"]


def test_null_strings_in_cv_entries():
    cv = CVDocument.model_validate({"experience": [
        {"title": "Data Engineer", "location": "null", "achievements": ["null", "Built X"]}]})
    assert cv.experience[0].location is None
    assert cv.experience[0].achievements == ["Built X"]