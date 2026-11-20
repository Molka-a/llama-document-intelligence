"""Rules added after the first real extraction (prompt cv-v2)."""

from app.schemas import CVDocument
from app.schemas.cv import Education, Experience


def test_employment_type_moved_out_of_title():
    job = Experience.model_validate({"title": "Machine Learning Engineer (Freelance)"})
    assert job.title == "Machine Learning Engineer"
    assert job.employment_type == "Freelance"


def test_title_without_type_is_unchanged():
    job = Experience.model_validate({"title": "AI Engineer Intern"})
    assert job.title == "AI Engineer Intern" and job.employment_type is None


def test_real_end_date_means_not_current():
    job = Experience.model_validate(
        {"title": "X", "start_date": "Jan 2026", "end_date": "Jul 2026", "is_current": True})
    assert job.is_current is False


def test_all_caps_name_is_title_cased():
    cv = CVDocument.model_validate({"contact": {"full_name": "MOLKA AMARA"}})
    assert cv.contact.full_name == "Molka Amara"


def test_mixed_case_name_is_kept():
    cv = CVDocument.model_validate({"contact": {"full_name": "Jean-Luc McDonald"}})
    assert cv.contact.full_name == "Jean-Luc McDonald"


def test_dates_come_right_after_title_and_company():
    """The model writes fields in schema order: dates must not be last."""
    for model, first in ((Experience, "title"), (Education, "degree")):
        keys = list(model.model_json_schema()["properties"])
        assert keys[0] == first
        assert keys.index("start_date") < keys.index("is_current") < len(keys) - 1

def test_skill_lines_copied_whole_are_split_into_groups():
    cv = CVDocument.model_validate({"skills": [{"category": "Skills", "skills": [
        "Cloud & tools: Docker, Kubeflow, Azure (basics)",
        "Software: Python (Pandas, NumPy), FastAPI",
    ]}]})
    assert [g.category for g in cv.skills] == ["Cloud & tools", "Software"]
    assert cv.skills[0].skills == ["Docker", "Kubeflow", "Azure (basics)"]
    assert cv.skills[1].skills == ["Python (Pandas, NumPy)", "FastAPI"]   # commas in () kept


def test_normal_skill_groups_are_unchanged():
    cv = CVDocument.model_validate({"skills": [{"category": "ML", "skills": ["PyTorch", "YOLOv8"]}]})
    assert len(cv.skills) == 1 and cv.skills[0].skills == ["PyTorch", "YOLOv8"]


def test_bullet_markers_are_removed():
    cv = CVDocument.model_validate({
        "experience": [{"title": "X", "achievements": ["- Built a pipeline", "• Cut costs by 40%"]}],
        "education": [{"degree": "MSc", "details": ["- Thesis: 17/20"]}],
    })
    assert cv.experience[0].achievements == ["Built a pipeline", "Cut costs by 40%"]
    assert cv.education[0].details == ["Thesis: 17/20"]


def test_negative_numbers_are_not_bullets():
    cv = CVDocument.model_validate({"experience": [{"title": "X", "achievements": ["-40% processing time"]}]})
    assert cv.experience[0].achievements == ["-40% processing time"]


def test_location_keeps_first_segment():
    cv = CVDocument.model_validate({"contact": {"location": "Rabat, Morocco · relocating to Berlin (visa)"}})
    assert cv.contact.location == "Rabat, Morocco"