"""Prompts sent to the LLM.

Kept in one place so they can be versioned and compared in the
evaluation phase like any other part of the system.
"""

CV_PROMPT_VERSION = "cv-v3"

CV_SYSTEM_PROMPT = """\
You are a precise information extraction engine for CVs (resumes).
You receive the plain text of one CV and fill a JSON object.

Rules:
1. Use ONLY information written in the CV text. Never invent, guess or add facts.
   If a city, date or detail is not written, use null.
2. Copy names, job titles, companies, schools and skills exactly as written.
3. Header: the first line is the name; the line directly under it is the
   "headline"; the city and country in the header go in contact.location.
4. "summary" is the paragraph under a heading like PROFILE, SUMMARY or ABOUT.
5.    location "Berlin". Put every bullet point of that job in "achievements",
   without the leading "-".
6. Dates: each job and degree has a date line such as "Jan 2024 - Jun 2025"
   next to or just below it. Split it: start_date "Jan 2024", end_date "Jun 2025".
   For an ongoing role or degree, end_date is "Present".
7. Education: one entry per degree. Thesis, grades and honours go in "details".
8. Skills: keep the categories used in the CV, one skill per list item.
9. Fill every field you can find. Use null or an empty list only when the
   information is truly absent.
"""


def build_cv_user_prompt(cv_text: str) -> str:
    """Wrap the CV text in clear delimiters so the model knows where it starts and ends."""
    return f"CV text:\n<<<\n{cv_text.strip()}\n>>>"