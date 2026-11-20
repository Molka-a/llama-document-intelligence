from app.services.text_cleaner import clean_text


def test_empty_text():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_collapses_spaces_and_blank_lines():
    raw = "Hello    world\t\tagain\n\n\n\n\nNext   paragraph   "
    assert clean_text(raw) == "Hello world again\n\nNext paragraph"


def test_keeps_single_paragraph_break():
    assert clean_text("A\n\nB") == "A\n\nB"


def test_normalizes_special_characters():
    raw = "\u201cQuoted\u201d \u2013 it\u2019s\u00a0fine \ufb01le"
    assert clean_text(raw) == "\"Quoted\" - it's fine file"


def test_leading_bullets_become_list_items():
    assert clean_text("\u2022 First\n\uf0b7 Second") == "- First\n- Second"


def test_inline_bullet_becomes_separator():
    raw = "Master A | 2025 - 2026   \u2022   Master B | 2022 - 2024"
    assert clean_text(raw) == "Master A | 2025 - 2026 | Master B | 2022 - 2024"


def test_rejoins_hyphenated_words():
    assert clean_text("intelli-\ngence") == "intelligence"


def test_windows_line_endings():
    assert clean_text("line1\r\nline2\rline3") == "line1\nline2\nline3"

def test_lone_bullet_joined_with_next_line():
    raw = "Job title\n\u2022\nFirst achievement\n\u2022\nSecond achievement"
    assert clean_text(raw) == "Job title\n- First achievement\n- Second achievement"