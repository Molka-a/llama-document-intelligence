"""Text cleaning and normalization for extracted document content."""

import re
import unicodedata

# Characters that PDF/Word extraction commonly leaves behind
_REPLACEMENTS = {
    "\u00a0": " ",   # non-breaking space
    "\u200b": "",    # zero-width space
    "\ufeff": "",    # byte-order mark
    "\u00ad": "",    # soft hyphen
    "\u2018": "'", "\u2019": "'",   # curly single quotes
    "\u201c": '"', "\u201d": '"',   # curly double quotes
    "\u2013": "-", "\u2014": "-",   # en dash, em dash
}

# Bullet characters: •  (Word Symbol font \uf0b7)  ●  ▪  ‣
_BULLETS = "\u2022\uf0b7\u25cf\u25aa\u2023"
_LEADING_BULLET = re.compile(rf"^[ \t]*[{_BULLETS}][ \t]*", re.MULTILINE)
_INLINE_BULLET = re.compile(rf"[ \t]*[{_BULLETS}][ \t]*")

_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_BLANK = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_LONE_BULLET = re.compile(r"^-\n(?=\S)", re.MULTILINE)


def clean_text(text: str) -> str:
    """Normalize whitespace and characters while keeping paragraph structure.

    - Unicode NFKC normalization (fixes ligatures such as the 'fi' ligature)
    - Replaces typographic quotes, dashes and invisible characters
    - Bullets at line start become '- '; bullets inside a line become ' | '
    - Removes remaining control characters
    - Re-joins words hyphenated across a line break
    - Collapses repeated spaces/tabs and strips each line
    - Keeps at most one blank line between paragraphs
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)

    # Bullets must be handled BEFORE removing control characters:
    # the Word bullet \uf0b7 is a private-use char (category "Co")
    # and would otherwise be deleted.
    text = _LEADING_BULLET.sub("- ", text)   # list item  -> "- "
    text = _INLINE_BULLET.sub(" | ", text)   # separator  -> " | "

    # Remove remaining control characters (keep \n and \t)
    text = "".join(
        ch for ch in text
        if ch in "\n\t" or unicodedata.category(ch)[0] != "C"
    )

    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    lines = [_MULTI_SPACE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTI_BLANK.sub("\n\n", text)
    text = _LONE_BULLET.sub("- ", text)   # "-\nText" -> "- Text"

    return text.strip()