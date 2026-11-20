"""Field-level comparison of an extraction with its gold answer.

For every field we count:
    TP  the model gave the right value
    FP  the model gave a value that is wrong or not in the gold answer
    FN  the gold answer has a value the model missed or got wrong
A wrong value counts as both FP and FN, the usual convention in
information extraction. From these:
    precision = TP / (TP + FP)   "when it answers, is it right?"
    recall    = TP / (TP + FN)   "does it find everything?"
    F1        = harmonic mean of both

Lists of entries (jobs, degrees, languages) are first aligned with the
gold entries by similarity, so a job extracted in a different order is
still compared with the right gold job.
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

FUZZY_THRESHOLD = 0.85      # long texts: 85 % similar counts as correct
ALIGN_THRESHOLD = 0.5       # entries less similar than this are not paired


# ---------------------------------------------------------------------------
# Counts and scorecards
# ---------------------------------------------------------------------------

@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def f1(self) -> float | None:
        # same as 2PR/(P+R), but also defined (0) when the model found nothing
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else None

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn


@dataclass
class Scorecard:
    fields: dict[str, Counts] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)   # what exactly was wrong

    def count(self, name: str, tp: int = 0, fp: int = 0, fn: int = 0) -> None:
        self.fields.setdefault(name, Counts()).add(Counts(tp, fp, fn))

    def error(self, name: str, gold: Any, pred: Any) -> None:
        self.errors.append({"field": name, "expected": gold, "got": pred})

    def merge(self, other: "Scorecard") -> None:
        for name, counts in other.fields.items():
            self.fields.setdefault(name, Counts()).add(counts)
        self.errors += other.errors

    @property
    def total(self) -> Counts:
        total = Counts()
        for counts in self.fields.values():
            total.add(counts)
        return total

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {
            name: {"tp": c.tp, "fp": c.fp, "fn": c.fn,
                   "precision": c.precision, "recall": c.recall, "f1": c.f1}
            for name, c in sorted(self.fields.items())
            if c.tp + c.fp + c.fn                 # nothing to score: not reported
        }


# ---------------------------------------------------------------------------
# Value comparison
# ---------------------------------------------------------------------------

def normalize(value: Any) -> str:
    text = str(value).lower()
    text = re.sub(r"https?://(www\.)?", "", text)
    text = re.sub(r"[\"'`´’]", "", text)
    text = re.sub(r"[^\w@./+%#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip(" .")


def is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, (str, list, dict)) and not value)


def same(gold: Any, pred: Any, fuzzy: bool = False) -> bool:
    g, p = normalize(gold), normalize(pred)
    if g == p:
        return True
    return fuzzy and SequenceMatcher(None, g, p).ratio() >= FUZZY_THRESHOLD


def similarity(gold: Any, pred: Any) -> float:
    if is_empty(gold) or is_empty(pred):
        return 0.0
    return SequenceMatcher(None, normalize(gold), normalize(pred)).ratio()


# ---------------------------------------------------------------------------
# Field comparators
# ---------------------------------------------------------------------------

def compare_value(card: Scorecard, name: str, gold: Any, pred: Any, mode: str = "exact") -> None:
    """mode: 'exact' (normalised), 'fuzzy' (long text) or 'presence' (free text
    written by the model, e.g. a summary: only filled / not filled is scored)."""
    g_empty, p_empty = is_empty(gold), is_empty(pred)
    if g_empty and p_empty:
        return
    if g_empty:
        card.count(name, fp=1)
        card.error(name, None, pred)
    elif p_empty:
        card.count(name, fn=1)
        card.error(name, gold, None)
    elif mode == "presence" or same(gold, pred, fuzzy=(mode == "fuzzy")):
        card.count(name, tp=1)
    else:
        card.count(name, fp=1, fn=1)
        card.error(name, gold, pred)


def compare_set(card: Scorecard, name: str, gold: list, pred: list, fuzzy: bool = False) -> None:
    """Unordered lists (skills, achievements...): each gold item matches at most one prediction."""
    remaining = [p for p in pred if not is_empty(p)]
    for g in (g for g in gold if not is_empty(g)):
        match = next((p for p in remaining if same(g, p, fuzzy)), None)
        if match is None:
            card.count(name, fn=1)
            card.error(name, g, None)
        else:
            card.count(name, tp=1)
            remaining.remove(match)
    card.count(name, fp=len(remaining))
    for extra in remaining:
        card.error(name, None, extra)


def align(gold: list[dict], pred: list[dict], anchors: list[str]) -> tuple[list, list, list]:
    """Pair gold and predicted entries by the similarity of their anchor fields.

    Returns (pairs, unmatched_gold, unmatched_pred).
    """
    def score(g, p):
        return sum(similarity(g.get(a), p.get(a)) for a in anchors) / len(anchors)

    candidates = sorted(
        ((score(g, p), gi, pi) for gi, g in enumerate(gold) for pi, p in enumerate(pred)),
        reverse=True,
    )
    used_g, used_p, pairs = set(), set(), []
    for s, gi, pi in candidates:
        if s < ALIGN_THRESHOLD or gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        pairs.append((gold[gi], pred[pi]))
    return (pairs,
            [g for i, g in enumerate(gold) if i not in used_g],
            [p for i, p in enumerate(pred) if i not in used_p])


def compare_entries(card: Scorecard, prefix: str, gold: list[dict], pred: list[dict],
                    anchors: list[str], values: dict[str, str], lists: dict[str, bool]) -> None:
    """values: field -> mode; lists: list field -> fuzzy."""
    pairs, lost, extra = align(gold or [], pred or [], anchors)
    empty: dict[str, Any] = {}
    for g, p in pairs + [(g, empty) for g in lost] + [(empty, p) for p in extra]:
        for name, mode in values.items():
            compare_value(card, f"{prefix}.{name}", _get(g, name), _get(p, name), mode)
        for name, fuzzy in lists.items():
            compare_set(card, f"{prefix}.{name}", g.get(name) or [], p.get(name) or [], fuzzy)


def _get(entry: dict, name: str) -> Any:
    # an ongoing job is stored as is_current=True and end_date=None: compare it as "present"
    if name == "end_date" and entry.get("is_current"):
        return "present"
    return entry.get(name)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

_CONTACT = ["full_name", "email", "phone", "location", "linkedin", "github", "website"]


def score_cv(gold: dict, pred: dict) -> Scorecard:
    card = Scorecard()
    g_contact, p_contact = gold.get("contact") or {}, pred.get("contact") or {}
    for name in _CONTACT:
        compare_value(card, f"contact.{name}", g_contact.get(name), p_contact.get(name))
    compare_value(card, "headline", gold.get("headline"), pred.get("headline"), "fuzzy")
    compare_value(card, "summary", gold.get("summary"), pred.get("summary"), "fuzzy")

    compare_set(card, "skills", _all_skills(gold), _all_skills(pred))
    compare_entries(card, "experience", gold.get("experience"), pred.get("experience"),
                    anchors=["title", "company"],
                    values={"title": "exact", "company": "exact", "location": "exact",
                            "employment_type": "exact", "start_date": "exact", "end_date": "exact"},
                    lists={"achievements": True})
    compare_entries(card, "education", gold.get("education"), pred.get("education"),
                    anchors=["degree", "institution"],
                    values={"degree": "exact", "institution": "exact", "location": "exact",
                            "start_date": "exact", "end_date": "exact"},
                    lists={"details": True})
    compare_entries(card, "languages", gold.get("languages"), pred.get("languages"),
                    anchors=["name"], values={"name": "exact", "level": "exact"}, lists={})
    compare_set(card, "certifications", gold.get("certifications") or [],
                pred.get("certifications") or [], fuzzy=True)
    compare_set(card, "extracurricular", gold.get("extracurricular") or [],
                pred.get("extracurricular") or [], fuzzy=True)
    return card


def score_cover_letter(gold: dict, pred: dict) -> Scorecard:
    card = Scorecard()
    g_sender, p_sender = gold.get("sender") or {}, pred.get("sender") or {}
    for name in _CONTACT:
        compare_value(card, f"sender.{name}", g_sender.get(name), p_sender.get(name))
    for name in ("recipient_name", "company", "position", "job_reference", "letter_date"):
        compare_value(card, name, gold.get(name), pred.get(name))
    compare_set(card, "key_skills", gold.get("key_skills") or [], pred.get("key_skills") or [])
    compare_value(card, "motivation_summary", gold.get("motivation_summary"),
                  pred.get("motivation_summary"), "presence")
    compare_value(card, "availability", gold.get("availability"),
                  pred.get("availability"), "presence")
    return card


SCORERS = {"cv": score_cv, "cover_letter": score_cover_letter}


def score_document(document_type: str, gold: dict | None, pred: dict | None) -> Scorecard:
    """Score one document. A missing prediction (wrong type, failure) counts every gold value as FN."""
    if gold is None:
        return Scorecard()
    return SCORERS[document_type](gold, pred or {})


def _all_skills(document: dict) -> list[str]:
    return [s for group in document.get("skills") or [] for s in group.get("skills") or []]