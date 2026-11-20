"""Metrics tests: each scoring rule on small, hand-checked cases."""

import copy

import pytest

from evaluation.dataset import load_samples
from evaluation.metrics import (
    Counts,
    Scorecard,
    align,
    compare_set,
    compare_value,
    score_cover_letter,
    score_cv,
)

SAMPLES = {s.id: s for s in load_samples(include_private=False)}
CV_GOLD = SAMPLES["cv_01_en"].gold
LETTER_GOLD = SAMPLES["letter_01_en"].gold


def counts(card: Scorecard, name: str) -> tuple[int, int, int]:
    c = card.fields.get(name, Counts())
    return c.tp, c.fp, c.fn


# --- Counts ---------------------------------------------------------------

def test_precision_recall_f1():
    c = Counts(tp=8, fp=2, fn=4)
    assert c.precision == 0.8
    assert c.recall == pytest.approx(8 / 12)
    assert c.f1 == pytest.approx(2 * 0.8 * (8 / 12) / (0.8 + 8 / 12))


def test_no_data_gives_none_not_zero():
    assert Counts().precision is None and Counts().f1 is None


def test_f1_is_zero_when_nothing_was_found():
    assert Counts(tp=0, fp=0, fn=3).f1 == 0.0


# --- Single values ----------------------------------------------------------

@pytest.mark.parametrize("gold, pred, expected", [
    ("Berlin", "Berlin", (1, 0, 0)),       # right
    ("Berlin", "berlin.", (1, 0, 0)),      # case and punctuation ignored
    ("Berlin", None, (0, 0, 1)),           # missed
    (None, "Berlin", (0, 1, 0)),           # invented
    ("Berlin", "Munich", (0, 1, 1)),       # wrong = both
    (None, None, (0, 0, 0)),               # nothing to score
])
def test_compare_value(gold, pred, expected):
    card = Scorecard()
    compare_value(card, "city", gold, pred)
    assert counts(card, "city") == expected


def test_fuzzy_accepts_small_differences_only():
    card = Scorecard()
    compare_value(card, "a", "Built a Kafka pipeline processing 3M events per day.",
                  "Built a Kafka pipeline processing 3M events a day", "fuzzy")
    compare_value(card, "b", "Built a Kafka pipeline", "Designed dbt models", "fuzzy")
    assert counts(card, "a") == (1, 0, 0)
    assert counts(card, "b") == (0, 1, 1)


def test_presence_mode_only_checks_that_something_was_written():
    card = Scorecard()
    compare_value(card, "summary", "Wants real-time data.", "Motivated by streaming work.", "presence")
    assert counts(card, "summary") == (1, 0, 0)


# --- Lists ------------------------------------------------------------------

def test_compare_set_counts_each_item_once():
    card = Scorecard()
    compare_set(card, "skills", ["Python", "SQL", "Spark"], ["python", "Spark", "Spark", "Rust"])
    assert counts(card, "skills") == (2, 2, 1)     # Python, Spark | extra Spark, Rust | SQL missed


def test_alignment_ignores_order():
    gold = [{"title": "Data Engineer", "company": "Acme"}, {"title": "Analyst", "company": "Beta"}]
    pred = [{"title": "Analyst", "company": "Beta"}, {"title": "Data Engineer", "company": "Acme"}]
    pairs, lost, extra = align(gold, pred, ["title", "company"])
    assert sorted((g["company"], p["company"]) for g, p in pairs) == [("Acme", "Acme"), ("Beta", "Beta")]
    assert lost == [] and extra == []


def test_unrelated_entries_are_not_paired():
    pairs, lost, extra = align([{"title": "Data Engineer", "company": "Acme"}],
                               [{"title": "Nurse", "company": "Hospital"}], ["title", "company"])
    assert pairs == [] and len(lost) == 1 and len(extra) == 1


# --- Whole documents --------------------------------------------------------

def test_perfect_cv_scores_100_percent():
    card = score_cv(CV_GOLD, copy.deepcopy(CV_GOLD))
    assert card.total.fp == 0 and card.total.fn == 0
    assert card.total.f1 == 1.0


def test_empty_prediction_has_zero_recall():
    card = score_cv(CV_GOLD, {})
    assert card.total.tp == 0 and card.total.recall == 0.0
    assert card.total.fn > 20


def test_missing_dates_are_counted_per_field():
    pred = copy.deepcopy(CV_GOLD)
    for job in pred["experience"]:
        job["start_date"] = job["end_date"] = None
        job["is_current"] = False
    card = score_cv(CV_GOLD, pred)
    assert counts(card, "experience.start_date") == (0, 0, 2)
    assert counts(card, "experience.end_date") == (0, 0, 2)
    assert counts(card, "experience.title") == (2, 0, 0)


def test_ongoing_job_compares_as_present():
    pred = copy.deepcopy(CV_GOLD)
    pred["experience"][0]["is_current"] = False
    pred["experience"][0]["end_date"] = "2025-06"          # wrong: the job is ongoing
    card = score_cv(CV_GOLD, pred)
    assert counts(card, "experience.end_date") == (1, 1, 1)


def test_cover_letter_with_missing_recipient_and_wrong_position():
    pred = copy.deepcopy(LETTER_GOLD)
    pred["recipient_name"] = None
    pred["position"] = "Senior Data Engineer, Hamburg (2026)"
    card = score_cover_letter(LETTER_GOLD, pred)
    assert counts(card, "recipient_name") == (0, 0, 1)
    assert counts(card, "position") == (0, 1, 1)
    assert counts(card, "company") == (1, 0, 0)