from __future__ import annotations

import joblib
import numpy as np
import pytest

from naive_bayes_text import config
from naive_bayes_text.data import clean_text
from naive_bayes_text.experiments import BENCHMARK_CANDIDATES, NB_VARIANTS
from naive_bayes_text.exporting import EDGE_CASES
from naive_bayes_text.modeling import (
    labels_at_threshold,
    make_pipeline,
    model_size_kb as size_kb,
    positive_proba,
)


# Cleaning

@pytest.mark.parametrize("raw,expected", [
    # Punctuation survives cleaning on purpose; the vectoriser's
    # token_pattern drops it later, so there is no reason to duplicate that here.
    ("Free entry!! http://spam.com win now", "free entry!! win now"),
    ("I did NOT like this.<br /><br />It was a waste.", "i did not like this. it was a waste."),
    # Digits go, the separator between them does not.
    ("Call 0905-123456 or mail a@b.com", "call - or mail"),
    ("", ""),
    (None, ""),
    (12345, ""),
    ("   lots    of\n\nspace  ", "lots of space"),
])
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_clean_text_removes_every_digit():
    assert not any(ch.isdigit() for ch in clean_text("order 12345 for 99 dollars"))


def test_clean_text_is_idempotent():
    once = clean_text("Hello <br /> WORLD http://x.com 42")
    assert clean_text(once) == once


# Configuration invariants

def test_stopwords_are_not_dropped():
    """Dropping "not" would flip sentiment; the config must never do it."""
    assert config.VEC_PARAMS["stop_words"] is None


def test_bigrams_enabled():
    assert config.VEC_PARAMS["ngram_range"] == (1, 2)


def test_alpha_grid_spans_two_orders_of_magnitude():
    assert min(config.ALPHAS) < 0.01 <= 1.0 < max(config.ALPHAS)


def test_threshold_grid_inside_unit_interval():
    assert all(0.0 < t < 1.0 for t in config.THRESHOLD_GRID)
    assert config.THRESHOLD_GRID == sorted(config.THRESHOLD_GRID)


def test_task_definitions_consistent():
    for name, task in config.TASKS.items():
        assert task["classes"] == sorted(task["classes"]), name
        assert task["positive"] == task["classes"][-1], name
        assert task["label_col"] and task["text_col"], name
        assert len(config.LABEL_MAP[name]) == len(task["classes"]), name


def test_benchmark_vocabulary_is_capped_below_the_shipped_one():
    assert config.BENCH_VEC_PARAMS["max_features"] < config.VEC_PARAMS["max_features"]


# Pipeline behaviour

@pytest.fixture(scope="module")
def fitted():
    X = ["free entry win a prize now", "claim your free gift today",
         "see you at the station for lunch", "can you pick up milk on the way"]
    y = ["spam", "spam", "ham", "ham"]
    pipe = make_pipeline(alpha=1.0).fit(X, y)
    return pipe


def test_pipeline_predicts_labels(fitted):
    assert fitted.predict(["free prize claim now"])[0] == "spam"
    assert fitted.predict(["lunch at the station with friends"])[0] == "ham"


def test_positive_proba_is_last_column(fitted):
    proba = positive_proba(fitted, ["free prize"], "spam")
    assert proba.shape == (1,)
    assert 0.0 <= proba[0] <= 1.0


def test_positive_proba_rejects_wrong_class_order(fitted):
    with pytest.raises(ValueError):
        positive_proba(fitted, ["free prize"], "ham")


def test_labels_at_threshold_respects_cut_off(fitted):
    high = labels_at_threshold(fitted, ["see you at the station"], "spam", 0.99)
    low = labels_at_threshold(fitted, ["see you at the station"], "spam", 0.01)
    assert high[0] == "ham"
    assert low[0] == "spam"


def test_model_size_is_positive(fitted):
    assert size_kb(fitted) > 0


def test_every_candidate_is_instantiable():
    for name, factory in BENCHMARK_CANDIDATES.items():
        assert factory() is not None, name


def test_nb_variants_are_all_listed():
    assert set(NB_VARIANTS) <= set(BENCHMARK_CANDIDATES)


# The two measurements the report quotes in prose

@pytest.fixture(scope="module")
def tiny_split():
    """A small real split, so the measurement code is exercised end to end."""
    from naive_bayes_text.experiments import run_latency_split, run_untuned_baseline

    rng = np.random.RandomState(0)
    words = [f"word{i}" for i in range(200)]
    X = np.array([" ".join(rng.choice(words, 6)) for _ in range(120)], dtype=object)
    y = np.array(["ham"] * 100 + ["spam"] * 20)
    order = rng.permutation(len(y))   # interleave, so every slice has both classes
    X, y = X[order], y[order]
    # keyed by a real task name: both functions read TASKS[name] for the classes
    task = "sms"
    splits = {task: (X[:80], X[80:], y[:80], y[80:])}
    best_alpha = {task: 0.1}
    return task, splits, best_alpha, run_untuned_baseline, run_latency_split


def test_untuned_baseline_reports_the_default_alpha(tiny_split):
    task, splits, _, run_untuned_baseline, _ = tiny_split
    out = run_untuned_baseline(splits, verbose=False)
    row = out[task]
    assert row["alpha"] == 1.0
    # the counts must add up, otherwise the prose quotes an impossible pair
    assert row["n_missed"] <= row["n_positive"]
    assert row["n_positive"] == int((splits[task][3] == "spam").sum())
    assert 0.0 <= row["overconfident_share"] <= 1.0
    for key in ("accuracy", "precision", "recall", "f1", "f1_macro"):
        assert 0.0 <= row[key] <= 1.0, key


def test_latency_split_is_positive_and_named_for_every_model(tiny_split):
    task, splits, best_alpha, _, run_latency_split = tiny_split
    row = run_latency_split(splits, best_alpha, verbose=False)[task]
    for key in ("MultinomialNB_e2e_ms", "LinearSVC_e2e_ms",
                "LogisticRegression_e2e_ms", "vectorise_ms"):
        assert row[key] > 0.0, key
    # The physical point of the measurement: an end-to-end call has to do the
    # vectorisation itself, so it cannot be cheaper than vectorisation alone.
    # The tolerance is there because these are sub-millisecond wall-clock means
    # and a busy machine can reorder two adjacent calls; an exact >= would make
    # this test flaky without catching a real regression.
    for key in ("MultinomialNB_e2e_ms", "LinearSVC_e2e_ms", "LogisticRegression_e2e_ms"):
        assert row[key] >= row["vectorise_ms"] * 0.75, (
            f"{key}={row[key]:.4f}ms below vectorise_ms={row['vectorise_ms']:.4f}ms")


# Shipped artefacts

@pytest.mark.parametrize("task", list(config.TASKS))
def test_bundle_loads_and_predicts(task):
    from naive_bayes_text.config import MODELS_DIR

    path = MODELS_DIR / f"{task}_bundle.joblib"
    if not path.exists():
        pytest.skip(f"{path.name} not built yet; run `python -m naive_bayes_text all`")
    bundle = joblib.load(path)
    assert set(bundle) >= {"task", "model", "threshold", "classes",
                           "label_map", "metrics", "meta"}
    assert bundle["task"] == task
    assert bundle["threshold"] == bundle["meta"]["threshold"]
    proba = bundle["model"].predict_proba(["a short english sentence"])[0]
    assert proba.shape == (len(bundle["classes"]),)
    assert np.isclose(proba.sum(), 1.0, atol=1e-6)


@pytest.mark.parametrize("task", list(config.TASKS))
def test_bundle_survives_hostile_input(task):
    from naive_bayes_text.config import MODELS_DIR

    path = MODELS_DIR / f"{task}_bundle.joblib"
    if not path.exists():
        pytest.skip(f"{path.name} not built yet")
    bundle = joblib.load(path)
    for text in EDGE_CASES:
        assert len(bundle["model"].predict([clean_text(text)])) == 1
